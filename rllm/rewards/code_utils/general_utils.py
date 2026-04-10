#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/03/20 13:08:39
@Author  :   wangjiakang 
@File    :   utils.py
'''

import ast
import json
import numpy as np
import os
import re
import subprocess
import time
import select
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Optional, Tuple, cast

from rllm.rewards import RewardConfig
reward_config = RewardConfig()

from .general_base import BASE_IMPORTS, BASE_LEETCODE_IMPORTS

_ERROR_MSG_PREFIX = "Execution error: "
_SYNTAX_ERROR_PREFIX = "Syntax error: "
_MAX_CHAR_DISPLAY = 2048
CLI_ARG_SIZE_LIMIT = 1024 * 3
DEFAULT_CODE_EXEC_MEM_MB = int(os.environ.get("CODE_EXEC_MEM_MB", "512"))


def check_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_msg = f"Line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n    {e.text.strip()}"
            if e.offset:
                error_msg += f"\n    {' ' * (e.offset - 1)}^"
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected parsing error: {str(e)}"


def code_exec(code, stdin: str | None = None, timeout=30, mem_limit_mb: int = DEFAULT_CODE_EXEC_MEM_MB):
    # Python syntax checking
    syntax_valid, syntax_error = check_python_syntax(code)
    if not syntax_valid:
        return False, _SYNTAX_ERROR_PREFIX + (syntax_error or "Unknown syntax error")
    
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env = {k: v for k, v in env.items() if not k.startswith("KML")}

    # Apply memory limit (address space and data segment) via prlimit
    memory_limit_bytes = int(mem_limit_mb) * 1024 * 1024
    command = [
        "prlimit",
        f"--as={memory_limit_bytes}",
        f"--data={memory_limit_bytes}",
        "--",
    ]
    # command = ["prlimit", "--as=1073741824", "--"]

    try:
        with TemporaryDirectory() as tmpdir:
            with NamedTemporaryFile(dir="/tmp", suffix=".py") as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.extend(["python", tmp.name])
                result = subprocess.run(command,
                                        cwd=tmpdir,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        timeout=timeout,
                                        env=env,
                                        check=False)

        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode()
        if result.returncode == 0:
            return True, stdout
        return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    except subprocess.TimeoutExpired:
        return False, _ERROR_MSG_PREFIX + "subprocess.TimeoutExpired"
    except Exception as e:
        return False, _ERROR_MSG_PREFIX + repr(e)


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def validate_format(processed_str: str):
    pattern = re.compile(r'<think>.*</think>.*```python\n.*```.*', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


def extract_answer(model_solution):
    pattern = r"```python\n(.*?)```"
    match = re.findall(pattern, model_solution, re.DOTALL)
    if len(match) == 0:
        return None
    else:
        code = match[-1]
        return code


def compute_score(solution_str, ground_truth, data_source='', extro_info=[], is_eval=False, debug=False, return_metadata: bool = False):
    format_correct = validate_format(solution_str) # 必须要有```python ``` ，好像不合理
    if not format_correct:
        extro_info.append("-" * 16 + "Bad format detected!" + "-" * 16)
        extro_info.append("-" * 16 + "Original Model Output" + "-" * 16)
        extro_info.append(solution_str)
        metadata = {"execution_time": None}
        if return_metadata:
            return reward_config.format_error_reward, "\n".join(extro_info), metadata
        return reward_config.format_error_reward, "\n".join(extro_info)

    solution_str = solution_str.split("</think>")[1]
    solution_code = extract_answer(solution_str)
    if solution_code is None:
        extro_info.append("-" * 16 + "Bad format detected!" + "-" * 16)
        extro_info.append("-" * 16 + "Original Model Output" + "-" * 16)
        extro_info.append(solution_str)
        metadata = {"execution_time": None}
        if return_metadata:
            return reward_config.format_error_reward, "\n".join(extro_info), metadata
        return reward_config.format_error_reward, "\n".join(extro_info)

    extro_info.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)

    return grade_answer_code(solution_code, ground_truth, data_source, extro_info, is_eval, debug, return_metadata=return_metadata)


def grade_answer_code(solution_code, ground_truth, data_source, extro_info=[], is_eval=False, debug=False, return_metadata: bool = False):
    t_start = time.time()
    metadata: dict[str, Any] = {"execution_time": None}
    output = ""

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    if "functional" in ground_truth:
        extro_info.append(solution_code + "\n" + ground_truth["functional"])
    else:
        extro_info.append(solution_code)

    if "functional" in ground_truth:
        solution_code = BASE_IMPORTS + "\n" + BASE_LEETCODE_IMPORTS + "\n" + solution_code
        succ, output = code_exec(solution_code + "\n" + ground_truth["functional"])

        if not succ:
            metadata["execution_time"] = time.time() - t_start
            extro_info.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            extro_info.append(output[:_MAX_CHAR_DISPLAY])
            extro_info.append("-" * 16 + "Failed Prompt" + "-" * 16)
            if return_metadata:
                return reward_config.incorrect_reward, "\n".join(extro_info), metadata
            return reward_config.incorrect_reward, "\n".join(extro_info)

    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list = ground_truth["inputs"]
        stdout_list = ground_truth["outputs"]

        if not is_eval: # 默认not is_eval
            max_test_case_num = 16
            selected_indices = range(len(stdin_list))
            if len(stdin_list) > max_test_case_num:
                sample_num = min(max_test_case_num, len(stdin_list))
                half = sample_num // 2
                selected_indices = list(range(half)) + list(range(len(stdin_list)-half, len(stdin_list)))
                selected_indices = selected_indices[:sample_num]
            stdin_list = [stdin_list[i] for i in selected_indices]
            stdout_list = [stdout_list[i] for i in selected_indices]

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(16, len(stdin_list))) as executor:
            futures = [
                executor.submit(remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if isinstance(stdin, list):
                    stdin = ", ".join(map(str, stdin))
                if isinstance(stdout, list):
                    stdout = ", ".join(map(str, stdout))

                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    metadata["execution_time"] = time.time() - t_start
                    extro_info.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    extro_info.append(f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
                    extro_info.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    if return_metadata:
                        return reward_config.incorrect_reward, "\n".join(extro_info), metadata
                    return reward_config.incorrect_reward, "\n".join(extro_info)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    metadata["execution_time"] = time.time() - t_start
    extro_info.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    extro_info.append(output)

    if return_metadata:
        return reward_config.correct_reward, "\n".join(extro_info), metadata
    return reward_config.correct_reward, "\n".join(extro_info)


if __name__ == "__main__":
    ground_truth = json.dumps({
        "functional": '''
def check(candidate):
    assert candidate(nums = [3,1,5,4,2], k = 2) == 4
    assert candidate(nums = [3,1,5,4,2], k = 5) == 5
    assert candidate(nums = [3,2,5,3,1], k = 3) == 4


check(Solution().minOperations)'''
    })

    solution = '''class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        is_added = [False] * k
        count = 0
        n = len(nums)
        for i in range(n - 1, -1, -1):
            if nums[i] > k or is_added[nums[i] - 1]:
                continue
            is_added[nums[i] - 1] = True
            count += 1
            if count == k:
                return n - i
'''
    solution_str = "<think> I am omniscient. </think> ```python\n{}\n```".format(solution)
    score, extro_info = cast(tuple[float, str], compute_score(solution_str, ground_truth, debug=True))

    marker = "✅" if score == reward_config.correct_reward else "❌"
    extro_info = marker * 16 + "Reward Calculation" + marker * 16 + "\n" + extro_info + "\n" + marker * 16 + f"Final Rward = {score}" + marker * 16

    print(extro_info + "\n\n")