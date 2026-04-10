# Taken from https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/evaluation/testing_util.py
import ast
from ftplib import all_errors
import json
import sys
import faulthandler
import platform
import multiprocessing
import queue
import re
import difflib  # 添加这个import用于计算编辑距离

import numpy as np
# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal


from io import StringIO

# used for testing the code that reads from input
from unittest.mock import patch, mock_open

# from pyext import RuntimeModule
from types import ModuleType

from enum import Enum
from decimal import Decimal
import time

from .utils import BASE_IMPORTS

import_string = BASE_IMPORTS
 
#"from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"

method_pattern = re.compile(r"def\s+([a-zA-Z_]\w*)\s*\((.*?)\)\s*:")


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None: # cat /proc/sys/fs/file-nr
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    all_errors = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            # handle floating point comparisons
            if tmp_result:
                all_results.append(1)
            else:
                # WA
                all_results.append(0)
                all_errors.append("Wrong Answer")

        except Exception as e:
            signal.alarm(0)
            if isinstance(e, TimeoutException):
                # TLE
                all_results.append(0)
                all_errors.append("Time Limit Exceeded")
            else:
                # Unkonwn Error
                all_results.append(0)
                all_errors.append(e)

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, all_errors, {"execution time": total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        print("compiled_sol is None")
        return

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        print("method is None")
        return

    all_results = []
    all_errors = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if isinstance(e, TimeoutException):
                    # TLE
                    all_results.append(0)
                    all_errors.append("Time Limit Exceeded")
                else:
                    # Unkonwn Error
                    all_results.append(0)
                    all_errors.append(e)
                continue

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            # WA
            all_results.append(0)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            all_errors.append(WA_send_args["error_message"])
            continue

        flag = 0
        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                # WA
                all_results.append(0)
                all_errors.append(WA_send_args["error_message"])
                flag = 1
                break
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                # WA
                all_results.append(0)
                all_errors.append(WA_send_args["error_message"])
                flag = 1
                break

            if abs(decimal_prediction_line - decimal_gtout_line) < 1e-6:
                continue
            
            # WA
            all_results.append(0)
            all_errors.append(WA_send_args["error_message"])
            flag = 1
            break
        if flag == 1:
            continue
        all_results.append(1)

    return all_results, all_errors, {"execution time": total_execution_time}


def find_all_functions(code: str) -> list[str]:
    """
    使用AST找到代码中所有的函数定义
    """
    try:
        tree = ast.parse(code)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return functions
    except:
        # 如果AST解析失败，回退到正则表达式
        return re.findall(r"def\s+([a-zA-Z_]\w*)\s*\(", code)

def find_best_matching_function(code: str, target_method_name: str) -> str:
    """
    在代码中找到与目标函数名最接近的函数
    """
    functions = find_all_functions(code)
    
    if not functions:
        return None
    
    # 过滤掉特殊方法（如__init__等）
    candidate_functions = [f for f in functions if not f.startswith('__')]
    
    if not candidate_functions:
        candidate_functions = functions  # 如果没有非特殊方法，就用所有函数
    
    # 计算编辑距离，找到最接近的函数
    best_match = None
    best_ratio = 0
    
    for func_name in candidate_functions:
        # 使用SequenceMatcher计算相似度
        ratio = difflib.SequenceMatcher(None, func_name, target_method_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = func_name
    
    return best_match

def run_test(tests, code=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
    except ValueError:
        pass  # Not in main thread

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(tests["input_output"])
    except ValueError as e:
        raise e
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]
            # todo老问题：prompt里都没有函数名，人家怎么做的对?  如果没有这段，LCB v6成绩会大幅下降
            if (' ' + method_name + '(') not in code: # 如果code里的method_name对不上，就换成想要的method_name。 空格是防止resp里的func名的后缀为目标method_name 此种特殊case，应替换的没被替换; (是防止前缀为包含目标method_name
                # 找到最接近的函数名
                print(f"not found {method_name} in code: {code}")
                best_match = find_best_matching_function(code, method_name)
                
                if best_match:
                    print(f"Warning: Could not find method '{method_name}' in code, but find closet '{best_match}', replace target to this function")
                    method_name = best_match  # 直接修改method_name
                    in_outs['fn_name'] = method_name
                else:
                    print(f"Warning: No suitable function found in code to replace with '{method_name}'")

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if code is None:
        assert False, "should not happen: test code is none"
        return in_outs, {"error": "no test code provided"}
    elif code is not None:
        results = []
        sol = import_string
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            signal.alarm(timeout)
            try:
                results, errors, metadata = grade_call_based(
                    code=code,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    fn_name=method_name,
                    timeout=timeout,
                )
                return results, errors, metadata
            except Exception as e:
                # print(f"Error during testing: {e}")
                return [-5], [e], {
                    "error_code": -5,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it
            signal.alarm(timeout)
            try:
                results, errors, metadata = grade_stdio(
                    code=code,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    timeout=timeout,
                )
                return results, errors, metadata
            except Exception as e:
                # print(f"Error during testing: {e}")
                return [-5], [e], {
                    "error_code": -5,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    # Keep putenv available: os.environ assignment and first-time numpy/pandas
    # imports depend on it, especially in the subprocess-based LCB runner.
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
    
    
