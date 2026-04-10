"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""
import json
import multiprocessing
import re
import time
from multiprocessing import Manager
from typing import Any, Dict, List, Union, cast
import random
import ast 
import subprocess
from tempfile import TemporaryDirectory
import os
import sys

#from rllm.rewards.code_utils.code_contests import run_test as code_contests_run_test
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.codeforces import run_test as codeforces_run_test
#from rllm.rewards.code_utils.swebench import swebench_check_correctness
from rllm.rewards.code_utils.humanevalplus import run_test as humanevalplus_run_test, get_num_test_cases
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.code_utils.firejail_exec import code_exec_firejail as lc_code_exec
from rllm.rewards.code_utils.kodcode import code_exec as kod_code_exec
from rllm.rewards.code_utils.opencodeinstruct import oci_run_assert_tests
from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardResult, RewardType


def extract_code_from_model(model_response: str) -> str | None:
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    
    # for code_block in code_blocks:
    #     if "def " in code_block:
    #         return code_block.strip()
    
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split('\n')
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def check_correctness(
    tests: Union[List[Dict[str, Any]], Dict[str, List[str]]],
    code: str,
    test_fn,
    timeout_per_test: int = 5,
    max_tests: int = 15,
) -> bool:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        bool: True if all tests pass, False otherwise

    Raises:
        AssertionError: If test results list is empty
    """
    manager = Manager()
    test_results = manager.list()
    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug, timeout=timeout_per_test))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")
    if isinstance(tests, list):
        tests_list = cast(List[Dict[str, Any]], tests)
        total_tests = len(tests_list)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(str(tests_list[i]["input"])), reverse=True)[:max_tests]
            tests_list = [tests_list[i] for i in selected_indices]
        tests = tests_list
    else:
        tests_dict = cast(Dict[str, List[str]], tests)
        total_tests = len(tests_dict['inputs'])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(tests_dict['inputs'][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests = {
                'inputs': [tests_dict['inputs'][i] for i in selected_indices],
                'outputs': [tests_dict['outputs'][i] for i in selected_indices]
            }
            tests = selected_tests
    
    process = multiprocessing.Process(
        target=evaluate_code,
        args=(tests, code, False, test_results, test_fn)
    )
    process.start()
    process.join()

    if process.is_alive():
        process.kill()
    test_results = test_results[:]
    if len(test_results) == 0:
        return False
    #assert len(test_results) == 1, f"Expected exactly one test result, but got {test_results}"
    test_results = test_results[0]
    test_results = [r==True for r in test_results]
    return all(test_results)


def oci_check_correctness(tests: object, code: str, timeout_per_test: int = 5, max_tests: int = 15) -> bool:
    """
    Check if generated code passes all OCI assert-based test cases.
    
    Args:
        tests: JSON string containing a list of assert statements
        code: Generated code to test
        timeout_per_test: Maximum execution time per test in seconds
        max_tests: Maximum number of tests to run
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        if isinstance(tests, str):
            test_statements = json.loads(tests)
        else:
            test_statements = tests
            
        if not isinstance(test_statements, list):
            print(f"Expected list of test statements, got {type(test_statements)}")
            return False
        if not all(isinstance(statement, str) for statement in test_statements):
            print("Expected all OCI test statements to be strings")
            return False
        test_statements = cast(List[str], test_statements)
            
        if len(test_statements) == 0:
            print("No test statements found")
            return False
            
        # Limit the number of tests if needed
        if len(test_statements) > max_tests:
            # Select the first max_tests statements
            test_statements = test_statements[:max_tests]
            
        # Clean up the code by removing main block if present
        code = clean_code_main_block(code)
        
        return oci_run_assert_tests(test_statements, code, timeout_per_test)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing test JSON: {e}")
        return False
    except Exception as e:
        print(f"Error in oci_check_correctness: {e}")
        return False

# todo timeout_per_test改回5
def am_distill_r1_check_correctness(tests_str: object, code: str, timeout_per_test: int = 5, max_tests: int = 15) -> bool:  
    if not isinstance(tests_str, str):
        print(f"Expected string tests for am, got {type(tests_str)}")
        return False
    tests = json.loads(json.loads(tests_str))
    
    def transfer_dict(d: Dict):
        if d['call_type'] != 'assert':
            return d
        inputs = []
        outputs = []
        fn_name = d['fn_name']
        for assert_case in d['assert_case']:
            # 解析assert语句，提取输入和输出
            try:
                # 移除assert关键字
                expr = assert_case.strip().replace('assert ', '')
                # 使用正则表达式匹配函数调用和期望输出
                # 匹配格式: function_name(args) == expected_output
                pattern = rf'{re.escape(fn_name)}\((.*?)\)\s*==\s*(.+)'
                match = re.match(pattern, expr)
                if match:
                    args_str = match.group(1).strip()
                    expected_output_str = match.group(2).strip()
                    # 安全地评估输入参数
                    try:
                        # 对于简单的参数，直接eval
                        input_value = ast.literal_eval(args_str)
                        if isinstance(input_value, tuple):
                            inputs.append('\n'.join([f'"{word}"' if isinstance(word, str) else json.dumps(word) for word in input_value]))
                        else:
                            inputs.append(json.dumps(input_value))
                    except:
                        # 如果eval失败，保持原字符串
                        inputs.append(args_str)
                    # 安全地评估期望输出
                    try:
                        output_value = ast.literal_eval(expected_output_str)
                        outputs.append(json.dumps(output_value))
                    except:
                        # 如果eval失败，保持原字符串
                        outputs.append(expected_output_str)
                else:
                    # 如果正则匹配失败，尝试其他解析方法
                    # 处理其他可能的assert格式
                    inputs.append(assert_case)
                    outputs.append(None)
            except Exception as e:
                print(f"Error parsing assert case: {assert_case}, error: {e}")
                inputs.append(assert_case)
                outputs.append(None)
        return {
            "call_type": "functional",
            "fn_name": fn_name,
            "inputs": inputs,
            "outputs": outputs
        }
    
    # tests = json.loads(tests_str)
    tests = transfer_dict(tests)
    # 转换格式，每个单测为一个dict,包含 'input' 和 'output' 这两个key
    tests = [{'input': input_sample, 'output': output_sample, 'testtype': tests['call_type'], 'metadata': {'func_name': tests.get('fn_name', None)}} for input_sample, output_sample in zip(tests['inputs'], tests['outputs'])]

    return lcb_check_correctness_v3(tests, code, timeout=timeout_per_test, debug=False)


def postprocess_lcb_sample(samples):
    for i in range(len(samples)):
        sam = samples[i]
        if isinstance(sam, str) and sam.strip().startswith("{"):
            try:
                samples[i] = json.loads(sam)
                print(f'convert sample: {samples[i]} to dict')
            except json.JSONDecodeError:
                pass
        
    sample_inputs = [sample['input'] for sample in samples]
    sample_outputs = [sample['output'] for sample in samples]
    
    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }
    
    if samples[0].get("testtype") == "functional":
        metadata = samples[0].get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        # Fill in the blank
        sample_dict['fn_name'] = fn_name
    
    sample = {
        'input_output': json.dumps(sample_dict),
    }
    return sample

# https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems
def primeintellect_check_correctness(tests, code):
    if isinstance(tests, str):
        try:
            tests =  ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {e}")
            return False

    assert len(tests) >= 1, "PrimeIntellect needs at least one test case"
    # Convert the tests to the format expected by the taco_run_test function
    inputs = [t['input'] for t in tests]
    outputs = [t['output'] for t in tests]
    fn_name = tests[0].get('fn_name', None)
    tests = {
        'inputs': inputs,
        'outputs': outputs,
    }
    if fn_name:
        tests['fn_name'] = fn_name
    return check_correctness(tests, code, taco_run_test)

def lcb_check_correctness_v2(sample, generation, timeout=5, debug=False):
    """某道题维度，针对多个单测的多进程校验，没法在外面套多进程"""
    if isinstance(sample, str) and sample.strip().startswith("["):
        sample = json.loads(sample)
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        run_result = lcb_run_test(sample, code=generation, debug=debug, timeout=timeout)
        if run_result is None:
            return
        res, metadata = run_result
        result.append(res)
        metadata_list.append(metadata)

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    if not result:
        return False
    # print(result[0], metadata_list)
    # Check if all elements in result[0] are True
    return all(x == True for x in result[0])

def lcb_check_correctness_v4(sample, generation, timeout=5, debug=False):
    """
    单测维度，单进程校验，可在外套多进程
    """
    if isinstance(sample, str) and sample.strip().startswith("["):
        sample = json.loads(sample)
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    global_timeout = (timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5

    with TemporaryDirectory() as tmpdir:
        # Prepare data for the subprocess
        input_data = {
            "sample": sample,
            "generation": generation,
            "timeout": timeout,
            "debug": debug,
        }
        input_path = os.path.join(tmpdir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_data, f)

        # This script will be run in a separate process
        runner_script = """
import json
import sys
import os
from rllm.rewards.code_utils.s_livecodebench import run_test as lcb_run_test

def main():
    input_path = sys.argv[1]
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    res, err, _ = lcb_run_test(
        data['sample'], 
        code=data['generation'], 
        debug=data['debug'], 
        timeout=data['timeout']
    )
    
    # We only care about the results, not metadata
    print(json.dumps({'result': res, 'error': err}))

if __name__ == "__main__":
    main()
"""
        script_path = os.path.join(tmpdir, "runner.py")
        with open(script_path, "w") as f:
            f.write(runner_script)
        
        # Set up environment for subprocess
        env = os.environ.copy()
        # The project root is two directories up from the rllm module's path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = project_root
        else:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"

        try:
            command = [sys.executable, script_path, input_path]
            process = subprocess.run(
                command,
                timeout=global_timeout,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            
            if process.returncode == 0 and process.stdout:
                last = process.stdout.strip().splitlines()[-1].strip().strip("'")
                data = json.loads(last) 
                return data
            
            if debug and process.returncode != 0:
                print(f"Subprocess for lcb_check_correctness_v4 failed with return code {process.returncode}")
                print(f"Stderr: {process.stderr}")
            return False

        except subprocess.TimeoutExpired:
            if debug:
                print(f"Global timeout of {global_timeout}s expired for lcb_check_correctness_v4.")
            return False
        except Exception as e:
            if debug:
                print(f"An exception occurred in lcb_check_correctness_v4: {e}")
            return False

def lcb_check_correctness_v3(sample, generation, timeout=5, debug=False):
    """
    单测维度，单进程校验，可在外套多进程
    """
    if isinstance(sample, str) and sample.strip().startswith("["):
        sample = json.loads(sample)
    assert len(sample) >= 1, "Sample must contain at least one test case"
    sample = postprocess_lcb_sample(sample)

    global_timeout = (timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5

    with TemporaryDirectory() as tmpdir:
        # Prepare data for the subprocess
        input_data = {
            "sample": sample,
            "generation": generation,
            "timeout": timeout,
            "debug": debug,
        }
        input_path = os.path.join(tmpdir, "input.json")
        with open(input_path, "w") as f:
            json.dump(input_data, f)

        # This script will be run in a separate process
        runner_script = """
import json
import sys
import os
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test

def main():
    input_path = sys.argv[1]
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    res, metadata = lcb_run_test(
        data['sample'], 
        code=data['generation'], 
        debug=data['debug'], 
        timeout=data['timeout']
    )
    
    # We only care about the results, not metadata
    print(json.dumps({'result': res, 'metadata': metadata}))

if __name__ == "__main__":
    main()
"""
        script_path = os.path.join(tmpdir, "runner.py")
        with open(script_path, "w") as f:
            f.write(runner_script)
        
        # Set up environment for subprocess
        env = os.environ.copy()
        # The project root is two directories up from the rllm module's path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = project_root
        else:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"

        try:
            command = [sys.executable, script_path, input_path]
            process = subprocess.run(
                command,
                timeout=global_timeout,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            
            if process.returncode == 0 and process.stdout:
                last = process.stdout.strip().splitlines()[-1].strip().strip("'")
                data = json.loads(last)
                if debug:
                    print(process.stdout)
                is_correct = all(x is True for x in data['result'])
                execution_time = data.get('metadata', {}).get('execution time') if is_correct else None
                return {'is_correct': is_correct, 'execution_time': execution_time}
            
            if debug and process.returncode != 0:
                print(f"Subprocess for lcb_check_correctness_v3 failed with return code {process.returncode}")
                print(f"Stderr: {process.stderr}")
            return {'is_correct': False, 'execution_time': None}

        except subprocess.TimeoutExpired:
            if debug:
                print(f"Global timeout of {global_timeout}s expired for lcb_check_correctness_v3.")
            return {'is_correct': False, 'execution_time': None}
        except Exception as e:
            if debug:
                print(f"An exception occurred in lcb_check_correctness_v3: {e}")
            return {'is_correct': False, 'execution_time': None}


def leetcode_check_correctness(tests: object, code: str) -> bool:
     """
     Check if generated code passes all LeetCode test cases.
    
     Args:
          tests: List of test cases, each containing input/output pairs
          code: Generated code to test
          timeout: Maximum execution time in seconds before killing process
          runtime_debug: Whether to print debug info during test execution
    
     Returns:
          bool: True if all tests pass and result list exists, False otherwise
     """
     if not isinstance(tests, dict):
         print(f"Expected dict tests for leetcode, got {type(tests)}")
         return False
     functional_test = tests.get("functional")
     if not isinstance(functional_test, str):
         print("Missing functional test for leetcode")
         return False
     succ, output = lc_code_exec(code + '\n' + functional_test)
     if not succ:
         print(f"Error in code execution: {output}")
     return succ

def kodcode_check_correctness(test: object, code: str, timeout_per_test: int = 5) -> bool:
    """
    Check if generated code passes all Kodcode test cases.
    
    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution
    
    Returns:
        bool: True if all tests pass and result list exists, False otherwise
    """
    if not isinstance(test, str):
        print(f"Expected string tests for kodcode, got {type(test)}")
        return False

    # Count the number of test functions in the test file
    num_tests = test.count('def test')

    # Remove 'if __name__ == "__main__":' block if present
    code = clean_code_main_block(code)
    
    succ, output = kod_code_exec(code, test, timeout_per_test * num_tests)
    if not succ:
        print(f"Error in code execution: {output}")
    return succ

def humanevalplus_check_correctness(test: object, code: str, timeout_per_test: int = 1) -> bool:
    """
    Check if generated code passes all HumanEvalPlus test cases.
    
    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution
    
    Returns:
        bool: True if all tests pass and result list exists, False otherwise
    """
    if not isinstance(test, str):
        print(f"Expected string tests for humanevalplus, got {type(test)}")
        return False

    code = clean_code_main_block(code)

    num_test_cases = get_num_test_cases(test)
    if not isinstance(num_test_cases, int):
        print(f"Failed to infer humanevalplus test count: {num_test_cases}")
        return False
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)
    if not succ:
        print(f"Error in code execution: {output}")
    return succ

class RewardCodeFn(RewardFn):
    """
    Reward function for evaluating code dataset answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the unit tests provided
    """
    def __call__(self, input: RewardInput) -> RewardResult:
        total_start_time = time.time()

        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)

        model_response = input.model_response
        metadata = input.metadata
        
        dataset_name = input.data_source
        tests = metadata
        if tests is None:
            print("No tests found in metadata")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            # print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        if dataset_name in ["taco", "apps", "code_contests"]:
            if not isinstance(tests, (list, dict)):
                print(f"Unexpected tests type for {dataset_name}: {type(tests)}")
                return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
            test_fn = taco_run_test
            is_correct = check_correctness(tests, model_code, test_fn)
        elif dataset_name == "codeforces":
            if not isinstance(tests, (list, dict)):
                print(f"Unexpected tests type for {dataset_name}: {type(tests)}")
                return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
            test_fn = codeforces_run_test
            is_correct = check_correctness(tests, model_code, test_fn)
        elif dataset_name == "leetcode":
            is_correct = leetcode_check_correctness(tests, model_code)
        elif dataset_name == "livecodebench":
            is_correct = lcb_check_correctness_v3(tests, model_code, debug=False) # todo: 改为v2
        elif dataset_name == "livecodebench_maj":
            is_correct = lcb_check_correctness_v4(tests, model_code, debug=False)
        elif dataset_name == "primeintellect":
            is_correct = primeintellect_check_correctness(tests, model_code)
        elif dataset_name == "kodcode":
            is_correct = kodcode_check_correctness(tests, model_code)
        elif dataset_name == "humanevalplus":
            is_correct = humanevalplus_check_correctness(tests, model_code)
        elif dataset_name == "am":
            is_correct = am_distill_r1_check_correctness(tests, model_code)
        elif dataset_name == "oci":
            is_correct = oci_check_correctness(tests, model_code)
        else:
            raise ValueError(f"Unsupported code dataset: {dataset_name}")

        total_time = time.time() - total_start_time
        # print(f"Total reward function execution time: {total_time:.2f} seconds")

        if is_correct and isinstance(is_correct, bool):
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        elif isinstance(is_correct, dict):
            return is_correct
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: object, **kwargs):
    """Evaluate code solutions against ground truth ansters
    
    This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: some tests for this llm_solution
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution passes all the test_case, False otherwise

    Example:
            model_response = '''
import sys
from itertools import permutations
def main():
    n,m=map(int, input().split()) 
    a=sum(list(map(int, input().split()))) 
    if a+(n-1)*10<=m: 
        print(5) 
    else: 
        print(5)
if __name__ == "__main__":
    main()
'''
    
    print(f"test the code_forces")
    # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ] 
    metadata = {
         "tests": tests,
    }
    True
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    try:
        reward_response = reward_fn(
            RewardInput(
                problem=None,
                problem_type=RewardType.CODE,
                data_source=data_source,
                model_response=llm_solution,
                metadata=ground_truth
            ))
        if isinstance(reward_response, dict):
            return reward_response
        return reward_response.is_correct
    except Exception as e:
        print(f"Error in reward function: {e}, data_source: {data_source}")
        
        # 留着用来debug
        debug_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(debug_dir, '1.txt'), 'w') as f:
            json.dump(ground_truth, f, ensure_ascii=False)
        with open(os.path.join(debug_dir, '1_gen.txt'), 'w') as f:
            json.dump(llm_solution, f, ensure_ascii=False)
        
        raise e
  
  
  
if __name__ == '__main__':
    # python -m rllm.rewards.code_reward
    
    # def postprocess_lcb_sample(sample):
    #     sample_inputs = [sample['input'] for sample in sample]
    #     sample_outputs = [sample['output'] for sample in sample]
        
    #     sample_dict = {
    #         'inputs': sample_inputs,
    #         'outputs': sample_outputs,
    #     }
        
    #     if sample[0].get("testtype") == "functional":
    #         metadata = sample[0].get("metadata", {})
    #         fn_name = metadata.get("func_name", None)
    #         # import ipdb; ipdb.set_trace()
    #         assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
    #         # Fill in the blank
    #         sample_dict['fn_name'] = fn_name
    #         print(f'extract fn_name: {fn_name}\n\n')
        
    #     sample = {
    #         'input_output': json.dumps(sample_dict),
    #     }
    #     return sample
    
    
    # with open('/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/rllm/rewards/1_tests.txt', 'r') as f:
    #     sample = json.load(f) 
    #     sample = postprocess_lcb_sample(sample)
    # # with open('/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/rllm/rewards/1_gen.txt', 'r') as f:
    # #     generation = json.load(f)
    # with open('/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/rllm/rewards/1_gen.txt', 'r') as f:
    #     generation = f.read()
    #     generation = extract_code_from_model(generation)
    #     print(f'extract code: \n{generation}\n\n')

    # ret = lcb_run_test(sample, code=generation, debug=False, timeout=6000)
    # print(ret)

    data_source = "livecodebench"
    llm_solution = """read from """
    ground_truth = [{"input": "3 4 1\nH...\n#..H\n.#.#", "output": "5", "testtype": "stdin"}, {"input": "5 6 2\n##...H\nH.....\n..H.#.\n.HH...\n.###..", "output": "21", "testtype": "stdin"}, {"input": "1 6 3\n...#..", "output": "0", "testtype": "stdin"}, {"input": "13 14 110\n.#..#H.##..#.H\n#....##..###H.\n.#..H###.#..#.\n##..H.H..#....\n.H#......H.H##\n#H.##.HH......\nH.#H##...HH...\n..#.#..#H##.H.\n.H..H...###H#.\n.H.H.#....#.#H\n#..#HHH##..#.H\nH..H..H#.H##..\n.....H..###..#", "output": "125\n", "testtype": "stdin"}, {"input": "4 5 8\n..H..\n.#..#\n#H..#\n...##", "output": "14\n", "testtype": "stdin"}, {"input": "5 3 4\n...\n#.#\n.#.\n.#H\n...", "output": "6\n", "testtype": "stdin"}, {"input": "12 14 19\n#..##.#H..#H.#\n##..#.H....##.\n#H.H##..HH.H##\n.#........#...\nH...H##..H.##.\nH#..H......H..\n#H#...###.#...\n##.#.#.#.H#H.H\n#.......#...##\n#.H..##.#..#..\n#H...###.H...H\n#...#..H......", "output": "117\n", "testtype": "stdin"}, {"input": "2 4 6\n.#..\n....", "output": "0\n", "testtype": "stdin"}, {"input": "2 1 1\n.\n#", "output": "0\n", "testtype": "stdin"}, {"input": "2 1 1\n.\nH", "output": "2\n", "testtype": "stdin"}, {"input": "4 4 10\n....\n...#\n#H#.\n....", "output": "13\n", "testtype": "stdin"}, {"input": "3 4 2\nHHHH\nHHHH\nHHHH", "output": "12\n", "testtype": "stdin"}, {"input": "1 5 1\n.#.#.", "output": "0\n", "testtype": "stdin"}, {"input": "5 5 7\n#....\n...H#\n.....\n##.H.\n....H", "output": "21\n", "testtype": "stdin"}, {"input": "12 15 28\n.H####H#HH...#.\n.###...#......#\nHH.##...H...##.\n##.......#.....\n...#.##H####.H#\n.###...##.#...#\n....##.#....#..\n#...###.##.#.H.\n.#....#HH#H#..#\nHH#H..H##......\n#...#..#H.H.##.\n...#H..H..#.###", "output": "117\n", "testtype": "stdin"}, {"input": "1 1 0\n#", "output": "0\n", "testtype": "stdin"}, {"input": "15 12 88\n#.##.H.##HH.\n...#.##..#.H\n.#...#.####.\nH#.H#H.H..H.\nHH.....#H...\n##.H#.#.H..#\nH###....#..#\n.#..#...#.H#\n...#..H.....\n.#.....HH.H.\n..#.#.####..\n..#H....H.#.\n...HH.##.#..\n#......#.#H#\n.#.HH..H....", "output": "127\n", "testtype": "stdin"}, {"input": "3 4 12\n.#..\nH#..\nH...", "output": "10\n", "testtype": "stdin"}, {"input": "13 12 90\n..#....H#...\n##..##..##.#\nH###.....#.#\n#H.....H##.#\n..H.#...#.##\n.H#HHHH.....\n#..#H.#...##\n.....H.#..#.\n...H..H##.##\n.####...##.H\n..H#..#.###.\n.#H.#.H#HH##\n..#H.#...##H", "output": "93\n", "testtype": "stdin"}, {"input": "5 5 4\n..##.\n....H\n.#...\n....#\n...H.", "output": "19\n", "testtype": "stdin"}, {"input": "1 2 1\n..", "output": "0\n", "testtype": "stdin"}, {"input": "3 5 0\nH....\n.#..#\n..##.", "output": "1\n", "testtype": "stdin"}, {"input": "1 4 3\n....", "output": "0\n", "testtype": "stdin"}, {"input": "3 3 2\n###\n###\n###", "output": "0\n", "testtype": "stdin"}, {"input": "8 7 9\n.#.#...\n#...##.\n#.###..\n.......\n.H##H..\n.H..H#.\n.#...HH\n.#.H##.", "output": "39\n", "testtype": "stdin"}, {"input": "1 2 1\n.#", "output": "0\n", "testtype": "stdin"}, {"input": "5 4 2\n#...\n....\n..#.\nH...\n....", "output": "8\n", "testtype": "stdin"}, {"input": "4 4 5\n....\n#..#\n....\n.##.", "output": "0\n", "testtype": "stdin"}, {"input": "1 2 2\n.#", "output": "0\n", "testtype": "stdin"}, {"input": "4 5 3\nH#.#.\n.....\n.#..#\n.....", "output": "6\n", "testtype": "stdin"}, {"input": "10 8 11\n#....#.#\n##..#..H\nH..#H..#\n#H#.#.#.\n#.#..#.#\n.#..H#..\n.#.#...H\n#H...#.H\n.#...##.\n#####...", "output": "46\n", "testtype": "stdin"}, {"input": "12 15 61\n#.H.H.##H..##.#\n.#....##......#\n###...#.#.###.#\nH##..#..#.#...#\n####H#H.H#....#\nH.###H#.....#..\n#.#H.....##...#\n.HH.#..#H..H...\n#.H.#....##..##\n###.H.H...#....\n.H.H.#.#..##.##\nH.#...####....H", "output": "114\n", "testtype": "stdin"}, {"input": "5 5 13\n.#.##\n#....\n...#.\n.##..\n..#..", "output": "0\n", "testtype": "stdin"}, {"input": "14 13 124\n..#.#...H.#.H\nH.H..#.#HH.H.\nH#H#.#H..#.H.\n.H......#####\n..H.....#.#.#\nH.....H..##.H\n.#####.....##\nH##H.........\n#....H#.H..#.\n#H....##HH...\n#.HH.H.####..\n##..H##.H#..H\n.H.H.##HH.#.#\n......####..#", "output": "128\n", "testtype": "stdin"}, {"input": "3 5 4\n.#..H\n.#...\n.###.", "output": "7\n", "testtype": "stdin"}, {"input": "2 4 3\nH...\n.#.#", "output": "6\n", "testtype": "stdin"}, {"input": "10 8 37\nHH.###H.\n.#.#####\n.......H\n.....HH.\n.#....##\n#....##.\n.H.....#\nH..#H...\n#....H..\n#H.##..#", "output": "57\n", "testtype": "stdin"}, {"input": "9 9 15\n.#H.....#\n......#..\n..H###...\n...#.#...\n......#..\n...#.....\n.#.......\n.###...H.\n#.#.#.###", "output": "59\n", "testtype": "stdin"}, {"input": "5 3 5\nH..\n#..\n...\n#..\n...", "output": "11\n", "testtype": "stdin"}, {"input": "7 8 2\n.#.#.#..\n......H.\n.###.H..\n.....H##\n#..HH...\n..H...##\n...H...#", "output": "31\n", "testtype": "stdin"}, {"input": "3 4 0\n.#.#\n.#..\n...#", "output": "0\n", "testtype": "stdin"}, {"input": "1 1 0\n.", "output": "0\n", "testtype": "stdin"}, {"input": "1 2 0\n..", "output": "0\n", "testtype": "stdin"}]
    
    passed = rllm_reward_fn_code(data_source, llm_solution, ground_truth)
    print(passed)
    