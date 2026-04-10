#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/03/20 18:52:57
@Author  :   wangjiakang 
@File    :   general_reward.py
'''

import re
import json
# import math_verify
import string
from typing import Any, List, Union, cast

import multiprocessing
from multiprocessing import Manager

# debug时用
# import sys
# sys.path.append("/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl")

from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

from rllm.rewards.code_utils.general_utils import grade_answer_code
from rllm.rewards.code_utils.general_utils import extract_answer as code_extract_answer

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def postprocess_lcb_sample(sample):
    sample = json.loads(sample)
    sample_inputs = [sample['input'] for sample in sample]
    sample_outputs = [sample['output'] for sample in sample]
    
    sample_dict = {
        'inputs': sample_inputs,
        'outputs': sample_outputs,
    }
    
    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
        # Fill in the blank
        sample_dict['fn_name'] = fn_name
    
    sample = {
        'input_output': json.dumps(sample_dict),
    }
    return sample



class RewardCodeFn(RewardFn):
    """
    Reward function for evaluating code answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig, data_source: str, is_eval: bool = False):
        super().__init__(config)
        self.data_source = data_source
        self.is_eval = is_eval
        self.last_metadata: dict[str, Any] = {"execution_time": None}

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.CODE, \
            "Invalid problem type: expected 'CODE', but got '{}'".format(input.problem_type)
            
        # # dump input # 报错 is not json serializable
        d = {
            "problem": input.problem,
            "model_response": input.model_response,
            "metadata": input.metadata,
        }
        # try:
        #     with open("/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/oexps/odebug/RewardCodeFnInput.json", "w") as f:
        #         json.dump(d, f, indent=4)
        # except Exception as e:
        #     pass

        model_response = input.model_response

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            model_solution = model_response
            # return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # todo extract_answer
        model_answer = code_extract_answer(model_solution)
        if model_answer is None:
            self.last_metadata = {"execution_time": None}
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        ground_truths = input.metadata

        is_correct, extro_info, metadata = cast(
            tuple[float, str, dict[str, Any]],
            grade_answer_code(
                model_answer,
                ground_truths,
                self.data_source,
                is_eval=self.is_eval,
                return_metadata=True,
            ),
        )
        self.last_metadata = metadata

        if is_correct == self.config.correct_reward:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def general_reward_fn(data_source: str, solution_str: str, ground_truth: Union[str, List[str]], extra_info=None, enable_llm=False, is_eval=False,skip_format_reward=False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm

    reward_fn = RewardCodeFn(reward_config, data_source=data_source, is_eval=is_eval)
    problem_type = RewardType.CODE

    reward_response = reward_fn(
        RewardInput(
            problem=solution_str,
            problem_type=problem_type,
            model_response=solution_str,
            data_source=data_source,
            metadata=ground_truth,
        )
    )
    return {
        "is_correct": reward_response.is_correct,
        "execution_time": reward_fn.last_metadata.get("execution_time") if reward_response.is_correct else None,
    }


if __name__ == "__main__":
    solution_str = '...'
    gd = ''
    reward = general_reward_fn("code", solution_str, gd)
    print(reward)
    