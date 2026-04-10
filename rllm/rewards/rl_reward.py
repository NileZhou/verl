from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.code_reward import rllm_reward_fn_code 
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.code_utils.general_reward import general_reward_fn
from typing import Any, List, Union, cast
import json 


class RLRewardFn(RewardFn):
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.math_reward_fn = RewardMathFn(config)
        self.cot_reward_fn = None

    def __call__(self, input: RewardInput) -> RewardOutput:
        reward_type = input.problem_type
        reward = 0
        is_correct = False
        if reward_type == RewardType.MATH:
            math_reward_output = self.math_reward_fn(input)
            reward += self.config.math_reward_weight * math_reward_output.reward
            is_correct = math_reward_output.is_correct
        elif reward_type == RewardType.CODE:
            pass
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        
        if self.config.cot_reward_weight != 0 and self.cot_reward_fn is not None:
            cot_reward_output = self.cot_reward_fn(input)
            reward += self.config.cot_reward_weight * cot_reward_output.reward
        
        return RewardOutput(
            reward=reward,
            is_correct=is_correct
        )

def rllm_reward_fn(data_source: str, solution_str: str = None, ground_truth: Union[str, List[str]] = None, extra_info={}, *, llm_solution: str = None, **kwargs):
    # Accept both solution_str (verl 0.8.0 DAPO) and llm_solution (legacy) as the response string
    llm_solution = solution_str if solution_str is not None else llm_solution
    if data_source in ["apps", "taco", "code_contests", "codeforces", "livecodebench", "kodcode", "leetcode", "primeintellect", "humanevalplus", "am", "oci"]:
        result = rllm_reward_fn_code(data_source, llm_solution, cast(Any, ground_truth), **kwargs)
    elif data_source == 'code':
        result = general_reward_fn(data_source, llm_solution, ground_truth, **kwargs)
    else:
        result = rllm_reward_fn_math(data_source, llm_solution, ground_truth, extra_info, **kwargs)

    # Normalize return value: DAPO reward manager expects either a scalar or dict with "score" key.
    if isinstance(result, dict):
        if "score" not in result:
            result["score"] = float(result.get("is_correct", 0))
        return result
    else:
        return float(result)

