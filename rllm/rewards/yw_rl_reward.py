from rllm.rewards.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.code_reward import rllm_reward_fn_code 
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.if_reward import rllm_reward_fn_instruct
from typing import Union, List, Dict
import json 
import re

# Check RewardConfig to understand the config values.
# 这个没用，走的是rllm_reward_fn 这个函数，各自的类别走各自的
class RLRewardFn(RewardFn): 
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.math_reward_fn = RewardMathFn(config)
        self.instruct_reward_fn = RewardInstructFn(config)
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
        elif reward_type == RewardType.INSTRUCT:
            instruct_reward_output = self.instruct_reward_fn(input)
            reward += self.config.instruct_reward_weight * instruct_reward_output.reward #instruct_reward_weight 决定单科在总分中的占比（rl_reward 混合用）
            is_correct = instruct_reward_output.is_correct
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        
        if self.config.cot_reward_weight != 0 and self.cot_reward_fn is not None:
            cot_reward_output = self.cot_reward_fn(input)
            reward += self.config.cot_reward_weight * cot_reward_output.reward
        
        return RewardOutput(
            reward=reward,
            is_correct=is_correct
        )

def rllm_reward_fn(data_source: str, llm_solution: str, ground_truth: Union[str, List[str], Dict], extra_info={}, **kwargs):
    #代码类数据源 → 走代码奖励，没用
    if data_source in ["apps", "taco", "code_contests", "codeforces", "livecodebench", "kodcode", "leetcode", "primeintellect", "humanevalplus"]:
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False 
        return rllm_reward_fn_code(data_source, llm_solution, ground_truth, **kwargs)
    
    elif data_source in ["ifeval", "instruction_following"]:
        prompt = ""
        response = llm_solution
        assistant_pattern = r'<\|im_start\|>assistant\s*(?:<think>.*?</think>)?\s*(.*?)(?:<\|endoftext\|>|$)' #抓模型对用户的可见回复（response，自动忽略可选的 <think>...</think> 思维链）
        user_pattern = r'<\|im_start\|>user\n(.*?)<\|im_end\|>'#抓用户给模型的指令/提示（prompt 
        user_match = re.search(user_pattern, llm_solution, re.DOTALL)
        response_match = re.search(assistant_pattern, llm_solution, re.DOTALL)
        
        #print(llm_solution)
        
        if response_match:
            response = response_match.group(1).strip()
            #print("+"*80)
            #print("Extracted response:", response)
        else:
            print("-"*80)
            print("Warning: Do not extract valiable **response**. May leading to wrong reward for instruct following")
        
        if user_match:
            prompt = user_match.group(1).strip()
            # print("+"*80)
            # print("Extracted prompt:", prompt)
        else:
            print("-"*80)
            print("Warning: Do not extract valiable prompt. May leading to wrong reward for instruct following")
        return rllm_reward_fn_instruct(data_source, response, ground_truth, prompt, **kwargs)
    else:
        return rllm_reward_fn_math(data_source, llm_solution, ground_truth, extra_info, **kwargs)