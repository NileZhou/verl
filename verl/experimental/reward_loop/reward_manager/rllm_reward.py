"""
RLLMRewardManager adapted for verl 0.8.0 experimental reward loop interface.
Ported from the original rllm_reward.py in verl 0.5.0.
"""

import os
import sys

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

# Ensure verl root is in sys.path so `rllm` package can be imported
_verl_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _verl_root not in sys.path:
    sys.path.insert(0, _verl_root)

from rllm.rewards.rl_reward import rllm_reward_fn  # noqa: E402


@register("rllm_reward")
class RLLMRewardManager(RewardManagerBase):
    """Reward manager that delegates to rllm reward functions for code/math tasks."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.verbose_debug = os.getenv("VERL_RLLM_REWARD_DEBUG", "0") == "1"

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode full sequence (prompt + response) as in the original implementation
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(sequences)
        )

        # Extract ground truth
        rm_data = data_item.non_tensor_batch["reward_model"]
        if isinstance(rm_data, str):
            import ujson as json
            ground_truth = json.loads(rm_data)["ground_truth"]
        else:
            ground_truth = rm_data["ground_truth"]

        data_source = data_item.non_tensor_batch["data_source"]

        # Debug: log decoded sequences and ground_truth for diagnosis
        if self.verbose_debug:
            import re as _re
            _code_blocks = _re.findall(r"```(?:\w+)?\n(.*?)```", sequences_str, _re.DOTALL)
            print(f"[RLLMReward DEBUG] data_source={data_source}, "
                  f"seq_len={len(sequences_str)}, "
                  f"gt_type={type(ground_truth).__name__}, "
                  f"gt_first100={str(ground_truth)[:100]}, "
                  f"n_code_blocks={len(_code_blocks)}, "
                  f"last_response_500={sequences_str[-500:]}")

        # Compute reward in executor to avoid blocking the event loop
        score = await self.loop.run_in_executor(
            None,
            lambda: rllm_reward_fn(
                data_source=data_source,
                llm_solution=sequences_str,
                ground_truth=ground_truth,
            ),
        )

        reward_extra_info = {}
        if isinstance(score, dict):
            reward = float(score.get("score", 0.0))
            reward_extra_info = score
        elif isinstance(score, bool):
            reward = 1.0 if score else 0.0
            reward_extra_info["acc"] = reward
        else:
            reward = float(score)
            reward_extra_info["acc"] = reward

        if self.verbose_debug:
            print(f"[RLLMReward DEBUG] score={score}, reward={reward}")

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
