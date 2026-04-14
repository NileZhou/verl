# coding=utf-8
"""Evaluation library for instruction following."""

import dataclasses
import re
from typing import Dict, Optional, Union

from .instructions_registry import INSTRUCTION_DICT
from . import instructions_util


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

#严格评测
def test_instruction_following_strict(inp, prompt_to_response):
    """Tests response to see if instructions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # 如果指令需要 prompt_to_repeat，先注入到 kwargs 中再一次性 build
        build_kwargs = dict(inp.kwargs[index])
        args_keys = instruction_cls(instruction_id).get_instruction_args_keys() if hasattr(instruction_cls, 'get_instruction_args_keys') else []
        if "prompt_to_repeat" in build_kwargs or "prompt_to_repeat" in args_keys:
            build_kwargs.setdefault("prompt_to_repeat", inp.prompt)

        # 提取 upper_bound，不传给 build_description（checker 不认识这个参数）
        upper_bound = build_kwargs.pop("upper_bound", None)

        instruction.build_description(**build_kwargs)

        if response.strip() and instruction.check_following(response):
            # 主判断通过后，额外检查 upper_bound（around 展开产生的上界）
            if upper_bound is not None:
                is_following_list.append(
                    _check_upper_bound(instruction_id, build_kwargs, response, upper_bound)
                )
            else:
                is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _check_upper_bound(instruction_id, build_kwargs, response, upper_bound):
    """检查 around 约束的上界。upper_bound 语义是 < upper_bound。"""
    upper_bound = int(upper_bound)

    if instruction_id == "length_constraints:number_sentences":
        return instructions_util.count_sentences(response) < upper_bound
    elif instruction_id == "length_constraints:number_words":
        return instructions_util.count_words(response) < upper_bound
    elif instruction_id == "length_constraints:number_paragraphs":
        return len(response.split("\n\n")) < upper_bound
    elif instruction_id == "length_constraints:letter_counting":
        return len(re.findall(r"[a-zA-Z]", response)) < upper_bound
    elif instruction_id in ("keywords:frequency", "keywords:word_count_different_numbers"):
        keyword = build_kwargs.get("keyword", "")
        return len(re.findall(keyword, response, flags=re.IGNORECASE)) < upper_bound
    elif instruction_id == "keywords:letter_frequency":
        import collections
        letter = build_kwargs.get("letter", "")
        return collections.Counter(response.lower())[letter] < upper_bound
    elif instruction_id == "change_case:capital_word_frequency":
        words = re.findall(r'\b\w+\b', response)
        return len([w for w in words if w.isupper()]) < upper_bound
    else:
        # 未知类型的 instruction，保守返回 True（不因上界阻断）
        return True


#宽松评测
def test_instruction_following_loose(inp, prompt_to_response):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # 如果指令需要 prompt_to_repeat，先注入到 kwargs 中再一次性 build
        build_kwargs = dict(inp.kwargs[index])
        args_keys = instruction_cls(instruction_id).get_instruction_args_keys() if hasattr(instruction_cls, 'get_instruction_args_keys') else []
        if "prompt_to_repeat" in build_kwargs or "prompt_to_repeat" in args_keys:
            build_kwargs.setdefault("prompt_to_repeat", inp.prompt)

        # 提取 upper_bound，不传给 build_description
        upper_bound = build_kwargs.pop("upper_bound", None)

        instruction.build_description(**build_kwargs)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                if upper_bound is not None:
                    if _check_upper_bound(instruction_id, build_kwargs, r, upper_bound):
                        is_following = True
                        break
                else:
                    is_following = True
                    break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )