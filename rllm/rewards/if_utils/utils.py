"""
Utility functions for instruction following evaluation.
This module provides helper functions to evaluate instruction following independently.
"""
from typing import List, Dict, Any

# Import from our local modules
from .instructions_registry import INSTRUCTION_DICT
from .evaluation_lib import InputExample, test_instruction_following_strict, OutputExample

#把原始参数组装成标准输入对象
def create_input_example(
    prompt: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
    key: int = 0
) -> InputExample:
    """
    Create an InputExample for instruction following evaluation.
    
    Args:
        prompt: The original prompt/problem text
        instruction_id_list: List of instruction IDs to evaluate
        kwargs: List of keyword arguments for each instruction
        key: Unique identifier for the example
    
    Returns:
        InputExample object ready for evaluation
    """
    if len(instruction_id_list) != len(kwargs):
        raise ValueError(f"instruction_id_list length ({len(instruction_id_list)}) "
                        f"must match kwargs length ({len(kwargs)})")
    
    return InputExample(
        key=key,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs
    )

#对某个 prompt/response 按给定的指令集进行评测。
def evaluate_instruction_following(
    prompt: str,
    response: str,
    instruction_id_list: List[str],
    kwargs: List[Dict[str, Any]],
    strict: bool = True
) -> OutputExample:
    """
    Evaluate instruction following for a given prompt and response.
    
    Args:
        prompt: The original prompt/problem text
        response: The model's response to evaluate
        instruction_id_list: List of instruction IDs to check
        kwargs: List of keyword arguments for each instruction
        strict: Whether to use strict evaluation (default: True)
    
    Returns:
        OutputExample containing evaluation results
    """
    # Create input example
    inp = create_input_example(prompt, instruction_id_list, kwargs)
    
    # Create prompt to response mapping
    prompt_to_response = {prompt: response}
    
    # Evaluate using strict method (as requested)
    if strict:
        return test_instruction_following_strict(inp, prompt_to_response)
    else:
        # For now, only implement strict evaluation as requested
        # Could add loose evaluation later if needed
        return test_instruction_following_strict(inp, prompt_to_response)

#简单返回 INSTRUCTION_DICT.keys() 的列表，也就是当前支持的全部指令ID
def get_supported_instructions() -> List[str]:
    """
    Get list of all supported instruction types.
    
    Returns:
        List of instruction IDs that are supported
    """
    return list(INSTRUCTION_DICT.keys())

#在评测前快速检查元数据是否看起来合理 评测的各类参数是否数量一样
def validate_instruction_metadata(instruction_id_list: List[str], kwargs: List[Dict[str, Any]]) -> bool:
    """
    Validate that instruction metadata is properly formatted.
    
    Args:
        instruction_id_list: List of instruction IDs
        kwargs: List of keyword arguments for each instruction
    
    Returns:
        True if metadata is valid, False otherwise
    """
    if len(instruction_id_list) != len(kwargs):
        return False
    
    # Check if all instruction IDs are supported
    supported = get_supported_instructions()
    for instruction_id in instruction_id_list:
        if instruction_id not in supported:
            print(f"Warning: Unsupported instruction ID: {instruction_id}")
            return False
    
    return True

#把评测结果转成几项汇总分
def calculate_instruction_score(output: OutputExample) -> Dict[str, float]:
    """
    Calculate various scores from instruction following evaluation output.
    
    Args:
        output: OutputExample from evaluation
    
    Returns:
        Dictionary containing different score metrics
    """
    total_instructions = len(output.follow_instruction_list)
    if total_instructions == 0:
        return {"overall": 0.0, "partial": 0.0, "strict": 0.0}
    
    num_followed = sum(output.follow_instruction_list)
    
    return {
        "overall": float(output.follow_all_instructions),  # 1.0 if all followed, 0.0 otherwise
        "partial": num_followed / total_instructions,      # Percentage of instructions followed
        "strict": 1.0 if output.follow_all_instructions else 0.0,  # Same as overall, for clarity
        "num_followed": num_followed,#通过的条数
        "total_instructions": total_instructions #指令总数
    }
