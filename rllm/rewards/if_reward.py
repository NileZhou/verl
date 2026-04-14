# coding=utf-8
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
# 修改导入路径，使用绝对路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from ifbench_utils.instructions_registry import INSTRUCTION_DICT
from ifbench_utils.evaluation_lib import InputExample, test_instruction_following_strict, OutputExample
from ifbench_utils.utils import (
    evaluate_instruction_following,
    validate_instruction_metadata,
    calculate_instruction_score
)

# 这些常量在多个函数里要用，提到类外更清晰
# 关系键 - 支持新旧版本的所有关系类型
REL_KEYS: Tuple[str, ...] = ("relation", "let_relation", "capital_relation")

# 计数键 - 支持新旧版本的所有计数类型
COUNT_KEYS: Tuple[str, ...] = (
    # 基础计数类型
    "num_words", "num_sentences", "num_paragraphs",
    "num_bullets", "num_highlights", "num_sections",
    "num_placeholders",
    
    # 频率相关
    "frequency", "let_frequency", "capital_frequency",
    
    # 关系参数
    "relation", "let_relation", "capital_relation",
    
    # 语言参数
    "language",
    
    # 数字参数
    "n", "m", "N", "small_n",
    "n_sent", "n_words",
    
    # 边界参数
    "low", "high", "lower_bound", "upper_bound",
    
    # 位置参数
    "nth_paragraph", "first_word", "last_word",
    "first_word_sent", "first_word_answer",
    "last_word_sent", "last_word_answer",
    "n_start", "n_end",
    
    # 关键词参数
    "keyword", "keywords", "keyword1", "keyword2",
    "num_keywords", "keyword_count",
    
    # 文本参数
    "phrase", "prompt_to_repeat", "end_phrase",
    "starter", "section_spliter", "postscript_marker",
    "original_message", "key_sentences", "title",
    "original_paragraph", "forbidden_words", "letter",
    
    # 选项参数
    "constrained_responses", "starter_options", "ending_options",
    "two_responses", "repeat_prompt",
    
    # 标点符号参数
    "comma", "quotation", "repeat_phrase",
    
    # 复制相关参数
    "copy_span_idx", "sentence_hyphens", "no_adjacent_consecutive",
    "square_brackets", "word_once", "word_count_different_numbers",
    "exclude_word_harder", "paragraphs", "paragraphs2",
    "bigram_wrapping", "copying_simple", "copying_multiple",
    "punctuation_dot", "punctuation_exclamation",
    
    # 计数相关参数
    "lowercase_counting", "letter_counting", "letter_counting2",
    "counting_composition", "count_unique", "count_increment_word",
    "palindrome", "keyword_specific_position", "start_end"
)
AROUND_TOLERANCE = 0.2  # ±20%


class RewardInstructFn(RewardFn):
    """评估模型在指令遵循任务中的表现。"""

    # === Step 1: 过滤空参数 ===
    def _filter_null_parameters(self, kwargs: List[Dict]) -> List[Dict]:
        """移除 kwargs 中的 None / ndarray 参数。"""
        if not kwargs:
            return kwargs
        filtered_kwargs = []
        for kwarg_dict in kwargs:
            if isinstance(kwarg_dict, dict):
                converted_dict = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in kwarg_dict.items()
                }
                filtered_dict = {k: v for k, v in converted_dict.items() if v is not None}
                filtered_kwargs.append(filtered_dict)
            else:
                filtered_kwargs.append(kwarg_dict)
        return filtered_kwargs

    # 小工具：把关系词与取值做归一化（不落盘，只返回规范化结果）
    def _canon_rel_and_val(
        self,
        rel: Optional[str],
        val: Optional[Union[int, float, str]]
    ) -> Tuple[Optional[str], Optional[Union[int, float, str, Tuple[int, int]]]]:
        def to_int_or_same(x):
            try:
                # 允许 "120"、120.0 这类
                return int(float(x))
            except Exception:
                return x

        if rel is None:
            return None, to_int_or_same(val)

        r = str(rel).strip().lower()
        v = to_int_or_same(val)

        # 精确关系 - 支持新旧版本的所有关系词
        if r in ("less than", "<", "lt"):
            return "less than", v
        if r in ("at least", ">=", "ge", "greater than or equal"):
            return "at least", v
        if r in ("at most", "<=", "le", "no more than", "less than or equal"):
            # 语义 ≤U → < U+1
            return "less than", (None if not isinstance(v, int) else v + 1)
        if r in ("more than", ">", "greater than", "gt"):
            # 语义 >N → ≥ N+1
            return "at least", (None if not isinstance(v, int) else v + 1)

        # 模糊关系 → "区间" - 支持更多模糊关系词
        if r in ("around", "approximately", "approx", "about", "roughly", "close to", "near", "circa"):
            if isinstance(v, int):
                low = max(1, int(v * (1 - AROUND_TOLERANCE)))
                up = int(v * (1 + AROUND_TOLERANCE))
                return "around", (low, up)
            else:
                return "around", v

        # 等于关系 - 新增支持
        if r in ("equal", "equals", "=", "exactly"):
            return "equal", v

        # 不等于关系 - 新增支持
        if r in ("not equal", "!=", "not equals", "unequal"):
            return "not equal", v

        # 其它原样返回
        return r, v

    # === Step 2: 关系归一化 + 模糊关系支持 ===
    def _normalize_relations_in_kwargs(self, kwargs_list: List[Dict]) -> List[Dict]:
        """
        统一关系词映射（在"原关系键位"上写回）：
          - at least, >=, ge → 'at least'
          - less than, <, lt → 'less than'
          - at most, <=, le, no more than → 'less than' +1
          - more than, >, greater than, gt → 'at least' +1
          - equal, equals, = → 'equal'
          - not equal, !=, unequal → 'not equal'
          - around / about / approximately / near → 在原键位上改为 'between'，并补 lower_bound/upper_bound
          
        支持新版本的所有关系类型和参数格式
        """
        if not kwargs_list:
            return kwargs_list

        out = []
        for kw in kwargs_list:
            if not isinstance(kw, dict):
                out.append(kw)
                continue

            # 寻找此样本采用的关系键位（支持所有关系键位）
            rel_key = next((rk for rk in REL_KEYS if rk in kw and kw[rk] is not None), None)
            if rel_key is None:
                out.append(kw)
                continue

            # 寻找计数字段（支持新版本的所有计数字段）
            cnt_key = next((ck for ck in COUNT_KEYS if ck in kw and kw[ck] is not None), None)
            cnt_val = kw[cnt_key] if cnt_key is not None else None

            canon_rel, canon_val = self._canon_rel_and_val(kw[rel_key], cnt_val)

            # 处理模糊关系 → between + 下/上界
            if canon_rel == "around" and isinstance(canon_val, tuple) and len(canon_val) == 2:
                lower, upper = canon_val
                kw[rel_key] = "between"
                kw["lower_bound"] = lower
                kw["upper_bound"] = upper
                # 中心值仅供观察；最终会在 _expand_bounds 中用上下界拆分
                if cnt_key:
                    kw[cnt_key] = (lower + upper) // 2
            else:
                kw[rel_key] = canon_rel
                if cnt_key is not None and canon_val is not None:
                    kw[cnt_key] = canon_val

            # 处理新增的参数类型（如 n, m, N 等）
            for param_key in ["n", "m", "N", "small_n", "n_sent", "n_words", "nth_paragraph", "first_word", "last_word"]:
                if param_key in kw and kw[param_key] is not None:
                    # 确保这些参数是整数类型
                    try:
                        kw[param_key] = int(float(kw[param_key])) if isinstance(kw[param_key], (int, float, str)) else kw[param_key]
                    except (ValueError, TypeError):
                        pass  # 保持原值如果转换失败

            # 处理关键词参数（如 keyword1, keyword2）
            for keyword_key in ["keyword1", "keyword2", "keyword"]:
                if keyword_key in kw and kw[keyword_key] is not None:
                    # 确保关键词是字符串类型
                    kw[keyword_key] = str(kw[keyword_key])

            out.append(kw)

        return out

    # === Step 2.5: 将 lower/upper/between 展开为标准比较（并复制对应的 instruction_id）===
    def _expand_bounds(
        self,
        instruction_ids: List[str],
        kwargs_list: List[Dict],
    ):
        """
        将包含 lower_bound / upper_bound（或 relation='between' / let_relation='between' / capital_relation='between'）
        的条目拆成两个标准比较：
          - lower_bound     => relation='at least',  value = lower_bound
          - upper_bound(≤U) => relation='less than', value = U+1
          
        支持新版本的所有关系类型和参数格式
        使用"原关系键位"写回，避免被各 Checker 的优先读取顺序绊倒。
        """
        if not kwargs_list:
            return instruction_ids, kwargs_list

        def _to_int(x):
            try:
                return int(float(x)) if isinstance(x, (int, float, str)) else None
            except Exception:
                return None

        new_ids, new_kwargs = [], []
        for ins_id, kw in zip(instruction_ids, kwargs_list):
            if not isinstance(kw, dict):
                new_ids.append(ins_id)
                new_kwargs.append(kw)
                continue

            # 找出原关系键位
            rel_key = next((rk for rk in REL_KEYS if rk in kw and kw[rk] is not None), None)

            # 取出并移除区间键
            lb = kw.pop("lower_bound", None)
            ub = kw.pop("upper_bound", None)

            # 如果任意一个关系键位 == 'between'，把它清掉，改用两条标准比较表示
            is_between = False
            for rk in REL_KEYS:
                if kw.get(rk, None) == "between":
                    is_between = True
                    kw.pop(rk, None)
                    # 以真实发现 'between' 的键作为写回键位
                    rel_key = rk
                    break

            # 找计数字段（支持新版本的所有计数字段）
            cnt_key = next((k for k in COUNT_KEYS if k in kw and kw[k] is not None), None)

            # 没有区间也没有 between → 原样保留
            if (lb is None and ub is None) and not is_between:
                new_ids.append(ins_id)
                new_kwargs.append(kw)
                continue

            # 对未知类型检查器（没有计数字段）出于稳妥考虑原样保留，避免误改
            if cnt_key is None:
                new_ids.append(ins_id)
                new_kwargs.append(kw)
                continue

            # 写回的键位：优先用原键位；没有则退回到通用 'relation'
            write_key = rel_key or "relation"

            # 处理 equal 关系 - 不需要拆分，直接保留
            if kw.get(write_key) == "equal":
                new_ids.append(ins_id)
                new_kwargs.append(kw)
                continue

            # 处理 not equal 关系 - 不需要拆分，直接保留
            if kw.get(write_key) == "not equal":
                new_ids.append(ins_id)
                new_kwargs.append(kw)
                continue

            # 不拆分：将上下界合并到同一条约束中，保持 instruction_id_list 长度不变
            kw_merged = {**kw}
            lb_i = _to_int(lb) if lb is not None else None
            ub_i = _to_int(ub) if ub is not None else None

            if lb_i is not None and ub_i is not None:
                # 有上下界：用 "at least" 下界，同时保留 upper_bound 供 checker 额外校验
                kw_merged[write_key] = "at least"
                kw_merged[cnt_key] = int(lb_i)
                kw_merged["upper_bound"] = int(ub_i + 1)  # 语义：< ub+1
            elif lb_i is not None:
                kw_merged[write_key] = "at least"
                kw_merged[cnt_key] = int(lb_i)
            elif ub_i is not None:
                kw_merged[write_key] = "less than"
                kw_merged[cnt_key] = int(ub_i + 1)

            new_ids.append(ins_id)
            new_kwargs.append(kw_merged)

        return new_ids, new_kwargs

    # === Step 3: 核心调用 ===
    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.INSTRUCT, \
            f"Invalid problem type: expected INSTRUCT, got {input.problem_type}"

        problem = input.problem
        model_response = input.model_response
        metadata = input.metadata

        instruction_id_list = metadata.get("instruction_id_list", [])
        if isinstance(instruction_id_list, np.ndarray):
            instruction_id_list = instruction_id_list.tolist()

        kwargs = metadata.get("kwargs", [])
        if isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()

        # === 1) 清理无效参数 ===
        kwargs = self._filter_null_parameters(kwargs)

        # === 2) 关系归一化（包含 around -> between + lower/upper）===
        kwargs = self._normalize_relations_in_kwargs(kwargs)

        # === 2.5) 展开 lower/upper/between 为两条标准比较，并保持 id/kwargs 对齐 ===
        instruction_id_list, kwargs = self._expand_bounds(instruction_id_list, kwargs)

        if not instruction_id_list:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # === 3) 校验元数据（长度/键/类型）===
        if not validate_instruction_metadata(instruction_id_list, kwargs):
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        try:
            output = evaluate_instruction_following(
                prompt=problem,
                response=model_response,
                instruction_id_list=instruction_id_list,
                kwargs=kwargs,
                strict=True
            )
            # 这个奖励设置意思是：答对为1，答错为0，如果有5个约束对3个就是0.6
            score = calculate_instruction_score(output)

            partial = score["partial"]  # num_followed / total, with zero-div guard
            reward = partial * self.config.correct_reward
            is_correct = output.follow_all_instructions
            return RewardOutput(reward=reward, is_correct=is_correct)

        except Exception as e:
            print(f"Error in instruction following evaluation: {e}")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)


def rllm_reward_fn_instruct(
    data_source: str,
    llm_solution: str,
    instruction_metadata: Dict,
    problem: str = "",
    **kwargs
):
    """外部调用接口"""
    reward_config = RewardConfig()
    reward_fn = RewardInstructFn(reward_config)
    reward_response = reward_fn(RewardInput(
        problem=problem,
        problem_type=RewardType.INSTRUCT,
        model_response=llm_solution,
        metadata=instruction_metadata,
        data_source=data_source
    ))
    return reward_response.reward


if __name__ == "__main__":
    # 简单测试：around 100 words → 应该被拆成 >=80 且 <121 两条约束
    reward_config = RewardConfig()
    reward_fn = RewardInstructFn(reward_config)
    test_input = RewardInput(
        problem="Write about AI in around 100 words.",
        problem_type=RewardType.INSTRUCT,
        model_response=(
            "Artificial Intelligence is revolutionizing industries. "
            "It automates processes, enhances productivity, and drives innovation worldwide. "
            "From healthcare to finance, AI-powered systems analyze data and assist decision-making. "
            "As models improve, responsible deployment and alignment become increasingly crucial."
        ),
        metadata={
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 100, "relation": "around"}]
        },
        data_source="test"
    )
    output = reward_fn(test_input)
    print(f"✅ Test output: {output}")
