# coding=utf-8
"""Library of instructions for instruction following evaluation."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union
from typing import Optional, Union
import math

try:
    import langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

from . import instructions_util

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES
_COMPARISON_RELATION = ("less than", "at least")

def _has_language_features(text: str) -> bool:
    """检查文本是否包含足够的语言特征进行语言检测。
    
    Args:
        text: 要检查的文本
        
    Returns:
        bool: 如果文本包含足够的语言特征返回True，否则返回False
    """
    if not text or not text.strip():
        return False
    
    # 移除空白字符
    cleaned_text = text.strip()
    
    # 检查是否只包含标点符号、数字或特殊字符
    import re
    # 只包含字母的字符数量
    alpha_chars = re.findall(r'[a-zA-Z]', cleaned_text)
    # 包含字母和数字的字符数量
    alnum_chars = re.findall(r'[a-zA-Z0-9]', cleaned_text)
    
    # 如果几乎没有字母字符，认为没有语言特征
    if len(alpha_chars) < 2:
        return False
    
    # 如果文本太短且字母字符比例太低，也认为没有足够的语言特征
    if len(cleaned_text) < 10 and len(alpha_chars) / len(cleaned_text) < 0.3:
        return False
        
    return True

def _to_int_safe(x: Optional[Union[int, float, str]]) -> Optional[int]:
    if x is None:
        return None
    try:
        # 允许 "120"、120.0 这类
        return int(float(x))
    except Exception:
        return None

def _canon_relation_and_value(
    relation: Optional[str],
    value: Optional[Union[int, float, str]],
    *,
    lower_bound: Optional[Union[int, float, str]] = None,
    upper_bound: Optional[Union[int, float, str]] = None,
):
    """
    归一化比较关系到:
      - ('at least', N)
      - ('less than', N)            # 注意: 已处理 <= / at most / no more than 等，做 +1 离散收敛
      - ('between', (L, U))         # 用于 around 或显式下/上界

    规则优先级（越上越高）:
      1) 显式传入 lower_bound / upper_bound
         - 同时有 L 和 U: 返回 ('between', (L, U))  # L/U 自动转 int，并纠正 L>U 的情况
         - 仅 L: 返回 ('at least', L)
         - 仅 U: 返回 ('less than', U + 1)
      2) 关系词 relation
         - '<' / 'less than'                 -> ('less than', value)
         - '>=' / 'ge' / 'at least'          -> ('at least', value)
         - '<=' / 'le' / 'at most' / 'no more than' -> ('less than', value + 1)
         - '>' / 'more than' / 'greater than'       -> ('at least', value + 1)
         - 'around' / 'approximately' / 'about' / 'approx'
              -> ('between', (floor(0.8*value), ceil(1.2*value)))
      3) 其它: 原样返回 (relation, value)
    """
    # ===== 1) 显式边界优先处理 =====
    L = _to_int_safe(lower_bound)
    U = _to_int_safe(upper_bound)
    if L is not None or U is not None:
        if L is not None and U is not None:
            # 纠正 L > U 的脏数据
            if L > U:
                L, U = U, L
            return "between", (L, U)
        if L is not None:
            return "at least", L
        if U is not None:
            return "less than", U + 1

    # ===== 2) 关系词归一化 =====
    if relation is None:
        return None, _to_int_safe(value)

    r = str(relation).strip().lower()
    v = _to_int_safe(value)

    if r in ("less than", "<"):
        return "less than", v
    if r in ("at least", ">=", "ge"):
        return "at least", v
    if r in ("at most", "<=", "le", "no more than"):
        return "less than", (None if v is None else v + 1)
    if r in ("more than", ">", "greater than"):
        return "at least", (None if v is None else v + 1)

    # ===== 3) 模糊关系 → 区间 =====
    if r in ("around", "approximately", "about", "approx"):
        if v is None:
            #return "between", None
            return None, _to_int_safe(value)
        low = max(0, math.floor(0.8 * v))
        up  = max(low, math.ceil(1.2 * v))  # 保证 up >= low
        return "between", (low, up)

    # 其它未识别关系，原样交由上层处理
    return r, v


# 其他常量
_MAX_NUM_SENTENCES = 20
_NUM_PLACEHOLDERS = 4
_NUM_BULLETS = 5
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.")
_STARTER_OPTIONS = ("I would say", "My answer is", "I believe",
                   "In my opinion", "I think", "I reckon", "I suppose")
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")
_HIGHLIGHTED_SECTION_PATTERN = r'\*([^*]+)\*'
_JSON_PATTERN = r'^\s*\{.*\}\s*$'


class Instruction:
    """An instruction template."""
    def __init__(self, instruction_id: Optional[str] = None):
        if instruction_id is None:
            instruction_id = getattr(self, "DEFAULT_INSTRUCTION_ID", None) \
                             or getattr(self, "instruction_id", None) \
                             or self.__class__.__name__
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        if language is None:
            language = random.choice(list(_LANGUAGES.keys()))
        language_name = _LANGUAGES.get(language, language)
        self._language = language
        self._description_pattern = f"Your entire response should be in {language_name}."
        self._instruction_args = {"language": language}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["language"]

    def check_following(self, value):
        # 如果文本太短或为空，直接返回True以避免语言检测错误
        if not value or not value.strip() or len(value.strip()) < 3:
            return True
            
        # 检查文本是否包含足够的语言特征
        if not _has_language_features(value):
            return True
            
        if not _LANGDETECT_AVAILABLE:
            if self._language == "en":
                words = re.findall(r'\b\w+\b', value)
                if not words:
                    return False
                ascii_words = sum(1 for word in words if all(ord(c) < 128 for c in word))
                return ascii_words / len(words) > 0.8
            else:
                return True
        try:
            detected_language = langdetect.detect(value)
            return detected_language == self._language
        except langdetect.LangDetectException as e:
            # 对于太短或无法检测的文本，默认返回True避免影响训练
            return True
        except Exception:
            return False


class KeywordChecker(Instruction):
    """Check if a keyword exists in the response."""

    def build_description(self, *, keyword=None, keywords=None):
        if keywords is not None:
            if isinstance(keywords, list) and keywords:
                keyword = keywords[0]
            elif isinstance(keywords, str):
                keyword = keywords
        if keyword is None:
            keyword = random.choice(instructions_util.WORD_LIST)
        self._keyword = keyword.lower()
        self._description_pattern = f'Include the keyword "{keyword}" in the response.'
        self._instruction_args = {"keyword": keyword}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["keyword"]

    def check_following(self, value):
        return bool(re.search(self._keyword, value, flags=re.IGNORECASE))

        
class KeywordFrequencyChecker(Instruction):
    def build_description(self, *, keyword=None, frequency=None, relation=None):
        if keyword is None:
            keyword = random.choice(instructions_util.WORD_LIST)
        if frequency is None:
            frequency = random.randint(2, 5)

        canon_rel, canon_freq = _canon_relation_and_value(relation, frequency)

        # ✅ between 支持
        if canon_rel == "between" and isinstance(canon_freq, tuple) and len(canon_freq) == 2:
            L, U = int(canon_freq[0]), int(canon_freq[1])
            if L > U: L, U = U, L
            self._keyword = keyword.lower()
            self._relation, self._lower, self._upper = "between", L, U
            self._description_pattern = (
                f'In your response, the word "{keyword}" should appear between {L} and {U} times (inclusive).'
            )
            self._instruction_args = {"keyword": keyword, "relation": "between", "lower_bound": L, "upper_bound": U}
            return

        if canon_rel is None:
            canon_rel = random.choice(_COMPARISON_RELATION)
        if canon_freq is not None:
            frequency = int(canon_freq)
        if canon_rel not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
                f"but {relation} is given (normalized to {canon_rel})."
            )

        self._keyword = keyword.lower()
        self._frequency = int(frequency)
        self._relation = canon_rel
        if self._relation == "less than":
            self._description_pattern = f'In your response, the word "{keyword}" should appear less than {self._frequency} times.'
        else:
            self._description_pattern = f'In your response, the word "{keyword}" should appear at least {self._frequency} times.'
        self._instruction_args = {"keyword": keyword, "frequency": self._frequency, "relation": self._relation}

    def get_instruction_args(self): return self._instruction_args
    #def get_instruction_args_keys(self): return ["keyword", "frequency", "relation", "lower_bound", "upper_bound"]
    def get_instruction_args_keys(self):
        if getattr(self, "_relation", None) == "between":
            return ["keyword", "relation", "lower_bound", "upper_bound"]
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        cnt = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        if getattr(self, "_relation", None) == "between":
            return (cnt >= self._lower) and (cnt <= self._upper)
        return cnt < self._frequency if self._relation == "less than" else cnt >= self._frequency



# class KeywordFrequencyChecker(Instruction):
#     """Check if a keyword appears with specified frequency."""

#     def build_description(self, *, keyword=None, frequency=None, relation=None):
#         if keyword is None:
#             keyword = random.choice(instructions_util.WORD_LIST)
#         if frequency is None:
#             frequency = random.randint(2, 5)

#         # === 归一化关系 ===
#         canon_rel, canon_freq = _canon_relation_and_value(relation, frequency)
#         if canon_rel is None:
#             canon_rel = random.choice(_COMPARISON_RELATION)
#         if canon_freq is not None:
#             frequency = canon_freq
#         if canon_rel not in _COMPARISON_RELATION:
#             raise ValueError(
#                 f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
#                 f"but {relation} is given (normalized to {canon_rel})."
#             )

#         self._keyword = keyword.lower()
#         self._frequency = frequency
#         self._relation = canon_rel

#         if self._relation == "less than":
#             self._description_pattern = f'In your response, the word "{keyword}" should appear less than {frequency} times.'
#         else:
#             self._description_pattern = f'In your response, the word "{keyword}" should appear at least {frequency} times.'
#         self._instruction_args = {"keyword": keyword, "frequency": frequency, "relation": self._relation}

#     def get_instruction_args(self):
#         return self._instruction_args

#     def get_instruction_args_keys(self):
#         return ["keyword", "frequency", "relation"]

#     def check_following(self, value):
#         actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
#         if self._relation == "less than":
#             return actual_occurrences < self._frequency
#         else:
#             return actual_occurrences >= self._frequency


class ForbiddenWords(Instruction):
    """Check that certain words do not appear in the response."""

    def build_description(self, *, forbidden_words=None):
        if forbidden_words is None:
            forbidden_words = random.sample(instructions_util.WORD_LIST, 3)
        self._forbidden_words = [word.lower() for word in forbidden_words]
        word_list_str = ", ".join(f'"{word}"' for word in forbidden_words)
        self._description_pattern = f"Do not include keywords {word_list_str} in the response."
        self._instruction_args = {"forbidden_words": forbidden_words}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["forbidden_words"]

    def check_following(self, value):
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True

class LetterFrequencyChecker(Instruction):
    def build_description(
        self, *,
        letter=None,
        let_frequency=None, let_relation=None,
        frequency=None, relation=None,
        **kwargs
    ):
        if let_frequency is None and frequency is not None:
            let_frequency = frequency
        if let_relation is None and relation is not None:
            let_relation = relation

        if not letter or len(letter) > 1:
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        if self._letter.isalpha():
            self._letter = self._letter.lower()

        freq = let_frequency if (let_frequency is not None and int(let_frequency) >= 0) else random.randint(1, 20)
        canon_rel, canon_freq = _canon_relation_and_value(let_relation, freq)

        # ✅ between 支持
        if canon_rel == "between" and isinstance(canon_freq, tuple) and len(canon_freq) == 2:
            L, U = int(canon_freq[0]), int(canon_freq[1])
            if L > U: L, U = U, L
            self._relation, self._lower, self._upper = "between", L, U
            self._description_pattern = f"In your response, the letter {self._letter} should appear between {L} and {U} times (inclusive)."
            self._instruction_args = {"letter": self._letter, "relation": "between", "lower_bound": L, "upper_bound": U}
            return

        if canon_rel is None:
            canon_rel = random.choice(_COMPARISON_RELATION)
        if canon_freq is not None:
            freq = int(canon_freq)
        if canon_rel not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
                f"but {let_relation} is given (normalized to {canon_rel})."
            )

        self._frequency = int(freq)
        self._comparison_relation = canon_rel
        self._description_pattern = (
            f"In your response, the letter {self._letter} should appear {self._comparison_relation} {self._frequency} times."
        )
        # ✅ 用 self._letter，避免 None
        self._instruction_args = {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation
        }

    def get_instruction_args(self): return self._instruction_args
    #def get_instruction_args_keys(self): return ["letter", "let_frequency", "let_relation", "lower_bound", "upper_bound"]
    def get_instruction_args_keys(self):
        if getattr(self, "_relation", None) == "between":
            return ["letter", "relation", "lower_bound", "upper_bound"]
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        s = value.lower() if self._letter.isalpha() else value
        ch = self._letter.lower() if self._letter.isalpha() else self._letter
        cnt = collections.Counter(s)[ch]
        if getattr(self, "_relation", None) == "between":
            return (cnt >= self._lower) and (cnt <= self._upper)
        return cnt < self._frequency if self._comparison_relation == "less than" else cnt >= self._frequency


# class LetterFrequencyChecker(Instruction):
#     """Check the frequency of a specific letter."""

#     def build_description(
#         self, *,
#         letter=None,
#         let_frequency=None, let_relation=None,   # 规范字段
#         frequency=None, relation=None,           # 兼容别名
#         **kwargs                                  # 吞掉其他未知键，避免再抛错
#     ):
#         # 兼容别名 -> 规范字段
#         if let_frequency is None and frequency is not None:
#             let_frequency = frequency
#         if let_relation is None and relation is not None:
#             let_relation = relation

#         if not letter or len(letter) > 1:
#             self._letter = random.choice(list(string.ascii_letters))
#         else:
#             self._letter = letter.strip()
#         if self._letter.isalpha():
#             self._letter = self._letter.lower()

#         frequency = let_frequency if (let_frequency is not None and let_frequency >= 0) else random.randint(1, 20)

#         # 关系归一化（沿用你已有的工具）
#         canon_rel, canon_freq = _canon_relation_and_value(let_relation, frequency)
#         if canon_rel is None:
#             canon_rel = random.choice(_COMPARISON_RELATION)
#         if canon_freq is not None:
#             frequency = canon_freq
#         if canon_rel not in _COMPARISON_RELATION:
#             raise ValueError(
#                 f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
#                 f"but {let_relation} is given (normalized to {canon_rel})."
#             )

#         self._frequency = frequency
#         self._comparison_relation = canon_rel
#         self._description_pattern = f"In your response, the letter {self._letter} should appear {self._comparison_relation} {self._frequency} times."
        
#         self._instruction_args = {
#             "letter": letter,
#             "let_frequency": self._frequency,
#             "let_relation": self._comparison_relation
#         }

#     def get_instruction_args(self):
#         return self._instruction_args

#     def get_instruction_args_keys(self):
#         return ["letter", "let_frequency", "let_relation"]

#     def check_following(self, value):
#         if self._letter.isalpha():
#             search_value = value.lower()
#             search_letter = self._letter.lower()
#         else:
#             search_value = value
#             search_letter = self._letter

#         letters = collections.Counter(search_value)
#         if self._comparison_relation == "less than":
#             return letters[search_letter] < self._frequency
#         else:
#             return letters[search_letter] >= self._frequency

class NumberOfSentences(Instruction):
    def build_description(self, *, num_sentences=None, relation=None):
        if num_sentences is None:
            num_sentences = random.randint(1, _MAX_NUM_SENTENCES)

        canon_rel, canon_val = _canon_relation_and_value(relation, num_sentences)

        if canon_rel == "between" and isinstance(canon_val, tuple) and len(canon_val) == 2:
            L, U = int(canon_val[0]), int(canon_val[1])
            if L > U:
                L, U = U, L
            self._relation = "between"
            self._lower = L
            self._upper = U
            self._description_pattern = f"Your response should contain between {L} and {U} sentences (inclusive)."
            self._instruction_args = {"relation": "between", "lower_bound": L, "upper_bound": U}
            return

        if canon_rel is None:
            canon_rel = random.choice(_COMPARISON_RELATION)
        if canon_val is not None:
            num_sentences = int(canon_val)
        if canon_rel not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
                f"but {relation} is given (normalized to {canon_rel})."
            )

        self._num_sentences = int(num_sentences)
        self._relation = canon_rel

        if self._relation == "less than":
            self._description_pattern = f"Your response should contain less than {self._num_sentences} sentences."
        else:
            self._description_pattern = f"Your response should contain at least {self._num_sentences} sentences."
        self._instruction_args = {"num_sentences": self._num_sentences, "relation": self._relation}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        if getattr(self, "_relation", None) == "between":
            return ["relation", "lower_bound", "upper_bound"]
        return ["num_sentences", "relation"]

    def check_following(self, value):
        sentence_count = instructions_util.count_sentences(value)

        if getattr(self, "_relation", None) == "between":
            return (sentence_count >= self._lower) and (sentence_count <= self._upper)

        if self._relation == "less than":
            return sentence_count < self._num_sentences
        else:
            return sentence_count >= self._num_sentences



# class NumberOfSentences(Instruction):
#     """Check the number of sentences in the response."""

#     def build_description(self, *, num_sentences=None, relation=None):
#         if num_sentences is None:
#             num_sentences = random.randint(1, _MAX_NUM_SENTENCES)

#         # === 归一化关系 ===
#         canon_rel, canon_val = _canon_relation_and_value(relation, num_sentences)
        
#         # # 若出现 between(L, U)，在本指令中落地成一侧约束（下界）
#         # if canon_rel == "between" and isinstance(canon_val, tuple) and len(canon_val) == 2:
#         #     canon_rel, canon_val = "at least", int(canon_val[0])

#         if canon_rel is None:
#             canon_rel = random.choice(_COMPARISON_RELATION)
#         if canon_val is not None:
#             num_sentences = canon_val
#         if canon_rel not in _COMPARISON_RELATION:
#             raise ValueError(
#                 f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
#                 f"but {relation} is given (normalized to {canon_rel})."
#             )

#         self._num_sentences = num_sentences
#         self._relation = canon_rel

#         if self._relation == "less than":
#             self._description_pattern = f"Your response should contain less than {num_sentences} sentences."
#         else:
#             self._description_pattern = f"Your response should contain at least {num_sentences} sentences."
#         self._instruction_args = {"num_sentences": num_sentences, "relation": self._relation}

#     def get_instruction_args(self):
#         return self._instruction_args

#     def get_instruction_args_keys(self):
#         return ["num_sentences", "relation"]

#     def check_following(self, value):
#         sentence_count = instructions_util.count_sentences(value)
#         if self._relation == "less than":
#             return sentence_count < self._num_sentences
#         else:
#             return sentence_count >= self._num_sentences


class ParagraphChecker(Instruction):
    """Check the number of paragraphs in the response."""

    def build_description(self, *, num_paragraphs=None):
        if num_paragraphs is None or num_paragraphs < 0:
            num_paragraphs = random.randint(1, 10)
        self._num_paragraphs = num_paragraphs
        self._description_pattern = (
            f"There should be {num_paragraphs} paragraphs. "
            "Paragraphs are separated with the markdown divider: ***"
        )
        self._instruction_args = {"num_paragraphs": num_paragraphs}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["num_paragraphs"]

    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False
        return num_paragraphs == self._num_paragraphs

class NumberOfWords(Instruction):
    def build_description(self, *, num_words=None, relation=None):
        if num_words is None:
            num_words = random.randint(50, 500)

        canon_rel, canon_val = _canon_relation_and_value(relation, num_words)

        # ✅ 新增：支持 between(L, U)
        if canon_rel == "between" and isinstance(canon_val, tuple) and len(canon_val) == 2:
            L, U = int(canon_val[0]), int(canon_val[1])
            if L > U:
                L, U = U, L  # 兜底纠正
            self._relation = "between"
            self._lower = L
            self._upper = U
            self._description_pattern = f"Answer with between {L} and {U} words (inclusive)."
            self._instruction_args = {"relation": "between", "lower_bound": L, "upper_bound": U}
            return  # 这里直接返回，避免走原来的两类比较分支

        # 原逻辑：只接受 less than / at least
        if canon_rel is None:
            canon_rel = random.choice(_COMPARISON_RELATION)
        if canon_val is not None:
            num_words = int(canon_val)
        if canon_rel not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
                f"but {relation} is given (normalized to {canon_rel})."
            )

        self._num_words = int(num_words)
        self._relation = canon_rel
        if self._relation == "less than":
            self._description_pattern = f"Answer with less than {self._num_words} words."
        else:
            self._description_pattern = f"Answer with at least {self._num_words} words."
        self._instruction_args = {"num_words": self._num_words, "relation": self._relation}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        if getattr(self, "_relation", None) == "between":
            return ["relation", "lower_bound", "upper_bound"]
        return ["num_words", "relation"]

    def check_following(self, value):
        word_count = instructions_util.count_words(value)

        # ✅ 新增：双侧约束
        if getattr(self, "_relation", None) == "between":
            return (word_count >= self._lower) and (word_count <= self._upper)

        if self._relation == "less than":
            return word_count < self._num_words
        else:
            return word_count >= self._num_words




class ParagraphFirstWordCheck(Instruction):
    """Check the first word of a specific paragraph."""

    def build_description(self, *, num_paragraphs=None, nth_paragraph=None, first_word=None):
        if num_paragraphs is None or num_paragraphs < 0:
            num_paragraphs = random.randint(1, 10)
        if (nth_paragraph is None or nth_paragraph <= 0 or nth_paragraph > num_paragraphs):
            nth_paragraph = random.randint(1, num_paragraphs + 1)
        if first_word is None:
            first_word = random.choice(instructions_util.WORD_LIST)

        self._num_paragraphs = int(num_paragraphs) if num_paragraphs is not None else num_paragraphs
        self._nth_paragraph = int(nth_paragraph) if nth_paragraph is not None else nth_paragraph
        self._first_word = first_word.lower()
        self._description_pattern = (
            f"There should be {num_paragraphs} paragraphs. "
            "Paragraphs and only paragraphs are separated with each other by two "
            "new lines as if it was '\\n\\n' in python. "
            f"Paragraph {nth_paragraph} must start with word {first_word}."
        )
        self._instruction_args = {"num_paragraphs": num_paragraphs, "nth_paragraph": nth_paragraph, "first_word": first_word}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}
        word = paragraph.split()[0].strip()
        word = word.lstrip("'").lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


class PlaceholderChecker(Instruction):
    """Check for placeholders in the response."""

    def build_description(self, *, num_placeholders=None):
        if num_placeholders is None or num_placeholders < 0:
            num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        self._num_placeholders = num_placeholders
        self._description_pattern = (
            f"The response must contain at least {num_placeholders} placeholders "
            "represented by square brackets, such as [address]."
        )
        self._instruction_args = {"num_placeholders": num_placeholders}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["num_placeholders"]

    def check_following(self, value):
        placeholders = re.findall(r"\[.*?\]", value)
        return len(placeholders) >= self._num_placeholders


class PostscriptChecker(Instruction):
    """Check for postscript in the response."""

    def build_description(self, *, postscript_marker=None):
        self._postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker
        if self._postscript_marker is None:
            self._postscript_marker = "P.S."
        self._description_pattern = (
            "At the end of your response, please explicitly add a postscript "
            f"starting with {self._postscript_marker}"
        )
        self._instruction_args = {"postscript_marker": self._postscript_marker}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["postscript_marker"]

    def check_following(self, value):
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class BulletListChecker(Instruction):
    """Check for bullet lists in the response."""

    def build_description(self, *, num_bullets=None):
        if num_bullets is None:
            num_bullets = random.randint(1, _NUM_BULLETS)
        self._num_bullets = num_bullets
        self._description_pattern = (
            f"Your answer must contain exactly {num_bullets} bullet points. "
            "Use the markdown bullet points such as: * This is a point."
        )
        self._instruction_args = {"num_bullets": num_bullets}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["num_bullets"]

    def check_following(self, value):
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Check if response is one of the constrained options."""

    def build_description(self, **kwargs):
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = f"Answer with one of the following options: {self._constrained_responses}"
        self._instruction_args = {}

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class HighlightSectionChecker(Instruction):
    """Check for highlighted sections in the response."""

    def build_description(self, *, num_highlights=None):
        if num_highlights is None:
            num_highlights = random.randint(1, 5)
        self._num_highlights = num_highlights
        self._description_pattern = (
            f"Highlight at least {num_highlights} sections in your answer with markdown, i.e. *highlighted section*."
        )
        self._instruction_args = {"num_highlights": num_highlights}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["num_highlights"]

    def check_following(self, value):
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            inner = highlight.removeprefix("**").removesuffix("**").strip()
            if inner:
                num_highlights += 1
        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Check for multiple sections in the response."""

    def build_description(self, *, section_splitter=None, section_spliter=None, num_sections=None):
        if section_spliter is not None:
            section_splitter = section_spliter
        if section_splitter is None:
            section_splitter = "SECTION"
        if num_sections is None:
            num_sections = random.randint(2, 5)

        self._section_splitter = section_splitter
        self._num_sections = num_sections
        self._description_pattern = (
            f"Your response should have {num_sections} sections. "
            f"Mark the beginning of each section with {section_splitter} X, such as: {section_splitter} 1"
        )
        self._instruction_args = {"section_splitter": section_splitter, "num_sections": num_sections}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["section_splitter", "num_sections"]

    def check_following(self, value):
        section_splitter_patten = r"\s?" + self._section_splitter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class JsonFormat(Instruction):
    """Check if response is in JSON format."""

    def build_description(self, **kwargs):
        self._description_pattern = (
            "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."
        )
        self._instruction_args = {}

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
            return True
        except (ValueError, json.JSONDecodeError):
            return False


class TitleChecker(Instruction):
    """Check if response has a title."""

    def build_description(self, **kwargs):
        self._description_pattern = (
            "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
        )
        self._instruction_args = {}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class TwoResponsesChecker(Instruction):
    """Check for two responses separated by asterisks."""

    def build_description(self, **kwargs):
        self._description_pattern = (
            "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******."
        )
        self._instruction_args = {}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        parts = value.split("******")
        return len(parts) == 2 and all(part.strip() for part in parts)


class RepeatPromptThenAnswer(Instruction):
    """Check if prompt is repeated before the answer."""

    def build_description(self, *, prompt=None, prompt_to_repeat=None):
        if prompt_to_repeat is not None:
            prompt = prompt_to_repeat
        if not prompt:
            # 使用默认值而不是抛出异常
            prompt = "Please repeat this prompt."
        self._prompt_to_repeat = prompt
        self._description_pattern = (
            "First repeat the request word for word without change,"
            " then give your answer (1. do not say any words or characters"
            " before repeating the request; 2. the request you need to repeat"
            " does not include this sentence)"
        )
        self._instruction_args = {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["prompt_to_repeat"]

    def check_following(self, value):
        return value.strip().lower().startswith(self._prompt_to_repeat.strip().lower())


class EndChecker(Instruction):
    """Check if response ends with specific words."""

    def build_description(self, *, end_phrase=None):
        if end_phrase is None:
            end_phrase = random.choice(_ENDING_OPTIONS)
        self._end_phrase = end_phrase
        self._description_pattern = (
            f'Finish your response with this exact phrase "{end_phrase}". No other words should follow this phrase.'
        )
        self._instruction_args = {"end_phrase": end_phrase}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return ["end_phrase"]

    def check_following(self, value):
        value = value.strip().strip('"').lower()
        end_phrase = self._end_phrase.strip().lower()
        return value.endswith(end_phrase)

class CapitalWordFrequencyChecker(Instruction):
    def build_description(
        self, *,
        capital_frequency=None, capital_relation=None,
        frequency=None, relation=None,
        **kwargs
    ):
        if capital_frequency is None and frequency is not None:
            capital_frequency = frequency
        if capital_relation is None and relation is not None:
            capital_relation = relation

        if capital_frequency is None:
            capital_frequency = random.randint(1, 10)

        canon_rel, canon_freq = _canon_relation_and_value(capital_relation, capital_frequency)

        # ✅ between 支持
        if canon_rel == "between" and isinstance(canon_freq, tuple) and len(canon_freq) == 2:
            L, U = int(canon_freq[0]), int(canon_freq[1])
            if L > U: L, U = U, L
            self._relation, self._lower, self._upper = "between", L, U
            self._description_pattern = (
                f"In your response, words with all capital letters should appear between {L} and {U} times (inclusive)."
            )
            self._instruction_args = {"capital_relation": "between", "lower_bound": L, "upper_bound": U}
            return

        if canon_rel is None:
            canon_rel = random.choice(_COMPARISON_RELATION)
        if canon_freq is not None:
            capital_frequency = int(canon_freq)
        if canon_rel not in _COMPARISON_RELATION:
            raise ValueError(
                f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
                f"but {capital_relation} is given (normalized to {canon_rel})."
            )

        self._frequency = int(capital_frequency)
        self._comparison_relation = canon_rel
        self._description_pattern = (
            f"In your response, words with all capital letters should appear "
            f"{'less than' if self._comparison_relation=='less than' else 'at least'} "
            f"{self._frequency} times."
        )
        self._instruction_args = {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation
        }

    def get_instruction_args(self): return self._instruction_args
    #def get_instruction_args_keys(self): return ["capital_frequency", "capital_relation", "lower_bound", "upper_bound"]
    def get_instruction_args_keys(self):
        if getattr(self, "_relation", None) == "between":
            return ["capital_relation", "lower_bound", "upper_bound"]
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        words = re.findall(r'\b\w+\b', value)
        cnt = sum(1 for w in words if w.isupper())
        if getattr(self, "_relation", None) == "between":
            return (cnt >= self._lower) and (cnt <= self._upper)
        return cnt < self._frequency if self._comparison_relation == "less than" else cnt >= self._frequency


# class CapitalWordFrequencyChecker(Instruction):
#     """Check frequency of capitalized words."""

#     def build_description(
#         self, *,
#         capital_frequency=None, capital_relation=None,  # 规范
#         frequency=None, relation=None,                 # 兼容
#         **kwargs
#     ):
#         if capital_frequency is None and frequency is not None:
#             capital_frequency = frequency
#         if capital_relation is None and relation is not None:
#             capital_relation = relation

#         if capital_frequency is None:
#             capital_frequency = random.randint(1, 10)

#         canon_rel, canon_freq = _canon_relation_and_value(capital_relation, capital_frequency)
#         if canon_rel is None:
#             canon_rel = random.choice(_COMPARISON_RELATION)
#         if canon_freq is not None:
#             capital_frequency = canon_freq
#         if canon_rel not in _COMPARISON_RELATION:
#             raise ValueError(
#                 f"The supported relation for comparison must be in {_COMPARISON_RELATION}, "
#                 f"but {capital_relation} is given (normalized to {canon_rel})."
#             )

#         self._frequency = capital_frequency
#         self._comparison_relation = canon_rel
#         self._description_pattern = (
#             f"In your response, words with all capital letters should appear "
#             f"{'less than' if self._comparison_relation=='less than' else 'at least'} "
#             f"{capital_frequency} times."
#         )
#         self._instruction_args = {
#             "capital_frequency": self._frequency,
#             "capital_relation": self._comparison_relation
#         }
#     def get_instruction_args(self):
#         return self._instruction_args

#     def get_instruction_args_keys(self):
#         return ["capital_frequency", "capital_relation"]

#     def check_following(self, value):
#         try:
#             import nltk
#             words = nltk.word_tokenize(value)
#         except (ImportError, LookupError):
#             words = re.findall(r'\b\w+\b', value)
#         capital_words = [word for word in words if word.isupper()]
#         capital_count = len(capital_words)
#         if self._comparison_relation == "less than":
#             return capital_count < self._frequency
#         else:
#             return capital_count >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Check if response is in all capital letters."""

    def build_description(self, **kwargs):
        self._description_pattern = "Your entire response should be in English, and in all capital letters."
        self._instruction_args = {}

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        # 如果文本太短或为空，直接检查大小写而不进行语言检测
        if not value or not value.strip() or len(value.strip()) < 3:
            return value.isupper() if value else True
            
        # 检查文本是否包含足够的语言特征
        if not _has_language_features(value):
            return value.isupper() if value else True
            
        try:
            if _LANGDETECT_AVAILABLE:
                return value.isupper() and langdetect.detect(value) == "en"
            else:
                return value.isupper()
        except langdetect.LangDetectException as e:
            # 对于太短或无法检测的文本，默认返回True避免影响训练
            return value.isupper() if value else True
        except Exception:
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Check if response is in all lowercase letters."""

    def build_description(self, **kwargs):
        self._description_pattern = "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        self._instruction_args = {}

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        # 如果文本太短或为空，直接检查大小写而不进行语言检测
        if not value or not value.strip() or len(value.strip()) < 3:
            return value.islower() if value else True
            
        # 检查文本是否包含足够的语言特征
        if not _has_language_features(value):
            return value.islower() if value else True
            
        try:
            if _LANGDETECT_AVAILABLE:
                return value.islower() and langdetect.detect(value) == "en"
            else:
                return value.islower()
        except langdetect.LangDetectException as e:
            # 对于太短或无法检测的文本，默认返回True避免影响训练
            return value.islower() if value else True
        except Exception:
            return True


class CommaChecker(Instruction):
    """Check that no commas appear in the response."""

    def build_description(self, **kwargs):
        self._description_pattern = "In your entire response, refrain from the use of any commas."
        self._instruction_args = {}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return not re.search(r"\,", value)


class QuotationChecker(Instruction):
    """Check if response is wrapped in quotation marks."""

    def build_description(self, **kwargs):
        self._description_pattern = 'Wrap your entire response with double quotation marks.'
        self._instruction_args = {}

    def get_instruction_args(self):
        return self._instruction_args

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'
