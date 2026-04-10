import os
# 代理
# os.environ['http_proxy'] = 'http://10.136.0.191:10890'
# os.environ['https_proxy'] = 'http://10.136.0.191:10890'


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import base64
import zlib
import json
import pickle

from datetime import datetime
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset
from enum import Enum
from tqdm import tqdm

import pandas as pd


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    # /njfs/train-aitech/datasets/code_generation
    # livecodebench/code_generation_lite
    dataset = load_dataset("/njfs/train-aitech/datasets/code_generation_lite", split="test", version_tag=release_version)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset



def load_code_generation_dataset(release_version="release_v6", start_date=None, end_date=None) -> list[Any]:
    #     dataset = load_dataset(
    #     "json",
    #     data_files={"test": "/njfs/train-aitech/datasets/code_generation/test.jsonl"},
    #     split="test",
    #     cache_dir="/njfs/train-aitech/datasets"  # Optional, but keeps caching local
    # )
    dataset = load_dataset("/njfs/train-aitech/datasets/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


# class CodeGenerationProblem:
#     question_title: str
#     question_content: str
#     platform: Platform
#     question_id: str
#     contest_id: str
#     contest_date: datetime
#     starter_code: str
#     difficulty: Difficulty
#     public_test_cases: list[Test]
#     private_test_cases: list[Test]
#     metadata: dict


def transfer2verl_format(d: CodeGenerationProblem) -> dict:
    ground_truth = []
    for t in d.public_test_cases + d.private_test_cases:
        # 对t.testtype.value == functional来说，d.metadata里有func_name，比如："{\"func_name\": \"minimumCost\"}"
        ground_truth.append({"input": t.input, "output": t.output, "testtype": t.testtype.value, "metadata": d.metadata})
        
    reward_model = {
        "style": "rule",
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
    }
    
    return {
        "data_source": "livecodebench",
        "prompt": [{"role": "user", "content": d.question_content}],
        "ability": "code",
        "reward_model": reward_model,
        "extra_info": {
            "source": d.platform.value,
            "source_title": d.question_title,
            "difficulty": d.difficulty.value,
            "index": d.question_id,
            "starter_code": d.starter_code,
        },
    }


def save_nest_parquet(df, fpath):
    save_df = df.copy() # 不然原df结构会变
    for col in ['prompt', 'reward_model', 'extra_info']:
        save_df[col] = save_df[col].apply(json.dumps)
    if 'responses' in save_df.columns:
        save_df['responses'] = save_df['responses'].apply(json.dumps)
    save_df.to_parquet(fpath)


if __name__ == "__main__":
    # # v5: 279道题
    # start_date = "2024-08-01"
    # tgt_parquet = f'/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/oexps/data/livecodebench_start_{start_date}.parquet'
    # data = load_code_generation_dataset(release_version="release_v5", start_date=start_date, end_date=None) # list[CodeGenerationProblem]
    
    # v6定义全量:2025-02-01: 131道题    v6:2025-03-01: 80道题    v6:2025-03-20: 39道题
    # AceReason-Nemotron 1.1论文定义: LiveCodeBench v5 (2024/08/01-2025/02/01, avg@8), and v6 (2025/02/01-2025/05/01, avg@8)
    start_date = "1980-01-01"
    tgt_parquet = f'/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/oexps/data/lcb_vpre.parquet'
    data = load_code_generation_dataset(release_version="release_v6", start_date=start_date, end_date="2024-07-31") # list[CodeGenerationProblem]
    
    # # 得到v6全量数据(错误做法，但有借鉴意义,v6-v5):
    # tgt_parquet = f'/njfs/train-aitech/projects/zhouyi9/projects/coderl/verl/oexps/data/livecodebench_v6.parquet'
    # data = load_code_generation_dataset(release_version="release_v6", start_date="2025-01-04", end_date=None) # list[CodeGenerationProblem]
    # old_data = load_code_generation_dataset(release_version="release_v5", start_date="2025-01-04", end_date=None) # list[CodeGenerationProblem]
    # # 从data里减去old_data
    # data = [d for d in data if d not in old_data]
    # print(f"total data: {len(data)}")
    
    ret_data = []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        ret_data.append(transfer2verl_format(d))
    print(f"total data: {len(ret_data)}")
    # 将ret_data写为parquet
    df = pd.DataFrame(ret_data)
    save_nest_parquet(df, tgt_parquet)

    # # 解析private_test_cases
    # ret = json.loads(
    #     pickle.loads(
    #         zlib.decompress(
    #             base64.b64decode(encoded_private_test_cases.encode("utf-8"))  # type: ignore
    #         )
    #     )
    # )
    # print(ret)
    
    


# 变成以下这种数据形式:
# {
#   "data_source": "livecodebench",
#   "prompt": "[{\"role\": \"user\", \"content\": \"...\"}]",
#   "ability": "code",
#   "reward_model": "{\"style\": \"rule\", \"ground_truth\": \"[{\\\"input\\\": \\\"4\\\\n8 2 5 1\\\\n\\\", \\\"output\\\": \\\"3\\\\n\\\", \\\"testtype\\\": \\\"stdin\\\"}, {...}]\"}",
#   "extra_info": "{\"split\": \"test\", \"index\": 0, \"reference\": null}"
# }
