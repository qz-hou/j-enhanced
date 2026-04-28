"""
Evaluate multihop QA results (EM/F1).

Converted from `eval_multihop.ipynb` with a small CLI wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class MultiHopEvaluator:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s: object) -> str:
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        if not isinstance(s, str):
            return ""
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str], None] = None,
    ) -> Dict[str, int]:
        if not prediction:
            return {"correct": 0, "incorrect": 1}
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = int(
            np.max([cls.normalize_answer(prediction) == cls.normalize_answer(gt) for gt in ground_truths])
        )
        return {"correct": correct, "incorrect": 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str], None] = None,
    ) -> Dict[str, float]:
        final_metric: Dict[str, float] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        if not prediction:
            return final_metric
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        normalized_prediction = cls.normalize_answer(prediction)
        for gt in ground_truths:
            normalized_ground_truth = cls.normalize_answer(gt)
            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue

            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            final_metric["f1"] = max(f1, final_metric["f1"])
            final_metric["precision"] = max(precision, final_metric["precision"])
            final_metric["recall"] = max(recall, final_metric["recall"])
        return final_metric

    def eval_answer(self, results_df: pd.DataFrame, answer_col: str = "Final Answer") -> None:
        em_list: List[int] = []
        f1_list: List[float] = []
        for _, row in results_df.iterrows():
            prediction = row.get(answer_col, "")
            ground_truth = row["ground_truth"]
            em_list.append(self.exact_match_score(prediction, ground_truth, None)["correct"])
            f1_list.append(self.f1_score(prediction, ground_truth, None)["f1"])

        em = float(sum(em_list) / max(len(em_list), 1))
        f1 = float(sum(f1_list) / max(len(f1_list), 1))
        print(f"EM: {em:.4f}\t F1: {f1:.4f}")


class WikiMultiHopEvaluator(MultiHopEvaluator):
    id_alias: Dict[str, List[str]] = {}

    def __init__(self, data_path: str = "data/multihop_data/2wikimultihopqa"):
        dataset = []
        with open(os.path.join(data_path, "dev.json"), "r", encoding="utf-8") as fin:
            js = json.load(fin)
            for example in tqdm(js, desc="Loading 2WikiMultiHopQA dev"):
                dataset.append(
                    {
                        "qid": example["_id"],
                        "question": example["question"],
                        "answer": example["answer"],
                        "answer_id": example["answer_id"],
                    }
                )
        self.dataset = dataset
        self.dataset_from_qid = {entry["qid"]: entry for entry in self.dataset}
        self.init_id_aliases(data_path)

    @classmethod
    def init_id_aliases(cls, data_path: str) -> None:
        cls.id_alias = {}
        with open(os.path.join(data_path, "id_aliases.json"), "r", encoding="utf-8") as fin:
            for l in fin:
                l = json.loads(l)
                cls.id_alias[l["Q_id"]] = l["aliases"]

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        return []

    def eval_answer(self, results_df: pd.DataFrame, answer_col: str = "Final Answer") -> None:
        em_list: List[int] = []
        f1_list: List[float] = []
        for _, row in results_df.iterrows():
            prediction = row.get(answer_col, "")
            ground_truth = row["ground_truth"]
            qid = row["qid"]
            ground_truth_id = self.dataset_from_qid[qid]["answer_id"]
            em_list.append(self.exact_match_score(prediction, ground_truth, ground_truth_id)["correct"])
            f1_list.append(self.f1_score(prediction, ground_truth, ground_truth_id)["f1"])

        em = float(sum(em_list) / max(len(em_list), 1))
        f1 = float(sum(f1_list) / max(len(f1_list), 1))
        print(f"EM: {em:.4f}\t F1: {f1:.4f}")


def load_results(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multihop results.jsonl with EM/F1.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["2wiki", "hotpotqa", "iirc"],
        help="Which dataset evaluator to use.",
    )
    parser.add_argument("--results", required=True, help="Path to results.jsonl (lines=True).")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["Final Answer", "Final Step Answer", "Final Read Answer"],
        help="Answer columns to evaluate.",
    )
    parser.add_argument(
        "--two_wiki_data_path",
        default="data/multihop_data/2wikimultihopqa",
        help="Only used when dataset=2wiki. Directory containing dev.json and id_aliases.json.",
    )
    args = parser.parse_args()

    results_df = load_results(args.results)
    print(len(results_df))

    if args.dataset == "2wiki":
        evaluator: MultiHopEvaluator = WikiMultiHopEvaluator(data_path=args.two_wiki_data_path)
    else:
        evaluator = MultiHopEvaluator()

    for column_name in args.columns:
        print(column_name)
        evaluator.eval_answer(results_df=results_df, answer_col=column_name)


if __name__ == "__main__":
    main()

