from __future__ import annotations
from collections.abc import Sequence
from math import log2
from typing import Any
import numpy as np
import pandas as pd


class RankingMetrics:
    def __init__(self, top_k: int = 20, binary_relevance: bool = True) -> None:
        self.top_k = top_k
        self.binary_relevance = binary_relevance

    def _resolve_top_k(self, top_k: int | None) -> int:
        value = self.top_k if top_k is None else top_k
        return value

    @staticmethod
    def _ensure_sequence(values: Any) -> list[int]:
        if values is None:
            return []
        if isinstance(values, list):
            return [int(v) for v in values]
        if isinstance(values, tuple):
            return [int(v) for v in values]
        if isinstance(values, np.ndarray):
            return [int(v) for v in values.tolist()]
        return [int(values)]

    def hit_rate(self,recommendations: dict[int, Sequence[int]],ground_truth: dict[int, set[int]],top_k: int | None = None) -> float:
        k = self._resolve_top_k(top_k)
        hits = 0
        total_users = len(recommendations)
        for user_id, recs in recommendations.items():
            user_recs = recs[:k]
            user_truth = ground_truth.get(user_id, set())
            if any(item in user_truth for item in user_recs):
                hits += 1
        return hits / total_users if total_users > 0 else 0.0

    def precision(self, recommendations: dict[int, Sequence[int]], ground_truth: dict[int, set[int]], top_k: int | None = None) -> float:
        k = self._resolve_top_k(top_k)
        precisions = []
        for user_id, recs in recommendations.items():
            user_recs = recs[:k]
            user_truth = ground_truth.get(user_id, set())
            relevant_count = sum(1 for item in user_recs if item in user_truth)
            precisions.append(relevant_count / k)
        return float(np.mean(precisions)) if precisions else 0.0

    def recall(self, recommendations: dict[int, Sequence[int]], ground_truth: dict[int, set[int]], top_k: int | None = None) -> float:
        k = self._resolve_top_k(top_k)
        recalls = []
        for user_id, recs in recommendations.items():
            user_recs = recs[:k]
            user_truth = ground_truth.get(user_id, set())

            if not user_truth:
                recalls.append(0.0)
                continue

            relevant_count = sum(1 for item in user_recs if item in user_truth)
            recalls.append(relevant_count / len(user_truth))

        return float(np.mean(recalls)) if recalls else 0.0

    def mrr(self, recommendations: dict[int, Sequence[int]], ground_truth: dict[int, set[int]], top_k: int | None = None) -> float:
        k = self._resolve_top_k(top_k)
        reciprocal_ranks = []

        for user_id, recs in recommendations.items():
            user_recs = recs[:k]
            user_truth = ground_truth.get(user_id, set())

            rr_value = 0.0
            for rank, item in enumerate(user_recs, 1):
                if item in user_truth:
                    rr_value = 1.0 / rank
                    break
            reciprocal_ranks.append(rr_value)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def ndcg(self, recommendations: dict[int, Sequence[int]], ground_truth: dict[int, set[int]] | dict[int, dict[int, float]], top_k: int | None = None) -> float:
        k = self._resolve_top_k(top_k)
        ndcg_scores = []

        for user_id, recs in recommendations.items():
            user_recs = recs[:k]
            user_truth = ground_truth.get(user_id, {})

            dcg = 0.0
            for rank, item in enumerate(user_recs, 1):
                if self.binary_relevance:
                    rel = 1.0 if item in user_truth else 0.0
                else:
                    rel = user_truth.get(item, 0.0)
                dcg += rel / (log2(rank + 1))

            if self.binary_relevance:
                num_relevant = len(user_truth)
                ideal_gains = [1.0] * min(k, num_relevant)
            else:
                ideal_gains = sorted(user_truth.values(), reverse=True)[:k]

            idcg = 0.0
            for rank, rel in enumerate(ideal_gains, 1):
                idcg += rel / (log2(rank + 1))

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def evaluate_frame(self, df: pd.DataFrame, preds_col: str, gt_col: str, top_k: int | None = None) -> dict[str, float]:
        k = self._resolve_top_k(top_k)
        eval_df = df[df[gt_col].map(len) > 0].copy()
        recommendations = dict(zip(eval_df['user_id'], eval_df[preds_col]))
        ground_truth = dict(zip(eval_df['user_id'], eval_df[gt_col].apply(set)))

        if not recommendations:
            return {metric: 0.0 for metric in self.metric_names(k)}

        metrics = {
            f'hit_rate@{k}': self.hit_rate(recommendations, ground_truth, top_k=k),
            f'precision@{k}': self.precision(recommendations, ground_truth, top_k=k),
            f'recall@{k}': self.recall(recommendations, ground_truth, top_k=k),
            f'mrr@{k}': self.mrr(recommendations, ground_truth, top_k=k),
            f'ndcg@{k}': self.ndcg(recommendations, ground_truth, top_k=k),
        }
        return metrics

    def evaluate_sequence(self, preds: Sequence[int], ground_truth: Sequence[int], top_k: int | None = None) -> dict[str, float]:
        k = self._resolve_top_k(top_k)
        recs = list(preds)[:k]
        truth_set = set(ground_truth)

        if not recs:
            return {metric: 0.0 for metric in self.metric_names(k)}

        hits = [item in truth_set for item in recs]
        hit = 1.0 if any(hits) else 0.0
        precision_val = sum(hits) / k
        recall_val = (sum(hits) / len(truth_set)) if truth_set else 0.0

        rr_val = 0.0
        for idx, item in enumerate(recs, 1):
            if item in truth_set:
                rr_val = 1.0 / idx
                break

        ideal_gains = [1.0] * min(len(truth_set), k)
        idcg = sum(rel / log2(idx + 1) for idx, rel in enumerate(ideal_gains, 1))
        dcg = 0.0
        for idx, rel in enumerate(hits, 1):
            dcg += float(rel) / log2(idx + 1)
        ndcg_val = dcg / idcg if idcg > 0 else 0.0

        return {
            f'hit_rate@{k}': hit,
            f'precision@{k}': precision_val,
            f'recall@{k}': recall_val,
            f'mrr@{k}': rr_val,
            f'ndcg@{k}': ndcg_val,
        }

    @staticmethod
    def metric_names(top_k: int) -> list[str]:
        return [
            f'hit_rate@{top_k}',
            f'precision@{top_k}',
            f'recall@{top_k}',
            f'mrr@{top_k}',
            f'ndcg@{top_k}',
        ]

    def get_metric_names(self, top_k: int | None = None) -> list[str]:
        return self.metric_names(self._resolve_top_k(top_k))