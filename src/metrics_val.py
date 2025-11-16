from math import log2


def hit_rate(recommendations, ground_truth, k=100):
    """
    Calculate HitRate@k

    Args:
        recommendations: dict {user_id: list of recommended item_ids}
        ground_truth: dict {user_id: set of relevant item_ids}
        k: cutoff level
    """
    hits = 0
    total_users = len(recommendations)

    for user_id, recs in recommendations.items():
        user_recs = recs[:k]
        user_truth = ground_truth.get(user_id, set())
        if any(item in user_truth for item in user_recs):
            hits += 1

    return hits / total_users if total_users > 0 else 0.0


def precision(recommendations, ground_truth, k=100):
    """
    Calculate Precision@k
    """
    precisions = []

    for user_id, recs in recommendations.items():
        user_recs = recs[:k]
        user_truth = ground_truth.get(user_id, set())
        relevant_count = sum(1 for item in user_recs if item in user_truth)
        user_precision = relevant_count / k
        precisions.append(user_precision)

    return np.mean(precisions) if precisions else 0.0


def recall(recommendations, ground_truth, k=100):
    """
    Calculate Recall@k
    """
    recalls = []

    for user_id, recs in recommendations.items():
        user_recs = recs[:k]
        user_truth = ground_truth.get(user_id, set())

        if not user_truth:  # If no ground truth items, recall is 0
            recalls.append(0.0)
            continue

        relevant_count = sum(1 for item in user_recs if item in user_truth)
        user_recall = relevant_count / len(user_truth)
        recalls.append(user_recall)

    return np.mean(recalls) if recalls else 0.0


def mrr(recommendations, ground_truth, k=100):
    """
    Calculate MRR@k
    """
    reciprocal_ranks = []

    for user_id, recs in recommendations.items():
        user_recs = recs[:k]
        user_truth = ground_truth.get(user_id, set())

        user_rr = 0.0
        for rank, item in enumerate(user_recs, 1):
            if item in user_truth:
                user_rr = 1.0 / rank
                break

        reciprocal_ranks.append(user_rr)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg(recommendations, ground_truth, k=100, binary_relevance=True):
    """
    Calculate NDCG@k

    Args:
        binary_relevance: if True, uses binary relevance (0/1),
                         if False, expects relevance scores in ground_truth
    """
    ndcg_scores = []

    for user_id, recs in recommendations.items():
        user_recs = recs[:k]
        user_truth = ground_truth.get(user_id, {})

        # Calculate DCG
        dcg = 0.0
        for rank, item in enumerate(user_recs, 1):
            if binary_relevance:
                rel = 1.0 if item in user_truth else 0.0
            else:
                rel = user_truth.get(item, 0.0)

            dcg += rel / (log2(rank + 1) if rank == 1 else 1)

        # Calculate IDCG
        if binary_relevance:
            # For binary relevance, ideal is all 1's sorted first
            num_relevant = len(user_truth)
            ideal_gains = [1.0] * min(k, num_relevant)
        else:
            # For graded relevance, take top-k relevance scores
            ideal_gains = sorted(user_truth.values(), reverse=True)[:k]

        idcg = 0.0
        for rank, rel in enumerate(ideal_gains, 1):
            idcg += rel / (log2(rank + 1) if rank == 1 else 1)

        user_ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(user_ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate_model(
    df: pd.DataFrame, preds_col: str, gt_col: str, top_k: int = 20
) -> dict:
    recommendations = pd.Series(df[preds_col].values, index=df["user_id"]).to_dict()

    ground_truth = pd.Series(
        df[gt_col].apply(set).values, index=df["user_id"]
    ).to_dict()

    hr = hit_rate(recommendations, ground_truth, k=top_k)
    p = precision(recommendations, ground_truth, k=top_k)
    r = recall(recommendations, ground_truth, k=top_k)
    m = mrr(recommendations, ground_truth, k=top_k)
    n = ndcg(recommendations, ground_truth, k=top_k)

    results = {
        f"hit_rate@{top_k}": hr,
        f"precision@{top_k}": p,
        f"recall@{top_k}": r,
        f"mrr@{top_k}": m,
        f"ndcg@{top_k}": n,
    }

    return results
