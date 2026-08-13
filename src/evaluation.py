from __future__ import annotations

import numpy as np

from .recommender import ContentRecommender


def _dcg(rels: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels * discounts))


def evaluate(
    recommender: ContentRecommender,
    relevance: dict[int, set[int]],
    k_values=(5, 10, 20),
    max_queries: int | None = None,
    seed: int = 42,
) -> dict:
    queries = list(relevance.keys())
    if max_queries is not None and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        queries = list(rng.choice(queries, size=max_queries, replace=False))

    kmax = max(k_values)
    topk = recommender.batch_topk(queries, kmax)

    agg = {f"precision@{k}": [] for k in k_values}
    agg.update({f"recall@{k}": [] for k in k_values})
    agg.update({f"map@{k}": [] for k in k_values})
    agg.update({f"ndcg@{k}": [] for k in k_values})

    for qi, q in enumerate(queries):
        rel = relevance[q]
        n_rel = len(rel)
        ranked = topk[qi]
        hits = np.array([1.0 if r in rel else 0.0 for r in ranked])
        for k in k_values:
            hk = hits[:k]
            n_hit = hk.sum()
            agg[f"precision@{k}"].append(n_hit / k)
            agg[f"recall@{k}"].append(n_hit / min(n_rel, k) if n_rel else 0.0)

            if n_hit > 0:
                csum = np.cumsum(hk)
                ranks = np.arange(1, k + 1)
                ap = np.sum((csum / ranks) * hk) / min(n_rel, k)
            else:
                ap = 0.0
            agg[f"map@{k}"].append(ap)

            idcg = _dcg(np.ones(min(n_rel, k)))
            agg[f"ndcg@{k}"].append(_dcg(hk) / idcg if idcg > 0 else 0.0)

    results = {m: round(float(np.mean(v)), 4) for m, v in agg.items()}
    results["n_queries_evaluated"] = len(queries)
    return results


def evaluate_all_signals(
    recommender: ContentRecommender,
    relevance_signals: dict[str, dict],
    k_values=(5, 10, 20),
    max_queries: int | None = 1000,
) -> dict:
    out = {}
    for signal_name, rel in relevance_signals.items():
        out[signal_name] = evaluate(
            recommender, rel, k_values=k_values, max_queries=max_queries
        )
    return out


def random_baseline(
    relevance: dict[int, set[int]], n_items: int, k_values=(5, 10, 20),
    max_queries: int | None = 1000, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    queries = list(relevance.keys())
    if max_queries and len(queries) > max_queries:
        queries = list(rng.choice(queries, size=max_queries, replace=False))
    out = {}
    for k in k_values:
        precs = [len(relevance[q]) / n_items for q in queries]  # E[P@k]
        out[f"precision@{k}"] = round(float(np.mean(precs)), 4)
    return out
