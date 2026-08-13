from __future__ import annotations

import numpy as np

from . import embeddings as emb_mod
from .recommender import ContentRecommender


def _rng(seed):
    return np.random.default_rng(seed)


def add_typos(text: str, rate: float, rng) -> str:
    chars = list(text)
    n = len(chars)
    n_edits = int(n * rate)
    for _ in range(n_edits):
        if n < 2:
            break
        i = rng.integers(0, n - 1)
        op = rng.integers(0, 3)
        if op == 0:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == 1:
            chars[i] = ""
        else:
            chars[i] = chars[i] * 2
    return "".join(chars)


def word_dropout(text: str, rate: float, rng) -> str:
    words = text.split()
    if not words:
        return text
    keep = [w for w in words if rng.random() > rate]
    return " ".join(keep) if keep else words[0]


def truncate(text: str, frac: float, rng=None) -> str:
    words = text.split()
    n = max(1, int(len(words) * frac))
    return " ".join(words[:n])


def shuffle_words(text: str, frac: float, rng) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


PERTURBATIONS = {
    "typos": add_typos,
    "word_dropout": word_dropout,
    "truncation": truncate,
    "shuffle": shuffle_words,
}


def perturb_documents(docs, kind: str, level: float, seed: int = 0):
    fn = PERTURBATIONS[kind]
    rng = _rng(seed)
    return [fn(d, level, rng) for d in docs]


def jaccard_topk(a, b) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def run_perturbation_study(
    df,
    base_recommender: ContentRecommender,
    fitted_embedder: dict,
    embed_fn,
    relevance: dict,
    kinds=("typos", "word_dropout", "truncation"),
    levels=(0.1, 0.2, 0.4),
    k: int = 10,
    n_queries: int = 400,
    seed: int = 42,
) -> dict:

    rng = _rng(seed)
    query_pool = [q for q in relevance.keys()]
    if len(query_pool) > n_queries:
        query_pool = list(rng.choice(query_pool, size=n_queries, replace=False))
    docs = df["document"].values

    clean_topk = base_recommender.batch_topk(query_pool, k)
    clean_prec = []
    for qi, q in enumerate(query_pool):
        rel = relevance[q]
        clean_prec.append(np.mean([1.0 if r in rel else 0.0 for r in clean_topk[qi]]))
    clean_prec = float(np.mean(clean_prec))

    results = {"clean_precision@%d" % k: round(clean_prec, 4), "perturbations": {}}

    for kind in kinds:
        results["perturbations"][kind] = {}
        for level in levels:
            pert_docs = [
                PERTURBATIONS[kind](docs[q], level, _rng(seed + int(q)))
                for q in query_pool
            ]
            qvecs = embed_fn(fitted_embedder, pert_docs)
            stabilities, precisions = [], []
            for qi, q in enumerate(query_pool):
                idx, _ = base_recommender.recommend_by_vector(qvecs[qi], k=k, exclude=q)
                stabilities.append(jaccard_topk(idx, clean_topk[qi]))
                rel = relevance[q]
                precisions.append(np.mean([1.0 if r in rel else 0.0 for r in idx]))
            results["perturbations"][kind][str(level)] = {
                "topk_jaccard_stability": round(float(np.mean(stabilities)), 4),
                "precision@%d" % k: round(float(np.mean(precisions)), 4),
                "precision_retention": round(
                    float(np.mean(precisions)) / clean_prec if clean_prec else 0.0, 4
                ),
            }
    return results
