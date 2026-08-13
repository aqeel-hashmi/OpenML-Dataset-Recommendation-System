from __future__ import annotations

import pandas as pd

from . import config


def _is_topical(tag: str) -> bool:
    t = tag.strip()
    if not t:
        return False
    low = t.lower()
    if any(low.startswith(p) for p in config.NON_TOPICAL_TAG_PREFIXES):
        return False
    if t in config.NOISE_TAGS:
        return False
    if t.isdigit() or len(t) < 3:
        return False
    return True


def _is_study(tag: str) -> bool:
    return tag.strip().lower().startswith("study_")


def topical_tags(tags: list[str]) -> set[str]:
    return {t for t in tags if _is_topical(t)}


def study_tags(tags: list[str]) -> set[str]:
    return {t for t in tags if _is_study(t)}


def build_relevance(
    df: pd.DataFrame, method: str = "topical", max_df_frac: float = 0.05
) -> dict[int, set[int]]:
    extractor = {"topical": topical_tags, "study": study_tags}[method]
    n_total = len(df)

    label_freq: dict[str, int] = {}
    for tags in df["tags"]:
        for lab in extractor(list(tags)):
            label_freq[lab] = label_freq.get(lab, 0) + 1

    df_cap = max_df_frac * n_total if method == "topical" else n_total + 1
    allowed = {lab for lab, f in label_freq.items() if f <= df_cap}

    label_to_rows: dict[str, set[int]] = {}
    row_labels: dict[int, set[str]] = {}
    for row_idx, tags in zip(df["row_idx"], df["tags"]):
        labels = {lab for lab in extractor(list(tags)) if lab in allowed}
        if not labels:
            continue
        row_labels[int(row_idx)] = labels
        for lab in labels:
            label_to_rows.setdefault(lab, set()).add(int(row_idx))

    relevance: dict[int, set[int]] = {}
    for row_idx, labels in row_labels.items():
        rel: set[int] = set()
        for lab in labels:
            rel |= label_to_rows[lab]
        rel.discard(row_idx)
        if rel:
            relevance[row_idx] = rel
    return relevance


def relevance_stats(relevance: dict[int, set[int]]) -> dict:
    sizes = [len(v) for v in relevance.values()]
    if not sizes:
        return {"n_queries": 0}
    return {
        "n_queries": len(relevance),
        "avg_relevant_per_query": round(sum(sizes) / len(sizes), 2),
        "median_relevant_per_query": int(sorted(sizes)[len(sizes) // 2]),
        "max_relevant_per_query": max(sizes),
    }
