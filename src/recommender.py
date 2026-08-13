from __future__ import annotations

import numpy as np


class ContentRecommender:
    def __init__(self, vectors: np.ndarray, ids=None, names=None):

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = (vectors / norms).astype(np.float32)
        self.n, self.dim = self.vectors.shape
        self.ids = np.asarray(ids) if ids is not None else np.arange(self.n)
        self.names = np.asarray(names) if names is not None else None

    def _scores_for_vector(self, qvec: np.ndarray) -> np.ndarray:
        q = qvec.astype(np.float32).ravel()
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        return self.vectors @ q  # (n,)

    @staticmethod
    def _topk(scores: np.ndarray, k: int, exclude: int | None = None):
        if exclude is not None:
            scores = scores.copy()
            scores[exclude] = -np.inf
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]


    def recommend_by_index(self, row_idx: int, k: int = 10):
        scores = self.vectors @ self.vectors[row_idx]
        idx, sims = self._topk(scores, k, exclude=row_idx)
        return idx, sims

    def recommend_by_vector(self, qvec: np.ndarray, k: int = 10, exclude=None):
        scores = self._scores_for_vector(qvec)
        idx, sims = self._topk(scores, k, exclude=exclude)
        return idx, sims

    def rank_all(self, row_idx: int) -> np.ndarray:
        scores = self.vectors @ self.vectors[row_idx]
        scores[row_idx] = -np.inf
        return np.argsort(-scores)

    def batch_topk(self, query_indices, k: int) -> np.ndarray:
        query_indices = np.asarray(query_indices)
        out = np.empty((len(query_indices), k), dtype=np.int64)
        chunk = 512
        for start in range(0, len(query_indices), chunk):
            qs = query_indices[start:start + chunk]
            sims = self.vectors[qs] @ self.vectors.T  # (c, n)
            # mask self-similarity
            for r, qi in enumerate(qs):
                sims[r, qi] = -np.inf
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            rows = np.arange(len(qs))[:, None]
            order = np.argsort(-sims[rows, part], axis=1)
            out[start:start + len(qs)] = part[rows, order]
        return out
