from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config


@dataclass
class EmbeddingResult:
    name: str
    vectors: np.ndarray            # dense, L2-normalised (n, d)
    dim: int
    params: dict = field(default_factory=dict)


def build_tfidf(documents, params: dict | None = None):
    p = {**config.TFIDF_DEFAULTS, **(params or {})}
    vectorizer = TfidfVectorizer(
        max_features=p["max_features"],
        ngram_range=tuple(p["ngram_range"]),
        min_df=p["min_df"],
        max_df=p["max_df"],
        sublinear_tf=p["sublinear_tf"],
        stop_words="english",
        strip_accents="unicode",
    )
    tfidf = vectorizer.fit_transform(documents)

    svd = None
    n_comp = int(p.get("svd_components", 0) or 0)
    if n_comp > 0:
        n_comp = min(n_comp, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=config.RANDOM_STATE)
        vectors = svd.fit_transform(tfidf)
    else:
        vectors = tfidf

    vectors = normalize(vectors)
    if sparse.issparse(vectors):
        vectors = np.asarray(vectors.todense())
    vectors = vectors.astype(np.float32)

    result = EmbeddingResult(
        name="tfidf_svd" if svd is not None else "tfidf",
        vectors=vectors,
        dim=vectors.shape[1],
        params=p,
    )
    return result, {"vectorizer": vectorizer, "svd": svd}


def transform_tfidf(fitted: dict, documents) -> np.ndarray:
    X = fitted["vectorizer"].transform(documents)
    if fitted["svd"] is not None:
        X = fitted["svd"].transform(X)
    X = normalize(X)
    if sparse.issparse(X):
        X = np.asarray(X.todense())
    return X.astype(np.float32)

_MODEL_CACHE: dict[str, object] = {}

def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def build_semantic(documents, model_name: str | None = None, batch_size: int = 64):
    model_name = model_name or config.SEMANTIC_MODEL_NAME
    model = _get_model(model_name)
    docs = [d[:2000] for d in documents]
    vectors = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    result = EmbeddingResult(
        name="semantic",
        vectors=vectors,
        dim=vectors.shape[1],
        params={"model_name": model_name},
    )
    return result, {"model": model}


def transform_semantic(fitted: dict, documents) -> np.ndarray:
    model = fitted["model"]
    docs = [d[:2000] for d in documents]
    return model.encode(
        docs, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def cache_path(name: str):
    return config.MODEL_DIR / f"emb_{name}.npy"


def save_embedding(result: EmbeddingResult):
    np.save(cache_path(result.name), result.vectors)


def load_embedding(name: str) -> np.ndarray | None:
    p = cache_path(name)
    return np.load(p) if p.exists() else None
