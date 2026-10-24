"""
dense.py: dense retrieval with a bi-encoder, hybrid fusion, and evaluation metrics.

Encoders, in order of preference:
  1. sentence-transformers/all-MiniLM-L6-v2 (384 dimensions), downloaded on first use.
  2. Offline fallback: average of GloVe 6B 50d word vectors (bundled in data/).

Hybrid fusion follows the Session 5 slides, where α is the weight of BM25:

    hybrid = α · BM25_norm + (1 − α) · dense

with BM25 divided by its maximum (as in the lecture code: sparse / (sparse.max() + 1e-8)).
α = 0.5 is balanced, 0.7 favors exact matches, 0.3 favors meaning.

This module does not depend on Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from weighting import STOPWORDS

DATA_DIR = Path(__file__).parent / "data"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DenseEncoder:
    name: str
    dims: int
    encode: Callable[[list[str]], np.ndarray]     # unit-length rows
    real: bool


@lru_cache(maxsize=1)
def _glove_encoder() -> DenseEncoder:
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    words = [str(w) for w in data["words"]]
    vecs = data["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = {w: i for i, w in enumerate(words)}

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        for k, text in enumerate(texts):
            ids = [index[w] for w in re.findall(r"[a-z]+", text.lower()) if w in index and w not in STOPWORDS]
            if ids:
                v = vecs[ids].mean(axis=0)
                out[k] = v / np.linalg.norm(v)
        return out

    return DenseEncoder("Averaged GloVe word vectors (offline fallback)", vecs.shape[1], encode, real=False)


def load_dense_encoder(prefer_minilm: bool) -> tuple[DenseEncoder, str | None]:
    if prefer_minilm:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MINILM, device="cpu")
            dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            return DenseEncoder("all-MiniLM-L6-v2", dim_fn(),
                                lambda t: np.asarray(model.encode(t, normalize_embeddings=True), dtype=np.float32),
                                real=True), None
        except ImportError:
            reason = "sentence-transformers is not installed"
        except Exception as e:
            reason = f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
        return _glove_encoder(), reason
    return _glove_encoder(), None


def normalize(scores: dict[str, float], method: str) -> dict[str, float]:
    """'max': divide by the maximum (lecture code). 'minmax': rescale to [0, 1]."""
    if not scores:
        return {}
    hi, lo = max(scores.values()), min(scores.values())
    if method == "max":
        return {d: s / (hi + 1e-8) for d, s in scores.items()}
    return {d: (s - lo) / (hi - lo) if hi > lo else 0.0 for d, s in scores.items()}


def hybrid(bm25: list[tuple[str, float]], dense: list[tuple[str, float]], alpha: float,
           method: str = "max") -> list[tuple[str, float]]:
    """α · BM25 + (1 − α) · dense, BM25 normalized; dense cosine normalized the same way only for 'minmax'."""
    s = normalize(dict(bm25), method)
    d = dict(dense) if method == "max" else normalize(dict(dense), method)
    # iterate in BM25 order so that ties keep BM25's order: α = 1 reproduces BM25 exactly
    order = [k for k, _ in bm25] + [k for k in d if k not in s]
    return sorted(((k, alpha * s.get(k, 0.0) + (1 - alpha) * d.get(k, 0.0)) for k in order), key=lambda x: -x[1])


def precision_at_k(ranking: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    """Relevant documents in the top k ÷ k. A document with score 0 was not retrieved and never counts."""
    return sum(1 for d, s in ranking[:k] if d in relevant and s > 0) / k


def average_precision(ranking: list[tuple[str, float]], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, (d, s) in enumerate(ranking, 1):
        if d in relevant and s > 0:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0
