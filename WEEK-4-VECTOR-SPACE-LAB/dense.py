"""
dense.py: dense retrieval with a bi-encoder, plus hybrid fusion with TF-IDF.

Bi-encoder: the query and each document are encoded separately into one vector,
    score(q, d) = cos(E(q), E(d))
so document vectors can be computed once, offline, and searched with an index such as FAISS.

Encoders, in order of preference:
  1. sentence-transformers/all-MiniLM-L6-v2 (384 dimensions), the model used in the lecture.
     Downloaded from the Hugging Face Hub the first time it is used.
  2. Offline fallback: the average of GloVe 6B 50d word vectors (bundled in data/). Much weaker,
     but needs no download, so the app still works without internet or without PyTorch.

This module does not depend on Streamlit. Try it from the terminal:
    python dense.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_STOP = set("""a an the and or of in on at to for from with by is are was were be been this that these
those it its as into about how what which who do does can will your you we our their they my i me""".split())


@dataclass
class Encoder:
    name: str
    dims: int
    encode: Callable[[list[str]], np.ndarray]     # returns unit-length rows
    note: str = ""


def _glove_encoder() -> Encoder:
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    words = [str(w) for w in data["words"]]
    vecs = data["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = {w: i for i, w in enumerate(words)}

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        for k, text in enumerate(texts):
            ids = [index[w] for w in re.findall(r"[a-z]+", text.lower()) if w in index and w not in _STOP]
            if ids:
                v = vecs[ids].mean(axis=0)
                out[k] = v / np.linalg.norm(v)
        return out

    return Encoder("Averaged GloVe word vectors (offline fallback)", vecs.shape[1], encode)


def load_encoder(prefer_minilm: bool = True) -> tuple[Encoder, str | None]:
    """Return (encoder, reason the preferred model could not be used, or None)."""
    if prefer_minilm:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MINILM, device="cpu")
            return Encoder("all-MiniLM-L6-v2 (sentence-transformers)",
                           (getattr(model, "get_embedding_dimension", None)
                            or model.get_sentence_embedding_dimension)(),
                           lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True),
                                                    dtype=np.float32)), None
        except ImportError:
            reason = "sentence-transformers is not installed"
        except Exception as e:                       # no internet, Hub unavailable, out of memory, …
            reason = f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
        return _glove_encoder(), reason
    return _glove_encoder(), None


class VectorIndex:
    """Exact inner-product search. Uses FAISS IndexFlatIP when installed, NumPy otherwise (same results)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.backend = "FAISS IndexFlatIP"
        except ImportError:
            self.index = None
            self.backend = "NumPy (exact inner product; install faiss-cpu to use FAISS)"

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return scores[0], ids[0]
        scores = self.vectors @ q[0]
        ids = np.argsort(-scores)[:k]
        return scores[ids], ids


def dense_ranking(encoder: Encoder, query: str, docs: dict[str, str]) -> list[tuple[str, float]]:
    names = list(docs)
    index = VectorIndex(encoder.encode([docs[n] for n in names]))
    scores, ids = index.search(encoder.encode([query])[0], len(names))
    return [(names[i], float(s)) for s, i in zip(scores, ids)]


def minmax(scores: dict[str, float]) -> dict[str, float]:
    lo, hi = min(scores.values()), max(scores.values())
    return {d: (s - lo) / (hi - lo) if hi > lo else 0.0 for d, s in scores.items()}


def hybrid_ranking(sparse: list[tuple[str, float]], dense: list[tuple[str, float]],
                   alpha: float) -> list[tuple[str, float]]:
    """score = α · dense + (1 − α) · sparse, each min-max normalized to [0, 1] first."""
    s, d = minmax(dict(sparse)), minmax(dict(dense))
    return sorted(((k, alpha * d[k] + (1 - alpha) * s.get(k, 0.0)) for k in d), key=lambda x: -x[1])


def reciprocal_rank(ranking: list[tuple[str, float]], relevant: set[str]) -> float:
    for pos, (d, _) in enumerate(ranking, 1):
        if d in relevant:
            return 1 / pos
    return 0.0


def hit_at_k(ranking: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d, _ in ranking[:k]))


if __name__ == "__main__":
    enc, why = load_encoder()
    print("Encoder:", enc.name, f"({enc.dims} dims)", "" if why is None else f"— fallback because {why}")
    docs = {"D1": "The cat sat on the mat", "D2": "Dogs are loyal pets",
            "D3": "Feline animals are independent", "D4": "Canine companions"}
    for d, s in dense_ranking(enc, "cat behavior", docs):
        print(f"  {d}  {s:.3f}  {docs[d]}")
