"""
systems.py: the retrieval systems that are evaluated, plus the dense encoder.

  Boolean AND   documents that contain every query term; an unranked set (shown in document-id order)
  TF-IDF        log TF × IDF (log10 N/df) weights, cosine similarity
  BM25          k1 = 1.2, b = 0.75, IDF = ln(1 + (N − n + 0.5) / (n + 0.5))
  Dense         all-MiniLM-L6-v2 bi-encoder, cosine similarity (offline fallback: averaged GloVe vectors)
  Hybrid (RRF)  reciprocal rank fusion of BM25 and Dense, RRF(d) = Σ 1 / (60 + rank)

Lexical systems return only documents that match at least one query term; Dense ranks every document.

This module does not depend on Streamlit.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEMS = ["Boolean AND", "TF-IDF", "BM25", "Dense", "Hybrid (RRF)"]

STOPWORDS = set("""a an the and or of in on at to for from with by is are was were be been this that these those it
its as into about how what which who do does can will your you we our their they my i me so than not but has have
each also when only every there here such like after before while far less them then one two much many vs
versus""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Dense encoder
# ---------------------------------------------------------------------------
@dataclass
class DenseEncoder:
    name: str
    dims: int
    encode: Callable[[list[str]], np.ndarray]     # unit-length rows
    real: bool


@lru_cache(maxsize=1)
def glove_encoder() -> DenseEncoder:
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


def load_dense_encoder(prefer_minilm: bool = True) -> tuple[DenseEncoder, str | None]:
    if not prefer_minilm:
        return glove_encoder(), None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MINILM, device="cpu")
    except ImportError:
        return glove_encoder(), "sentence-transformers is not installed"
    except Exception as e:                                   # no internet, Hub unavailable, out of memory, …
        return glove_encoder(), f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return DenseEncoder("all-MiniLM-L6-v2", dim_fn(),
                        lambda t: np.asarray(model.encode(t, normalize_embeddings=True), dtype=np.float32),
                        real=True), None


# ---------------------------------------------------------------------------
# Lexical index
# ---------------------------------------------------------------------------
class LexicalIndex:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.ids = list(docs)
        self.tf = {d: Counter(tokenize(t)) for d, t in docs.items()}
        self.length = {d: sum(c.values()) for d, c in self.tf.items()}
        self.N = len(docs)
        self.avgdl = sum(self.length.values()) / self.N
        self.df: Counter = Counter()
        for c in self.tf.values():
            self.df.update(c.keys())
        self.k1, self.b = k1, b
        # TF-IDF document vectors (log TF × IDF), with their norms for the cosine
        self.tfidf = {d: {t: (1 + math.log10(f)) * self.idf_log(t) for t, f in c.items()} for d, c in self.tf.items()}
        self.norm = {d: math.sqrt(sum(w * w for w in v.values())) for d, v in self.tfidf.items()}

    def idf_log(self, t: str) -> float:
        return math.log10(self.N / self.df[t]) if self.df[t] else 0.0

    def boolean_and(self, q: str) -> list[tuple[str, float]]:
        terms = set(tokenize(q))
        if not terms:
            return []
        return [(d, 1.0) for d in self.ids if terms <= set(self.tf[d])]

    def tfidf_rank(self, q: str) -> list[tuple[str, float]]:
        qv = {t: (1 + math.log10(f)) * self.idf_log(t) for t, f in Counter(tokenize(q)).items()}
        qn = math.sqrt(sum(w * w for w in qv.values()))
        out = []
        for d in self.ids:
            dot = sum(w * self.tfidf[d].get(t, 0.0) for t, w in qv.items())
            if dot > 0:
                out.append((d, dot / (qn * self.norm[d])))
        return sorted(out, key=lambda x: -x[1])

    def bm25_rank(self, q: str) -> list[tuple[str, float]]:
        out = []
        terms = tokenize(q)
        for d in self.ids:
            s = 0.0
            for t in terms:
                f = self.tf[d].get(t, 0)
                if f:
                    n = self.df[t]
                    idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.length[d] / self.avgdl))
            if s > 0:
                out.append((d, s))
        return sorted(out, key=lambda x: -x[1])


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in rankings:
        for r, (d, _) in enumerate(lst, 1):
            scores[d] = scores.get(d, 0.0) + 1 / (k + r)
    return sorted(scores.items(), key=lambda x: -x[1])


@dataclass
class Runs:
    rankings: dict[str, dict[str, list[tuple[str, float]]]]     # system → topic → ranking
    latency_ms: dict[str, float]                                 # mean time per query
    index_ms: dict[str, float]                                   # time to build the index


def run_all(docs: dict[str, str], topics: dict[str, str], encoder: DenseEncoder,
            doc_vectors: np.ndarray | None = None) -> Runs:
    ids = list(docs)
    t = time.perf_counter()
    lex = LexicalIndex(docs)
    lex_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    D = doc_vectors if doc_vectors is not None else encoder.encode([docs[d] for d in ids])
    dense_ms = (time.perf_counter() - t) * 1000

    def dense_rank(q: str) -> list[tuple[str, float]]:
        s = D @ encoder.encode([q])[0]
        return sorted(zip(ids, map(float, s)), key=lambda x: -x[1])

    fns = {"Boolean AND": lex.boolean_and, "TF-IDF": lex.tfidf_rank, "BM25": lex.bm25_rank, "Dense": dense_rank}
    rankings: dict[str, dict[str, list]] = {s: {} for s in SYSTEMS}
    latency: dict[str, float] = {}
    for name, fn in fns.items():
        t = time.perf_counter()
        for tid, q in topics.items():
            rankings[name][tid] = fn(q)
        latency[name] = (time.perf_counter() - t) * 1000 / len(topics)
    t = time.perf_counter()
    for tid in topics:
        rankings["Hybrid (RRF)"][tid] = rrf([rankings["BM25"][tid], rankings["Dense"][tid]])
    latency["Hybrid (RRF)"] = latency["BM25"] + latency["Dense"] + (time.perf_counter() - t) * 1000 / len(topics)
    index_ms = {"Boolean AND": lex_ms, "TF-IDF": lex_ms, "BM25": lex_ms, "Dense": dense_ms,
                "Hybrid (RRF)": lex_ms + dense_ms}
    return Runs(rankings, latency, index_ms)


if __name__ == "__main__":
    import json
    import metrics
    data = json.loads((DATA_DIR / "collection.json").read_text())
    docs = {d["id"]: d["text"] for d in data["documents"]}
    topics = {t["id"]: t["query"] for t in data["topics"]}
    enc, why = load_dense_encoder(False)
    runs = run_all(docs, topics, enc)
    qrels = data["qrels"]["Judge A"]
    for s in SYSTEMS:
        res = [metrics.evaluate([d for d, _ in runs.rankings[s][t]], qrels[t]) for t in topics]
        print(f"{s:13s}", {m: round(np.mean([r[m] for r in res]), 3) for m in res[0]})
