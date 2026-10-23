"""
vsm.py: the Vector Space Model (Salton et al., 1975), step by step.

Documents and the query become vectors of TF-IDF weights, one dimension per term:

    w(t, d) = TF(t, d) × IDF(t)          IDF(t) = log(N / df(t))
    cos(q, d) = Σ w(t,q)·w(t,d) / ( √Σ w(t,q)² × √Σ w(t,d)² )
    euclid(a, b) = √Σ (w(t,a) − w(t,b))²

Every choice the lecture examples make differently is an option:
  - TF:    raw count, binary (0/1), or 1 + log10(count)
  - IDF:   log base 10, 2, or e
  - query as a document: the old slides count the query in N and df (N = 4 for 3 documents)

This module does not depend on Streamlit. Try it from the terminal:
    python vsm.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

STOPWORDS = {
    "id": set("""yang dan di ke dari untuk dengan pada adalah ini itu atau juga dalam oleh sebagai
                 akan tidak bisa ada saya kami kita mereka para""".split()),
    "en": set("""a an the and or of in on at to for from with by is are was were be been this that
                 these those it its as into about how what which who do does can will your you we
                 our their they my i me must may up per""".split()),
}
LOG = {"10": math.log10, "2": math.log2, "e": math.log}
LOG_LABEL = {"10": "log₁₀", "2": "log₂", "e": "ln"}
QUERY_ID = "Q"


def load_collections() -> dict:
    return json.loads((DATA_DIR / "collections.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@dataclass(frozen=True)
class Settings:
    lang: str = "id"
    stopwords: bool = True
    stemming: bool = True                    # Sastrawi, Indonesian only
    tf: str = "raw"                          # "raw" | "binary" | "log"
    idf_base: str = "10"                     # "10" | "2" | "e"
    query_as_document: bool = False


def preprocess(text: str, s: Settings) -> list[str]:
    tokens = re.findall(r"[^\W\d_]+|\d+(?:\.\d+)?", text.lower())
    if s.stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS.get(s.lang, set())]
    if s.stemming and s.lang == "id":
        tokens = [_sastrawi().stem(t) for t in tokens]
    return tokens


def tf_weight(count: int, scheme: str) -> float:
    if count == 0:
        return 0.0
    return {"raw": float(count), "binary": 1.0, "log": 1 + math.log10(count)}[scheme]


@dataclass
class VSMResult:
    terms: list[str]
    columns: list[str]                       # Q first, then documents
    counts: np.ndarray                       # terms × columns, raw counts
    tf: np.ndarray                           # terms × columns, after the TF scheme
    df: np.ndarray                           # per term
    idf: np.ndarray                          # per term
    weights: np.ndarray                      # terms × columns, TF × IDF
    n: int                                   # N used in IDF
    ranking: list[tuple[str, float]] = field(default_factory=list)

    def col(self, name: str) -> np.ndarray:
        return self.weights[:, self.columns.index(name)]

    def norm(self, name: str) -> float:
        return float(np.sqrt((self.col(name) ** 2).sum()))

    def dot(self, a: str, b: str) -> float:
        return float(self.col(a) @ self.col(b))

    def cosine(self, a: str, b: str) -> float:
        na, nb = self.norm(a), self.norm(b)
        return self.dot(a, b) / (na * nb) if na and nb else 0.0

    def euclidean(self, a: str, b: str) -> float:
        return float(np.sqrt(((self.col(a) - self.col(b)) ** 2).sum()))


def build(documents: dict[str, str], query: str | None, s: Settings) -> VSMResult:
    """Build the term × document table for the documents (and the query, if given)."""
    texts = ({QUERY_ID: query} if query is not None else {}) | documents
    tokens = {name: preprocess(t, s) for name, t in texts.items()}
    terms = sorted({t for toks in tokens.values() for t in toks})
    columns = list(texts)
    counts = np.array([[Counter(tokens[c])[t] for c in columns] for t in terms], dtype=float).reshape(
        len(terms), len(columns))
    tf = np.vectorize(lambda x: tf_weight(int(x), s.tf))(counts) if counts.size else counts

    in_df = [c for c in columns if c != QUERY_ID or s.query_as_document]
    idx = [columns.index(c) for c in in_df]
    df = (counts[:, idx] > 0).sum(axis=1) if counts.size else np.zeros(0)
    n = len(in_df)
    log = LOG[s.idf_base]
    idf = np.array([log(n / d) if d else 0.0 for d in df])
    weights = tf * idf[:, None] if counts.size else counts

    res = VSMResult(terms, columns, counts, tf, df, idf, weights, n)
    if query is not None:
        res.ranking = sorted(((d, res.cosine(QUERY_ID, d)) for d in documents), key=lambda x: -x[1])
    return res


def kmeans_trace(res: VSMResult, docs: list[str], init: list[str], max_iter: int = 10) -> list[dict]:
    """
    K-means with Euclidean distance on the TF-IDF vectors, starting from the given documents as
    centroids. Returns one record per iteration: distances, assignment, and the new centroids.
    """
    centroids = [res.col(d).copy() for d in init]
    iterations = []
    previous = None
    for it in range(1, max_iter + 1):
        dist = {d: [float(np.sqrt(((res.col(d) - c) ** 2).sum())) for c in centroids] for d in docs}
        assign = {d: int(np.argmin(v)) for d, v in dist.items()}
        iterations.append({"iteration": it, "distances": dist, "assignment": assign,
                           "centroids": [c.copy() for c in centroids]})
        if assign == previous:
            break
        previous = assign
        new = []
        for k in range(len(centroids)):
            members = [res.col(d) for d in docs if assign[d] == k]
            new.append(np.mean(members, axis=0) if members else centroids[k])
        centroids = new
    return iterations


if __name__ == "__main__":
    cols = load_collections()
    c = cols["Rice plant symptoms (lecture worked example)"]
    r = build(c["documents"], c["query"], Settings(**c["settings"], lang=c["lang"]))
    print("terms:", r.terms, " N =", r.n)
    for t, d, i in zip(r.terms, r.df, r.idf):
        print(f"  {t:<8} df={int(d)}  idf={i:.5f}")
    for d, s in r.ranking:
        print(f"  cos(Q, {d}) = {r.dot('Q', d):.5f} / ({r.norm('Q'):.6f} × {r.norm(d):.6f}) = {s:.6f}")
