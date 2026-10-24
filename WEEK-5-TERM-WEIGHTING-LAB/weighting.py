"""
weighting.py: term weighting with TF-IDF variants and BM25 (Okapi), with every intermediate value.

TF variants (from the lecture):
    raw          f(t,d)
    normalized   f(t,d) / Σ f(t',d)          (share of the document)
    max          f(t,d) / max f(t',d)
    log          1 + log10 f(t,d)            (0 if f = 0)
    augmented    0.5 + 0.5 · f(t,d) / max f  (0 if f = 0)
    boolean      1 if f(t,d) > 0 else 0

    IDF(t)      = log10(N / df(t))
    TF-IDF(t,d) = TF(t,d) × IDF(t)

BM25:
    BM25(q,d) = Σ IDF(q_i) · f(q_i,d)·(k1+1) / ( f(q_i,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(q_i)  = ln( (N − n(q_i) + 0.5) / (n(q_i) + 0.5) + 1 )
    k1 = 1.2, b = 0.75 by default

This module does not depend on Streamlit. Try it from the terminal:
    python weighting.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
STOPWORDS = set("""a an the and or of in on at to for from with by is are was were be been this that these
those it its as into about how what which who do does can will your you we our their they my i me
while had has have one under next""".split())

TF_VARIANTS = {
    "raw": "Raw: f(t,d)",
    "normalized": "Normalized: f(t,d) / Σf(t',d)",
    "max": "Max-normalized: f(t,d) / max f",
    "log": "Log: 1 + log₁₀ f(t,d)",
    "augmented": "Augmented: 0.5 + 0.5 · f / max f",
    "boolean": "Boolean: 1 if f > 0",
}


def load_collections() -> dict:
    return json.loads((DATA_DIR / "collections.json").read_text(encoding="utf-8"))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if not (remove_stopwords and t in STOPWORDS)]


def tf_value(f: int, variant: str, total: int, max_f: int) -> float:
    if f == 0:
        return 0.0
    return {
        "raw": float(f),
        "normalized": f / total,
        "max": f / max_f,
        "log": 1 + math.log10(f),
        "augmented": 0.5 + 0.5 * f / max_f,
        "boolean": 1.0,
    }[variant]


class Collection:
    """Tokenized documents plus the statistics both TF-IDF and BM25 need."""

    def __init__(self, documents: dict[str, str], remove_stopwords: bool = True):
        self.docs = documents
        self.remove_stopwords = remove_stopwords
        self.tokens = {d: tokenize(t, remove_stopwords) for d, t in documents.items()}
        self.tf = {d: Counter(toks) for d, toks in self.tokens.items()}
        self.length = {d: len(toks) for d, toks in self.tokens.items()}
        self.N = len(documents)
        self.avgdl = sum(self.length.values()) / self.N if self.N else 0.0
        self.df = Counter(t for c in self.tf.values() for t in c)
        self.vocab = sorted(self.df)

    def query_terms(self, query: str) -> list[str]:
        return list(dict.fromkeys(tokenize(query, self.remove_stopwords)))

    # --- TF-IDF ---------------------------------------------------------------
    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log10(self.N / n) if n else 0.0

    def tfidf_vector(self, doc: str, variant: str) -> dict[str, float]:
        c = self.tf[doc]
        total, max_f = sum(c.values()), max(c.values(), default=1)
        return {t: tf_value(f, variant, total, max_f) * self.idf(t) for t, f in c.items()}

    def tfidf_table(self, doc: str) -> list[dict]:
        """One row per term of the document, with every TF variant and the IDF."""
        c = self.tf[doc]
        total, max_f = sum(c.values()), max(c.values(), default=1)
        rows = []
        for t, f in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            row = {"Term": t, "f(t,d)": f, "df": self.df[t], "IDF": self.idf(t)}
            for v in TF_VARIANTS:
                row[v] = tf_value(f, v, total, max_f)
            rows.append(row)
        return rows

    def tfidf_ranking(self, query: str, variant: str) -> list[tuple[str, float]]:
        """Cosine similarity between the TF-IDF vectors of the query and each document."""
        qc = Counter(tokenize(query, self.remove_stopwords))
        total, max_f = sum(qc.values()) or 1, max(qc.values(), default=1)
        q = {t: tf_value(f, variant, total, max_f) * self.idf(t) for t, f in qc.items()}
        qn = math.sqrt(sum(w * w for w in q.values()))
        out = []
        for d in self.docs:
            v = self.tfidf_vector(d, variant)
            vn = math.sqrt(sum(w * w for w in v.values()))
            dot = sum(q.get(t, 0.0) * w for t, w in v.items())
            out.append((d, dot / (qn * vn) if qn and vn else 0.0))
        return sorted(out, key=lambda x: -x[1])

    # --- BM25 -----------------------------------------------------------------
    def bm25_idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def bm25_trace(self, query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> list[dict]:
        """Every quantity in the BM25 formula, one row per query term."""
        rows = []
        dl = self.length[doc]
        norm = k1 * (1 - b + b * dl / self.avgdl) if self.avgdl else k1
        for t in self.query_terms(query):
            f = self.tf[doc].get(t, 0)
            idf = self.bm25_idf(t)
            num = f * (k1 + 1)
            den = f + norm
            rows.append({"Term": t, "f(t,d)": f, "n(t)": self.df.get(t, 0), "IDF": idf,
                         "|d|": dl, "avgdl": self.avgdl, "f·(k1+1)": num,
                         "f + k1·(1−b+b·|d|/avgdl)": den, "Contribution": idf * num / den if f else 0.0})
        return rows

    def bm25(self, query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> float:
        return sum(r["Contribution"] for r in self.bm25_trace(query, doc, k1, b))

    def bm25_ranking(self, query: str, k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
        return sorted(((d, self.bm25(query, d, k1, b)) for d in self.docs), key=lambda x: -x[1])


def substring_df(term: str, documents: dict[str, str]) -> int:
    """The df computed by the lecture code: `term in d.lower()` matches substrings, not words."""
    return sum(1 for d in documents.values() if term in d.lower())


if __name__ == "__main__":
    col = load_collections()["BM25 by hand (4 short documents)"]
    c = Collection(col["documents"])
    print(f"N = {c.N}, avgdl = {c.avgdl:.3f}, lengths = {c.length}")
    for d in c.docs:
        rows = c.bm25_trace("cat", d)
        r = rows[0]
        print(f"{d}: f={r['f(t,d)']} |d|={r['|d|']} IDF={r['IDF']:.4f} "
              f"num={r['f·(k1+1)']:.3f} den={r['f + k1·(1−b+b·|d|/avgdl)']:.4f} -> {r['Contribution']:.4f}")
    print("TF-IDF (log TF) ranking:", [(d, round(s, 4)) for d, s in c.tfidf_ranking("cat", "log")])
