"""
embeddings.py: word embeddings and the three retrieval approaches of the lecture case study.

Vectors: the 30,000 most frequent words of GloVe 6B, 50 dimensions (Pennington et al., 2014),
stored in data/glove_50d_30k.npz. GloVe is released under the Public Domain Dedication and
License (PDDL).

This module does not depend on Streamlit. Try it from the terminal:
    python embeddings.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# small English stopword list, used only to average document embeddings and for TF-IDF
STOPWORDS = set("""a an the and or of in on at to for from with by is are was were be been this that
these those it its as into about of over under than then so such not no nor can will just do does
did has have had their there they them he she we you your our i me my his her""".split())


@lru_cache(maxsize=1)
def load() -> tuple[list[str], dict[str, int], np.ndarray]:
    """Return (words, word -> row, unit-length vectors as float32)."""
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    words = [str(w) for w in data["words"]]
    vectors = data["vectors"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return words, {w: i for i, w in enumerate(words)}, vectors


def raw_vector(word: str) -> np.ndarray | None:
    """The original (not normalized) GloVe vector, for display."""
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    _, index, _ = load()
    i = index.get(word.lower())
    return None if i is None else data["vectors"][i].astype(np.float32)


def has(word: str) -> bool:
    return word.lower() in load()[1]


def vector(word: str) -> np.ndarray:
    words, index, vecs = load()
    return vecs[index[word.lower()]]


def similarity(a: str, b: str) -> float:
    """Cosine similarity between two words (vectors are unit length, so it is a dot product)."""
    return float(vector(a) @ vector(b))


def nearest(v: np.ndarray, k: int = 10, exclude: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    words, _, vecs = load()
    v = v / (np.linalg.norm(v) or 1.0)
    scores = vecs @ v
    out = []
    for i in np.argsort(-scores):
        if words[i] not in exclude:
            out.append((words[i], float(scores[i])))
        if len(out) == k:
            break
    return out


def neighbors(word: str, k: int = 10) -> list[tuple[str, float]]:
    return nearest(vector(word), k, exclude=(word.lower(),))


def analogy(a: str, b: str, c: str, k: int = 5) -> list[tuple[str, float]]:
    """a − b + c, e.g. king − man + woman ≈ queen (3CosAdd)."""
    v = vector(a) - vector(b) + vector(c)
    return nearest(v, k, exclude=(a.lower(), b.lower(), c.lower()))


def project_2d(words: list[str]) -> np.ndarray:
    """PCA to two dimensions, for plotting a handful of words."""
    m = np.stack([vector(w) for w in words])
    m = m - m.mean(axis=0)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    return m @ vt[:2].T


def char_ngrams(word: str, n_min: int = 3, n_max: int = 6) -> list[str]:
    """FastText-style character n-grams with < and > as word boundaries."""
    w = f"<{word.lower()}>"
    return [w[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)]


# ---------------------------------------------------------------------------
# Document retrieval: Boolean vs TF-IDF vs averaged embeddings
# ---------------------------------------------------------------------------

def words_of(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def doc_embedding(text: str) -> tuple[np.ndarray | None, list[str]]:
    """Average of the word vectors (bag of embeddings). Returns (vector, out-of-vocabulary words)."""
    ws = words_of(text)
    known = [w for w in ws if has(w)]
    oov = [w for w in ws if not has(w)]
    if not known:
        return None, oov
    v = np.mean([vector(w) for w in known], axis=0)
    return v / np.linalg.norm(v), oov


def embedding_ranking(query: str, docs: dict[str, str]) -> list[tuple[str, float]]:
    q, _ = doc_embedding(query)
    if q is None:
        return []
    scores = []
    for d, text in docs.items():
        v, _ = doc_embedding(text)
        scores.append((d, float(q @ v) if v is not None else 0.0))
    return sorted(scores, key=lambda x: -x[1])


def tfidf_ranking(query: str, docs: dict[str, str]) -> list[tuple[str, float]]:
    """Cosine similarity of TF-IDF vectors, w = (1 + log10 tf) * log10(N / df)."""
    tfs = {d: Counter(words_of(t)) for d, t in docs.items()}
    n = len(docs)
    df = Counter(w for tf in tfs.values() for w in tf)
    idf = {w: math.log10(n / c) for w, c in df.items()}

    def vec(tf: Counter) -> dict[str, float]:
        return {w: (1 + math.log10(f)) * idf.get(w, 0.0) for w, f in tf.items()}

    q = vec(Counter(words_of(query)))
    qn = math.sqrt(sum(x * x for x in q.values()))
    out = []
    for d, tf in tfs.items():
        v = vec(tf)
        vn = math.sqrt(sum(x * x for x in v.values()))
        dot = sum(q.get(w, 0.0) * x for w, x in v.items())
        out.append((d, dot / (qn * vn) if qn and vn else 0.0))
    return sorted(out, key=lambda x: -x[1])


if __name__ == "__main__":
    print("king · queen      =", round(similarity("king", "queen"), 3))
    print("car · automobile  =", round(similarity("car", "automobile"), 3))
    print("king − man + woman ≈", [(w, round(s, 3)) for w, s in analogy("king", "man", "woman", 3)])
    print("neighbors of bank:", [w for w, _ in neighbors("bank", 8)])
