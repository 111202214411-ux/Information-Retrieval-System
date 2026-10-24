"""
vectordb.py: the index types inside a vector database, written in NumPy so every step is visible.

The database is the 30,000 bundled GloVe word vectors (50 dimensions, unit length, so inner product = cosine).

  FlatIndex   compares the query with every vector. Exact, cost grows linearly with the collection.
  IVFIndex    inverted file: k-means splits the vectors into `nlist` clusters; a query scans only the
              `nprobe` clusters whose centroids are closest. Fewer comparisons, may miss neighbours.
  PQIndex     product quantization: each vector is cut into `m` sub-vectors and every sub-vector is stored as
              the id (1 byte) of its nearest of 256 centroids. 200 bytes → m bytes. The query's similarity to
              every centroid is computed once (a lookup table) and a database score is a sum of m lookups.

FAISS implements the same ideas in C++ (IndexFlatIP, IndexIVFFlat, IndexPQ, plus HNSW graphs and GPU support);
ChromaDB and Weaviate wrap an index like these (HNSW) in a database with an API, storage and metadata filters.

This module does not depend on Streamlit. Try it from the terminal:
    python vectordb.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def kmeans(X: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd k-means (squared Euclidean). Returns (centroids, assignment)."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    x2 = (X ** 2).sum(1, keepdims=True)
    for _ in range(iters):
        d = x2 - 2 * X @ C.T + (C ** 2).sum(1)
        a = d.argmin(1)
        for j in range(k):
            members = X[a == j]
            C[j] = members.mean(0) if len(members) else X[rng.integers(len(X))]
    d = x2 - 2 * X @ C.T + (C ** 2).sum(1)
    return C.astype(np.float32), d.argmin(1)


@dataclass
class SearchResult:
    ids: np.ndarray
    scores: np.ndarray
    compared: int          # how many database vectors (or codes) were scored
    ms: float


class FlatIndex:
    kind = "Flat (exact)"

    def __init__(self, X: np.ndarray):
        self.X = X

    @property
    def bytes_per_vector(self) -> int:
        return self.X.shape[1] * 4

    def search(self, q: np.ndarray, k: int = 10) -> SearchResult:
        t = time.perf_counter()
        s = self.X @ q
        ids = np.argpartition(-s, k)[:k]
        ids = ids[np.argsort(-s[ids])]
        return SearchResult(ids, s[ids], len(self.X), (time.perf_counter() - t) * 1000)


class IVFIndex:
    kind = "IVF (clustered)"

    def __init__(self, X: np.ndarray, nlist: int = 100, seed: int = 0):
        self.X = X
        self.nlist = nlist
        self.centroids, assign = kmeans(X, nlist, seed=seed)
        self.lists = [np.flatnonzero(assign == j) for j in range(nlist)]

    @property
    def bytes_per_vector(self) -> int:
        return self.X.shape[1] * 4 + 8         # the vector plus its id in the inverted list

    def search(self, q: np.ndarray, k: int = 10, nprobe: int = 4) -> SearchResult:
        t = time.perf_counter()
        probe = np.argsort(-(self.centroids @ q))[:nprobe]
        cand = np.concatenate([self.lists[j] for j in probe])
        s = self.X[cand] @ q
        top = np.argsort(-s)[:k]
        return SearchResult(cand[top], s[top], len(cand), (time.perf_counter() - t) * 1000)


class PQIndex:
    kind = "PQ (compressed)"

    def __init__(self, X: np.ndarray, m: int = 10, ksub: int = 256, train_size: int = 10000, seed: int = 0):
        assert X.shape[1] % m == 0, "m must divide the dimension"
        self.m, self.ksub, self.dsub = m, ksub, X.shape[1] // m
        rng = np.random.default_rng(seed)
        train = X[rng.choice(len(X), size=min(train_size, len(X)), replace=False)]
        self.codebooks = []
        codes = np.empty((len(X), m), dtype=np.uint8)
        for j in range(m):
            sl = slice(j * self.dsub, (j + 1) * self.dsub)
            C, _ = kmeans(train[:, sl], ksub, iters=12, seed=seed + j)
            self.codebooks.append(C)
            sub = X[:, sl]
            d = (sub ** 2).sum(1, keepdims=True) - 2 * sub @ C.T + (C ** 2).sum(1)
            codes[:, j] = d.argmin(1)
        self.codes = codes
        self.n = len(X)

    @property
    def bytes_per_vector(self) -> int:
        return self.m                              # one byte per sub-vector

    def search(self, q: np.ndarray, k: int = 10) -> SearchResult:
        t = time.perf_counter()
        lut = np.stack([self.codebooks[j] @ q[j * self.dsub:(j + 1) * self.dsub] for j in range(self.m)])
        s = lut[np.arange(self.m), self.codes].sum(1)          # Σ_j LUT[j, code_j]
        ids = np.argpartition(-s, k)[:k]
        ids = ids[np.argsort(-s[ids])]
        return SearchResult(ids, s[ids], self.n, (time.perf_counter() - t) * 1000)

    def reconstruct(self, i: int) -> np.ndarray:
        return np.concatenate([self.codebooks[j][self.codes[i, j]] for j in range(self.m)])


def recall_at_k(found: np.ndarray, exact: np.ndarray) -> float:
    return len(set(found.tolist()) & set(exact.tolist())) / len(exact)


if __name__ == "__main__":
    import glove
    g = glove.load()
    X = g.vectors
    t = time.perf_counter(); flat = FlatIndex(X)
    ivf = IVFIndex(X, 100); pq = PQIndex(X, 10)
    print(f"built indexes in {time.perf_counter() - t:.1f}s")
    for word in ["search", "ranking", "rice"]:
        q = g.vec(word)
        ex = flat.search(q, 10)
        for idx, kw in [(ivf, {"nprobe": 4}), (pq, {})]:
            r = idx.search(q, 10, **kw)
            print(f"{word:8s} {idx.kind:16s} recall@10 {recall_at_k(r.ids, ex.ids):.1f}  compared {r.compared}")
        print("   exact:", [g.words[i] for i in ex.ids])
