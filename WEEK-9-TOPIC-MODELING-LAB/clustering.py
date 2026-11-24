"""
clustering.py: K-Means written out step by step, and the tools for choosing K.

K-Means (P09 slide):

    objective   J = Σ_{i=1..K} Σ_{x ∈ C_i} ‖x − μ_i‖²
    1. initialise K centroids μ_1 … μ_K (here: K documents chosen at random)
    2. assignment: put every document in the cluster of its nearest centroid
    3. update: move every centroid to the mean of its documents, μ_i = (1/|C_i|) Σ_{x ∈ C_i} x
    4. repeat 2–3 until no document changes cluster

Choosing K: the elbow method (J against K) and the silhouette score
    s(x) = (b − a) / max(a, b), a = mean distance to its own cluster, b = mean distance to the nearest other cluster,
    from −1 (wrong cluster) to 1 (well separated); the score of a clustering is the mean over documents.

This module does not depend on Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


@dataclass
class Step:
    iteration: int
    centroids: np.ndarray          # centroids used for this assignment
    distances: np.ndarray          # documents × K, Euclidean
    labels: np.ndarray
    J: float
    changed: int                   # documents that changed cluster in this assignment


def kmeans_steps(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 50) -> list[Step]:
    """Lloyd's algorithm with every iteration recorded. Initial centroids are k distinct random documents."""
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].astype(float)
    steps: list[Step] = []
    prev = None
    for it in range(1, max_iter + 1):
        sq = (X ** 2).sum(axis=1)[:, None] - 2 * X @ centroids.T + (centroids ** 2).sum(axis=1)[None, :]
        d = np.sqrt(np.maximum(sq, 0))
        labels = d.argmin(axis=1)
        J = float(sum(((X[labels == j] - centroids[j]) ** 2).sum() for j in range(k)))
        changed = len(X) if prev is None else int((labels != prev).sum())
        steps.append(Step(it, centroids.copy(), d, labels, J, changed))
        if prev is not None and changed == 0:
            break
        prev = labels
        new = centroids.copy()
        for j in range(k):
            if (labels == j).any():
                new[j] = X[labels == j].mean(axis=0)
        centroids = new
    return steps


@dataclass
class KChoice:
    k: int
    inertia: float
    silhouette: float


def choose_k(X, ks: list[int], seed: int = 42, metric: str = "euclidean") -> list[KChoice]:
    out = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_, metric=metric) if 1 < k < X.shape[0] else float("nan")
        out.append(KChoice(k, float(km.inertia_), float(sil)))
    return out


def kmeans(X, k: int, seed: int = 42) -> KMeans:
    """The slide's call: KMeans(n_clusters=k, random_state=42, n_init=10)."""
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)


def silhouettes(X, labels, metric: str = "euclidean") -> np.ndarray:
    return silhouette_samples(X, labels, metric=metric)


if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer
    import corpora
    X = TfidfVectorizer(max_features=100).fit_transform(corpora.SLIDE_DOCS)
    km = kmeans(X, 3)
    print("slide code: labels", km.labels_, "silhouette", round(silhouette_score(X, km.labels_), 4))
    steps = kmeans_steps(X.toarray(), 3, seed=0)
    for s in steps:
        print(f"  iteration {s.iteration}: J = {s.J:.4f}, changed {s.changed}, labels {s.labels}")
    for c in choose_k(X, list(range(2, 8))):
        print(f"  K={c.k}: J={c.inertia:.4f} silhouette={c.silhouette:.4f}")
