"""
topics.py: topic models and how to judge them.

  LDA          Latent Dirichlet Allocation (Blei, Ng & Jordan, 2003): every document is a mixture of topics, every
               topic a distribution over words; fitted with scikit-learn on word counts
  c-TF-IDF     class-based TF-IDF from BERTopic (P09 slide):  w_{t,c} = tf_{t,c} · log(1 + A / tf_t)
               tf_{t,c} = frequency of t in cluster c (its documents joined), tf_t = frequency of t in the whole
               corpus, A = average number of words per cluster
  BERTopic     (Grootendorst, 2022) documents → sentence embeddings → UMAP → HDBSCAN → c-TF-IDF
               UMAP and HDBSCAN use BERTopic's defaults: n_neighbors 15, n_components 5, min_dist 0, cosine;
               HDBSCAN with the euclidean metric and "eom" cluster selection; documents HDBSCAN cannot place
               get the outlier topic −1

Quality measures
  NPMI coherence   for the top words of a topic, mean over word pairs of
                   log(P(wi, wj) / (P(wi) P(wj))) / −log P(wi, wj), with probabilities from document
                   co-occurrence in the same corpus (−1: never together, 0: independent, 1: always together)
  topic diversity  share of distinct words among the top words of all topics (Dieng et al., 2020)
  silhouette       how well separated the clusters are, measured in the sentence-embedding space
  ARI / NMI        agreement with known categories, when the corpus has them

This module does not depend on Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------------
# c-TF-IDF
# ---------------------------------------------------------------------------
@dataclass
class CTfidf:
    clusters: list[int]
    vocab: np.ndarray
    tf: np.ndarray                 # clusters × terms, raw frequency in the joined documents of each cluster
    tf_t: np.ndarray               # terms, frequency in the whole corpus
    A: float                       # average number of words per cluster
    weights: np.ndarray            # clusters × terms

    def top(self, cluster: int, n: int = 10) -> list[tuple[str, float]]:
        i = self.clusters.index(cluster)
        order = np.argsort(-self.weights[i])[:n]
        return [(self.vocab[j], float(self.weights[i, j])) for j in order if self.tf[i, j] > 0]


def c_tfidf(texts: list[str], labels, stopwords=frozenset()) -> CTfidf:
    labels = np.asarray(labels)
    clusters = sorted(set(labels.tolist()))
    cv = CountVectorizer(stop_words=sorted(stopwords) or None, token_pattern=r"(?u)\b\w\w+\b")
    X = cv.fit_transform(texts)
    tf = np.vstack([np.asarray(X[labels == c].sum(axis=0)).ravel() for c in clusters]).astype(float)
    tf_t = tf.sum(axis=0)
    A = tf.sum(axis=1).mean()
    weights = tf * np.log(1 + A / np.where(tf_t == 0, 1, tf_t))
    return CTfidf(clusters, cv.get_feature_names_out(), tf, tf_t, float(A), weights)


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------
@dataclass
class LDAResult:
    model: LatentDirichletAllocation
    vocab: np.ndarray
    doc_topic: np.ndarray          # documents × topics, rows sum to 1
    labels: np.ndarray             # most probable topic per document

    def top(self, topic: int, n: int = 10) -> list[tuple[str, float]]:
        comp = self.model.components_[topic]
        p = comp / comp.sum()
        return [(self.vocab[j], float(p[j])) for j in np.argsort(-p)[:n]]


def lda(texts: list[str], k: int, stopwords=frozenset(), seed: int = 42, alpha: float | None = None,
        beta: float | None = None) -> LDAResult:
    cv = CountVectorizer(stop_words=sorted(stopwords) or None, token_pattern=r"(?u)\b\w\w+\b")
    X = cv.fit_transform(texts)
    m = LatentDirichletAllocation(n_components=k, random_state=seed, learning_method="batch", max_iter=50,
                                  doc_topic_prior=alpha, topic_word_prior=beta).fit(X)
    dt = m.transform(X)
    return LDAResult(m, cv.get_feature_names_out(), dt, dt.argmax(axis=1))


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------
def reduce(emb: np.ndarray, method: str = "UMAP", n_components: int = 5, seed: int = 42) -> np.ndarray:
    n = len(emb)
    dims = max(2, min(n_components, n - 2))
    if method == "UMAP":
        import umap
        return umap.UMAP(n_neighbors=max(2, min(15, n - 1)), n_components=dims, min_dist=0.0, metric="cosine",
                         random_state=seed).fit_transform(emb)
    return PCA(n_components=min(dims, emb.shape[1]), random_state=seed).fit_transform(emb)


def hdbscan(reduced: np.ndarray, min_cluster_size: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric="euclidean",
                   cluster_selection_method="eom").fit_predict(reduced)


@dataclass
class BERTopicResult:
    reduced: np.ndarray
    labels: np.ndarray
    ctfidf: CTfidf

    @property
    def topics(self) -> list[int]:
        return [c for c in self.ctfidf.clusters if c != -1]


def bertopic(emb: np.ndarray, texts: list[str], min_cluster_size: int, reducer: str = "UMAP",
             stopwords=frozenset(), seed: int = 42, reduced: np.ndarray | None = None) -> BERTopicResult:
    red = reduce(emb, reducer, seed=seed) if reduced is None else reduced
    labels = hdbscan(red, min_cluster_size)
    return BERTopicResult(red, labels, c_tfidf(texts, labels, stopwords))


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
def doc_sets(texts: list[str]) -> list[set[str]]:
    return [set(t.split()) for t in texts]


def npmi(words: list[str], docs: list[set[str]]) -> float:
    N = len(docs)
    df = {w: sum(w in d for d in docs) for w in words}
    vals = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            a, b = words[i], words[j]
            if not df[a] or not df[b]:
                continue
            both = sum(a in d and b in d for d in docs)
            if both == 0:
                vals.append(-1.0)
                continue
            p_ab, p_a, p_b = both / N, df[a] / N, df[b] / N
            vals.append(1.0 if p_ab == 1 else math.log(p_ab / (p_a * p_b)) / -math.log(p_ab))
    return float(np.mean(vals)) if vals else float("nan")


def coherence(topics: dict[int, list[str]], texts: list[str]) -> dict[int, float]:
    docs = doc_sets(texts)
    return {t: npmi(w, docs) for t, w in topics.items()}


def diversity(topics: dict[int, list[str]]) -> float:
    words = [w for ws in topics.values() for w in ws]
    return len(set(words)) / len(words) if words else float("nan")


def map_2d(reduced: np.ndarray, seed: int = 0) -> np.ndarray:
    """A 2-D view for plotting: PCA of the reduced embeddings."""
    return PCA(n_components=2, random_state=seed).fit_transform(reduced)


if __name__ == "__main__":
    import corpora
    import embeddings
    c = corpora.load(corpora.NAMES[1])
    emb = embeddings.lsa(c.clean, c.stopwords)
    for red in ("PCA", "UMAP"):
        res = bertopic(emb, c.clean, c.default_min_size, red, c.stopwords)
        print(red, "topics", res.topics, "outliers", int((res.labels == -1).sum()))
        for t in res.topics:
            print("  ", t, [w for w, _ in res.ctfidf.top(t, 6)])
    L = lda(c.clean, 3, c.stopwords)
    tops = {t: [w for w, _ in L.top(t, 10)] for t in range(3)}
    print("LDA", tops, "coherence", {t: round(v, 3) for t, v in coherence(tops, c.clean).items()},
          "diversity", round(diversity(tops), 3))
