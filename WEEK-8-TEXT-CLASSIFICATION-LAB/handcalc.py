"""
handcalc.py: Naive Bayes and K-Nearest Neighbor written out step by step, for the worked examples of the lecture.

Multinomial Naive Bayes (slides 8.2 and P08):

    P(c | d) ∝ P(c) · Π_i P(w_i | c)
    P(c)     = documents of class c / all documents
    P(w | c) = (count(w, c) + α) / (count(c) + α·|V|)

    count(w, c) = occurrences of w in the documents of class c
    count(c)    = number of words (tokens) in the documents of class c
    |V|         = number of distinct words in the training documents
    α           = 1 is Laplace smoothing

K-Nearest Neighbor (slide 9.2):

    weight w_{t,d} = tf_{t,d} · log10(N / df_t)      (N and df count every document, the test document included)
    cos(d_i, d_j)  = Σ_t w_{t,i} w_{t,j} / (‖d_i‖ · ‖d_j‖)
    the class of the test document = the majority among its k most similar training documents

This module does not depend on Streamlit. Try it from the terminal:
    python handcalc.py
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------
@dataclass
class NBModel:
    classes: list[str]
    vocab: list[str]
    prior: dict[str, float]
    n_docs: dict[str, int]
    counts: dict[str, Counter]          # class → word → count
    tokens: dict[str, int]              # class → count(c), number of tokens
    alpha: float

    def likelihood(self, w: str, c: str) -> float:
        return (self.counts[c][w] + self.alpha) / (self.tokens[c] + self.alpha * len(self.vocab))


def nb_train(docs: list[tuple[list[str], str]], alpha: float = 1.0) -> NBModel:
    classes = list(dict.fromkeys(c for _, c in docs))
    vocab = list(dict.fromkeys(w for toks, _ in docs for w in toks))
    counts = {c: Counter(w for toks, cc in docs if cc == c for w in toks) for c in classes}
    n_docs = {c: sum(cc == c for _, cc in docs) for c in classes}
    return NBModel(classes, vocab, {c: n_docs[c] / len(docs) for c in classes}, n_docs, counts,
                   {c: sum(counts[c].values()) for c in classes}, alpha)


@dataclass
class NBScore:
    cls: str
    prior: float
    terms: list[tuple[str, int, float]]     # word, count(w, c), P(w | c)
    product: float
    log_score: float


def nb_classify(model: NBModel, test: list[str], unseen: str = "laplace") -> list[NBScore]:
    """unseen = 'laplace': a word that is not in the vocabulary still gets α / (count(c) + α|V|), as on the slide;
    unseen = 'ignore': such words are skipped, as scikit-learn does."""
    words = [w for w in test if unseen == "laplace" or w in model.vocab]
    out = []
    for c in model.classes:
        terms = [(w, model.counts[c][w], model.likelihood(w, c)) for w in words]
        prod = model.prior[c] * math.prod(p for _, _, p in terms)
        out.append(NBScore(c, model.prior[c], terms, prod,
                           math.log(model.prior[c]) + sum(math.log(p) for _, _, p in terms)))
    return out


def nb_posterior(scores: list[NBScore]) -> dict[str, float]:
    """Normalize the scores so they sum to 1 (P(c | d) without the constant P(d))."""
    m = max(s.log_score for s in scores)
    e = {s.cls: math.exp(s.log_score - m) for s in scores}
    z = sum(e.values())
    return {c: v / z for c, v in e.items()}


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------
@dataclass
class TfIdfTable:
    names: list[str]
    terms: list[str]
    tf: dict[str, Counter]
    df: dict[str, int]
    idf: dict[str, float]
    weights: dict[str, dict[str, float]]


def tfidf_table(docs: dict[str, list[str]]) -> TfIdfTable:
    names = list(docs)
    terms = list(dict.fromkeys(t for d in names for t in docs[d]))
    tf = {d: Counter(docs[d]) for d in names}
    df = {t: sum(t in tf[d] for d in names) for t in terms}
    idf = {t: math.log10(len(names) / df[t]) for t in terms}
    weights = {d: {t: tf[d][t] * idf[t] for t in terms} for d in names}
    return TfIdfTable(names, terms, tf, df, idf, weights)


@dataclass
class Neighbor:
    name: str
    label: str
    dot: float
    length: float
    score: float          # cosine similarity, or a distance for Euclidean / Manhattan


def knn_scores(table: TfIdfTable, test: str, labels: dict[str, str], metric: str = "cosine") -> list[Neighbor]:
    q = table.weights[test]
    q_len = math.sqrt(sum(v * v for v in q.values()))
    out = []
    for d in table.names:
        if d == test:
            continue
        w = table.weights[d]
        dot = sum(q[t] * w[t] for t in table.terms)
        length = math.sqrt(sum(v * v for v in w.values()))
        if metric == "cosine":
            score = dot / (length * q_len) if length and q_len else 0.0
        elif metric == "euclidean":
            score = math.sqrt(sum((q[t] - w[t]) ** 2 for t in table.terms))
        else:
            score = sum(abs(q[t] - w[t]) for t in table.terms)
        out.append(Neighbor(d, labels[d], dot, length, score))
    # most similar first: highest cosine, or smallest distance; a stable sort keeps document order on ties
    return sorted(out, key=lambda n: -n.score if metric == "cosine" else n.score)


def vote(ranked: list[Neighbor], k: int) -> tuple[dict[str, int], list[str]]:
    """Votes of the k nearest neighbours, and the class or classes with the most votes."""
    votes = Counter(n.label for n in ranked[:k])
    best = max(votes.values())
    return dict(votes), [c for c, v in votes.items() if v == best]


if __name__ == "__main__":
    train = [("sepakbola indah menyerang", "Olahraga"), ("presiden menaikan harga bbm", "Politik"),
             ("partai politik indonesia berburu suara", "Politik"), ("manchester united juara liga inggris", "Olahraga"),
             ("timnas indonesia gagal juara afc", "Olahraga")]
    m = nb_train([(t.split(), c) for t, c in train])
    print("|V| =", len(m.vocab), "count(c) =", m.tokens)
    for s in nb_classify(m, "psis berburu juara liga indonesia".split()):
        print(f"  {s.cls}: prior {s.prior} · {[round(p, 5) for _, _, p in s.terms]} = {s.product:.6e}")
    docs = {"D1": "partai golkar demokrat tanding kampanye 2009", "D2": "tanding pertama persema persebaya malang",
            "D3": "besar wasit tanding sepakbola adil", "D4": "partai demokrat menang pemilu 2009 figur sby",
            "D5": "tanding sepakbola persebaya kampanye pemilu 2009 tunda"}
    tab = tfidf_table({k: v.split() for k, v in docs.items()})
    labels = {"D1": "Politik", "D2": "Olahraga", "D3": "Olahraga", "D4": "Politik"}
    ranked = knn_scores(tab, "D5", labels)
    print([(n.name, round(n.dot, 6), round(n.score, 5)) for n in ranked], vote(ranked, 3))
