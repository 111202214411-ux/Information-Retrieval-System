"""
metrics.py: every evaluation measure in the lecture, written out so each step can be shown.

Notation: `ranked` is a ranked list of document ids (best first); `qrels` maps document id → relevance grade
(0 = not relevant, 1 = marginal, 2 = relevant, 3 = highly relevant); a document is *relevant* for the binary
measures when its grade ≥ `threshold` (1 by default, the TREC convention). Unjudged documents count as 0.

Set-based (unranked retrieval)
    Precision  P = |relevant ∩ retrieved| / |retrieved| = tp / (tp + fp)
    Recall     R = |relevant ∩ retrieved| / |relevant|  = tp / (tp + fn)
    F1         = 2PR / (P + R);   Fβ = (1 + β²) P R / (β² P + R)
    Accuracy   = (tp + tn) / (tp + fp + fn + tn)

Ranked
    P@k        = |relevant in top k| / k
    R@k        = |relevant in top k| / |relevant|
    AP         = (1 / |Rel|) Σ_k P@k · rel(k)        (|Rel| = all relevant documents, retrieved or not)
    MAP        = mean of AP over the queries
    RR         = 1 / rank of the first relevant document;  MRR = mean of RR
    DCG@k      = Σ_{i=1..k} gain(rel_i) / log2(i + 1)
                 gain = rel            ("linear", the notebook code and trec_eval / ir_measures)
                 gain = 2^rel − 1      ("exponential", the slide and the notebook's formula cell)
    nDCG@k     = DCG@k / IDCG@k, IDCG = DCG of the ideal ordering of all judged documents

Agreement and significance
    Cohen's κ  = (P(A) − P(E)) / (1 − P(E))
    paired randomization test on per-query scores (Smucker, Allan & Carterette, 2007)

This module does not depend on Streamlit. Its results match ir_measures (see `python metrics.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def relevant_set(qrels: dict[str, int], threshold: int = 1) -> set[str]:
    return {d for d, g in qrels.items() if g >= threshold}


# ---------------------------------------------------------------------------
# Set-based
# ---------------------------------------------------------------------------
@dataclass
class SetScores:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def f_beta(self, beta: float = 1.0) -> float:
        p, r = self.precision, self.recall
        return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0

    @property
    def f1(self) -> float:
        return self.f_beta(1.0)

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0


def set_scores(retrieved: set[str], relevant: set[str], collection_size: int) -> SetScores:
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    return SetScores(tp, fp, fn, collection_size - tp - fp - fn)


# ---------------------------------------------------------------------------
# Ranked
# ---------------------------------------------------------------------------
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranked[:k]) / len(relevant) if relevant else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1 / i
    return 0.0


def r_precision(ranked: list[str], relevant: set[str]) -> float:
    return precision_at_k(ranked, relevant, len(relevant)) if relevant else 0.0


def gain(rel: float, kind: str = "linear") -> float:
    return float(rel) if kind == "linear" else 2.0 ** rel - 1


def dcg(grades: list[float], k: int, kind: str = "linear") -> float:
    return sum(gain(g, kind) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_from_grades(grades: list[float], k: int, kind: str = "linear",
                     unretrieved: list[float] | None = None) -> float:
    """nDCG@k from the grades of a ranked list. The ideal ranking uses the retrieved grades plus the grades of
    relevant documents that were not retrieved (the notebook leaves those out, which inflates nDCG)."""
    ideal = sorted(list(grades) + list(unretrieved or []), reverse=True)
    idcg = dcg(ideal, k, kind)
    return dcg(grades, k, kind) / idcg if idcg > 0 else 0.0


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int, kind: str = "linear") -> float:
    grades = [qrels.get(d, 0) for d in ranked]
    idcg = dcg(sorted(qrels.values(), reverse=True), k, kind)
    return dcg(grades, k, kind) / idcg if idcg > 0 else 0.0


def pr_points(ranked: list[str], relevant: set[str]) -> list[tuple[float, float]]:
    """(recall, precision) after every retrieved document."""
    pts, hits = [], 0
    for i, d in enumerate(ranked, 1):
        hits += d in relevant
        pts.append((hits / len(relevant) if relevant else 0.0, hits / i))
    return pts


def interpolated_11pt(ranked: list[str], relevant: set[str]) -> list[float]:
    """Interpolated precision at recall 0.0, 0.1, …, 1.0: the highest precision at any recall ≥ r."""
    pts = pr_points(ranked, relevant)
    out = []
    for r in np.linspace(0, 1, 11):
        ps = [p for rec, p in pts if rec >= r - 1e-12]
        out.append(max(ps) if ps else 0.0)
    return out


def evaluate(ranked: list[str], qrels: dict[str, int], threshold: int = 1, kind: str = "linear") -> dict[str, float]:
    rel = relevant_set(qrels, threshold)
    return {"P@5": precision_at_k(ranked, rel, 5), "P@10": precision_at_k(ranked, rel, 10),
            "R@10": recall_at_k(ranked, rel, 10), "R-Prec": r_precision(ranked, rel),
            "AP": average_precision(ranked, rel), "RR": reciprocal_rank(ranked, rel),
            "nDCG@10": ndcg_at_k(ranked, qrels, 10, kind)}


# ---------------------------------------------------------------------------
# Agreement and significance
# ---------------------------------------------------------------------------
@dataclass
class Kappa:
    both_yes: int
    a_yes_b_no: int
    a_no_b_yes: int
    both_no: int

    @property
    def n(self) -> int:
        return self.both_yes + self.a_yes_b_no + self.a_no_b_yes + self.both_no

    @property
    def p_agree(self) -> float:
        return (self.both_yes + self.both_no) / self.n if self.n else 0.0

    @property
    def p_chance(self) -> float:
        """Cohen: P(E) = P_A(yes)·P_B(yes) + P_A(no)·P_B(no)."""
        if not self.n:
            return 0.0
        a_yes = (self.both_yes + self.a_yes_b_no) / self.n
        b_yes = (self.both_yes + self.a_no_b_yes) / self.n
        return a_yes * b_yes + (1 - a_yes) * (1 - b_yes)

    @property
    def kappa(self) -> float:
        pe = self.p_chance
        return (self.p_agree - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_from_labels(a: list[bool], b: list[bool]) -> Kappa:
    return Kappa(sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                 sum(not x and y for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b)))


def randomization_test(a: list[float], b: list[float], trials: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomization (sign-flip) test of mean(a) − mean(b). Returns the p-value."""
    d = np.asarray(a) - np.asarray(b)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(trials, len(d)))
    observed = abs(d.mean())
    return float((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).mean())


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall's τ between two orderings of the same items (no ties)."""
    pos = {x: i for i, x in enumerate(order_b)}
    n, s = len(order_a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if pos[order_a[i]] < pos[order_a[j]] else -1
    return s / (n * (n - 1) / 2) if n > 1 else 1.0


if __name__ == "__main__":
    rel = {"D1", "D3", "D5"}
    ret = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    for k in (1, 3, 5, 7):
        print(f"P@{k}={precision_at_k(ret, rel, k):.3f}, R@{k}={recall_at_k(ret, rel, k):.3f}")
    print("AP", round(average_precision(ret, rel), 4))
    g = [3, 0, 2, 0, 1, 0, 0]
    print("nDCG@5 linear", round(ndcg_from_grades(g, 5), 4), "exponential", round(ndcg_from_grades(g, 5, "exp"), 4))
    q = {"d1": 2, "d3": 1, "d5": 1}
    run = ["d1", "d2", "d3", "d4", "d5"]
    print("slide example:", {m: round(v, 4) for m, v in evaluate(run, q).items()},
          "R@5", recall_at_k(run, relevant_set(q), 5))
    k = Kappa(300, 20, 10, 70)
    print(f"IIR kappa example: P(A)={k.p_agree:.3f} P(E)={k.p_chance:.3f} kappa={k.kappa:.3f}")
