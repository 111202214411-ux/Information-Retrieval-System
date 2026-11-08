"""
evaluate.py — Retrieval experiment for the Bookshelf search engine
==================================================================

Compares five scenarios and three ablations on the test queries in queries.json,
whose relevance judgments were fixed before any system was run.

  S1  Substring baseline   : case-insensitive substring match (like SQL LIKE '%q%'),
                             unranked, sorted alphabetically
  S2  Boolean AND          : every query word must appear in the title, unranked
  S3  VSM, no stemming     : TF-IDF + cosine similarity, stopword removal only
  S4  VSM + stemming       : the proposed ranking model (cosine results only)
  S5  S4 + partial match   : the full system as deployed (ir.search)

  Ablations of S5: without stopword removal, without stemming, without partial matching (= S4)

Metrics: P@5, recall, MAP, MRR, and mean latency per query (ms).
Significance: Wilcoxon signed-rank test on per-query average precision.

Usage:
    python evaluation/evaluate.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ir  # noqa: E402

BOOKS = json.loads((HERE.parent / "books.json").read_text(encoding="utf-8"))
QUERIES = json.loads((HERE / "queries.json").read_text(encoding="utf-8"))
TITLES = [b["title"] for b in BOOKS]
N = len(BOOKS)


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------
def alphabetical(ids):
    return sorted(ids, key=lambda d: (ir.fold(TITLES[d]), d))


def s1_substring(query: str) -> list[int]:
    q = re.sub(r"\s+", " ", ir.fold(query.strip()))
    return alphabetical(d for d in range(N) if q and q in ir.fold(TITLES[d]))


def _tokens(text: str, stop: bool) -> list[str]:
    tokens = ir.preprocess(text).tokens
    return [t for t in tokens if not (stop and t in ir.STOPWORDS)]


_BOOL_DOCS = [set(_tokens(t, stop=True)) for t in TITLES]


def s2_boolean(query: str) -> list[int]:
    q = set(_tokens(query, stop=True))
    return alphabetical(d for d in range(N) if q and q <= _BOOL_DOCS[d])


class VSM:
    """TF-IDF + cosine similarity with switchable stemming and stopword removal.

    With stem=True and stop=True it reproduces the ranking of ir.search exactly.
    """

    def __init__(self, stem: bool, stop: bool):
        self.stem, self.stop = stem, stop
        tf = [self._tf(t) for t in TITLES]
        df: dict[str, int] = {}
        for m in tf:
            for t in m:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log10(N / n) for t, n in df.items()}
        self.w = [{t: f * self.idf[t] for t, f in m.items()} for m in tf]
        self.norm = [math.sqrt(sum(x * x for x in w.values())) for w in self.w]

    def _terms(self, text: str) -> list[str]:
        return [ir.stem(t) if self.stem else t for t in _tokens(text, self.stop)]

    def _tf(self, text: str) -> dict[str, int]:
        m: dict[str, int] = {}
        for t in self._terms(text):
            m[t] = m.get(t, 0) + 1
        return m

    def rank(self, query: str) -> list[int]:
        q = {t: f * self.idf[t] for t, f in self._tf(query).items() if t in self.idf}
        qn = math.sqrt(sum(x * x for x in q.values()))
        scored = []
        for d in range(N):
            dot = sum(wq * self.w[d].get(t, 0.0) for t, wq in q.items())
            if dot > 0:
                scored.append((-(dot / (qn * self.norm[d])), ir.fold(TITLES[d]), d))
        return [d for *_, d in sorted(scored)]


def with_partial(ranked: list[int], query: str) -> list[int]:
    """Append partial matches after the ranked list, in the same order as ir.search."""
    text = re.sub(r"\s+", " ", ir.fold(query.strip()))
    order = {k: i for i, (k, _) in enumerate(ir.PARTIAL_KINDS)}
    hits = []
    for d in range(N):
        if d in ranked:
            continue
        m = ir.partial_match(TITLES[d], text)
        if m:
            hits.append((order[m[0]], ir.fold(TITLES[d]), d))
    return ranked + [d for *_, d in sorted(hits)]


INDEX = ir.build_index(BOOKS, "title")
VSM_NOSTEM = VSM(stem=False, stop=True)
VSM_STEM = VSM(stem=True, stop=True)
VSM_NOSTOP = VSM(stem=True, stop=False)

SCENARIOS = {
    "S1 Substring (baseline)": s1_substring,
    "S2 Boolean AND": s2_boolean,
    "S3 VSM tanpa stemming": VSM_NOSTEM.rank,
    "S4 VSM + stemming": VSM_STEM.rank,
    "S5 Usulan lengkap": lambda q: [r.doc_id for r in ir.search(INDEX, q).results],
}
ABLATIONS = {
    "S5 tanpa stopword removal": lambda q: with_partial(VSM_NOSTOP.rank(q), q),
    "S5 tanpa stemming": lambda q: with_partial(VSM_NOSTEM.rank(q), q),
    "S5 tanpa pencocokan potongan kata (= S4)": VSM_STEM.rank,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    hits, precisions, rr = 0, [], 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            precisions.append(hits / i)
            if not rr:
                rr = 1 / i
    return {
        "P@5": sum(d in relevant for d in ranked[:5]) / 5,
        "Recall": hits / len(relevant),
        "AP": sum(precisions) / len(relevant),
        "RR": rr,
    }


def latency_ms(fn, repeats: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        for q in QUERIES:
            fn(q["query"])
    return (time.perf_counter() - start) / (repeats * len(QUERIES)) * 1000


def evaluate(systems: dict) -> dict:
    out = {}
    for name, fn in systems.items():
        per_query = [metrics(fn(q["query"]), set(q["relevant"])) for q in QUERIES]
        out[name] = {
            "per_query": per_query,
            "P@5": statistics.mean(m["P@5"] for m in per_query),
            "Recall": statistics.mean(m["Recall"] for m in per_query),
            "MAP": statistics.mean(m["AP"] for m in per_query),
            "MRR": statistics.mean(m["RR"] for m in per_query),
            "Latency (ms)": latency_ms(fn),
        }
    return out


def by_category(result: dict) -> dict[str, float]:
    cats: dict[str, list[float]] = {}
    for q, m in zip(QUERIES, result["per_query"]):
        cats.setdefault(q["category"], []).append(m["AP"])
    return {c: statistics.mean(v) for c, v in cats.items()}


def wilcoxon(a: list[float], b: list[float]) -> str:
    try:
        from scipy.stats import wilcoxon as w
    except ImportError:
        return "scipy not installed"
    diffs = [x - y for x, y in zip(a, b)]
    nonzero = sum(1 for x in diffs if x)
    if nonzero == 0:
        return "no difference on any query"
    res = w(a, b, zero_method="wilcox", alternative="two-sided")
    return f"W = {res.statistic:.1f}, p = {res.pvalue:.4f}, n(non-zero) = {nonzero}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # sanity check: the reimplemented VSM must match the deployed engine exactly
    for q in QUERIES:
        deployed = [r.doc_id for r in ir.search(INDEX, q["query"]).results if r.match == "term"]
        assert deployed == VSM_STEM.rank(q["query"]), q["query"]

    main = evaluate(SCENARIOS)
    abl = evaluate(ABLATIONS)
    cols = ["P@5", "Recall", "MAP", "MRR", "Latency (ms)"]

    print(f"{len(QUERIES)} queries, {N} books\n")
    print(f"{'Scenario':42}" + "".join(f"{c:>14}" for c in cols))
    for name, r in {**main, **abl}.items():
        print(f"{name:42}" + "".join(f"{r[c]:>14.4f}" for c in cols))

    print("\nMAP per category")
    cats = list(by_category(main["S1 Substring (baseline)"]))
    print(f"{'Category':36}" + "".join(f"{n.split()[0]:>8}" for n in main))
    for c in cats:
        print(f"{c:36}" + "".join(f"{by_category(main[n])[c]:>8.3f}" for n in main))

    ap = {n: [m["AP"] for m in r["per_query"]] for n, r in main.items()}
    print("\nWilcoxon signed-rank on per-query AP")
    for other in ["S1 Substring (baseline)", "S2 Boolean AND", "S3 VSM tanpa stemming", "S4 VSM + stemming"]:
        print(f"  S5 vs {other}: {wilcoxon(ap['S5 Usulan lengkap'], ap[other])}")

    print("\nQueries where S5 is not perfect (AP < 1):")
    for q, m in zip(QUERIES, main["S5 Usulan lengkap"]["per_query"]):
        if m["AP"] < 1:
            print(f"  {q['query']!r}: AP = {m['AP']:.3f}, RR = {m['RR']:.3f}")

    json.dump({"scenarios": {n: {k: v for k, v in r.items()} for n, r in {**main, **abl}.items()}},
              open(HERE / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
