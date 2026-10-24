"""
engine.py: components 2 to 4 of a search engine, without the neural models.

  Indexer (offline)      Index: inverted index with term frequencies, plus a metadata index
                         (title, host, URL, file type, language) used by the search operators
  Query processor        parse_query(): intitle, allintitle, inurl, allinurl, site, filetype
  (online)               correct_spelling(): "Did you mean" by edit distance over the index vocabulary
                         expand_query(): related index terms from word-embedding neighbours
                         detect_intent(): navigational, transactional or informational
  Retrieval engine       BM25, then fusion with dense scores:
                         weighted_hybrid(): S = α · minmax(BM25) + (1 − α) · S_dense   (lecture notebook)
                         rrf():             RRF(d) = Σ_r 1 / (k + rank_r(d)),  k = 60   (Cormack et al., 2009)
  Evaluation             precision_at_k, average_precision, reciprocal_rank

BM25 comes in two flavours:
  * "lucene": IDF = ln(1 + (N − n + 0.5) / (n + 0.5)), always positive (used for the mini-web, as in Session 5),
  * "okapi":  the rank_bm25.BM25Okapi formula used in the lecture notebook: IDF = ln((N − n + 0.5) / (n + 0.5)),
              negative values replaced by 0.25 × the average IDF, k1 = 1.5, b = 0.75.

This module does not depend on Streamlit. Try it from the terminal:
    python engine.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = set("""a an the and or of in on at to for from with by is are was were be been this that these those
it its as into about how what which who do does can will your you we our their they my i me so than not but
has have each also when only every there here such like after before while far less them then one two
yang dan di ke dari untuk dengan pada adalah ini itu dalam tidak atau juga akan oleh agar lalu karena seperti
bagian ketika sama setiap""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if not (remove_stopwords and t in STOPWORDS)]


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
class Index:
    """Inverted index + metadata index over a list of documents (objects with url, title, text, type, lang, host)."""

    def __init__(self, docs: list, k1: float = 1.2, b: float = 0.75):
        self.docs = {d.url: d for d in docs}
        self.urls = [d.url for d in docs]
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = {}
        self.length: dict[str, int] = {}
        for d in docs:
            toks = tokenize(d.title + " " + d.text)
            self.length[d.url] = len(toks)
            for t, f in Counter(toks).items():
                self.postings.setdefault(t, {})[d.url] = f
        self.N = len(docs)
        self.avgdl = sum(self.length.values()) / self.N if self.N else 0.0
        self.title_terms = {d.url: set(re.findall(r"[a-z0-9]+", d.title.lower())) for d in docs}

    # -- statistics --------------------------------------------------------
    def df(self, t: str) -> int:
        return len(self.postings.get(t, {}))

    def idf(self, t: str) -> float:
        n = self.df(t)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    @property
    def vocabulary(self) -> dict[str, int]:
        return {t: len(p) for t, p in self.postings.items()}

    # -- BM25 ----------------------------------------------------------------
    def term_score(self, t: str, url: str) -> float:
        f = self.postings.get(t, {}).get(url, 0)
        if not f:
            return 0.0
        norm = 1 - self.b + self.b * self.length[url] / self.avgdl
        return self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * norm)

    def bm25(self, weights: dict[str, float], candidates: list[str] | None = None) -> dict[str, float]:
        """Weighted BM25 for every candidate (0 for documents that match no query term)."""
        cands = self.urls if candidates is None else candidates
        return {u: sum(w * self.term_score(t, u) for t, w in weights.items()) for u in cands}

    # -- metadata index (search operators) -------------------------------------
    def apply_filters(self, pq: "ParsedQuery") -> list[str]:
        out = []
        for u in self.urls:
            d = self.docs[u]
            ok = True
            for op, val in pq.filters:
                v = val.lower()
                if op == "intitle":
                    ok = v in self.title_terms[u]
                elif op == "allintitle":
                    ok = all(w in self.title_terms[u] for w in v.split())
                elif op == "inurl":
                    ok = v in u.lower()
                elif op == "allinurl":
                    ok = all(w in u.lower() for w in v.split())
                elif op == "site":
                    ok = d.host == v or d.host.endswith("." + v.lstrip("."))
                elif op == "filetype":
                    ok = d.type == v.lstrip(".")
                if not ok:
                    break
            if ok:
                out.append(u)
        return out


def okapi_scores(docs_tokens: list[list[str]], query_tokens: list[str], k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25) -> tuple[list[float], dict[str, float]]:
    """The exact formula of rank_bm25.BM25Okapi (the library in the lecture notebook). Returns (scores, idf)."""
    N = len(docs_tokens)
    avgdl = sum(map(len, docs_tokens)) / N
    nd: dict[str, int] = {}
    for toks in docs_tokens:
        for w in set(toks):
            nd[w] = nd.get(w, 0) + 1
    idf, negative = {}, []
    for w, n in nd.items():
        idf[w] = math.log(N - n + 0.5) - math.log(n + 0.5)
        if idf[w] < 0:
            negative.append(w)
    eps = epsilon * sum(idf.values()) / len(idf)
    for w in negative:
        idf[w] = eps
    scores = []
    for toks in docs_tokens:
        tf = Counter(toks)
        s = 0.0
        for q in query_tokens:
            f = tf.get(q, 0)
            s += idf.get(q, 0.0) * (f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avgdl)))
        scores.append(s)
    return scores, idf


# ---------------------------------------------------------------------------
# Query processor
# ---------------------------------------------------------------------------
OPERATORS = ("allintitle", "allinurl", "intitle", "inurl", "site", "filetype")


@dataclass
class ParsedQuery:
    raw: str
    text: list[str] = field(default_factory=list)              # words that are ranked
    filters: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_query(q: str) -> ParsedQuery:
    """Parse the Google-style operators from the lecture.

    intitle:w     the title must contain w (one word; the rest of the query is matched in the content)
    allintitle:   every following word must be in the title; cannot be combined with other operators
    inurl:w       the URL must contain w
    allinurl:     every following word must be in the URL; cannot be combined with other operators
    site:x        the host must be x or end with .x  (site:ac.id)
    filetype:x    the document type must be x       (filetype:pdf)
    """
    pq = ParsedQuery(q)
    toks = q.split()
    for i, tok in enumerate(toks):
        low = tok.lower()
        for op in ("allintitle", "allinurl"):
            if low.startswith(op + ":"):
                rest = [tok[len(op) + 1:]] + toks[i + 1:]
                words = [w for w in rest if w]
                others = [w for w in toks[:i] + words if any(w.lower().startswith(o + ":") for o in OPERATORS)]
                if others or toks[:i]:
                    pq.warnings.append(f"{op}: cannot be combined with other operators or words before it; "
                                       f"everything after it is treated as {'title' if op == 'allintitle' else 'URL'} words.")
                words = [w.split(":", 1)[-1] for w in words]
                pq.filters.append((op, " ".join(words)))
                pq.text = words if op == "allintitle" else []
                if op == "allinurl":
                    pq.text = []
                return pq
    for tok in toks:
        if ":" in tok and tok.split(":", 1)[0].lower() in OPERATORS and tok.split(":", 1)[1]:
            op, val = tok.split(":", 1)
            pq.filters.append((op.lower(), val))
            if op.lower() == "intitle":
                pq.text.append(val)
        else:
            pq.text.append(tok)
    return pq


def edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment): insert, delete, substitute, swap adjacent letters."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def correct_spelling(terms: list[str], vocab: dict[str, int], known: set[str] | frozenset = frozenset()
                     ) -> dict[str, tuple[str, int]]:
    """{wrong: (suggestion, distance)} for terms that are neither in the index nor known English words."""
    out = {}
    for t in terms:
        t = t.lower()
        if t in vocab or t in known or t in STOPWORDS or len(t) < 4 or not t.isalpha():
            continue
        max_d = 1 if len(t) <= 5 else 2
        best = None
        for v, df in vocab.items():
            if abs(len(v) - len(t)) > max_d:
                continue
            dist = edit_distance(t, v)
            if dist <= max_d and (best is None or (dist, -df) < (best[1], -best[2])):
                best = (v, dist, df)
        if best:
            out[t] = (best[0], best[1])
    return out


def expand_query(terms: list[str], glove, vocab: dict[str, int], per_term: int = 2, min_sim: float = 0.6
                 ) -> dict[str, list[tuple[str, float]]]:
    """For each query word, the closest GloVe neighbours that also occur in the index."""
    import numpy as np
    cand = [w for w in vocab if glove.has(w)]
    if not cand:
        return {}
    M = np.stack([glove.vec(w) for w in cand])
    out = {}
    for t in terms:
        if not glove.has(t):
            continue
        sims = M @ glove.vec(t)
        order = np.argsort(-sims)
        picks = [(cand[i], float(sims[i])) for i in order if cand[i] != t and cand[i] not in terms
                 and sims[i] >= min_sim][:per_term]
        if picks:
            out[t] = picks
    return out


NAV_CUES = {"homepage", "home", "login", "website", "beranda", "searchpedia", "vectorlab", "fusionsearch",
            "unikampus", "techharian", "wiki"}
TRANS_CUES = {"download", "unduh", "pdf", "ppt", "slides", "slide", "manual", "install", "buy", "beli", "file"}
INFO_CUES = {"how", "what", "why", "when", "which", "who", "apa", "bagaimana", "mengapa", "cara", "kenapa"}


def detect_intent(pq: ParsedQuery) -> tuple[str, list[str]]:
    """Rule-based intent detection. Returns (intent, cues that triggered it)."""
    words = [w.lower() for w in pq.raw.replace(":", " ").split()]
    ops = {op for op, _ in pq.filters}
    nav = [w for w in words if w in NAV_CUES] + (["site:"] if "site" in ops else [])
    trans = [w for w in words if w in TRANS_CUES] + (["filetype:"] if "filetype" in ops else [])
    info = [w for w in words if w in INFO_CUES]
    if trans:
        return "transactional", trans
    if nav:
        return "navigational", nav
    return "informational", info or ["no special cue (default)"]


def snippet(text: str, terms: set[str], width: int = 28) -> tuple[list[tuple[str, bool]], bool, bool]:
    """Best window of `width` words (most query-term hits).
    Returns ((word, is_hit) pairs, text was cut on the left, text was cut on the right)."""
    words = text.split()
    hits = [re.sub(r"[^a-z0-9]", "", w.lower()) in terms for w in words]
    if len(words) <= width:
        return list(zip(words, hits)), False, False
    best = max(range(len(words) - width + 1), key=lambda i: (sum(hits[i:i + width]), -i))
    return list(zip(words[best:best + width], hits[best:best + width])), best > 0, best + width < len(words)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def minmax(scores: dict[str, float]) -> dict[str, float]:
    """(s − min) / (max − min + 1e-8), exactly as in the lecture notebook."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {u: (s - lo) / (hi - lo + 1e-8) for u, s in scores.items()}


def ranking(scores: dict[str, float], keep_zero: bool = True) -> list[tuple[str, float]]:
    """Sort by score, highest first; stable, so ties keep the input order."""
    items = [(u, s) for u, s in scores.items() if keep_zero or s > 0]
    return sorted(items, key=lambda x: -x[1])


def weighted_hybrid(bm25: dict[str, float], dense: dict[str, float], alpha: float,
                    normalize_dense: bool = False) -> tuple[list[tuple[str, float]], dict[str, float], dict[str, float]]:
    """S = α · minmax(BM25) + (1 − α) · S_dense. Returns (ranking, BM25 normalized, dense used)."""
    b = minmax(bm25)
    d = minmax(dense) if normalize_dense else dict(dense)
    # iterate in BM25 order so ties keep BM25's order: α = 1 reproduces the BM25 ranking exactly
    order = [u for u, _ in ranking(bm25)]
    fused = {u: alpha * b.get(u, 0.0) + (1 - alpha) * d.get(u, 0.0) for u in order}
    return ranking(fused), b, d


def rrf(rankings: dict[str, list[str]], k: int = 60) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    """Reciprocal rank fusion. `rankings` maps a retriever name to its ranked list of URLs (best first).
    Returns (fused ranking, {url: {retriever: rank}})."""
    ranks: dict[str, dict[str, int]] = {}
    for name, lst in rankings.items():
        for r, u in enumerate(lst, 1):
            ranks.setdefault(u, {})[name] = r
    scores = {u: sum(1 / (k + r) for r in rs.values()) for u, rs in ranks.items()}
    return ranking(scores), ranks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(u in relevant for u in ranked[:k]) / k


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, u in enumerate(ranked, 1):
        if u in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, u in enumerate(ranked, 1):
        if u in relevant:
            return 1 / i
    return 0.0


if __name__ == "__main__":
    import web
    _, docs, _ = web.crawl(web.WEB["seeds"])
    kept, _ = web.find_duplicates(docs)
    idx = Index(kept)
    print(f"{idx.N} documents, {len(idx.postings)} terms, avgdl {idx.avgdl:.1f}")
    for q in ["intitle:hybrid search", "allintitle:hybrid search", "bm25 site:ac.id", "filetype:pdf bm25",
              "hybird serach"]:
        pq = parse_query(q)
        cands = idx.apply_filters(pq)
        print(q, "->", pq.filters, pq.text, len(cands), "candidates", correct_spelling(pq.text, idx.vocabulary),
              detect_intent(pq))
    nb = web.WEB["notebook"]
    s, _ = okapi_scores([d.lower().split() for d in nb["docs"]], nb["query"].split())
    print("notebook BM25:", [round(x, 4) for x in s])
