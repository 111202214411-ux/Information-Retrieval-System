from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

STOPWORDS = set(
    """a an the and or of in on at to for from with by is are was were be been
    this that these those it its as into about how what which who whom do does
    can will your you we our their they his her including using""".split()
)
BOOLEAN_OPS = {"and", "or", "not"}
_TOKEN_RE = re.compile(r"[a-z0-9\+]+")


def tokenize(text: str) -> list[str]:
    """Case folding (lowercase) + tokenization. The '+' sign is kept so 'c++' stays one token."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize, then (optionally) remove stopwords."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def query_terms(query: str, remove_stopwords: bool = True) -> list[str]:
    """Unique query terms (Boolean operators excluded), in their original order."""
    seen: list[str] = []
    for t in preprocess(query, remove_stopwords):
        if t not in BOOLEAN_OPS and t not in seen:
            seen.append(t)
    return seen


# ---------------------------------------------------------------------------
# 2. Data and inverted index
# ---------------------------------------------------------------------------

@dataclass
class Book:
    id: str
    call_number: str
    title: str
    author: str
    description: str

    @property
    def text(self) -> str:
        """The text that gets indexed: title + description."""
        return f"{self.title} {self.description}"


def load_books(path: str | Path) -> list[Book]:
    with open(path, encoding="utf-8") as f:
        return [Book(**row) for row in json.load(f)]


@dataclass
class Result:
    id: str
    score: float | None                                           # None for Boolean (unranked)
    parts: list[tuple[str, float]] = field(default_factory=list)  # score contribution per term


class SearchEngine:
    """Builds the inverted index once, then serves searches with three models."""

    def __init__(self, books: list[Book], remove_stopwords: bool = True):
        self.books = books
        self.by_id = {b.id: b for b in books}
        self.remove_stopwords = remove_stopwords

        self.index: dict[str, list[tuple[str, int]]] = {}   # term -> [(id, tf)]
        self.doc_tf: dict[str, Counter] = {}                  # id -> Counter(term)
        self.doc_len: dict[str, int] = {}                     # id -> number of tokens

        for b in books:
            terms = preprocess(b.text, remove_stopwords)
            tf = Counter(terms)
            self.doc_tf[b.id] = tf
            self.doc_len[b.id] = len(terms)
            for term, freq in tf.items():
                self.index.setdefault(term, []).append((b.id, freq))

        self.N = len(books)
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0.0

    # --- term statistics ----------------------------------------------------
    def df(self, term: str) -> int:
        """Document frequency: number of books that contain the term."""
        return len(self.index.get(term, []))

    def idf_log(self, term: str) -> float:
        """IDF for TF-IDF: log10(N / df)."""
        n = self.df(term)
        return math.log10(self.N / n) if n else 0.0

    def idf_bm25(self, term: str) -> float:
        """BM25 IDF (always positive): ln(1 + (N - df + 0.5) / (df + 0.5))."""
        n = self.df(term)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def postings(self, term: str) -> list[tuple[str, int]]:
        return self.index.get(term, [])

    # --- model 1: Boolean ---------------------------------------------------
    def search_boolean(self, query: str) -> list[Result]:
        """
        Evaluated left to right: 'a AND b', 'a OR b', 'a AND NOT b'.
        Terms written without an operator between them are joined with AND.
        Results are unranked (sorted by ID).
        """
        all_ids = [b.id for b in self.books]
        result: set[str] | None = None
        op, negate = "and", False

        for tok in tokenize(query):
            if tok in ("and", "or"):
                op = tok
                continue
            if tok == "not":
                negate = True
                continue
            if self.remove_stopwords and tok in STOPWORDS:
                continue

            posting = {doc_id for doc_id, _ in self.postings(tok)}
            if negate:
                posting = set(all_ids) - posting
                negate = False

            if result is None:
                result = posting
            elif op == "and":
                result &= posting          # intersection of posting lists
            else:
                result |= posting          # union of posting lists
            op = "and"

        return [Result(doc_id, None) for doc_id in sorted(result or [])]

    # --- model 2: TF-IDF + Cosine -------------------------------------------
    def weight(self, tf: int, term: str) -> float:
        """Weight (1 + log10 tf) x log10(N/df)."""
        return (1 + math.log10(tf)) * self.idf_log(term) if tf > 0 else 0.0

    def doc_norm(self, doc_id: str) -> float:
        """Length of the document's TF-IDF vector, |d|."""
        return math.sqrt(sum(self.weight(f, t) ** 2 for t, f in self.doc_tf[doc_id].items()))

    def search_tfidf(self, query: str) -> list[Result]:
        q_tf = Counter(t for t in preprocess(query, self.remove_stopwords) if t not in BOOLEAN_OPS)
        q_vec = {t: self.weight(f, t) for t, f in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if q_norm == 0:
            return []

        results = []
        for b in self.books:
            tf = self.doc_tf[b.id]
            d_norm = self.doc_norm(b.id)
            if d_norm == 0:
                continue
            parts = [
                (t, self.weight(tf[t], t) * qw / (q_norm * d_norm))
                for t, qw in q_vec.items() if tf.get(t)
            ]
            score = sum(c for _, c in parts)
            if score > 0:
                results.append(Result(b.id, score, parts))
        return sorted(results, key=lambda r: r.score, reverse=True)

    # --- model 3: BM25 ------------------------------------------------------
    def search_bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[Result]:
        terms = query_terms(query, self.remove_stopwords)
        results = []
        for book in self.books:
            tf, dl = self.doc_tf[book.id], self.doc_len[book.id]
            parts = []
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                norm = k1 * (1 - b + b * dl / self.avgdl)
                parts.append((t, self.idf_bm25(t) * f * (k1 + 1) / (f + norm)))
            score = sum(c for _, c in parts)
            if score > 0:
                results.append(Result(book.id, score, parts))
        return sorted(results, key=lambda r: r.score, reverse=True)

    def search(self, query: str, model: str, k1: float = 1.5, b: float = 0.75) -> list[Result]:
        if not query.strip():
            return []
        if model == "boolean":
            return self.search_boolean(query)
        if model == "tfidf":
            return self.search_tfidf(query)
        return self.search_bm25(query, k1, b)


DATA_PATH = Path(__file__).parent / "data" / "books.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the sample book catalog from the terminal.")
    parser.add_argument("query", nargs="?", default="search engine ranking")
    parser.add_argument("--model", choices=["bm25", "tfidf", "boolean"], default="bm25")
    args = parser.parse_args()

    engine = SearchEngine(load_books(DATA_PATH))
    print(f"{engine.N} books, {len(engine.index)} terms, avgdl = {engine.avgdl:.2f}")
    print(f"Query terms: {query_terms(args.query)}\n")
    print(f"[{args.model}] {args.query}")
    for rank, r in enumerate(engine.search(args.query, args.model), 1):
        score = "-" if r.score is None else f"{r.score:.4f}"
        print(f"  {rank:>2}. {r.id}  {score:>7}  {engine.by_id[r.id].title}")
