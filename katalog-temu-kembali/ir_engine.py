"""
ir_engine.py — Mesin temu kembali informasi (Information Retrieval) sederhana.

Modul ini tidak bergantung pada Streamlit, sehingga bisa dipakai ulang di
notebook atau skrip lain. Isinya:

1. Preprocessing   : case folding -> tokenisasi -> stopword removal
2. Inverted index  : istilah -> posting list [(id_buku, tf), ...]
3. Tiga model      : Boolean, TF-IDF + Cosine Similarity, BM25
"""

from __future__ import annotations

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
    """dan yang di ke dari untuk dengan pada adalah ini itu serta atau dalam oleh
    cara belajar menggunakan hingga mulai termasuk sebagai juga bagi para tiap
    the of in to for a an""".split()
)
BOOLEAN_OPS = {"and", "or", "not"}
_TOKEN_RE = re.compile(r"[a-z0-9\+]+")


def tokenize(text: str) -> list[str]:
    """Case folding (huruf kecil) + tokenisasi. Tanda '+' dipertahankan agar 'c++' tetap utuh."""
    return _TOKEN_RE.findall(text.lower())


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenisasi lalu (opsional) buang stopword."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def query_terms(query: str, remove_stopwords: bool = True) -> list[str]:
    """Istilah unik dari query (tanpa operator Boolean), urutan dipertahankan."""
    seen = []
    for t in preprocess(query, remove_stopwords):
        if t not in BOOLEAN_OPS and t not in seen:
            seen.append(t)
    return seen


# ---------------------------------------------------------------------------
# 2. Data & inverted index
# ---------------------------------------------------------------------------

@dataclass
class Book:
    id: str
    nomor_panggil: str
    judul: str
    pengarang: str
    deskripsi: str

    @property
    def text(self) -> str:
        """Teks yang diindeks: judul + deskripsi."""
        return f"{self.judul} {self.deskripsi}"


def load_books(path: str | Path) -> list[Book]:
    with open(path, encoding="utf-8") as f:
        return [Book(**row) for row in json.load(f)]


@dataclass
class Result:
    id: str
    score: float | None                     # None untuk Boolean (tanpa ranking)
    parts: list[tuple[str, float]] = field(default_factory=list)  # sumbangan skor per istilah


class SearchEngine:
    """Membangun inverted index sekali, lalu melayani pencarian dengan tiga model."""

    def __init__(self, books: list[Book], remove_stopwords: bool = True):
        self.books = books
        self.by_id = {b.id: b for b in books}
        self.remove_stopwords = remove_stopwords

        self.index: dict[str, list[tuple[str, int]]] = {}   # istilah -> [(id, tf)]
        self.doc_tf: dict[str, Counter] = {}                  # id -> Counter(istilah)
        self.doc_len: dict[str, int] = {}                     # id -> jumlah token

        for b in books:
            terms = preprocess(b.text, remove_stopwords)
            tf = Counter(terms)
            self.doc_tf[b.id] = tf
            self.doc_len[b.id] = len(terms)
            for term, freq in tf.items():
                self.index.setdefault(term, []).append((b.id, freq))

        self.N = len(books)
        self.avgdl = sum(self.doc_len.values()) / self.N if self.N else 0.0

    # --- statistik istilah ---------------------------------------------------
    def df(self, term: str) -> int:
        """Document frequency: jumlah buku yang memuat istilah."""
        return len(self.index.get(term, []))

    def idf_log(self, term: str) -> float:
        """IDF untuk TF-IDF: log10(N / df)."""
        n = self.df(term)
        return math.log10(self.N / n) if n else 0.0

    def idf_bm25(self, term: str) -> float:
        """IDF versi BM25 (selalu positif): ln(1 + (N - df + 0.5) / (df + 0.5))."""
        n = self.df(term)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def postings(self, term: str) -> list[tuple[str, int]]:
        return self.index.get(term, [])

    # --- model 1: Boolean ----------------------------------------------------
    def search_boolean(self, query: str) -> list[Result]:
        """
        Evaluasi kiri-ke-kanan: 'a AND b', 'a OR b', 'a AND NOT b'.
        Istilah tanpa operator di antaranya digabung dengan AND.
        Hasil tidak diranking (urut berdasarkan ID).
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
                result &= posting          # irisan posting list
            else:
                result |= posting          # gabungan posting list
            op = "and"

        return [Result(doc_id, None) for doc_id in sorted(result or [])]

    # --- model 2: TF-IDF + Cosine -------------------------------------------
    def _weight(self, tf: int, term: str) -> float:
        """Bobot (1 + log10 tf) x log10(N/df)."""
        return (1 + math.log10(tf)) * self.idf_log(term) if tf > 0 else 0.0

    def search_tfidf(self, query: str) -> list[Result]:
        q_tf = Counter(t for t in preprocess(query, self.remove_stopwords) if t not in BOOLEAN_OPS)
        q_vec = {t: self._weight(f, t) for t, f in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
        if q_norm == 0:
            return []

        results = []
        for b in self.books:
            tf = self.doc_tf[b.id]
            d_norm = math.sqrt(sum(self._weight(f, t) ** 2 for t, f in tf.items()))
            if d_norm == 0:
                continue
            parts = []
            for t, qw in q_vec.items():
                if tf.get(t):
                    parts.append((t, self._weight(tf[t], t) * qw / (q_norm * d_norm)))
            score = sum(c for _, c in parts)
            if score > 0:
                results.append(Result(b.id, score, parts))
        return sorted(results, key=lambda r: r.score, reverse=True)

    # --- model 3: BM25 -------------------------------------------------------
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


if __name__ == "__main__":
    # Uji cepat dari terminal: python ir_engine.py
    engine = SearchEngine(load_books(Path(__file__).parent / "data" / "buku.json"))
    print(f"{engine.N} buku, {len(engine.index)} istilah, avgdl={engine.avgdl:.2f}\n")
    for q, m in [("pencarian dokumen teks", "bm25"), ("pemrograman AND NOT java", "boolean")]:
        print(f"[{m}] {q}")
        for r in engine.search(q, m):
            score = "-" if r.score is None else f"{r.score:.3f}"
            print(f"   {r.id}  {score:>6}  {engine.by_id[r.id].judul}")
        print()
