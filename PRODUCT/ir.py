"""
ir.py — Information Retrieval engine (Vector Space Model)
=========================================================

Pipeline:
  1. Preprocessing : case folding -> tokenization -> stopword removal -> stemming
  2. Indexing      : build an inverted index with term frequencies (tf)
  3. Weighting     : TF-IDF,  w(t,d) = tf(t,d) x idf(t),  idf(t) = log10(N / df(t))
  4. Ranking       : cosine similarity between the query vector and each document vector
                     cos(q,d) = sum( w(t,q) * w(t,d) ) / ( |q| x |d| )
  5. Title prefix  : titles that start with the typed text are also returned
                     (after the cosine results), so results appear from the first letter

Each document is a book title. The unstructured data is the title text.
This module does not depend on Streamlit, so it can be tested on its own:

    python ir.py "hujan bulan"
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

# ---------------------------------------------------------------------------
# 1a. Stopwords: Indonesian, plus common English words
# ---------------------------------------------------------------------------
STOPWORDS = set("""
    ada adalah agar akan aku amat anda antara apa apakah apalagi atau bagi bahwa begitu belum
    bersama bisa boleh dahulu dalam dan dapat dari daripada demi demikian dengan di dia dimana
    dll dsb dst engkau guna hal hanya harus ia ingin ini itu itulah jika juga kah kalau kamu
    karena ke kecuali kemana kenapa kepada ketika kita lagi lain maka mari masih melainkan
    menurut mengapa mereka namun nanti nya oleh pada para pula pun saat saja sambil sampai
    saya sebab sebagai sebelum secara sedangkan sehingga sekitar selagi selain sementara serta
    sesudah setelah setiap seperti si sudah supaya tanpa tapi telah tentang terhadap tetapi
    toh untuk walau yaitu yakni yang
    a an and at by for from in is of on the to with
""".split())

# ---------------------------------------------------------------------------
# 1b. English root words (for translated titles).
#     A plural "-s" is reduced to the singular only when the singular form
#     is in this dictionary: brothers -> brother.
# ---------------------------------------------------------------------------
ROOT_WORDS_EN = set("""
    attack brother crime dragon note miserable piece prince punishment saga shore titan wood
""".split())

# ---------------------------------------------------------------------------
# 1c. Indonesian root-word dictionary for the stemmer (simplified Nazief-Adriani).
#     The stemmer only strips an affix when the result is in this dictionary,
#     so root words such as "menara" or "pelangi" are not cut incorrectly.
# ---------------------------------------------------------------------------
ROOT_WORDS_ID = set("""
    air ajar alam amba anak angin api asuh atheis ayah ayat baca bahagia baik bangsa bangun
    bapak baru bawah benar besar biasa bintang biru buku bulan bumi bunga cahaya cantik cari
    cerita cinta damai dasar datang dengar desa dilan doa dukuh dunia edensor gadis gelap
    gunung habibie ainun harap hari harimau hati hidup hijau hilang hitam hujan hutan ibu
    iman indah ingat istri jalan jambu jantan jatuh jejak juni kaca kabah kambing kapal karpov
    kasih kata kembang kenang kertas kecil kisah koala kota kretek ksatria kumal kuning laut
    langit langkah larung laskar layar lelaki lihat lindung lupa luka main malam manusia
    marmut maryamah matahari menara merah milea mimpi muara murid negara negeri nurbaya orang
    pagi panjang pantai paruk pelangi perahu perang perempuan pergi pohon pulang puteri putih
    rahasia ranah rantau rindu ronggeng rumah sabtu sahabat sakit salah saman sayang sehat
    sekolah semua seni senyum siti suami suara sungai supernova tahun tanah tangis tasbih tawa
    teman temu tenggelam terang terbang terbit tidur tinggal tulis tunggu waktu warna
""".split())

VOWELS = "aiueo"


def _prefix_candidates(word: str) -> list[str]:
    """Possible forms of a word after removing a prefix (with meN-/peN- sound changes)."""
    out: list[str] = []

    def assimilate(rest: str, a: str, b: str) -> list[str]:
        return [a + rest, b + rest] if rest[:1] in VOWELS else [rest]

    rules = [
        ("meng", lambda r: [r, "k" + r]), ("peng", lambda r: [r, "k" + r]),
        ("meny", lambda r: ["s" + r]),    ("peny", lambda r: ["s" + r]),
        ("mem", lambda r: assimilate(r, "m", "p")), ("pem", lambda r: assimilate(r, "m", "p")),
        ("men", lambda r: assimilate(r, "n", "t")), ("pen", lambda r: assimilate(r, "n", "t")),
        ("ber", lambda r: [r]), ("ter", lambda r: [r]), ("per", lambda r: [r]),
        ("me", lambda r: [r]),  ("pe", lambda r: [r]),  ("be", lambda r: [r]), ("te", lambda r: [r]),
        ("di", lambda r: [r]),  ("ke", lambda r: [r]),  ("se", lambda r: [r]),
    ]
    for prefix, rule in rules:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            out.extend(s for s in rule(word[len(prefix):]) if len(s) >= 2)
    return out


def _strip_prefixes(word: str, depth: int = 0) -> str | None:
    """Strip prefixes one layer at a time (up to 3), stopping at a known root word."""
    if word in ROOT_WORDS_ID:
        return word
    if depth >= 3:
        return None
    for candidate in _prefix_candidates(word):
        root = _strip_prefixes(candidate, depth + 1)
        if root:
            return root
    return None


def _suffix_candidates(word: str) -> list[str]:
    """Possible forms of a word after removing suffixes."""
    forms = [word]
    w = word
    for p in ("lah", "kah", "tah", "pun"):              # particles
        if w.endswith(p) and len(w) > len(p) + 2:
            w = w[: -len(p)]
            forms.append(w)
            break
    for p in ("nya", "ku", "mu"):                       # possessive pronouns
        if w.endswith(p) and len(w) > len(p) + 2:
            w = w[: -len(p)]
            forms.append(w)
            break
    for p in ("kan", "an", "i"):                        # derivational suffixes
        if w.endswith(p) and len(w) > len(p) + 2:
            forms.append(w[: -len(p)])
            break
    return forms


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Reduce a word to its root form. Unknown words are returned unchanged."""
    if word in ROOT_WORDS_EN:
        return word
    if word.endswith("s") and word[:-1] in ROOT_WORDS_EN:
        return word[:-1]                                 # English plural
    if word in ROOT_WORDS_ID or word.isdigit():
        return word
    for form in _suffix_candidates(word):
        root = _strip_prefixes(form)
        if root:
            return root
    return word


_ACCENTS = re.compile(r"[̀-ͯ]")


def fold(text: str) -> str:
    """Lowercase and strip accents (Misérables -> miserables)."""
    return _ACCENTS.sub("", unicodedata.normalize("NFD", text.lower()))


# ---------------------------------------------------------------------------
# 1d. Full preprocessing, returning every step (so the app can display them)
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    folded: str
    tokens: list[str]
    removed: list[str]
    stems: list[tuple[str, str]]          # (token, root word)

    @property
    def terms(self) -> list[str]:
        return [root for _, root in self.stems]


def preprocess(text: str) -> PreprocessResult:
    folded = fold(text)                                              # case folding
    tokens = [t for t in re.split(r"[^a-z0-9]+",                     # tokenization
                                  re.sub(r"['’`]", "", folded)) if t]  # ka'bah -> kabah
    removed = [t for t in tokens if t in STOPWORDS]
    kept = [t for t in tokens if t not in STOPWORDS]                 # stopword removal
    stems = [(t, stem(t)) for t in kept]                             # stemming
    return PreprocessResult(folded, tokens, removed, stems)


# ---------------------------------------------------------------------------
# 2-3. Build the inverted index and TF-IDF weights
# ---------------------------------------------------------------------------
@dataclass
class Index:
    docs: list[dict]
    field: str
    N: int
    index: dict[str, dict[int, int]]      # term -> {doc_id: tf}
    idf: dict[str, float]
    tf: list[dict[str, int]]              # tf[d] : term -> frequency
    weights: list[dict[str, float]]       # w(t,d)
    norms: list[float]                    # |d|


def build_index(docs: list[dict], field: str = "title") -> Index:
    N = len(docs)
    tf: list[dict[str, int]] = [{} for _ in docs]
    index: dict[str, dict[int, int]] = {}

    for d, doc in enumerate(docs):
        for term in preprocess(doc[field]).terms:
            tf[d][term] = tf[d].get(term, 0) + 1
        for term, f in tf[d].items():
            index.setdefault(term, {})[d] = f

    idf = {term: math.log10(N / len(postings)) for term, postings in index.items()}

    # document weights and vector lengths |d|
    weights = [{term: f * idf[term] for term, f in m.items()} for m in tf]
    norms = [math.sqrt(sum(x * x for x in w.values())) for w in weights]

    return Index(docs, field, N, index, idf, tf, weights, norms)


# ---------------------------------------------------------------------------
# 4-5. Search: query weights -> cosine similarity -> sort -> title prefix
# ---------------------------------------------------------------------------
@dataclass
class Result:
    doc_id: int
    doc: dict
    score: float
    dot: float
    match: str                            # "term" (cosine) or "prefix"


@dataclass
class SearchResult:
    steps: PreprocessResult
    qtf: dict[str, int]
    q_weights: dict[str, float]
    q_norm: float
    unknown: list[str]
    results: list[Result] = field(default_factory=list)
    prefix: str = ""
    prefix_matches: list[int] = field(default_factory=list)


def search(idx: Index, query: str) -> SearchResult:
    steps = preprocess(query)

    qtf: dict[str, int] = {}
    for t in steps.terms:
        qtf[t] = qtf.get(t, 0) + 1

    q_weights: dict[str, float] = {}
    unknown: list[str] = []
    for t, f in qtf.items():
        if t in idx.idf:
            q_weights[t] = f * idx.idf[t]
        else:
            unknown.append(t)
    q_norm = math.sqrt(sum(x * x for x in q_weights.values()))

    # only score documents that contain at least one query term (via the inverted index)
    candidates: set[int] = set()
    for t in q_weights:
        candidates.update(idx.index[t].keys())

    results: list[Result] = []
    for d in candidates:
        dot = sum(wq * idx.weights[d].get(t, 0.0) for t, wq in q_weights.items())
        denom = q_norm * idx.norms[d]
        score = dot / denom if denom else 0.0
        if score > 0:
            results.append(Result(d, idx.docs[d], score, dot, "term"))
    results.sort(key=lambda r: (-r.score, fold(r.doc[idx.field]), r.doc_id))

    # 5. Title prefix matching: titles that start with the typed text
    #    (case-insensitive), so results appear from the first letter typed.
    #    Appended after the cosine results, in alphabetical order.
    prefix = re.sub(r"\s+", " ", fold(query.strip()))
    prefix_matches: list[int] = []
    if prefix:
        found = {r.doc_id for r in results}
        prefix_matches = [d for d, doc in enumerate(idx.docs) if fold(doc[idx.field]).startswith(prefix)]
        extra = sorted((d for d in prefix_matches if d not in found),
                       key=lambda d: (fold(idx.docs[d][idx.field]), d))
        results.extend(Result(d, idx.docs[d], 0.0, 0.0, "prefix") for d in extra)

    return SearchResult(steps, qtf, q_weights, q_norm, unknown, results, prefix, prefix_matches)


# ---------------------------------------------------------------------------
# Quick test from the terminal:  python ir.py "your query"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    books = json.loads((Path(__file__).parent / "books.json").read_text(encoding="utf-8"))
    idx = build_index(books)
    q = " ".join(sys.argv[1:]) or "hujan bulan"
    res = search(idx, q)
    print(f"Query: {q!r}   N = {idx.N}, terms in index = {len(idx.index)}")
    print("Query terms:", res.steps.terms, "| |q| =", round(res.q_norm, 4))
    for i, r in enumerate(res.results, 1):
        score = f"{r.score:.4f}" if r.match == "term" else "prefix"
        print(f"{i:>2}. {r.doc['title']}  ({r.doc['author']})  {score}")
