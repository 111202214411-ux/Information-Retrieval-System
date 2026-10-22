"""
ir.py — Mesin Temu Kembali Informasi (Vector Space Model)
=========================================================

Alur:
  1. Preprocessing : case folding -> tokenisasi -> stopword removal -> stemming
  2. Indexing      : membangun indeks terbalik (inverted index) + tf
  3. Pembobotan    : TF-IDF,  w(t,d) = tf(t,d) x idf(t),  idf(t) = log10(N / df(t))
  4. Perankingan   : cosine similarity antara vektor query dan vektor dokumen
                     cos(q,d) = sum( w(t,q) * w(t,d) ) / ( |q| x |d| )
  5. Awalan judul  : judul yang diawali teks yang diketik ikut ditampilkan
                     (setelah hasil cosine), sehingga hasil muncul per huruf

Dokumen = judul buku. Data tidak terstruktur = teks judul buku.
Modul ini tidak bergantung pada Streamlit, jadi bisa diuji sendiri:

    python ir.py "hujan bulan"
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

# ---------------------------------------------------------------------------
# 1a. Daftar stopword bahasa Indonesia (+ beberapa kata Inggris)
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
# 1b'. Kata dasar bahasa Inggris (untuk judul buku terjemahan).
#      Bentuk jamak "-s" dikembalikan ke bentuk tunggal hanya jika
#      bentuk tunggalnya ada di kamus ini: brothers -> brother.
# ---------------------------------------------------------------------------
KATA_DASAR_EN = set("""
    attack brother crime dragon note miserable piece prince punishment saga shore titan wood
""".split())

# ---------------------------------------------------------------------------
# 1b. Kamus kata dasar untuk stemmer (Nazief-Adriani sederhana).
#     Stemmer hanya memotong imbuhan jika hasilnya ada di kamus,
#     sehingga kata dasar seperti "menara" atau "pelangi" tidak rusak.
# ---------------------------------------------------------------------------
KATA_DASAR = set("""
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

VOKAL = "aiueo"


def _kandidat_awalan(word: str) -> list[str]:
    """Kemungkinan bentuk kata setelah awalan dilepas (dengan peluluhan meN-/peN-)."""
    out: list[str] = []

    def luluh(r: str, a: str, b: str) -> list[str]:
        return [a + r, b + r] if r[:1] in VOKAL else [r]

    aturan = [
        ("meng", lambda r: [r, "k" + r]), ("peng", lambda r: [r, "k" + r]),
        ("meny", lambda r: ["s" + r]),    ("peny", lambda r: ["s" + r]),
        ("mem", lambda r: luluh(r, "m", "p")), ("pem", lambda r: luluh(r, "m", "p")),
        ("men", lambda r: luluh(r, "n", "t")), ("pen", lambda r: luluh(r, "n", "t")),
        ("ber", lambda r: [r]), ("ter", lambda r: [r]), ("per", lambda r: [r]),
        ("me", lambda r: [r]),  ("pe", lambda r: [r]),  ("be", lambda r: [r]), ("te", lambda r: [r]),
        ("di", lambda r: [r]),  ("ke", lambda r: [r]),  ("se", lambda r: [r]),
    ]
    for awalan, fn in aturan:
        if word.startswith(awalan) and len(word) > len(awalan) + 1:
            out.extend(s for s in fn(word[len(awalan):]) if len(s) >= 2)
    return out


def _lepas_awalan(word: str, depth: int = 0) -> str | None:
    """Lepas awalan bertahap (maksimal 3 lapis), berhenti saat ketemu kata dasar."""
    if word in KATA_DASAR:
        return word
    if depth >= 3:
        return None
    for kandidat in _kandidat_awalan(word):
        hasil = _lepas_awalan(kandidat, depth + 1)
        if hasil:
            return hasil
    return None


def _kandidat_akhiran(word: str) -> list[str]:
    """Kemungkinan bentuk kata setelah akhiran dilepas."""
    forms = [word]
    w = word
    for p in ("lah", "kah", "tah", "pun"):              # partikel
        if w.endswith(p) and len(w) > len(p) + 2:
            w = w[: -len(p)]
            forms.append(w)
            break
    for p in ("nya", "ku", "mu"):                       # kata ganti milik
        if w.endswith(p) and len(w) > len(p) + 2:
            w = w[: -len(p)]
            forms.append(w)
            break
    for p in ("kan", "an", "i"):                        # akhiran turunan
        if w.endswith(p) and len(w) > len(p) + 2:
            forms.append(w[: -len(p)])
            break
    return forms


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Ubah kata ke bentuk dasarnya. Kata yang tidak dikenali dibiarkan apa adanya."""
    if word in KATA_DASAR_EN:
        return word
    if word.endswith("s") and word[:-1] in KATA_DASAR_EN:
        return word[:-1]                                 # jamak Inggris
    if word in KATA_DASAR or word.isdigit():
        return word
    for form in _kandidat_akhiran(word):
        hasil = _lepas_awalan(form)
        if hasil:
            return hasil
    return word


_AKSEN = re.compile(r"[̀-ͯ]")


def fold(text: str) -> str:
    """Huruf kecil + buang tanda aksen (Misérables -> miserables)."""
    return _AKSEN.sub("", unicodedata.normalize("NFD", text.lower()))


# ---------------------------------------------------------------------------
# 1c. Preprocessing lengkap, mengembalikan setiap tahap (untuk ditampilkan)
# ---------------------------------------------------------------------------
@dataclass
class HasilPreprocessing:
    folded: str
    tokens: list[str]
    removed: list[str]
    stems: list[tuple[str, str]]          # (token, kata dasar)

    @property
    def terms(self) -> list[str]:
        return [s for _, s in self.stems]


def preprocess(text: str) -> HasilPreprocessing:
    folded = fold(text)                                              # case folding
    tokens = [t for t in re.split(r"[^a-z0-9]+",                     # tokenisasi
                                  re.sub(r"['’`]", "", folded)) if t]  # ka'bah -> kabah
    removed = [t for t in tokens if t in STOPWORDS]
    kept = [t for t in tokens if t not in STOPWORDS]                 # stopword removal
    stems = [(t, stem(t)) for t in kept]                             # stemming
    return HasilPreprocessing(folded, tokens, removed, stems)


# ---------------------------------------------------------------------------
# 2-3. Membangun indeks terbalik dan bobot TF-IDF
# ---------------------------------------------------------------------------
@dataclass
class Indeks:
    docs: list[dict]
    field: str
    N: int
    index: dict[str, dict[int, int]]      # term -> {doc_id: tf}
    idf: dict[str, float]
    tf: list[dict[str, int]]              # tf[d] : term -> frekuensi
    weights: list[dict[str, float]]       # w(t,d)
    norms: list[float]                    # |d|


def build_index(docs: list[dict], field: str = "judul") -> Indeks:
    N = len(docs)
    tf: list[dict[str, int]] = [{} for _ in docs]
    index: dict[str, dict[int, int]] = {}

    for d, doc in enumerate(docs):
        for term in preprocess(doc[field]).terms:
            tf[d][term] = tf[d].get(term, 0) + 1
        for term, f in tf[d].items():
            index.setdefault(term, {})[d] = f

    idf = {term: math.log10(N / len(postings)) for term, postings in index.items()}

    # bobot dokumen dan panjang vektornya |d|
    weights = [{term: f * idf[term] for term, f in m.items()} for m in tf]
    norms = [math.sqrt(sum(x * x for x in w.values())) for w in weights]

    return Indeks(docs, field, N, index, idf, tf, weights, norms)


# ---------------------------------------------------------------------------
# 4-5. Pencarian: bobot query -> cosine similarity -> urutkan -> awalan judul
# ---------------------------------------------------------------------------
@dataclass
class Hasil:
    doc_id: int
    doc: dict
    score: float
    dot: float
    match: str                            # "kata" (cosine) atau "awalan"


@dataclass
class HasilPencarian:
    steps: HasilPreprocessing
    qtf: dict[str, int]
    q_weights: dict[str, float]
    q_norm: float
    unknown: list[str]
    results: list[Hasil] = field(default_factory=list)
    prefix: str = ""
    prefix_matches: list[int] = field(default_factory=list)


def search(idx: Indeks, query: str) -> HasilPencarian:
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

    # hanya dokumen yang memuat minimal satu term query (lewat indeks terbalik)
    candidates: set[int] = set()
    for t in q_weights:
        candidates.update(idx.index[t].keys())

    results: list[Hasil] = []
    for d in candidates:
        dot = sum(wq * idx.weights[d].get(t, 0.0) for t, wq in q_weights.items())
        denom = q_norm * idx.norms[d]
        score = dot / denom if denom else 0.0
        if score > 0:
            results.append(Hasil(d, idx.docs[d], score, dot, "kata"))
    results.sort(key=lambda r: (-r.score, fold(r.doc[idx.field]), r.doc_id))

    # 5. Pencocokan awalan judul: judul yang diawali teks yang diketik
    #    (tidak peka huruf besar/kecil), agar hasil muncul sejak huruf pertama.
    #    Ditambahkan setelah hasil cosine, urut abjad.
    prefix = re.sub(r"\s+", " ", fold(query.strip()))
    prefix_matches: list[int] = []
    if prefix:
        found = {r.doc_id for r in results}
        prefix_matches = [d for d, doc in enumerate(idx.docs) if fold(doc[idx.field]).startswith(prefix)]
        tambahan = sorted((d for d in prefix_matches if d not in found),
                          key=lambda d: (fold(idx.docs[d][idx.field]), d))
        results.extend(Hasil(d, idx.docs[d], 0.0, 0.0, "awalan") for d in tambahan)

    return HasilPencarian(steps, qtf, q_weights, q_norm, unknown, results, prefix, prefix_matches)


# ---------------------------------------------------------------------------
# Uji cepat dari terminal:  python ir.py "kata kunci"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    buku = json.loads((Path(__file__).parent / "buku.json").read_text(encoding="utf-8"))
    idx = build_index(buku)
    q = " ".join(sys.argv[1:]) or "hujan bulan"
    hasil = search(idx, q)
    print(f"Query: {q!r}   N = {idx.N}, jumlah term = {len(idx.index)}")
    print("Term query:", hasil.steps.terms, "| |q| =", round(hasil.q_norm, 4))
    for i, r in enumerate(hasil.results, 1):
        skor = f"{r.score:.4f}" if r.match == "kata" else "awalan"
        print(f"{i:>2}. {r.doc['judul']}  ({r.doc['pengarang']})  {skor}")
