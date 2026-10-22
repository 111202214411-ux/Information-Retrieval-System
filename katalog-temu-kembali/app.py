"""
app.py — Antarmuka Streamlit untuk "Katalog Temu Kembali".

Jalankan secara lokal:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from ir_engine import BOOLEAN_OPS, STOPWORDS, Book, SearchEngine, load_books, query_terms, tokenize

DATA_PATH = Path(__file__).parent / "data" / "buku.json"

MODELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF + Cosine",
    "boolean": "Boolean",
}
FORMULAS = {
    "boolean": r"R(q,d)=\begin{cases}1 & d \text{ memenuhi ekspresi AND / OR / NOT}\\ 0 & \text{selain itu}\end{cases}",
    "tfidf": r"\text{sim}(q,d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert},\qquad w_{t,d}=(1+\log_{10} tf_{t,d})\cdot\log_{10}\frac{N}{df_t}",
    "bm25": r"\text{score}(q,d)=\sum_{t\in q}\text{idf}(t)\cdot\frac{tf_{t,d}\,(k_1+1)}{tf_{t,d}+k_1\left(1-b+b\cdot\frac{|d|}{avgdl}\right)}",
}
EXAMPLES = [
    ("pencarian dokumen teks", "bm25"),
    ("pemrograman AND objek", "boolean"),
    ("java OR c++", "boolean"),
    ("pemrograman AND NOT java", "boolean"),
    ("mesin pencari web", "bm25"),
    ("apel", "bm25"),
    ("rumah makan semarang", "bm25"),
    ("evaluasi pencarian", "tfidf"),
]

st.set_page_config(page_title="Katalog Temu Kembali", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Gaya kartu katalog (warna transparan agar terbaca di tema terang & gelap)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.kartu{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:12px 16px 12px 16px;margin-bottom:12px;
  border-top:3px solid #cf3a3f;font-family:"Courier Prime","Courier New",monospace}
.kartu .top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
.kartu .call{font-size:12.5px;opacity:.7}
.kartu .judul{font-size:17px;font-weight:700;margin:2px 0 0}
.kartu .pengarang{font-size:13.5px;opacity:.75}
.kartu .desc{font-size:14px;margin-top:6px;line-height:1.6}
.stamp{border:2px solid #7a5fd0;color:#7a5fd0;border-radius:4px;padding:2px 8px;font-weight:700;transform:rotate(-3deg);
  text-align:center;line-height:1.2;font-variant-numeric:tabular-nums}
.stamp small{display:block;font-size:10px;font-weight:400;letter-spacing:.08em;text-transform:uppercase}
.why{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.why span{background:rgba(122,95,208,.15);border-radius:3px;padding:1px 7px;font-size:12.5px}
mark{background:#ffe680;color:#2b2400;padding:0 2px;border-radius:2px}
.tok{display:inline-block;font-family:"Courier New",monospace;font-size:13px;border:1px solid rgba(128,140,160,.45);
  border-radius:3px;padding:1px 7px;margin:2px 4px 2px 0}
.tok.stop{opacity:.45;text-decoration:line-through}
.tok.term{font-weight:700;border-color:currentColor}
.tok.op{border-style:dashed;color:#7a5fd0}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "added" not in st.session_state:
    st.session_state.added = []          # buku tambahan pengguna (hanya selama sesi)
if "query" not in st.session_state:
    st.session_state.query = "pencarian dokumen teks"
if "model" not in st.session_state:
    st.session_state.model = "bm25"


def use_example(q: str, m: str) -> None:
    st.session_state.query = q
    st.session_state.model = m


@st.cache_data
def base_books() -> list[dict]:
    return [b.__dict__ for b in load_books(DATA_PATH)]


@st.cache_resource
def get_engine(books_key: tuple, remove_stopwords: bool) -> SearchEngine:
    books = [Book(*row) for row in books_key]
    return SearchEngine(books, remove_stopwords)


# ---------------------------------------------------------------------------
# Sidebar: pengaturan model & koleksi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")
    model = st.radio(
        "Model temu kembali",
        options=list(MODELS),
        format_func=MODELS.get,
        key="model",
    )
    k1 = st.slider("k₁ (saturasi tf)", 0.0, 3.0, 1.5, 0.1, disabled=model != "bm25",
                   help="Makin kecil, makin cepat tf 'jenuh'. k₁ = 0 berarti tf diabaikan.")
    b = st.slider("b (normalisasi panjang)", 0.0, 1.0, 0.75, 0.05, disabled=model != "bm25",
                  help="b = 0: panjang dokumen diabaikan. b = 1: normalisasi penuh.")
    remove_sw = st.checkbox("Buang stopword", value=True)

    st.divider()
    st.subheader("Tambah buku ke koleksi")
    with st.form("tambah", clear_on_submit=True):
        f_judul = st.text_input("Judul", placeholder="Pengantar Data Mining")
        f_pengarang = st.text_input("Pengarang")
        f_desc = st.text_area("Deskripsi singkat", placeholder="Isi buku dalam satu-dua kalimat")
        if st.form_submit_button("Tambah & indeks", width="stretch"):
            if f_judul.strip() and f_desc.strip():
                new_id = f"B{len(base_books()) + len(st.session_state.added) + 1:02d}"
                st.session_state.added.append(
                    dict(id=new_id, nomor_panggil="000 BARU", judul=f_judul.strip(),
                         pengarang=f_pengarang.strip() or "-", deskripsi=f_desc.strip())
                )
                st.success(f"{new_id} ditambahkan dan diindeks.")
            else:
                st.error("Judul dan deskripsi wajib diisi.")
    if st.session_state.added and st.button("Kembalikan koleksi awal", width="stretch"):
        st.session_state.added = []
        st.rerun()
    st.caption("Buku tambahan hanya tersimpan selama sesi browser ini.")

# ---------------------------------------------------------------------------
# Bangun / ambil engine
# ---------------------------------------------------------------------------
all_books = base_books() + st.session_state.added
books_key = tuple((d["id"], d["nomor_panggil"], d["judul"], d["pengarang"], d["deskripsi"]) for d in all_books)
engine = get_engine(books_key, remove_sw)

# ---------------------------------------------------------------------------
# Header & input query
# ---------------------------------------------------------------------------
st.caption("STKI · PERTEMUAN 1 · LABORATORIUM PENCARIAN")
st.title("Katalog Temu Kembali")
st.write(
    "Mesin pencari mini untuk katalog buku contoh. Ketik query, pilih model di sidebar, "
    "lalu lihat bagaimana query diproses, dicari lewat **inverted index**, dan diberi skor."
)

query = st.text_input("Query", key="query", placeholder="Contoh: pemrograman berorientasi objek")

st.write("Coba:")
cols = st.columns(4)
for i, (q, m) in enumerate(EXAMPLES):
    cols[i % 4].button(q, key=f"ex{i}", on_click=use_example, args=(q, m), width="stretch")

st.latex(FORMULAS[model])

results = engine.search(query, model, k1, b)
terms = query_terms(query, remove_sw)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Buku ditemukan", f"{len(results)} / {engine.N}")
m2.metric("Istilah di indeks", len(engine.index))
m3.metric("avgdl (token)", f"{engine.avgdl:.2f}")
m4.metric("Model", MODELS[model])

# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
tab_hasil, tab_proses, tab_index, tab_banding = st.tabs(
    ["Hasil pencarian", "1 · Pemrosesan query", "2 · Inverted index", "3 · Perbandingan model"]
)


def highlight(text: str, hit_terms: set[str]) -> str:
    """Bungkus kata yang cocok dengan <mark> (setelah teks di-escape)."""
    return re.sub(
        r"[A-Za-z0-9\+]+",
        lambda mo: f"<mark>{mo.group(0)}</mark>" if mo.group(0).lower() in hit_terms else mo.group(0),
        html.escape(text),
    )


with tab_hasil:
    if not query.strip():
        st.info("Ketik query untuk mulai mencari.")
    elif not results:
        unknown = [t for t in terms if not engine.df(t)]
        msg = "Tidak ada buku yang cocok. "
        if unknown:
            msg += "Istilah " + ", ".join(f"“{t}”" for t in unknown) + " tidak ada di indeks. "
        msg += ("Pencarian keyword hanya menemukan kata yang sama persis, jadi sinonim seperti "
                "“rumah makan” tidak dikenali sebagai “restoran”.")
        st.warning(msg)
    else:
        hit = set(terms)
        for rank, r in enumerate(results, 1):
            book = engine.by_id[r.id]
            if r.score is None:
                stamp = '<div class="stamp">COCOK<small>boolean</small></div>'
                label = "·"
            else:
                stamp = f'<div class="stamp">{r.score:.3f}<small>{"bm25" if model == "bm25" else "cosine"}</small></div>'
                label = str(rank)
            why = "".join(f"<span>{html.escape(t)} +{c:.3f}</span>" for t, c in r.parts)
            st.markdown(
                f"""<div class="kartu">
  <div class="top">
    <div><div class="call">{label} &nbsp;·&nbsp; {html.escape(book.nomor_panggil)} &nbsp;·&nbsp; {book.id}</div>
      <div class="judul">{highlight(book.judul, hit)}</div>
      <div class="pengarang">{html.escape(book.pengarang)}</div></div>
    {stamp}
  </div>
  <div class="desc">{highlight(book.deskripsi, hit)}</div>
  {f'<div class="why">{why}</div>' if why else ''}
</div>""",
                unsafe_allow_html=True,
            )

with tab_proses:
    toks = tokenize(query)

    def chips(items: list[str]) -> str:
        return "".join(items) or "–"

    st.markdown("**Query asli**")
    st.code(query or " ", language=None)
    st.markdown("**Case folding + tokenisasi**")
    st.markdown(chips([f'<span class="tok">{html.escape(t)}</span>' for t in toks]), unsafe_allow_html=True)
    st.markdown("**Stopword removal**" + ("" if remove_sw else " _(dimatikan)_"))
    marked = []
    for t in toks:
        if remove_sw and t in STOPWORDS:
            marked.append(f'<span class="tok stop">{html.escape(t)}</span>')
        elif t in BOOLEAN_OPS:
            marked.append(f'<span class="tok op">{t.upper()}</span>')
        else:
            marked.append(f'<span class="tok term">{html.escape(t)}</span>')
    st.markdown(chips(marked), unsafe_allow_html=True)
    st.markdown("**Istilah yang dicari di indeks**")
    st.markdown(chips([f'<span class="tok term">{html.escape(t)}</span>' for t in terms]), unsafe_allow_html=True)
    with st.expander("Daftar stopword yang dipakai"):
        st.write(", ".join(sorted(STOPWORDS)))

with tab_index:
    st.markdown("Posting list tiap istilah query: daftar buku yang memuat istilah itu beserta frekuensinya (**tf**).")
    if terms:
        rows = []
        for t in terms:
            rows.append({
                "istilah": t,
                "df": engine.df(t),
                "idf (log10 N/df)": round(engine.idf_log(t), 3),
                "idf BM25": round(engine.idf_bm25(t), 3),
                "posting list (id:tf)": " → ".join(f"{d}:{f}" for d, f in engine.postings(t)) or "tidak ada di indeks",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Belum ada istilah untuk dicari.")

    with st.expander(f"Lihat seluruh inverted index ({len(engine.index)} istilah)"):
        full = pd.DataFrame(
            [{"istilah": t, "df": len(p), "posting list": " → ".join(f"{d}:{f}" for d, f in p)}
             for t, p in engine.index.items()]
        ).sort_values(["df", "istilah"], ascending=[False, True])
        st.dataframe(full, hide_index=True, width="stretch", height=360)

    with st.expander("Lihat seluruh koleksi buku"):
        st.dataframe(pd.DataFrame(all_books), hide_index=True, width="stretch")

with tab_banding:
    st.markdown("Peringkat dari ketiga model untuk query yang sama. Tanda ✓ berarti posisi sama dengan BM25.")
    res_b = engine.search(query, "boolean")
    res_t = engine.search(query, "tfidf")
    res_m = engine.search(query, "bm25", k1, b)
    n = max(len(res_b), len(res_t), len(res_m))

    def cell(res, i, mark_same=False):
        if i >= len(res):
            return "–"
        r = res[i]
        txt = f"{r.id} {engine.by_id[r.id].judul}"
        if r.score is not None:
            txt += f" ({r.score:.2f})"
        if mark_same and i < len(res_m) and res_m[i].id == r.id:
            txt = "✓ " + txt
        return txt

    if n:
        table = pd.DataFrame({
            "Posisi": range(1, n + 1),
            "Boolean (tanpa ranking)": [cell(res_b, i) for i in range(n)],
            "TF-IDF + Cosine": [cell(res_t, i, True) for i in range(n)],
            "BM25": [cell(res_m, i) for i in range(n)],
        })
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("Tidak ada hasil dari ketiga model.")

st.divider()
st.caption(
    "Koleksi berisi deskripsi buku contoh untuk latihan, dengan nomor panggil gaya Dewey (DDC). "
    "Preprocessing: case folding → tokenisasi → stopword removal, tanpa stemming."
)
