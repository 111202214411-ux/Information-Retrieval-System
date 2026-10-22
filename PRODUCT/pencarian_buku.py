"""
Rak Buku — Mesin Pencarian Judul Buku (Streamlit)
=================================================
Tugas Sistem Temu Kembali Informasi (STKI).

- Mesin temu kembali (preprocessing, indeks terbalik, TF-IDF, cosine similarity)
  ada di ir.py; file ini hanya mengurus tampilan.
- Korpus dokumen ada di buku.json (yang diindeks: teks judul buku).
- Sampul asli diletakkan di folder covers/ sesuai field "cover". Jika belum ada,
  sampul sederhana dibuat otomatis.

Menjalankan di komputer sendiri:
    pip install -r requirements.txt
    streamlit run pencarian_buku.py
"""

from __future__ import annotations

import html
import json
import re
import zlib
from pathlib import Path

import pandas as pd
import streamlit as st

import ir

try:  # kolom pencarian yang memperbarui hasil di setiap ketikan
    from st_keyup import st_keyup
except ImportError:  # cadangan: hasil diperbarui setelah Enter
    st_keyup = None

BASE_DIR = Path(__file__).parent
COVER_DIR = BASE_DIR / "covers"

st.set_page_config(page_title="Rak Buku — Pencarian Judul Buku", page_icon="📚", layout="wide")

# Warna kain sampul untuk punggung buku & sampul pengganti
BINDINGS = [
    ("#7A2E2E", "#F4E9D8"),  # merah marun
    ("#2F4F6F", "#EEF0E6"),  # biru tua
    ("#3E6B48", "#F1EEDC"),  # hijau
    ("#C99A3B", "#2A2014"),  # mustard
    ("#5B3F6B", "#F2EADF"),  # ungu
    ("#2E6E6E", "#EDF2E9"),  # teal
    ("#A0522D", "#F7EDDF"),  # tembaga
    ("#3F444D", "#E9E4D6"),  # abu batu
]


# ---------------------------------------------------------------------------
# Data & indeks (dibangun sekali, lalu disimpan di cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def muat_indeks() -> tuple[list[dict], ir.Indeks]:
    buku = json.loads((BASE_DIR / "buku.json").read_text(encoding="utf-8"))
    return buku, ir.build_index(buku, "judul")


BOOKS, INDEX = muat_indeks()


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text))


def fmt(x: float, d: int = 4) -> str:
    """Format angka gaya Indonesia: 0,5868"""
    return f"{x:.{d}f}".replace(".", ",")


def binding(doc_id: int) -> tuple[str, str]:
    return BINDINGS[doc_id % len(BINDINGS)]


def nama_belakang(nama: str) -> str:
    return nama.split()[-1]


_MD_KHUSUS = re.compile(r"([\\`*_{}\[\]()#+\-.!:|~<>&])")


def md_escape(text: str) -> str:
    return _MD_KHUSUS.sub(r"\\\1", text)


def label_punggung(judul: str, terms: set[str], prefix_len: int) -> str:
    """Label tombol (Markdown) dengan bagian judul yang cocok ditandai."""
    if prefix_len:
        return f":orange-background[{md_escape(judul[:prefix_len])}]{md_escape(judul[prefix_len:])}"
    if not terms:
        return md_escape(judul)
    out = []
    for i, part in enumerate(re.split(r"([\w'’]+)", judul)):
        if i % 2 == 1:
            token = ir.fold(part).replace("'", "").replace("’", "")
            if token not in ir.STOPWORDS and ir.stem(token) in terms:
                out.append(f":orange-background[{md_escape(part)}]")
                continue
        out.append(md_escape(part))
    return "".join(out)


def tandai_html(judul: str, prefix_len: int) -> str:
    return f"<mark>{esc(judul[:prefix_len])}</mark>{esc(judul[prefix_len:])}"


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{esc(text)}</span>'


# ---------------------------------------------------------------------------
# Gaya tampilan
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Libre+Caslon+Text:wght@400;700&display=swap');

:root {
  --wall: #E3E7DC; --cabinet: #D3D9C6; --ink: #1F2B25; --ink-soft: #55625A;
  --paper: #F6F7F2; --line: #C4CBB8; --wood: #84573A; --wood-light: #A3714D;
  --wood-dark: #5E3C27; --accent: #2F5D50; --mark: #F1D27A;
  --row: 232px; --plank: 18px;
  --serif: "Libre Caslon Text", Georgia, serif;
  --sans: "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
}

.stApp, .stApp p, .stApp li, .stApp td, .stApp th, .stApp label, .stApp input, .stApp button {
  font-family: var(--sans);
}
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }

.masthead h1 {
  font-family: var(--serif) !important; font-weight: 700; color: var(--ink);
  font-size: clamp(2.6rem, 7vw, 4.25rem); line-height: 1; letter-spacing: -0.02em;
  margin: 0; padding: 0;
}
.masthead p { margin: 14px 0 6px; max-width: 56ch; color: var(--ink-soft); font-size: 1.0625rem; }
.status { color: var(--ink-soft); font-size: .9375rem; margin: -4px 2px 6px; }
.status strong { color: var(--ink); font-weight: 600; }

/* ---------- Lemari & rak: kontainer tombol disusun menyamping ---------- */
.st-key-rak {
  display: flex !important; flex-direction: row !important; flex-wrap: wrap !important;
  gap: 0 4px !important; align-items: flex-end;
  border: 14px solid var(--wood); border-bottom-width: 18px; border-radius: 6px;
  padding: 0 14px !important; min-height: var(--row);
  background-color: var(--cabinet);
  background-image: linear-gradient(to bottom,
    transparent calc(var(--row) - var(--plank)),
    var(--wood-light) calc(var(--row) - var(--plank)),
    var(--wood) calc(var(--row) - var(--plank) + 4px),
    var(--wood-dark) var(--row));
  background-size: 100% var(--row);
  box-shadow: inset 0 10px 24px rgba(0,0,0,.14), 0 2px 0 var(--wood-dark);
}
.st-key-rak > div {
  width: auto !important; flex: 0 0 auto !important;
  height: var(--row); display: flex; align-items: flex-end;
  padding-bottom: var(--plank); position: relative;
}
.st-key-rak > div::after {   /* nomor peringkat di papan */
  position: absolute; left: 0; right: 0; bottom: 0; height: var(--plank);
  display: grid; place-items: center; color: #F6E7D2;
  font: 600 .6875rem/1 var(--sans); font-variant-numeric: tabular-nums;
}

/* ---------- Punggung buku ---------- */
.st-key-rak button {
  --bg: #2F4F6F; --fg: #F4EEDF;
  writing-mode: vertical-rl;
  display: flex; justify-content: space-between; align-items: center; gap: 10px;
  min-width: 46px !important; min-height: 0 !important; width: auto !important;
  padding: 16px 10px !important; margin: 0;
  border: 0 !important; border-radius: 3px 3px 1px 1px !important;
  background: linear-gradient(90deg, rgba(0,0,0,.28) 0%, rgba(255,255,255,.14) 18%,
      rgba(255,255,255,0) 42%, rgba(0,0,0,.08) 80%, rgba(0,0,0,.30) 100%), var(--bg) !important;
  color: var(--fg) !important;
  box-shadow: 2px 0 3px rgba(0,0,0,.18);
  transition: transform .22s cubic-bezier(.2,.8,.2,1);
  position: relative;
}
.st-key-rak button::before {
  content: ""; position: absolute; left: 5px; right: 5px; top: 8px; bottom: 8px;
  border-top: 2px solid color-mix(in srgb, var(--fg) 55%, transparent);
  border-bottom: 2px solid color-mix(in srgb, var(--fg) 55%, transparent);
  pointer-events: none;
}
.st-key-rak button:hover { transform: translateY(-6px); }
.st-key-rak button:focus-visible { outline: 3px solid var(--accent) !important; outline-offset: 3px; }
.st-key-rak button p {
  font-family: var(--serif) !important; font-weight: 700; font-size: .9375rem !important;
  line-height: 1.2; text-align: start; overflow: hidden;
}
/* lapisan pembungkus bawaan Streamlit dibuat "transparan" agar judul
   menjadi item flex langsung dari tombol dan bisa terlipat ke beberapa kolom */
.st-key-rak button > div,
.st-key-rak button > div > span,
.st-key-rak button > div > span > span,
.st-key-rak button [data-testid="stMarkdownContainer"] { display: contents !important; }
.st-key-rak button p {
  flex: 0 1 auto; min-inline-size: 0; margin: 0 !important;
  white-space: normal !important; text-overflow: clip !important;
}
.st-key-rak button::after {          /* nama pengarang di punggung */
  content: var(--author); flex: 0 1 auto; max-inline-size: 40%;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
  font: 500 .6875rem/1.2 var(--sans); letter-spacing: .04em; opacity: .85;
}
.st-key-rak button .stMarkdownColoredBackground {   /* bagian judul yang cocok */
  background: var(--mark) !important; color: #1F2B25 !important;
  border-radius: 2px; padding: 0 1px;
}

/* ---------- Keadaan kosong ---------- */
.st-key-rak:empty, .st-key-rak:not(:has(button)) { display: none !important; }
.st-key-kosong { border: 2px dashed var(--line); border-radius: 12px; padding: 8px 16px 24px; }
.empty { text-align: center; padding: 20px 12px 4px; }
.empty__title { font-family: var(--serif); font-size: 1.375rem; margin: 0 0 6px; color: var(--ink); }
.empty__text { color: var(--ink-soft); margin: 0 auto !important; max-width: 46ch; }
.empty p { text-align: center !important; }

/* ---------- Detail buku (dialog) ---------- */
.cover-art {
  position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px; width: 100%; max-width: 240px; margin: 0 auto; aspect-ratio: 2 / 3;
  padding: 22px 18px 22px 24px; border-radius: 3px 8px 8px 3px; text-align: center;
  background: linear-gradient(90deg, rgba(0,0,0,.22), rgba(0,0,0,0) 9%), var(--bg);
  color: var(--fg);
  box-shadow: inset 6px 0 0 rgba(0,0,0,.18), 0 18px 30px -12px rgba(0,0,0,.45);
  animation: tarik .45s cubic-bezier(.2,.8,.2,1) both;
}
.cover-art::after {
  content: ""; position: absolute; inset: 10px 10px 10px 16px;
  border: 1.5px solid color-mix(in srgb, var(--fg) 55%, transparent); border-radius: 2px;
}
.cover-art__title { font-family: var(--serif); font-weight: 700; font-size: 1.375rem; line-height: 1.15; }
.cover-art__rule { width: 36px; border-top: 1.5px solid currentColor; opacity: .7; }
.cover-art__author { font-size: .8125rem; font-weight: 500; opacity: .9; }
@keyframes tarik { from { opacity: 0; transform: translateY(24px) rotateY(-28deg); } to { opacity: 1; transform: none; } }

.rank { color: var(--ink-soft); font-size: .875rem; margin: 0 0 4px; font-variant-numeric: tabular-nums; }
.rank strong { color: var(--accent); }
.book-title { font-family: var(--serif) !important; font-weight: 700 !important; font-size: 1.625rem !important; line-height: 1.15 !important; margin: 0 0 14px !important; padding: 0 !important; color: var(--ink); }
.facts { margin: 0; }
.facts div { display: grid; grid-template-columns: 7.5rem 1fr; gap: 12px; padding: 10px 0; border-top: 1px solid var(--line); }
.facts div:last-child { border-bottom: 1px solid var(--line); }
.facts dt { color: var(--ink-soft); font-size: .875rem; padding-top: 2px; }
.facts dd { margin: 0; font-weight: 600; color: var(--ink); }
.tag { display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0; border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 14%, transparent); font-size: .8125rem; font-weight: 500; }

/* ---------- Tahapan temu kembali ---------- */
.process-title { font-family: var(--serif) !important; font-weight: 700 !important; font-size: clamp(1.5rem, 3.5vw, 2rem) !important; padding: 0 !important;
  margin: 2.5rem 0 0; color: var(--ink); }
.process-intro { color: var(--ink-soft); margin: 8px 0 20px; max-width: 64ch; }
ol.steps { list-style: none !important; margin: 0 !important; padding: 0 !important; counter-reset: step; max-width: 880px; }
ol.steps > li.step { position: relative; counter-increment: step; padding: 0 0 26px 52px !important;
  margin: 0 !important; list-style: none !important; }
.step::after { content: ""; position: absolute; left: 15px; top: 34px; bottom: 6px; border-left: 2px solid var(--line); }
.step:last-child::after { display: none; }
.step::before { content: counter(step); position: absolute; left: 0; top: 0; display: grid; place-items: center;
  width: 32px; height: 32px; border-radius: 50%; background: var(--accent); color: var(--paper);
  font: 600 .875rem/1 var(--sans); }
.step h3 { margin: 4px 0 2px !important; padding: 0 !important; font-size: 1.0625rem !important; font-weight: 600; color: var(--ink); font-family: var(--sans) !important; }
.step > p { margin: 0 0 10px; color: var(--ink-soft); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { padding: 4px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); font-size: .9375rem; }
.chip--removed { color: var(--ink-soft); text-decoration: line-through; border-style: dashed; background: transparent; }
.chip--stem strong { color: var(--accent); }
.formula { font-family: var(--serif); font-style: italic; color: var(--ink) !important; }
.muted { color: var(--ink-soft); }
.small { font-size: .875rem; margin: 8px 0 0; }
mark { background: var(--mark); color: #1F2B25; border-radius: 2px; padding: 0 1px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--paper); }
.data-table { width: 100%; border-collapse: collapse; font-size: .9375rem; margin: 0; }
.data-table th, .data-table td { padding: 8px 12px; text-align: left; vertical-align: top; border: 0; border-bottom: 1px solid var(--line); }
.data-table tbody tr:last-child td { border-bottom: 0; }
.data-table th { color: var(--ink-soft); font-size: .8125rem; font-weight: 500; white-space: nowrap; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.data-table .is-muted td { color: var(--ink-soft); }

.footnote { color: var(--ink-soft); font-size: .8125rem; margin-top: 28px; }
.footnote code { font-size: .8125rem; }

@media (max-width: 640px) {
  :root { --row: 206px; --plank: 16px; }
  .st-key-rak { border-width: 10px 10px 14px; padding: 0 6px !important; }
  .st-key-rak button { min-width: 40px !important; padding: 14px 8px !important; }
  .st-key-rak button p { font-size: .8125rem !important; }
  ol.steps > li.step { padding-left: 44px !important; }
  .data-table th, .data-table td { padding: 7px 9px; }
  .facts div { grid-template-columns: 6.5rem 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Keadaan (state)
# ---------------------------------------------------------------------------
st.session_state.setdefault("versi_cari", 0)   # ganti key = isi ulang kolom pencarian
st.session_state.setdefault("isi_awal", "")
st.session_state.setdefault("dibuka", None)     # doc_id buku yang diklik


def pakai_saran(kata: str) -> None:
    st.session_state.versi_cari += 1
    st.session_state.isi_awal = kata


def pilih_buku(doc_id: int) -> None:
    st.session_state.dibuka = doc_id


# ---------------------------------------------------------------------------
# Kepala halaman & kolom pencarian
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>Rak Buku</h1>'
    "<p>Ketik huruf awal judul atau kata dari judul buku. Hasilnya diurutkan menurut "
    "kemiripan dengan kata kuncimu, lalu ambil bukunya dari rak untuk melihat sampul "
    "dan keterangannya.</p></div>",
    unsafe_allow_html=True,
)

key_cari = f"cari_{st.session_state.versi_cari}"
if st_keyup is not None:
    query = st_keyup("Kata kunci judul buku", value=st.session_state.isi_awal, key=key_cari,
                     placeholder="Contoh: hujan bulan", label_visibility="collapsed", debounce=150)
else:
    query = st.text_input("Kata kunci judul buku", value=st.session_state.isi_awal, key=key_cari,
                          placeholder="Contoh: hujan bulan (tekan Enter)", label_visibility="collapsed")
query = (query or "").strip()

# ---------------------------------------------------------------------------
# Pencarian
# ---------------------------------------------------------------------------
hasil = ir.search(INDEX, query) if query else None
if hasil is None:
    view = [ir.Hasil(d, b, 0.0, 0.0, "semua") for d, b in enumerate(BOOKS)]
else:
    view = hasil.results

if not query:
    status = f"<strong>{len(BOOKS)} buku</strong> di rak. Ketik huruf atau kata dari judul untuk mencari."
elif view:
    status = f"<strong>{len(view)} buku</strong> ditemukan untuk “{esc(query)}”."
elif not hasil.steps.terms:
    status = "Semua kata pada kata kunci termasuk stopword. Coba kata lain dari judul."
elif not hasil.q_weights:
    status = f"Kata “{esc(query)}” tidak ada di indeks judul."
else:
    status = f"Tidak ada judul yang relevan untuk “{esc(query)}”."
st.markdown(f'<p class="status">{status}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Rak buku: setiap punggung buku adalah tombol
# ---------------------------------------------------------------------------
terms = set(hasil.q_weights) if hasil else set()
gaya_buku = []
if not view:
    with st.container(key="kosong"):
        st.markdown(
            '<div class="empty"><p class="empty__title">Belum ada judul yang relevan.</p>'
            '<p class="empty__text">Tidak ada judul yang diawali teks ini atau memuat kata ini. '
            "Coba kata lain dari judul, misalnya:</p></div>",
            unsafe_allow_html=True,
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.button("hujan", on_click=pakai_saran, args=("hujan",))
            st.button("bumi manusia", on_click=pakai_saran, args=("bumi manusia",))

with st.container(key="rak", horizontal=True, wrap=True, gap=None, vertical_alignment="bottom"):
    for rank, item in enumerate(view, 1):
        d, buku = item.doc_id, item.doc
        bg, fg = binding(d)
        h = zlib.crc32(f"{buku['judul']}{buku['pengarang']}".encode())
        kata_terpanjang = max(len(w) for w in buku["judul"].split())
        turun = h % 6 if len(buku["judul"]) > 20 or kata_terpanjang >= 7 else h % 34
        pengarang = nama_belakang(buku["pengarang"]).replace("\\", "").replace('"', "")
        gaya = (f".st-key-b{d} button{{--bg:{bg};--fg:{fg};--author:\"{pengarang}\";"
                f"height:calc(var(--row) - var(--plank) - 26px - {turun}px) !important;}}")
        if item.match != "semua":
            gaya += f'.st-key-rak > .st-key-b{d}::after{{content:"{rank}";}}'
        gaya_buku.append(gaya)

        prefix_len = len(hasil.prefix) if item.match == "awalan" else 0
        label = label_punggung(buku["judul"], terms, prefix_len)
        bantuan = (f"Peringkat {rank} · " if item.match != "semua" else "") + f"{buku['judul']} — {buku['pengarang']}"
        st.button(label, key=f"b{d}", help=bantuan, on_click=pilih_buku, args=(d,), wrap=True)

st.markdown(f"<style>{''.join(gaya_buku)}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Detail buku (muncul sebagai jendela saat punggung buku diklik)
# ---------------------------------------------------------------------------
@st.dialog("Detail buku", width="large")
def tampilkan_detail(doc_id: int) -> None:
    buku = BOOKS[doc_id]
    bg, fg = binding(doc_id)
    pos = next((i for i, v in enumerate(view, 1) if v.doc_id == doc_id), None)
    item = view[pos - 1] if pos else None

    kiri, kanan = st.columns([2, 3], gap="large")
    with kiri:
        file_cover = COVER_DIR / buku.get("cover", "")
        if buku.get("cover") and file_cover.is_file():
            st.image(str(file_cover), width="stretch")
        else:
            st.markdown(
                f'<div class="cover-art" style="--bg:{bg};--fg:{fg}">'
                f'<span class="cover-art__title">{esc(buku["judul"])}</span>'
                f'<span class="cover-art__rule"></span>'
                f'<span class="cover-art__author">{esc(buku["pengarang"])}</span></div>',
                unsafe_allow_html=True,
            )
    with kanan:
        if item and item.match == "kata":
            st.markdown(f'<p class="rank">Peringkat <strong>{pos}</strong> dari {len(view)}, '
                        f'skor cosine <strong>{fmt(item.score)}</strong></p>', unsafe_allow_html=True)
        elif item and item.match == "awalan":
            st.markdown(f'<p class="rank">Peringkat <strong>{pos}</strong> dari {len(view)}, '
                        "cocok dengan awalan judul</p>", unsafe_allow_html=True)
        tags = "".join(f'<span class="tag">{esc(g)}</span>' for g in buku["genre"])
        st.markdown(
            f'<h2 class="book-title">{esc(buku["judul"])}</h2>'
            '<dl class="facts">'
            f'<div><dt>Nama buku</dt><dd>{esc(buku["judul"])}</dd></div>'
            f'<div><dt>Pengarang</dt><dd>{esc(buku["pengarang"])}</dd></div>'
            f"<div><dt>Genre</dt><dd>{tags}</dd></div>"
            f'<div><dt>Tahun terbit</dt><dd>{esc(buku["tahun"])}</dd></div>'
            f'<div><dt>Penerbit</dt><dd>{esc(buku["penerbit"])}</dd></div>'
            "</dl>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Kembalikan ke rak"):
            st.rerun()


if st.session_state.dibuka is not None:
    dibuka = st.session_state.dibuka
    st.session_state.dibuka = None
    tampilkan_detail(dibuka)


# ---------------------------------------------------------------------------
# Tahapan temu kembali untuk query yang sedang diketik
# ---------------------------------------------------------------------------
def html_tahapan(h: ir.HasilPencarian) -> str:
    s = h.steps
    tidak_ada = '<span class="muted">tidak ada token</span>'

    baris_bobot = "".join(
        (f'<tr><td>{esc(t)}</td><td class="num">{h.qtf[t]}</td>'
         f'<td class="num">{len(INDEX.index[t])}</td><td class="num">{fmt(INDEX.idf[t])}</td>'
         f'<td class="num">{fmt(h.q_weights[t])}</td></tr>')
        if t in INDEX.idf else
        (f'<tr class="is-muted"><td>{esc(t)}</td><td class="num">{h.qtf[t]}</td>'
         '<td class="num">–</td><td class="num">–</td><td class="num">tidak di indeks</td></tr>')
        for t in h.qtf
    ) or '<tr><td colspan="5" class="muted">Tidak ada term.</td></tr>'

    hasil_kata = [(i, r) for i, r in enumerate(h.results, 1) if r.match == "kata"]
    baris_rank = "".join(
        f'<tr><td class="num">{i}</td><td>{esc(r.doc["judul"])} '
        f'<span class="muted">{esc(r.doc["pengarang"])}</span></td>'
        f'<td class="num">{fmt(r.dot)}</td><td class="num">{fmt(INDEX.norms[r.doc_id])}</td>'
        f'<td class="num"><strong>{fmt(r.score)}</strong></td></tr>'
        for i, r in hasil_kata
    )
    tabel_rank = (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        '<th class="num">#</th><th>Judul</th><th class="num">Σ w(q)·w(d)</th>'
        '<th class="num">|d|</th><th class="num">Skor</th></tr></thead>'
        f"<tbody>{baris_rank}</tbody></table></div>"
        if hasil_kata else '<p class="muted">Tidak ada dokumen dengan skor di atas 0.</p>'
    )

    posisi = {r.doc_id: i for i, r in enumerate(h.results, 1)}
    kata_ids = {r.doc_id for _, r in hasil_kata}
    awalan = sorted(h.prefix_matches, key=lambda d: posisi[d])
    chips_awalan = "".join(
        f'<span class="chip">{tandai_html(BOOKS[d]["judul"], len(h.prefix))} '
        f'<span class="muted">#{posisi[d]}{", sudah dari cosine" if d in kata_ids else ""}</span></span>'
        for d in awalan
    ) or '<span class="muted">Tidak ada judul yang diawali teks ini.</span>'

    stem_chips = "".join(
        chip(dasar) if tok == dasar else
        f'<span class="chip chip--stem">{esc(tok)} › <strong>{esc(dasar)}</strong></span>'
        for tok, dasar in s.stems
    ) or '<span class="muted">tidak ada term tersisa</span>'
    ada_imbuhan = any(tok != dasar for tok, dasar in s.stems)

    return f"""
<ol class="steps">
  <li class="step"><h3>Case folding</h3><p>Semua huruf diubah menjadi huruf kecil.</p>
    <div class="chips">{chip(s.folded)}</div></li>
  <li class="step"><h3>Tokenisasi</h3><p>Teks dipecah menjadi kata; tanda baca dibuang.</p>
    <div class="chips">{"".join(chip(t) for t in s.tokens) or tidak_ada}</div></li>
  <li class="step"><h3>Stopword removal</h3><p>Kata umum yang tidak membedakan judul dihapus.</p>
    <div class="chips">{"".join(chip(t, "chip--removed" if t in ir.STOPWORDS else "") for t in s.tokens) or tidak_ada}</div></li>
  <li class="step"><h3>Stemming</h3><p>Imbuhan dilepas sehingga kata kembali ke bentuk dasarnya.</p>
    <div class="chips">{stem_chips}</div>
    {"" if ada_imbuhan else '<p class="muted small">Tidak ada kata berimbuhan pada kata kunci ini.</p>'}</li>
  <li class="step"><h3>Bobot TF-IDF query</h3>
    <p class="formula">w = tf × idf,&nbsp; idf = log<sub>10</sub>(N / df)</p>
    <div class="table-wrap"><table class="data-table"><thead><tr><th>Term</th><th class="num">tf</th>
      <th class="num">df</th><th class="num">idf</th><th class="num">w(q)</th></tr></thead>
      <tbody>{baris_bobot}</tbody></table></div>
    {f'<p class="small muted">Panjang vektor query |q| = {fmt(h.q_norm)}</p>' if h.q_weights else ""}
    {'<p class="small muted">Term yang tidak ada di indeks diabaikan.</p>' if h.unknown else ""}</li>
  <li class="step"><h3>Cosine similarity &amp; peringkat</h3>
    <p class="formula">cos(q, d) = Σ w(q)·w(d) / (|q| × |d|)</p>{tabel_rank}</li>
  <li class="step"><h3>Pencocokan awalan judul</h3>
    <p>Judul yang diawali “{esc(h.prefix)}” (huruf besar/kecil tidak dibedakan) ikut ditampilkan,
    agar hasil sudah muncul sejak huruf pertama diketik. Yang belum ditemukan lewat cosine
    ditambahkan di akhir peringkat, urut abjad.</p>
    <div class="chips">{chips_awalan}</div></li>
</ol>"""


st.markdown('<h2 class="process-title">Cara hasil ini ditemukan</h2>', unsafe_allow_html=True)
if hasil is None:
    st.markdown(
        '<p class="process-intro">Ketik kata kunci untuk melihat tahapan temu kembali yang dijalankan: '
        "preprocessing, pembobotan TF-IDF, perankingan dengan cosine similarity, lalu pencocokan "
        "awalan judul.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p class="process-intro">Tahapan untuk kata kunci “{esc(query)}”. '
        f"Koleksi berisi N = {INDEX.N} judul buku.</p>{html_tahapan(hasil)}",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Indeks terbalik
# ---------------------------------------------------------------------------
with st.expander(f"Lihat indeks terbalik ({len(INDEX.index)} term dari {INDEX.N} judul)"):
    baris = []
    for term in sorted(INDEX.index):
        postings = INDEX.index[term]
        baris.append({
            "Term": term,
            "df": len(postings),
            "idf": round(INDEX.idf[term], 4),
            "Muncul di judul": "; ".join(
                BOOKS[d]["judul"] + (f" (tf {f})" if f > 1 else "") for d, f in postings.items()
            ),
        })
    st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")

st.markdown(
    '<p class="footnote">Sampul diambil dari folder <code>covers/</code>. '
    "Jika gambarnya belum ada, sampul sederhana dibuat otomatis.</p>",
    unsafe_allow_html=True,
)
