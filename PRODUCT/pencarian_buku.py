from __future__ import annotations

import html
import json
import re
import zlib
from pathlib import Path

import pandas as pd
import streamlit as st

import ir

try:  # search box that updates results on every keystroke
    from st_keyup import st_keyup
except ImportError:  # fallback: results update after pressing Enter
    st_keyup = None

BASE_DIR = Path(__file__).parent
COVER_DIR = BASE_DIR / "covers"

st.set_page_config(page_title="Bookshelf: Book Title Search", page_icon="📚", layout="wide")

# Cloth colors for book spines and generated covers: (background, text)
BINDINGS = [
    ("#7A2E2E", "#F4E9D8"),  # maroon
    ("#2F4F6F", "#EEF0E6"),  # navy
    ("#3E6B48", "#F1EEDC"),  # green
    ("#C99A3B", "#2A2014"),  # mustard
    ("#5B3F6B", "#F2EADF"),  # purple
    ("#2E6E6E", "#EDF2E9"),  # teal
    ("#A0522D", "#F7EDDF"),  # copper
    ("#3F444D", "#E9E4D6"),  # slate
]


# ---------------------------------------------------------------------------
# Data and index (built once, then cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_index() -> tuple[list[dict], ir.Index]:
    books = json.loads((BASE_DIR / "books.json").read_text(encoding="utf-8"))
    return books, ir.build_index(books, "title")


BOOKS, INDEX = load_index()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text))


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def binding(doc_id: int) -> tuple[str, str]:
    return BINDINGS[doc_id % len(BINDINGS)]


def last_name(name: str) -> str:
    return name.split()[-1]


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!:|~<>&])")


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def spine_label(title: str, terms: set[str], span: tuple[int, int] | None) -> str:
    """Button label (Markdown) with the matching part of the title highlighted.

    span is the (start, end) of a partial match; otherwise words whose root form
    is one of the query terms are highlighted.
    """
    if span:
        a, b = span
        return (f"{md_escape(title[:a])}:orange-background[{md_escape(title[a:b])}]"
                f"{md_escape(title[b:])}")
    if not terms:
        return md_escape(title)
    out = []
    for i, part in enumerate(re.split(r"([\w'’]+)", title)):
        if i % 2 == 1:
            token = ir.fold(part).replace("'", "").replace("’", "")
            if token not in ir.STOPWORDS and ir.stem(token) in terms:
                out.append(f":orange-background[{md_escape(part)}]")
                continue
        out.append(md_escape(part))
    return "".join(out)


def mark_span(title: str, start: int, length: int) -> str:
    end = start + length
    return f"{esc(title[:start])}<mark>{esc(title[start:end])}</mark>{esc(title[end:])}"


# How each partial-match kind is described in the interface
PARTIAL_LABELS = {
    "prefix": ("Start of the title", "matched by the start of the title"),
    "word": ("Start of a word", "matched by the start of a word"),
    "inside": ("Inside a word", "matched by part of a word"),
}
MIN_LENGTH = dict(ir.PARTIAL_KINDS)


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{esc(text)}</span>'


# ---------------------------------------------------------------------------
# Styles
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
  font-size: clamp(2.6rem, 7vw, 4.25rem); line-height: 1.25 !important; letter-spacing: -0.02em;
  margin: 0; padding: 0 !important; overflow: visible;
}
/* hide the link icon Streamlit adds next to headings */
[data-testid="stHeaderActionElements"] { display: none !important; }
.masthead p { margin: 14px 0 6px; max-width: 56ch; color: var(--ink-soft); font-size: 1.0625rem; }
.status { color: var(--ink-soft); font-size: .9375rem; margin: -4px 2px 6px; }
.status strong { color: var(--ink); font-weight: 600; }

/* ---------- Bookcase: the button container laid out as shelf rows ---------- */
.st-key-shelf {
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
.st-key-shelf > div {
  width: auto !important; flex: 0 0 auto !important;
  height: var(--row); display: flex; align-items: flex-end;
  padding-bottom: var(--plank); position: relative;
}
.st-key-shelf > div::after {   /* rank number on the plank */
  position: absolute; left: 0; right: 0; bottom: 0; height: var(--plank);
  display: grid; place-items: center; color: #F6E7D2;
  font: 600 .6875rem/1 var(--sans); font-variant-numeric: tabular-nums;
}

/* ---------- Book spines ---------- */
.st-key-shelf button {
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
.st-key-shelf button::before {
  content: ""; position: absolute; left: 5px; right: 5px; top: 8px; bottom: 8px;
  border-top: 2px solid color-mix(in srgb, var(--fg) 55%, transparent);
  border-bottom: 2px solid color-mix(in srgb, var(--fg) 55%, transparent);
  pointer-events: none;
}
.st-key-shelf button:hover { transform: translateY(-6px); }
.st-key-shelf button:focus-visible { outline: 3px solid var(--accent) !important; outline-offset: 3px; }
/* Streamlit's wrapper elements are made "transparent" so the title becomes a
   direct flex item of the button and can wrap into several vertical columns */
.st-key-shelf button > div,
.st-key-shelf button > div > span,
.st-key-shelf button > div > span > span,
.st-key-shelf button [data-testid="stMarkdownContainer"] { display: contents !important; }
.st-key-shelf button p {
  font-family: var(--serif) !important; font-weight: 700; font-size: .9375rem !important;
  line-height: 1.2; text-align: start; overflow: hidden;
  flex: 0 1 auto; min-inline-size: 0; margin: 0 !important;
  white-space: normal !important; text-overflow: clip !important;
}
.st-key-shelf button::after {          /* author name on the spine */
  content: var(--author); flex: 0 1 auto; max-inline-size: 40%;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
  font: 500 .6875rem/1.2 var(--sans); letter-spacing: .04em; opacity: .85;
}
.st-key-shelf button .stMarkdownColoredBackground {   /* matching part of the title */
  background: var(--mark) !important; color: #1F2B25 !important;
  border-radius: 2px; padding: 0 1px;
}

/* ---------- No results ---------- */
.st-key-shelf:empty, .st-key-shelf:not(:has(button)) { display: none !important; }
.st-key-empty_state { border: 2px dashed var(--line); border-radius: 12px; padding: 8px 16px 24px; }
.empty { text-align: center; padding: 20px 12px 4px; }
.empty__title { font-family: var(--serif); font-size: 1.375rem; margin: 0 0 6px; color: var(--ink); }
.empty__text { color: var(--ink-soft); margin: 0 auto !important; max-width: 46ch; }
.empty p { text-align: center !important; }

/* ---------- Book details (dialog) ---------- */
.cover-art {
  position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px; width: 100%; max-width: 240px; margin: 0 auto; aspect-ratio: 2 / 3;
  padding: 22px 18px 22px 24px; border-radius: 3px 8px 8px 3px; text-align: center;
  background: linear-gradient(90deg, rgba(0,0,0,.22), rgba(0,0,0,0) 9%), var(--bg);
  color: var(--fg);
  box-shadow: inset 6px 0 0 rgba(0,0,0,.18), 0 18px 30px -12px rgba(0,0,0,.45);
  animation: pull-out .45s cubic-bezier(.2,.8,.2,1) both;
}
.cover-art::after {
  content: ""; position: absolute; inset: 10px 10px 10px 16px;
  border: 1.5px solid color-mix(in srgb, var(--fg) 55%, transparent); border-radius: 2px;
}
.cover-art__title { font-family: var(--serif); font-weight: 700; font-size: 1.375rem; line-height: 1.15; }
.cover-art__rule { width: 36px; border-top: 1.5px solid currentColor; opacity: .7; }
.cover-art__author { font-size: .8125rem; font-weight: 500; opacity: .9; }
@keyframes pull-out { from { opacity: 0; transform: translateY(24px) rotateY(-28deg); } to { opacity: 1; transform: none; } }

.rank { color: var(--ink-soft); font-size: .875rem; margin: 0 0 4px; font-variant-numeric: tabular-nums; }
.rank strong { color: var(--accent); }
.book-title { font-family: var(--serif) !important; font-weight: 700 !important; font-size: 1.625rem !important;
  line-height: 1.15 !important; margin: 0 0 14px !important; padding: 0 !important; color: var(--ink); }
.facts { margin: 0; }
.facts div { display: grid; grid-template-columns: 7.5rem 1fr; gap: 12px; padding: 10px 0; border-top: 1px solid var(--line); }
.facts div:last-child { border-bottom: 1px solid var(--line); }
.facts dt { color: var(--ink-soft); font-size: .875rem; padding-top: 2px; }
.facts dd { margin: 0; font-weight: 600; color: var(--ink); }
.tag { display: inline-block; padding: 2px 10px; margin: 0 6px 4px 0; border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 14%, transparent); font-size: .8125rem; font-weight: 500; }

/* ---------- Retrieval steps ---------- */
.process-title { font-family: var(--serif) !important; font-weight: 700 !important;
  font-size: clamp(1.5rem, 3.5vw, 2rem) !important; padding: 0 !important; margin: 2.5rem 0 0; color: var(--ink); }
.process-intro { color: var(--ink-soft); margin: 8px 0 20px; max-width: 64ch; }
ol.steps { list-style: none !important; margin: 0 !important; padding: 0 !important; counter-reset: step; max-width: 880px; }
ol.steps > li.step { position: relative; counter-increment: step; padding: 0 0 26px 52px !important;
  margin: 0 !important; list-style: none !important; }
.step::after { content: ""; position: absolute; left: 15px; top: 34px; bottom: 6px; border-left: 2px solid var(--line); }
.step:last-child::after { display: none; }
.step::before { content: counter(step); position: absolute; left: 0; top: 0; display: grid; place-items: center;
  width: 32px; height: 32px; border-radius: 50%; background: var(--accent); color: var(--paper);
  font: 600 .875rem/1 var(--sans); }
.step h3 { margin: 4px 0 2px !important; padding: 0 !important; font-size: 1.0625rem !important;
  font-weight: 600; color: var(--ink); font-family: var(--sans) !important; }
.step > p { margin: 0 0 10px; color: var(--ink-soft); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.partial-group { margin: 0 0 12px; }
.partial-label { margin: 0 0 6px !important; font-weight: 600; font-size: .9375rem; color: var(--ink); }
.partial-label .muted { font-weight: 400; }
.chip { padding: 4px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); font-size: .9375rem; }
.chip--removed { color: var(--ink-soft); text-decoration: line-through; border-style: dashed; background: transparent; }
.chip--stem strong { color: var(--accent); }
.formula { font-family: var(--serif); font-style: italic; color: var(--ink) !important; }
.muted { color: var(--ink-soft); }
.small { font-size: .875rem; margin: 8px 0 0; }
mark { background: var(--mark); color: #1F2B25; border-radius: 2px; padding: 0 1px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--paper); }
.data-table { width: 100%; border-collapse: collapse; font-size: .9375rem; margin: 0; }
.data-table th, .data-table td { padding: 8px 12px; text-align: left; vertical-align: top; border: 0;
  border-bottom: 1px solid var(--line); }
.data-table tbody tr:last-child td { border-bottom: 0; }
.data-table th { color: var(--ink-soft); font-size: .8125rem; font-weight: 500; white-space: nowrap; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.data-table .is-muted td { color: var(--ink-soft); }

.footnote { color: var(--ink-soft); font-size: .8125rem; margin-top: 28px; }
.footnote code { font-size: .8125rem; }

@media (max-width: 640px) {
  :root { --row: 206px; --plank: 16px; }
  .st-key-shelf { border-width: 10px 10px 14px; padding: 0 6px !important; }
  .st-key-shelf button { min-width: 40px !important; padding: 14px 8px !important; }
  .st-key-shelf button p { font-size: .8125rem !important; }
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
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("search_version", 0)   # a new key resets the search box
st.session_state.setdefault("initial_query", "")
st.session_state.setdefault("opened_book", None)   # doc_id of the clicked book


def use_suggestion(text: str) -> None:
    st.session_state.search_version += 1
    st.session_state.initial_query = text


def open_book(doc_id: int) -> None:
    st.session_state.opened_book = doc_id


# ---------------------------------------------------------------------------
# Header and search box
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>Bookshelf</h1>'
    "<p>Type any part of a book title. Results are ranked by how closely they match "
    "your query. Pick a book from the shelf to see its cover and details.</p></div>",
    unsafe_allow_html=True,
)

search_key = f"search_{st.session_state.search_version}"
if st_keyup is not None:
    query = st_keyup("Book title search", value=st.session_state.initial_query, key=search_key,
                     placeholder="e.g. hujan bulan", label_visibility="collapsed", debounce=150)
else:
    query = st.text_input("Book title search", value=st.session_state.initial_query, key=search_key,
                          placeholder="e.g. hujan bulan (press Enter)", label_visibility="collapsed")
query = (query or "").strip()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
search = ir.search(INDEX, query) if query else None
if search is None:
    view = [ir.Result(d, b, 0.0, 0.0, "all") for d, b in enumerate(BOOKS)]
else:
    view = search.results

if not query:
    status = (f"<strong>{plural(len(BOOKS), 'book')}</strong> on the shelf. "
              "Type any part of a title to search.")
elif view:
    status = f"<strong>{plural(len(view), 'book')}</strong> found for “{esc(query)}”."
elif not search.steps.terms:
    status = "Every word in this query is a stopword. Try another word from a title."
elif not search.q_weights:
    status = f"No titles contain “{esc(query)}”."
else:
    status = f"No relevant titles for “{esc(query)}”."
st.markdown(f'<p class="status">{status}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bookshelf: every book spine is a button
# ---------------------------------------------------------------------------
terms = set(search.q_weights) if search else set()
spine_styles = []
if not view:
    with st.container(key="empty_state"):
        st.markdown(
            '<div class="empty"><p class="empty__title">No matching titles yet.</p>'
            '<p class="empty__text">No title contains this text. '
            "Try another word from a title, for example:</p></div>",
            unsafe_allow_html=True,
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.button("bumi manusia", on_click=use_suggestion, args=("bumi manusia",))
            st.button("slam dunk", on_click=use_suggestion, args=("slam dunk",))

with st.container(key="shelf", horizontal=True, wrap=True, gap=None, vertical_alignment="bottom"):
    for rank, item in enumerate(view, 1):
        d, book = item.doc_id, item.doc
        bg, fg = binding(d)
        h = zlib.crc32(f"{book['title']}{book['author']}".encode())
        longest_word = max(len(w) for w in book["title"].split())
        # long titles get taller spines so they are not clipped
        drop = h % 6 if len(book["title"]) > 20 or longest_word >= 7 else h % 34
        author = last_name(book["author"]).replace("\\", "").replace('"', "")
        style = (f".st-key-b{d} button{{--bg:{bg};--fg:{fg};--author:\"{author}\";"
                 f"height:calc(var(--row) - var(--plank) - 26px - {drop}px) !important;}}")
        if item.match != "all":
            style += f'.st-key-shelf > .st-key-b{d}::after{{content:"{rank}";}}'
        spine_styles.append(style)

        span = (item.start, item.start + len(search.text)) if item.match in PARTIAL_LABELS else None
        label = spine_label(book["title"], terms, span)
        tooltip = (f"Rank {rank} · " if item.match != "all" else "") + f"{book['title']} — {book['author']}"
        st.button(label, key=f"b{d}", help=tooltip, on_click=open_book, args=(d,), wrap=True)

st.markdown(f"<style>{''.join(spine_styles)}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Book details (opens as a dialog when a spine is clicked)
# ---------------------------------------------------------------------------
@st.dialog("Book details", width="large")
def show_details(doc_id: int) -> None:
    book = BOOKS[doc_id]
    bg, fg = binding(doc_id)
    pos = next((i for i, v in enumerate(view, 1) if v.doc_id == doc_id), None)
    item = view[pos - 1] if pos else None

    left, right = st.columns([2, 3], gap="large")
    with left:
        cover_file = COVER_DIR / book.get("cover", "")
        if book.get("cover") and cover_file.is_file():
            st.image(str(cover_file), width="stretch")
        else:
            st.markdown(
                f'<div class="cover-art" style="--bg:{bg};--fg:{fg}">'
                f'<span class="cover-art__title">{esc(book["title"])}</span>'
                f'<span class="cover-art__rule"></span>'
                f'<span class="cover-art__author">{esc(book["author"])}</span></div>',
                unsafe_allow_html=True,
            )
    with right:
        if item and item.match == "term":
            st.markdown(f'<p class="rank">Rank <strong>{pos}</strong> of {len(view)}, '
                        f'cosine score <strong>{fmt(item.score)}</strong></p>', unsafe_allow_html=True)
        elif item and item.match in PARTIAL_LABELS:
            st.markdown(f'<p class="rank">Rank <strong>{pos}</strong> of {len(view)}, '
                        f"{PARTIAL_LABELS[item.match][1]}</p>", unsafe_allow_html=True)
        tags = "".join(f'<span class="tag">{esc(g)}</span>' for g in book["genre"])
        st.markdown(
            f'<h2 class="book-title">{esc(book["title"])}</h2>'
            '<dl class="facts">'
            f'<div><dt>Title</dt><dd>{esc(book["title"])}</dd></div>'
            f'<div><dt>Author</dt><dd>{esc(book["author"])}</dd></div>'
            f"<div><dt>Genre</dt><dd>{tags}</dd></div>"
            f'<div><dt>Year published</dt><dd>{esc(book["year"])}</dd></div>'
            f'<div><dt>Publisher</dt><dd>{esc(book["publisher"])}</dd></div>'
            "</dl>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Put back on the shelf"):
            st.rerun()


if st.session_state.opened_book is not None:
    opened = st.session_state.opened_book
    st.session_state.opened_book = None
    show_details(opened)


# ---------------------------------------------------------------------------
# Retrieval steps for the current query
# ---------------------------------------------------------------------------
def steps_html(s: ir.SearchResult) -> str:
    pre = s.steps
    no_tokens = '<span class="muted">no tokens</span>'

    weight_rows = "".join(
        (f'<tr><td>{esc(t)}</td><td class="num">{s.qtf[t]}</td>'
         f'<td class="num">{len(INDEX.index[t])}</td><td class="num">{fmt(INDEX.idf[t])}</td>'
         f'<td class="num">{fmt(s.q_weights[t])}</td></tr>')
        if t in INDEX.idf else
        (f'<tr class="is-muted"><td>{esc(t)}</td><td class="num">{s.qtf[t]}</td>'
         '<td class="num">–</td><td class="num">–</td><td class="num">not in index</td></tr>')
        for t in s.qtf
    ) or '<tr><td colspan="5" class="muted">No terms.</td></tr>'

    term_hits = [(i, r) for i, r in enumerate(s.results, 1) if r.match == "term"]
    rank_rows = "".join(
        f'<tr><td class="num">{i}</td><td>{esc(r.doc["title"])} '
        f'<span class="muted">{esc(r.doc["author"])}</span></td>'
        f'<td class="num">{fmt(r.dot)}</td><td class="num">{fmt(INDEX.norms[r.doc_id])}</td>'
        f'<td class="num"><strong>{fmt(r.score)}</strong></td></tr>'
        for i, r in term_hits
    )
    rank_table = (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        '<th class="num">#</th><th>Title</th><th class="num">Σ w(q)·w(d)</th>'
        '<th class="num">|d|</th><th class="num">Score</th></tr></thead>'
        f"<tbody>{rank_rows}</tbody></table></div>"
        if term_hits else '<p class="muted">No documents scored above 0.</p>'
    )

    position = {r.doc_id: i for i, r in enumerate(s.results, 1)}
    term_ids = {r.doc_id for _, r in term_hits}
    partial_groups = []
    for kind, min_len in ir.PARTIAL_KINDS:
        name = PARTIAL_LABELS[kind][0]
        hits = sorted((m for m in s.partial_matches if m.kind == kind), key=lambda m: position[m.doc_id])
        if len(s.text) < min_len:
            body = f'<span class="muted">Needs at least {min_len} characters.</span>'
        elif hits:
            body = "".join(
                f'<span class="chip">{mark_span(BOOKS[m.doc_id]["title"], m.start, len(s.text))} '
                f'<span class="muted">#{position[m.doc_id]}'
                f'{", already found by cosine" if m.doc_id in term_ids else ""}</span></span>'
                for m in hits
            )
        else:
            body = '<span class="muted">No matches.</span>'
        partial_groups.append(
            f'<div class="partial-group"><p class="partial-label">{name} '
            f'<span class="muted">({min_len}+ characters)</span></p><div class="chips">{body}</div></div>'
        )

    stem_chips = "".join(
        chip(root) if token == root else
        f'<span class="chip chip--stem">{esc(token)} › <strong>{esc(root)}</strong></span>'
        for token, root in pre.stems
    ) or '<span class="muted">no terms left</span>'
    has_affixes = any(token != root for token, root in pre.stems)

    return f"""
<ol class="steps">
  <li class="step"><h3>Case folding</h3><p>All letters are converted to lowercase.</p>
    <div class="chips">{chip(pre.folded)}</div></li>
  <li class="step"><h3>Tokenization</h3><p>The text is split into words; punctuation is removed.</p>
    <div class="chips">{"".join(chip(t) for t in pre.tokens) or no_tokens}</div></li>
  <li class="step"><h3>Stopword removal</h3><p>Common words that do not distinguish titles are removed.</p>
    <div class="chips">{"".join(chip(t, "chip--removed" if t in ir.STOPWORDS else "") for t in pre.tokens) or no_tokens}</div></li>
  <li class="step"><h3>Stemming</h3><p>Affixes are removed so each word returns to its root form.</p>
    <div class="chips">{stem_chips}</div>
    {"" if has_affixes else '<p class="muted small">This query has no affixed words.</p>'}</li>
  <li class="step"><h3>Query TF-IDF weights</h3>
    <p class="formula">w = tf × idf,&nbsp; idf = log<sub>10</sub>(N / df)</p>
    <div class="table-wrap"><table class="data-table"><thead><tr><th>Term</th><th class="num">tf</th>
      <th class="num">df</th><th class="num">idf</th><th class="num">w(q)</th></tr></thead>
      <tbody>{weight_rows}</tbody></table></div>
    {f'<p class="small muted">Query vector length |q| = {fmt(s.q_norm)}</p>' if s.q_weights else ""}
    {'<p class="small muted">Terms that are not in the index are ignored.</p>' if s.unknown else ""}</li>
  <li class="step"><h3>Cosine similarity &amp; ranking</h3>
    <p class="formula">cos(q, d) = Σ w(q)·w(d) / (|q| × |d|)</p>{rank_table}</li>
  <li class="step"><h3>Partial matching</h3>
    <p>Titles that contain “{esc(s.text)}” (case-insensitive) are also shown, so results appear
    from the first letter typed and partial words are found anywhere in a title. Titles not already
    found by cosine similarity are added at the end of the ranking: start of the title first, then
    start of a word, then inside a word, each in alphabetical order.</p>
    {"".join(partial_groups)}</li>
</ol>"""


st.markdown('<h2 class="process-title">How these results were found</h2>', unsafe_allow_html=True)
if search is None:
    st.markdown(
        '<p class="process-intro">Type a query to see each retrieval step: preprocessing, '
        "TF-IDF weighting, cosine similarity ranking, and partial matching.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<p class="process-intro">Steps for the query “{esc(query)}”. '
        f"The collection has N = {INDEX.N} book titles.</p>{steps_html(search)}",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------
with st.expander(f"View the inverted index ({len(INDEX.index)} terms from {INDEX.N} titles)"):
    rows = []
    for term in sorted(INDEX.index):
        postings = INDEX.index[term]
        rows.append({
            "Term": term,
            "df": len(postings),
            "idf": round(INDEX.idf[term], 4),
            "Found in titles": "; ".join(
                BOOKS[d]["title"] + (f" (tf {f})" if f > 1 else "") for d, f in postings.items()
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown(
    '<p class="footnote">Covers are loaded from the <code>covers/</code> folder. '
    "If an image is missing, a simple cover is generated automatically.</p>",
    unsafe_allow_html=True,
)
