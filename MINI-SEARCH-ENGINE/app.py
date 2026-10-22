from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from ir_engine import (BOOLEAN_OPS, DATA_PATH, STOPWORDS, Book, SearchEngine,
                       load_books, query_terms, tokenize)

MODELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF + Cosine",
    "boolean": "Boolean",
}
FORMULAS = {
    "boolean": r"R(q,d)=\begin{cases}1 & d \text{ satisfies the AND / OR / NOT expression}\\ 0 & \text{otherwise}\end{cases}",
    "tfidf": r"\text{sim}(q,d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\,\lVert\vec d\rVert},\qquad w_{t,d}=(1+\log_{10} tf_{t,d})\cdot\log_{10}\frac{N}{df_t}",
    "bm25": r"\text{score}(q,d)=\sum_{t\in q}\text{idf}(t)\cdot\frac{tf_{t,d}\,(k_1+1)}{tf_{t,d}+k_1\left(1-b+b\cdot\frac{|d|}{avgdl}\right)}",
}
EXAMPLES = [
    ("search engine ranking", "bm25"),
    ("programming AND object", "boolean"),
    ("java OR c++", "boolean"),
    ("programming AND NOT java", "boolean"),
    ("document search evaluation", "tfidf"),
    ("apple", "bm25"),
    ("places to eat in semarang", "bm25"),
    ("text classification", "bm25"),
]

st.set_page_config(page_title="Retrieval Catalog", page_icon="📇", layout="wide")

# ---------------------------------------------------------------------------
# Catalog card styles (translucent colors so they read in light and dark themes)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.card{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:12px 16px;margin-bottom:12px;
  border-top:3px solid #cf3a3f;font-family:"Courier Prime","Courier New",monospace}
.card .top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}
.card .call{font-size:12.5px;opacity:.7}
.card .title{font-size:17px;font-weight:700;margin:2px 0 0}
.card .author{font-size:13.5px;opacity:.75}
.card .desc{font-size:14px;margin-top:6px;line-height:1.6}
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
# Session state
# ---------------------------------------------------------------------------
if "added" not in st.session_state:
    st.session_state.added = []          # books added by the user (this session only)
if "query" not in st.session_state:
    st.session_state.query = "search engine ranking"
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
# Sidebar: model settings and collection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model = st.radio("Retrieval model", options=list(MODELS), format_func=MODELS.get, key="model")
    k1 = st.slider("k₁ (tf saturation)", 0.0, 3.0, 1.5, 0.1, disabled=model != "bm25",
                   help="Lower values make tf saturate sooner. k₁ = 0 ignores tf entirely.")
    b = st.slider("b (length normalization)", 0.0, 1.0, 0.75, 0.05, disabled=model != "bm25",
                  help="b = 0 ignores document length. b = 1 applies full normalization.")
    remove_sw = st.checkbox("Remove stopwords", value=True)

    st.divider()
    st.subheader("Add a book to the collection")
    with st.form("add_book", clear_on_submit=True):
        f_title = st.text_input("Title", placeholder="Introduction to Data Mining")
        f_author = st.text_input("Author")
        f_desc = st.text_area("Short description", placeholder="One or two sentences about the book")
        if st.form_submit_button("Add & index", width="stretch"):
            if f_title.strip() and f_desc.strip():
                new_id = f"B{len(base_books()) + len(st.session_state.added) + 1:02d}"
                st.session_state.added.append(
                    dict(id=new_id, call_number="000 NEW", title=f_title.strip(),
                         author=f_author.strip() or "-", description=f_desc.strip())
                )
                st.success(f"{new_id} added and indexed.")
            else:
                st.error("Title and description are required.")
    if st.session_state.added and st.button("Restore original collection", width="stretch"):
        st.session_state.added = []
        st.rerun()
    st.caption("Added books are kept only for this browser session.")

# ---------------------------------------------------------------------------
# Build or reuse the engine
# ---------------------------------------------------------------------------
all_books = base_books() + st.session_state.added
books_key = tuple((d["id"], d["call_number"], d["title"], d["author"], d["description"]) for d in all_books)
engine = get_engine(books_key, remove_sw)

# ---------------------------------------------------------------------------
# Header and query input
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 1 · SEARCH LAB")
st.title("📇 Retrieval Catalog")
st.write(
    "A mini search engine for a sample book catalog. Type a query, pick a model in the sidebar, "
    "and see how the query is processed, looked up in the **inverted index**, and scored."
)

query = st.text_input("Query", key="query", placeholder="e.g. object-oriented programming")

st.write("Try:")
cols = st.columns(4)
for i, (q, m) in enumerate(EXAMPLES):
    cols[i % 4].button(q, key=f"ex{i}", on_click=use_example, args=(q, m), width="stretch")

st.latex(FORMULAS[model])

results = engine.search(query, model, k1, b)
terms = query_terms(query, remove_sw)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Books found", f"{len(results)} / {engine.N}")
m2.metric("Terms in index", len(engine.index))
m3.metric("avgdl (tokens)", f"{engine.avgdl:.2f}")
m4.metric("Model", MODELS[model])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_results, tab_process, tab_index, tab_compare = st.tabs(
    ["Search results", "1 · Query processing", "2 · Inverted index", "3 · Model comparison"]
)


def highlight(text: str, hit_terms: set[str]) -> str:
    """Wrap matching words in <mark> (after escaping the text)."""
    return re.sub(
        r"[A-Za-z0-9\+]+",
        lambda mo: f"<mark>{mo.group(0)}</mark>" if mo.group(0).lower() in hit_terms else mo.group(0),
        html.escape(text),
    )


with tab_results:
    if not query.strip():
        st.info("Type a query to start searching.")
    elif not results:
        unknown = [t for t in terms if not engine.df(t)]
        msg = "No books match. "
        if unknown:
            msg += "The term(s) " + ", ".join(f"“{t}”" for t in unknown) + " are not in the index. "
        msg += ("Keyword search only finds exact words, so a synonym such as “eat” is not "
                "recognized as “restaurants”. Try other words or another model.")
        st.warning(msg)
    else:
        hit = set(terms)
        for rank, r in enumerate(results, 1):
            book = engine.by_id[r.id]
            if r.score is None:
                stamp = '<div class="stamp">MATCH<small>boolean</small></div>'
                label = "·"
            else:
                kind = "bm25" if model == "bm25" else "cosine"
                stamp = f'<div class="stamp">{r.score:.3f}<small>{kind}</small></div>'
                label = f"#{rank}"
            why = "".join(f"<span>{html.escape(t)} +{c:.3f}</span>" for t, c in r.parts)
            st.markdown(
                f"""<div class="card">
  <div class="top">
    <div><div class="call">{label} &nbsp;·&nbsp; {html.escape(book.call_number)} &nbsp;·&nbsp; {book.id}</div>
      <div class="title">{highlight(book.title, hit)}</div>
      <div class="author">{html.escape(book.author)}</div></div>
    {stamp}
  </div>
  <div class="desc">{highlight(book.description, hit)}</div>
  {f'<div class="why">{why}</div>' if why else ''}
</div>""",
                unsafe_allow_html=True,
            )

with tab_process:
    toks = tokenize(query)

    def chips(items: list[str]) -> str:
        return "".join(items) or "–"

    st.markdown("**Original query**")
    st.code(query or " ", language=None)
    st.markdown("**Case folding + tokenization**")
    st.markdown(chips([f'<span class="tok">{html.escape(t)}</span>' for t in toks]), unsafe_allow_html=True)
    st.markdown("**Stopword removal**" + ("" if remove_sw else " _(turned off)_"))
    marked = []
    for t in toks:
        if remove_sw and t in STOPWORDS:
            marked.append(f'<span class="tok stop">{html.escape(t)}</span>')
        elif t in BOOLEAN_OPS:
            marked.append(f'<span class="tok op">{t.upper()}</span>')
        else:
            marked.append(f'<span class="tok term">{html.escape(t)}</span>')
    st.markdown(chips(marked), unsafe_allow_html=True)
    st.markdown("**Terms looked up in the index**")
    st.markdown(chips([f'<span class="tok term">{html.escape(t)}</span>' for t in terms]), unsafe_allow_html=True)
    with st.expander("Stopword list"):
        st.write(", ".join(sorted(STOPWORDS)))

with tab_index:
    st.markdown("Posting list for each query term: the books that contain the term, with its frequency (**tf**).")
    if terms:
        rows = [{
            "term": t,
            "df": engine.df(t),
            "idf (log10 N/df)": round(engine.idf_log(t), 4),
            "idf (BM25)": round(engine.idf_bm25(t), 4),
            "posting list (id:tf)": " → ".join(f"{d}:{f}" for d, f in engine.postings(t)) or "not in index",
        } for t in terms]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No terms to look up yet.")

    with st.expander(f"Browse the full inverted index ({len(engine.index)} terms)"):
        full = pd.DataFrame(
            [{"term": t, "df": len(p), "posting list": " → ".join(f"{d}:{f}" for d, f in p)}
             for t, p in engine.index.items()]
        ).sort_values(["df", "term"], ascending=[False, True])
        st.dataframe(full, hide_index=True, width="stretch", height=360)

    with st.expander("Browse the book collection"):
        st.dataframe(pd.DataFrame(all_books), hide_index=True, width="stretch")

with tab_compare:
    st.markdown("Rankings from all three models for the same query. ✓ means the same position as BM25.")
    res_b = engine.search(query, "boolean")
    res_t = engine.search(query, "tfidf")
    res_m = engine.search(query, "bm25", k1, b)
    n = max(len(res_b), len(res_t), len(res_m))

    def cell(res, i, mark_same=False):
        if i >= len(res):
            return "–"
        r = res[i]
        txt = f"{r.id} {engine.by_id[r.id].title}"
        if r.score is not None:
            txt += f" ({r.score:.2f})"
        if mark_same and i < len(res_m) and res_m[i].id == r.id:
            txt = "✓ " + txt
        return txt

    if n:
        table = pd.DataFrame({
            "Rank": range(1, n + 1),
            "Boolean (unranked)": [cell(res_b, i) for i in range(n)],
            "TF-IDF + Cosine": [cell(res_t, i, True) for i in range(n)],
            "BM25": [cell(res_m, i) for i in range(n)],
        })
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("None of the three models returned a result.")

st.divider()
st.caption(
    "The collection contains sample book descriptions written for practice, with Dewey-style (DDC) "
    "call numbers. Preprocessing: case folding → tokenization → stopword removal, no stemming."
)
