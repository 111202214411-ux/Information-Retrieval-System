"""
app.py: Streamlit interface for Search Engine Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html
from urllib.parse import urlparse

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import engine
import glove
import neural
import vectordb
import web

st.set_page_config(page_title="Search Engine Lab", page_icon="🔎", layout="wide")

ACCENT = "#0b7285"
AMBER = "#e8a028"
WEB = web.WEB
ALL_PAGES = web.PAGES

st.markdown(
    """
<style>
.pipe{display:flex;gap:4px;align-items:stretch;margin:4px 0 14px}
.stage{flex:1 1 0;min-width:0;border:1px solid rgba(128,140,160,.4);border-top:3px solid rgba(11,114,133,.85);
  border-radius:4px;padding:5px 7px;font-size:12px;line-height:1.35}
.stage b{display:block;font-size:13.5px}
.stage.on{border-top-color:#e8a028}
.arrow{align-self:center;opacity:.55;font-size:13px}
@media (max-width:900px){.pipe{flex-wrap:wrap}.stage{flex:1 1 30%}}
.card{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:8px 12px;font-size:14px;height:100%}
.card h5{margin:0 0 6px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;opacity:.75}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}
.chip{font-family:"Courier New",monospace;font-size:13px;border:1px solid rgba(128,140,160,.45);border-radius:4px;
  padding:1px 7px}
.chip.orig{border-color:rgba(11,114,133,.85);background:rgba(11,114,133,.14);font-weight:700}
.chip.exp{border-color:rgba(232,160,40,.85);background:rgba(232,160,40,.16)}
.chip.op{border-color:rgba(120,90,200,.7);background:rgba(120,90,200,.12)}
.serp{padding:8px 0 12px;border-bottom:1px solid rgba(128,140,160,.2);max-width:860px}
.serp .u{font-size:13px;opacity:.8}
.serp .t{font-size:18px;color:#1a5fb4;margin:1px 0 2px}
.serp .s{font-size:14px;line-height:1.5}
.serp .s b{background:rgba(232,160,40,.25);font-weight:600}
.serp .w{font-size:12.5px;opacity:.7;margin-top:3px;font-family:"Courier New",monospace}
.badge{font-size:11px;border:1px solid rgba(128,140,160,.6);border-radius:3px;padding:0 5px;margin-left:6px}
.callout{border-left:3px solid rgba(11,114,133,.7);padding:6px 12px;font-size:14px;margin:6px 0}
@media (prefers-color-scheme: dark){.serp .t{color:#8cb8f0}}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(str(s))


def heat(df: pd.DataFrame, cols: list[str], digits: int = 3, color: str = "11,114,133", extra: dict | None = None):
    """Shade numeric columns from transparent (column minimum) to the accent color (column maximum)."""
    def shade(col: pd.Series):
        lo, hi = col.min(), col.max()
        return [f"background-color: rgba({color},{0.08 + 0.42 * ((v - lo) / (hi - lo) if hi > lo else 0):.2f})"
                for v in col]
    sty = df.style.format({**{c: f"{{:.{digits}f}}" for c in cols}, **(extra or {})})
    return sty.apply(shade, subset=cols)


def short(url: str) -> str:
    u = urlparse(url)
    return u.netloc + (u.path if u.path != "/" else "")


# ---------------------------------------------------------------------------
# Cached heavy work
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_crawl(seeds: tuple[str, ...], max_pages: int, dedup: bool, threshold: float):
    log, docs, frontier = web.crawl(list(seeds), max_pages)
    kept, dups = web.find_duplicates(docs, threshold) if dedup else (docs, [])
    return log, docs, frontier, kept, dups


@st.cache_resource(show_spinner="Loading all-MiniLM-L6-v2 (first run downloads about 90 MB)…")
def get_encoders(prefer: bool):
    return neural.load_encoders(prefer)


@st.cache_resource(show_spinner="Loading the ms-marco cross-encoder (first run downloads about 90 MB)…")
def get_cross(prefer: bool):
    return neural.load_cross_encoder(prefer)


@st.cache_data(show_spinner=False)
def sentence_vectors(prefer: bool, texts: tuple[str, ...]) -> np.ndarray:
    return get_encoders(prefer)[0].encode(list(texts))


@st.cache_data(show_spinner=False)
def token_vectors(prefer: bool, text: str):
    return get_encoders(prefer)[0].tokens(text)


@st.cache_data(show_spinner=False)
def cross_scores(prefer: bool, query: str, texts: tuple[str, ...]) -> np.ndarray:
    ce = get_cross(prefer)[0]
    return ce.predict([(query, t) for t in texts])


@st.cache_resource(show_spinner="Building the vector indexes (k-means over 30,000 vectors)…")
def get_vector_indexes(nlist: int, m: int):
    X = glove.load().vectors
    return vectordb.FlatIndex(X), vectordb.IVFIndex(X, nlist), vectordb.PQIndex(X, m)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
SEED_CHOICES = [WEB["seeds"][0], WEB["seeds"][1], "https://wiki.searchpedia.org/",
                "https://blog.vectorlab.dev/posts/chromadb-semantic-search-in-five-lines",
                "https://docs.fusionsearch.io/guide/ranking-pipeline.html"]

with st.sidebar:
    st.header("1 · Crawler")
    seeds = st.multiselect("Seed URLs", SEED_CHOICES, default=WEB["seeds"], format_func=short, key="seeds")
    max_pages = st.slider("Maximum URLs to fetch", 3, 40, 40, key="max_pages",
                          help="The crawler stops when the frontier is empty or this budget is used up.")
    dedup = st.checkbox("Remove duplicate pages", value=True, key="dedup")
    threshold = st.slider("Near-duplicate threshold (Jaccard of word 3-shingles)", 0.5, 1.0, 0.8, 0.05,
                          key="threshold", disabled=not dedup)

    st.header("2 · Models")
    dense_choice = st.radio("Bi-encoder and token vectors", ["all-MiniLM-L6-v2", "Offline word vectors"],
                            key="dense_choice",
                            help="MiniLM downloads about 90 MB on first use. The offline option uses the bundled "
                                 "GloVe vectors, so the app also works without internet.")
    cross_choice = st.radio("Reranker", ["ms-marco cross-encoder", "MaxSim (late interaction)"], key="cross_choice",
                            help="The cross-encoder downloads about 90 MB. MaxSim reranks with the token vectors "
                                 "above and needs no extra download.")

    st.header("3 · Retrieval")
    k1 = st.slider("BM25 k₁", 0.0, 3.0, 1.2, 0.1, key="k1")
    b = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05, key="b")
    fusion = st.radio("Hybrid fusion", ["Reciprocal rank fusion", "Weighted sum"], key="fusion",
                      help="RRF combines ranks; the weighted sum combines min-max normalized BM25 with cosine.")
    alpha = st.slider("α, weight of BM25 in the weighted sum", 0.0, 1.0, 0.5, 0.05, key="alpha",
                      disabled=fusion != "Weighted sum")
    rrf_k = st.slider("RRF constant k", 1, 100, 60, key="rrf_k", disabled=fusion != "Reciprocal rank fusion")
    do_rerank = st.checkbox("Rerank the top candidates", value=True, key="rerank")
    depth = st.slider("Reranking depth", 3, 20, 10, key="depth", disabled=not do_rerank)

if not seeds:
    st.warning("Choose at least one seed URL in the sidebar so the crawler has somewhere to start.")
    st.stop()

prefer_dense = dense_choice == "all-MiniLM-L6-v2"
prefer_cross = cross_choice == "ms-marco cross-encoder"
log, fetched, frontier_left, kept, dups = run_crawl(tuple(seeds), max_pages, dedup, threshold)
if not kept:
    st.warning("The crawl fetched no documents. Raise the URL budget or choose other seeds.")
    st.stop()
idx = engine.Index(kept, k1, b)
URLS = [d.url for d in kept]
DOCS = {d.url: d for d in kept}
TEXT = {d.url: f"{d.title}. {d.text}" for d in kept}

enc, enc_why = get_encoders(prefer_dense)
cross, cross_why = get_cross(prefer_cross)
DOC_VECS = sentence_vectors(prefer_dense, tuple(TEXT[u] for u in URLS))
reranker_name = cross.name if cross else "MaxSim over " + ("MiniLM token outputs" if enc.real else "GloVe word vectors")

if enc_why:
    st.warning(f"all-MiniLM-L6-v2 is unavailable because {enc_why}. Using averaged GloVe word vectors instead. "
               f"They are much weaker and know no Indonesian, but need no download.")
if cross_why:
    st.warning(f"The cross-encoder is unavailable because {cross_why}. Reranking uses MaxSim instead.")


# ---------------------------------------------------------------------------
# Retrieval used by the Search and Evaluation tabs
# ---------------------------------------------------------------------------
def dense_scores(query: str, cands: list[str]) -> dict[str, float]:
    qv = sentence_vectors(prefer_dense, (query,))[0]
    pos = {u: i for i, u in enumerate(URLS)}
    return {u: float(DOC_VECS[pos[u]] @ qv) for u in cands}


def maxsim_score(query: str, url: str) -> float:
    _, Q = token_vectors(prefer_dense, query)
    _, D = token_vectors(prefer_dense, TEXT[url])
    return neural.maxsim(Q, D)[0]


def rerank_scores(query: str, urls: list[str]) -> dict[str, float]:
    if not urls:
        return {}
    if cross:
        return dict(zip(urls, map(float, cross_scores(prefer_cross, query, tuple(TEXT[u] for u in urls)))))
    return {u: maxsim_score(query, u) for u in urls}


def retrieve(weights: dict[str, float], dense_query: str, cands: list[str], method: str,
             rerank: bool, a: float = 0.5, k: int = 60, rdepth: int = 10) -> dict:
    """method: BM25, Dense, Weighted, RRF. Returns the stages so the UI can show them."""
    bm = idx.bm25(weights, cands)
    de = dense_scores(dense_query, cands) if dense_query.strip() else dict.fromkeys(cands, 0.0)
    bm_rank = [u for u, _ in engine.ranking(bm, keep_zero=False)]
    de_rank = [u for u, _ in engine.ranking(de)]
    out = {"bm25": bm, "dense": de, "bm_rank": bm_rank, "de_rank": de_rank, "rrf_ranks": {}, "bm_norm": {}}
    if method == "BM25":
        fused = [(u, bm[u]) for u in bm_rank]
    elif method == "Dense":
        fused = [(u, de[u]) for u in de_rank]
    elif method == "Weighted":
        fused, out["bm_norm"], _ = engine.weighted_hybrid(bm, de, a)
    else:
        fused, out["rrf_ranks"] = engine.rrf({"BM25": bm_rank, "Dense": de_rank}, k)
    out["fused"] = fused
    out["rerank"] = {}
    final = fused
    if rerank and fused:
        top = [u for u, _ in fused[:rdepth]]
        rs = rerank_scores(dense_query, top)
        out["rerank"] = rs
        final = sorted(((u, rs[u]) for u in top), key=lambda x: -x[1]) + fused[rdepth:]
    out["final"] = final
    return out


def method_label(method: str) -> str:
    return {"Weighted": f"weighted sum (α = {alpha:.2f})", "RRF": f"RRF (k = {rrf_k})"}.get(method, method)


FUSION_METHOD = "RRF" if fusion == "Reciprocal rank fusion" else "Weighted"

# ---------------------------------------------------------------------------
# Header and pipeline strip
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 6 · SEARCH ENGINE ARCHITECTURE & HYBRID SEARCH")
st.title("🔎 Search Engine Lab")
st.write("A complete, miniature search engine over a simulated web of 37 pages: it **crawls** from seed URLs, "
         "**processes** and deduplicates the pages, builds **inverted, dense and metadata indexes**, understands "
         "queries with **operators, spelling correction, expansion and intent**, retrieves with **BM25 + dense**, "
         "merges them by **hybrid fusion**, and **reranks**. Separate tabs open up **late interaction (ColBERT)** "
         "and the **index types of a vector database**.")

ok = sum(s.status == 200 for s in log)
stages = [
    ("Crawl", f"{len(log)} URLs fetched<br>{len(log) - ok} broken"),
    ("Process", f"{len(fetched)} documents<br>{len(dups)} duplicates removed"),
    ("Index", f"{len(idx.postings)} terms · {len(kept)} docs<br>{len(kept)} × {enc.dims}-d vectors"),
    ("Query", "operators · spelling<br>expansion · intent"),
    ("Retrieve", "BM25 + dense<br>bi-encoder"),
    ("Fuse", method_label(FUSION_METHOD).replace(" (", "<br>(")),
    ("Rerank", (f"top {depth} with<br>{'cross-encoder' if cross else 'MaxSim'}") if do_rerank else "off"),
    ("Results", "ranked list<br>with snippets"),
]
show('<div class="pipe">' + '<span class="arrow">→</span>'.join(
    f'<div class="stage{" on" if name in ("Fuse", "Rerank") else ""}"><b>{name}</b>{body}</div>'
    for name, body in stages) + "</div>")

tab_crawl, tab_search, tab_fuse, tab_late, tab_vdb, tab_eval = st.tabs(
    ["1 · Crawl & index", "2 · Search", "3 · Hybrid fusion", "4 · Late interaction", "5 · Vector database",
     "6 · Evaluation"])

# ===========================================================================
# Tab 1: crawl, process, index
# ===========================================================================
with tab_crawl:
    st.subheader("Crawler: breadth-first traversal from the seeds")
    st.write("The crawler takes the next URL from the front of the **frontier** (a queue), fetches it, extracts its "
             "links and appends every link it has not seen before. Change the seeds and the budget in the sidebar.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("URLs fetched", len(log))
    c2.metric("Documents (HTTP 200)", ok)
    c3.metric("Still in the frontier", len(frontier_left))
    undiscovered = sorted(set(ALL_PAGES) - {s.url for s in log} - set(frontier_left))
    c4.metric("Never discovered", len(undiscovered))
    crawl_df = pd.DataFrame([{"#": s.order, "URL": short(s.url), "Status": s.status,
                              "Type": {"text/html": "HTML", "application/pdf": "PDF",
                                       "application/vnd.ms-powerpoint": "PPT"}.get(s.content_type, "—"),
                              "Links found": s.links_found, "New links queued": s.new_links,
                              "Frontier after": s.frontier_after} for s in log])
    st.dataframe(crawl_df, hide_index=True, width="stretch", height=323)
    notes = []
    if any(s.status == 404 for s in log):
        notes.append("A **404** is a broken link: the course page links to a syllabus that no longer exists.")
    if frontier_left:
        notes.append(f"The budget ran out with {len(frontier_left)} URLs still waiting in the frontier.")
    if undiscovered:
        notes.append("Never discovered: " + ", ".join(f"`{short(u)}`" for u in undiscovered) +
                     ". No crawled page links to it, so a crawler that only follows links cannot find it; real "
                     "engines also read sitemaps and accept URL submissions.")
    for n in notes:
        st.markdown("- " + n)

    st.subheader("Document processor: parsing, language identification, duplicates")
    pick = st.selectbox("Inspect a fetched document", [d.url for d in fetched], format_func=short, key="inspect")
    resp = web.fetch(pick)
    doc = next(d for d in fetched if d.url == pick)
    a, bcol = st.columns(2)
    with a:
        st.markdown(f"**Raw response** · `{resp.content_type}`")
        st.code(resp.body, language="html" if resp.content_type == "text/html" else None, height=360,
                wrap_lines=True)
    with bcol:
        st.markdown("**Extracted document**")
        en_c, id_c = doc.lang_counts
        show(f'<div class="card"><h5>Title</h5>{esc(doc.title)}<h5 style="margin-top:10px">Main text</h5>'
             f'{esc(doc.text)}<h5 style="margin-top:10px">Out-links ({len(doc.links)})</h5>'
             f'{"<br>".join(esc(short(l)) for l in doc.links) or "none"}'
             f'<h5 style="margin-top:10px">Language</h5><b>{"Indonesian" if doc.lang == "id" else "English"}</b> · '
             f'{en_c} English and {id_c} Indonesian function words</div>')
        if resp.content_type == "text/html":
            st.caption("The parser keeps `<title>` and `<main>`, skips `<nav>`, `<aside>`, `<footer>` and "
                       "`<script>` (boilerplate), and collects every `<a href>` as an out-link.")
        else:
            st.caption("PDF and PPT files have no HTML. A real crawler runs a text extractor; here the extracted "
                       "text is served directly, and such files have no out-links.")

    if dedup:
        if dups:
            st.markdown(f"**Duplicates removed** (threshold {threshold:.2f}). The copy with the most in-links is kept.")
            st.dataframe(pd.DataFrame([{"Removed": short(d.url), "Kept (canonical)": short(d.kept), "Kind": d.kind,
                                        "Jaccard": round(d.similarity, 3)} for d in dups]),
                         hide_index=True, width="stretch")
        else:
            st.info("No duplicates among the fetched documents at this threshold.")
    else:
        st.info("Duplicate removal is off, so mirrors and copies are indexed and can appear twice in the results.")

    st.subheader("Indexer: three indexes, built offline")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("**Inverted index** (term → postings)")
        term = st.text_input("Look up a term", "search", key="term").strip().lower()
        t = engine.tokenize(term, remove_stopwords=False)
        t = t[0] if t else ""
        post = idx.postings.get(t, {})
        if post:
            st.caption(f"df = {len(post)} of N = {idx.N}, IDF = {idx.idf(t):.3f}")
            st.dataframe(pd.DataFrame([{"Document": short(u), "tf": f} for u, f in
                                       sorted(post.items(), key=lambda x: -x[1])]), hide_index=True,
                         width="stretch", height=230)
        else:
            st.caption(f"'{term}' is not in the index (stopwords are not indexed).")
    with i2:
        st.markdown("**Dense index** (document → vector)")
        st.caption(f"{len(kept)} vectors × {enc.dims} dimensions from {enc.name}, unit length, so the inner "
                   f"product is the cosine similarity.")
        st.dataframe(pd.DataFrame(DOC_VECS[:, :6], index=[short(u)[:34] for u in URLS],
                                  columns=[f"dim {i}" for i in range(6)]).round(3), width="stretch", height=230)
    with i3:
        st.markdown("**Metadata index** (used by the operators)")
        st.dataframe(pd.DataFrame([{"Host": d.host, "Type": d.type, "Lang": d.lang, "Title": d.title}
                                   for d in kept]), hide_index=True, width="stretch", height=268)

# ===========================================================================
# Tab 2: search
# ===========================================================================
EXAMPLES = ["combine keyword and semantc search", "how to merge rankings from several retrievers",
            "hybird serach", "intitle:hybrid search", "allintitle:hybrid search", "bm25 site:ac.id",
            "filetype:pdf bm25", "inurl:posts vector", "allinurl:guide ranking", "download the fusionsearch manual",
            "cara kerja mesin pencari"]


def use_example():
    st.session_state.q = st.session_state.example


with tab_search:
    st.session_state.setdefault("q", EXAMPLES[0])
    l, r = st.columns([3, 2])
    with r:
        st.selectbox("Example queries", EXAMPLES, key="example", on_change=use_example)
    with l:
        query = st.text_input("Search the mini-web", key="q")
    o1, o2, _ = st.columns([2, 2.4, 1.6])
    auto_fix = o1.checkbox("Correct spelling automatically", value=True, key="auto_fix")
    expand = o2.checkbox("Expand the query with related terms", value=False, key="expand")

    pq = engine.parse_query(query)
    words = [w for w in engine.tokenize(" ".join(pq.text))]
    known = set(glove.load().words)
    fixes = engine.correct_spelling(words, idx.vocabulary, known)
    words_fixed = [fixes[w][0] if (auto_fix and w in fixes) else w for w in words]
    expansions = engine.expand_query(words_fixed, glove.load(), idx.vocabulary) if expand else {}
    weights = {w: 1.0 for w in words_fixed}
    for w, lst in expansions.items():
        for n, _ in lst:
            weights.setdefault(n, 0.5)
    intent, cues = engine.detect_intent(pq)
    cands = idx.apply_filters(pq)
    dense_q = " ".join(words_fixed)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        body = "".join(f'<span class="chip op">{esc(op)}:{esc(v)}</span>' for op, v in pq.filters) or "none"
        warn = "".join(f"<br><small>⚠️ {esc(w)}</small>" for w in pq.warnings)
        show(f'<div class="card"><h5>Operators</h5><div class="chips">{body}</div>'
             f'{len(cands)} of {idx.N} documents pass the filters{warn}</div>')
    with q2:
        if fixes:
            corrected = " ".join(fixes[w][0] if w in fixes else w for w in words)
            detail = ", ".join(f"{esc(w)} → <b>{esc(s)}</b> (distance {d})" for w, (s, d) in fixes.items())
            status = "applied" if auto_fix else "not applied"
            show(f'<div class="card"><h5>Spelling</h5>Did you mean <i><b>{esc(corrected)}</b></i>?<br>'
                 f'<small>{detail} · {status}</small></div>')
        else:
            show('<div class="card"><h5>Spelling</h5>No corrections: every word is in the index or is a known '
                 'English word.</div>')
    with q3:
        chips = "".join(f'<span class="chip orig">{esc(w)}</span>' for w in words_fixed)
        chips += "".join(f'<span class="chip exp" title="cos {s:.2f} to {esc(w)}">{esc(n)}</span>'
                         for w, lst in expansions.items() for n, s in lst)
        note = ("Orange terms come from GloVe neighbours that occur in the index; weight 0.5 in BM25." if expand
                else "Turn on expansion to add related index terms.")
        show(f'<div class="card"><h5>Ranked terms</h5><div class="chips">{chips or "—"}</div><small>{note}</small></div>')
    with q4:
        show(f'<div class="card"><h5>Intent</h5><b>{intent}</b><br><small>cues: {esc(", ".join(cues))}</small><br>'
             f'<small>A production engine routes on this, for example showing a homepage first for a '
             f'navigational query.</small></div>')

    if not cands:
        st.info("No document passes the operators.")
    elif not words_fixed:
        st.info("The query has only filter operators and no words to rank by, so the matching documents are listed "
                "in crawl order.")
        for u in cands:
            d = DOCS[u]
            show(f'<div class="serp"><div class="u">{esc(short(u))}<span class="badge">{d.type.upper()}</span></div>'
                 f'<div class="t">{esc(d.title)}</div><div class="s">{esc(d.text[:200])}…</div></div>')
    else:
        res = retrieve(weights, dense_q, cands, FUSION_METHOD, do_rerank, alpha, rrf_k, depth)
        n_bm = len(res["bm_rank"])
        steps = [f"{len(cands)} candidates", f"BM25 matched {n_bm}", f"dense scored {len(res['de_rank'])}",
                 f"{method_label(FUSION_METHOD)} → {len(res['fused'])}"]
        if do_rerank:
            steps.append(f"{reranker_name} reranked the top {min(depth, len(res['fused']))}")
        st.caption(" → ".join(steps))
        bm_pos = {u: i for i, u in enumerate(res["bm_rank"], 1)}
        de_pos = {u: i for i, u in enumerate(res["de_rank"], 1)}
        fu = dict(res["fused"])
        hl = set(weights) | set(words_fixed)
        for rank, (u, _) in enumerate(res["final"][:10], 1):
            d = DOCS[u]
            pairs, cut_l, cut_r = engine.snippet(d.text, hl)
            snip = ("… " if cut_l else "") + " ".join(f"<b>{esc(w)}</b>" if hit else esc(w) for w, hit in pairs) + \
                (" …" if cut_r else "")
            why = [f"#{rank}", f"BM25 {'rank ' + str(bm_pos[u]) if u in bm_pos else 'no match'} "
                               f"({res['bm25'][u]:.2f})",
                   f"dense rank {de_pos[u]} ({res['dense'][u]:.3f})"]
            if FUSION_METHOD == "RRF":
                why.append(f"RRF {fu[u]:.4f}")
            else:
                why.append(f"hybrid {fu[u]:.3f}")
            if u in res["rerank"]:
                why.append(f"rerank {res['rerank'][u]:.3f}")
            path = " › ".join([d.host] + [p for p in urlparse(u).path.split("/") if p])
            lang = '<span class="badge">ID</span>' if d.lang == "id" else ""
            show(f'<div class="serp"><div class="u">{esc(path)}<span class="badge">{d.type.upper()}</span>{lang}</div>'
                 f'<div class="t">{esc(d.title)}</div><div class="s">{snip}</div>'
                 f'<div class="w">{" · ".join(esc(x) for x in why)}</div></div>')

# ===========================================================================
# Tab 3: hybrid fusion
# ===========================================================================
with tab_fuse:
    st.markdown(r"""Two ways to merge a lexical and a semantic ranking:

* **Weighted sum** (lecture notebook): $S_{hybrid} = \alpha \cdot S_{BM25} + (1-\alpha)\cdot S_{dense}$, with BM25
  min-max normalized as $\frac{s-\min}{\max-\min+10^{-8}}$ and $S_{dense}$ the raw cosine.
* **Reciprocal rank fusion**: $RRF(d) = \sum_r \frac{1}{k + rank_r(d)}$ with $k = 60$. Scores are never compared,
  only positions, so no normalization is needed.""")
    nb = WEB["notebook"]
    src = st.radio("Collection", ["Lecture notebook (5 documents)", "Mini-web index"], horizontal=True, key="fsrc")
    is_nb = src.startswith("Lecture")
    f1, f2, f3 = st.columns([3, 1, 1])
    fq = f1.text_input("Query", nb["query"] if is_nb else "how to merge rankings from several retrievers",
                       key=f"fq_{is_nb}")
    fa = f2.slider("α (BM25 weight)", 0.0, 1.0, 0.5, 0.05, key="fa")
    fk = f3.slider("RRF k", 1, 100, 60, key="fk")

    if is_nb:
        names = [f"doc {i}" for i in range(len(nb["docs"]))]
        texts = dict(zip(names, nb["docs"]))
        raw, _ = engine.okapi_scores([t.lower().split() for t in nb["docs"]], fq.lower().split())
        bm = dict(zip(names, raw))
        E = sentence_vectors(prefer_dense, tuple(nb["docs"]))
        qv = sentence_vectors(prefer_dense, (fq,))[0]
        de = dict(zip(names, map(float, E @ qv)))
        bm_label = "BM25Okapi (rank_bm25 formula, k₁ = 1.5, b = 0.75)"
        if not enc.real:
            st.warning("The offline GloVe vectors contain almost no Indonesian words, so most dense scores are 0 "
                       "here. With all-MiniLM-L6-v2 (as in the notebook) the dense column is meaningful.")
    else:
        names = URLS
        texts = {u: DOCS[u].title for u in URLS}
        bm = idx.bm25({t: 1.0 for t in engine.tokenize(fq)})
        de = dense_scores(fq, URLS) if fq.strip() else dict.fromkeys(URLS, 0.0)
        bm_label = f"BM25 (k₁ = {k1}, b = {b})"

    fused, bm_n, _ = engine.weighted_hybrid(bm, de, fa)
    bm_rank = [u for u, _ in engine.ranking(bm, keep_zero=False)]
    de_rank = [u for u, _ in engine.ranking(de)]
    rrf_rank, rrf_ranks = engine.rrf({"BM25": bm_rank, "Dense": de_rank}, fk)
    rrf_s = dict(rrf_rank)
    w_pos = {u: i for i, (u, _) in enumerate(fused, 1)}
    r_pos = {u: i for i, (u, _) in enumerate(rrf_rank, 1)}

    if is_nb:
        st.markdown("**The notebook's printout** (`hybrid_search(query, alpha)`, top 3)")
        st.dataframe(pd.DataFrame([{"Rank": i, "Hybrid": h, "BM25": bm_n[u], "Dense": de[u], "Document": texts[u]}
                                   for i, (u, h) in enumerate(fused[:3], 1)]).style.format(
            {"Hybrid": "{:.4f}", "BM25": "{:.4f}", "Dense": "{:.4f}"}), hide_index=True, width="stretch")

    st.markdown(f"**Every document, both fusions** · {bm_label}; dense = {enc.name}")
    rows = []
    for u in sorted(names, key=lambda x: w_pos[x]):
        rk = rrf_ranks.get(u, {})
        rows.append({"Document": texts[u][:60], "BM25 raw": bm[u], "BM25 min-max": bm_n[u], "Dense cos": de[u],
                     "Weighted": dict(fused)[u], "Weighted rank": w_pos[u],
                     "BM25 rank": str(rk["BM25"]) if "BM25" in rk else "—",
                     "Dense rank": str(rk["Dense"]) if "Dense" in rk else "—",
                     "RRF": rrf_s.get(u, 0.0), "RRF rank": str(r_pos[u]) if u in r_pos else "—"})
    tbl = pd.DataFrame(rows)
    st.dataframe(heat(tbl, ["BM25 raw", "BM25 min-max", "Dense cos", "Weighted", "RRF"], 4),
                 hide_index=True, width="stretch", height=min(420, 38 + 35 * len(rows)))
    st.caption("A — in the BM25 rank column means the document contains no query word, so BM25 did not retrieve it and it "
               "gets no RRF contribution from that list. In the weighted sum it still gets min-max BM25 = 0.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**How α moves the weighted ranking**")
        top = [u for u, _ in fused[:6]]
        curve = []
        for al in np.round(np.linspace(0, 1, 21), 2):
            for u in top:
                curve.append({"α": al, "Document": texts[u][:40], "Hybrid score": al * bm_n[u] + (1 - al) * de[u]})
        base = alt.Chart(pd.DataFrame(curve)).mark_line().encode(
            x=alt.X("α:Q", title="α  (0 = dense only, 1 = BM25 only)"), y="Hybrid score:Q",
            color=alt.Color("Document:N", legend=alt.Legend(orient="bottom", columns=1, labelLimit=400, title=None)),
            tooltip=["Document", "α", alt.Tooltip("Hybrid score:Q", format=".3f")])
        rule = alt.Chart(pd.DataFrame({"α": [fa]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="α:Q")
        st.altair_chart((base + rule).properties(height=300), width="stretch")
        st.caption("Where two lines cross, the two documents swap places. Each line runs from the dense score "
                   "(α = 0) to the min-max BM25 score (α = 1).")
    with c2:
        st.markdown("**What one list position is worth in RRF**")
        rr = [{"Rank": r, "k": f"k = {kk}", "1 / (k + rank)": 1 / (kk + r)} for kk in (1, 10, fk) for r in range(1, 21)]
        st.altair_chart(alt.Chart(pd.DataFrame(rr).drop_duplicates()).mark_line(point=True).encode(
            x="Rank:Q", y=alt.Y("1 / (k + rank):Q", scale=alt.Scale(type="log")),
            color=alt.Color("k:N", scale=alt.Scale(scheme="teals"), legend=alt.Legend(orient="bottom")),
            tooltip=["k", "Rank", alt.Tooltip("1 / (k + rank):Q", format=".4f")]).properties(height=300),
            width="stretch")
        st.caption(f"With k = {fk}, rank 1 is worth {1 / (fk + 1):.4f} and rank 10 is worth {1 / (fk + 10):.4f}. "
                   f"A large k flattens the curve, so a document that both retrievers place reasonably high beats "
                   f"one that only a single retriever puts first.")

    st.markdown("**Top 5 side by side**")
    s1, s2, s3, s4 = st.columns(4)
    for col_, title, lst in [(s1, "BM25", bm_rank), (s2, "Dense", de_rank),
                             (s3, f"Weighted (α = {fa:.2f})", [u for u, _ in fused]),
                             (s4, f"RRF (k = {fk})", [u for u, _ in rrf_rank])]:
        with col_:
            st.markdown(f"*{title}*")
            for i, u in enumerate(lst[:5], 1):
                st.markdown(f"{i}. {esc(texts[u][:48])}")
            if not lst:
                st.caption("no document matches")

    with st.expander("The notebook code, and the same computation in this app"):
        st.code('''bm25_scores = bm25.get_scores(query.split())
bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
q_emb = model.encode([query], normalize_embeddings=True)
dense_scores = (doc_embs @ q_emb.T).flatten()
hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_scores

# engine.py
fused, bm25_norm, dense = engine.weighted_hybrid(bm25, dense, alpha)          # same formula
fused, ranks = engine.rrf({"BM25": bm25_ranking, "Dense": dense_ranking}, k=60)''', language="python")
        st.caption("engine.okapi_scores reproduces rank_bm25.BM25Okapi exactly, including its rule that replaces a "
                   "negative IDF (a word in more than half of the documents) with 0.25 × the average IDF.")

# ===========================================================================
# Tab 4: late interaction
# ===========================================================================
with tab_late:
    st.markdown(r"""Three ways a neural model can score a query against a document:

| Architecture | Score | Stores per document | Speed |
|---|---|---|---|
| Bi-encoder | $\cos(E(q), E(d))$ | 1 vector | fastest; documents encoded offline |
| **Late interaction (ColBERT)** | $\sum_i \max_j \cos(q_i, d_j)$ | 1 vector **per token** | fast; documents encoded offline |
| Cross-encoder | $\text{BERT}([q; \text{SEP}; d])$ | nothing | slow; runs once per query–document pair |""")
    lq1, lq2 = st.columns([2, 3])
    lq = lq1.text_input("Query", "late interaction token level matching", key="lq")
    default_doc = next((u for u in URLS if u.endswith("/ColBERT")), URLS[0])
    ld = lq2.selectbox("Document", URLS, index=URLS.index(default_doc), format_func=lambda u: DOCS[u].title,
                       key="ld")
    qt, Q = token_vectors(prefer_dense, lq)
    dt, D = token_vectors(prefer_dense, TEXT[ld])
    score, S, best = neural.maxsim(Q, D)
    bi = float(sentence_vectors(prefer_dense, (lq,))[0] @ DOC_VECS[URLS.index(ld)])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bi-encoder cosine", f"{bi:.3f}")
    m2.metric("MaxSim score Σᵢ maxⱼ", f"{score:.3f}")
    m3.metric("MaxSim ÷ query tokens", f"{score / max(len(qt), 1):.3f}")
    if cross:
        m4.metric("Cross-encoder score", f"{float(cross_scores(prefer_cross, lq, (TEXT[ld],))[0]):.3f}")
    else:
        m4.metric("Cross-encoder score", "not loaded")
    st.caption(f"Token vectors: {enc.token_note}. Stopwords and punctuation are dropped to keep the matrix readable.")

    if len(qt) and len(dt):
        cells = [{"Query token": f"{i + 1}. {qt[i]}", "Doc token": f"{j + 1:02d} {dt[j]}", "cos": float(S[i, j]),
                  "max": "●" if best[i] == j else ""} for i in range(len(qt)) for j in range(len(dt))]
        cdf = pd.DataFrame(cells)
        base = alt.Chart(cdf).encode(
            x=alt.X("Doc token:N", sort=None, title="document tokens", axis=alt.Axis(labelAngle=-60, labelLimit=120)),
            y=alt.Y("Query token:N", sort=None, title=None))
        hm = base.mark_rect().encode(color=alt.Color("cos:Q", scale=alt.Scale(scheme="teals"), title="cos"),
                                     tooltip=["Query token", "Doc token", alt.Tooltip("cos:Q", format=".3f")])
        dots = base.mark_text(fontSize=13, color="#e8a028").encode(text="max:N")
        st.altair_chart((hm + dots).properties(height=max(150, 34 * len(qt) + 120)), width="stretch")
        st.dataframe(pd.DataFrame([{"Query token": qt[i], "Best document token": dt[best[i]],
                                    "max cos": float(S[i, best[i]])} for i in range(len(qt))]
                                  ).style.format({"max cos": "{:.3f}"}), hide_index=True, width="stretch")
        st.caption(f"Each orange dot is the MaxSim of a row. The score is the sum of the {len(qt)} row maxima "
                   f"= {score:.3f}. A bi-encoder would first squash all {len(dt)} document tokens into one vector.")
    else:
        st.info("No query or document tokens left after removing stopwords (the offline vectors know no Indonesian).")

    st.markdown("**Rank the whole index three ways**")
    if lq.strip():
        bi_all = dense_scores(lq, URLS)
        ms_all = {u: maxsim_score(lq, u) for u in URLS}
        cols = [("Bi-encoder", bi_all), ("MaxSim", ms_all)]
        if cross:
            cols.append(("Cross-encoder", dict(zip(URLS, map(float, cross_scores(prefer_cross, lq,
                                                                                  tuple(TEXT[u] for u in URLS)))))))
        rc = st.columns(len(cols))
        for c, (name, sc) in zip(rc, cols):
            with c:
                st.markdown(f"*{name}*")
                st.dataframe(pd.DataFrame([{"Document": DOCS[u].title, "Score": s} for u, s in
                                           engine.ranking(sc)[:5]]).style.format({"Score": "{:.3f}"}),
                             hide_index=True, width="stretch")
        if not cross:
            st.caption("Choose the cross-encoder in the sidebar to add the third column (downloads about 90 MB).")

    toks = [len(token_vectors(prefer_dense, TEXT[u])[0]) for u in URLS]
    avg_t = float(np.mean(toks))
    st.markdown("**The storage trade-off**")
    st.dataframe(pd.DataFrame([
        {"Representation": f"Bi-encoder ({enc.dims}-d float32)", "Vectors per document": 1,
         "Bytes per document": enc.dims * 4},
        {"Representation": f"Late interaction, this lab ({enc.dims}-d float32)",
         "Vectors per document": round(avg_t, 1), "Bytes per document": round(avg_t * enc.dims * 4)},
        {"Representation": "ColBERTv2 (128-d, 2-bit residual compression ≈ 36 bytes per token)",
         "Vectors per document": round(avg_t, 1), "Bytes per document": round(avg_t * 36)},
    ]), hide_index=True, width="stretch")
    st.caption("ColBERT approaches cross-encoder quality at close to bi-encoder speed, but stores a vector for every "
               "token. ColBERTv2 (Santhanam et al., NAACL 2022) cuts that cost with residual compression. This lab's "
               "token vectors come from MiniLM (or GloVe), not from a trained ColBERT checkpoint, so they show the "
               "mechanics, not ColBERTv2's accuracy.")

# ===========================================================================
# Tab 5: vector database
# ===========================================================================
with tab_vdb:
    st.write("A vector database stores embeddings and finds the nearest ones to a query vector. Here the database "
             "is the **30,000 bundled GloVe word vectors** (50 dimensions), and three index types answer the same "
             "query. ChromaDB, FAISS and Weaviate build on the same ideas (HNSW graphs, IVF clusters, quantization).")
    v1, v2, v3, v4 = st.columns([2, 1, 1, 1])
    vq = v1.text_input("Query word or phrase (English)", "search engine", key="vq")
    nlist = v2.select_slider("IVF clusters (nlist)", [25, 50, 100, 200, 400], 100, key="nlist")
    nprobe = v3.select_slider("Clusters probed (nprobe)", [1, 2, 4, 8, 16, 32, 64], 4, key="nprobe")
    m = v4.select_slider("PQ sub-vectors (m)", [5, 10, 25, 50], 10, key="pqm")
    g = glove.load()
    qv = g.text_vector(vq)
    if not np.any(qv):
        st.info("None of these words is in the 30,000-word vocabulary. Try an English word such as *search*.")
    else:
        flat, ivf, pq_ = get_vector_indexes(nlist, m)
        def timed(fn):          # best of 5 runs, so the first call's warm-up does not distort the timing
            runs = [fn() for _ in range(5)]
            return min(runs, key=lambda r: r.ms)
        ex = timed(lambda: flat.search(qv, 10))
        r_ivf = timed(lambda: ivf.search(qv, 10, min(nprobe, nlist)))
        r_pq = timed(lambda: pq_.search(qv, 10))
        exact = set(ex.ids.tolist())
        cols = st.columns(3)
        for c, (idx_, r) in zip(cols, [(flat, ex), (ivf, r_ivf), (pq_, r_pq)]):
            with c:
                st.markdown(f"**{idx_.kind}**")
                st.dataframe(pd.DataFrame([{"Word": g.words[i], "Score": float(s), "In exact top 10": "✓" if i in exact else "✗"}
                                           for i, s in zip(r.ids, r.scores)]).style.format({"Score": "{:.3f}"}),
                             hide_index=True, width="stretch", height=388)
        N = len(g.words)
        st.dataframe(pd.DataFrame([
            {"Index": f"Flat", "Vectors scored": ex.compared, "Share of database": 1.0,
             "Bytes per vector": flat.bytes_per_vector, "Memory": f"{N * flat.bytes_per_vector / 1e6:.1f} MB",
             "Recall@10": 1.0, "Time (ms)": ex.ms},
            {"Index": f"IVF (nlist {nlist}, nprobe {min(nprobe, nlist)})", "Vectors scored": r_ivf.compared,
             "Share of database": r_ivf.compared / N, "Bytes per vector": ivf.bytes_per_vector,
             "Memory": f"{N * ivf.bytes_per_vector / 1e6:.1f} MB", "Recall@10": vectordb.recall_at_k(r_ivf.ids, ex.ids),
             "Time (ms)": r_ivf.ms},
            {"Index": f"PQ (m {m}, 256 centroids each)", "Vectors scored": r_pq.compared, "Share of database": 1.0,
             "Bytes per vector": pq_.bytes_per_vector, "Memory": f"{N * pq_.bytes_per_vector / 1e6:.2f} MB",
             "Recall@10": vectordb.recall_at_k(r_pq.ids, ex.ids), "Time (ms)": r_pq.ms},
        ]).style.format({"Share of database": "{:.1%}", "Recall@10": "{:.1f}", "Time (ms)": "{:.2f}"}),
            hide_index=True, width="stretch")
        st.caption("Recall@10 = how many of the exact top 10 the index found. IVF saves work by scanning only the "
                   "probed clusters; PQ saves memory by scoring 1-byte codes against a lookup table of "
                   f"{m} × 256 sub-vector similarities. Times (best of 5 runs) are for NumPy on this small database and "
                   "only roughly indicative; at millions of vectors the savings dominate.")

        @st.cache_data(show_spinner="Measuring recall over 200 sample queries…")
        def tradeoff(nlist_: int, m_: int) -> pd.DataFrame:
            fl, iv, pq2 = get_vector_indexes(nlist_, m_)
            rng = np.random.default_rng(0)
            sample = rng.choice(5000, 200, replace=False)
            truth = [fl.search(g.vectors[i], 10).ids for i in sample]
            rows = []
            for npb in [p for p in [1, 2, 4, 8, 16, 32, 64, 128] if p <= nlist_]:
                rec, share = [], []
                for i, t in zip(sample, truth):
                    r = iv.search(g.vectors[i], 10, npb)
                    rec.append(vectordb.recall_at_k(r.ids, t))
                    share.append(r.compared / len(g.words))
                rows.append({"nprobe": npb, "Recall@10": np.mean(rec), "Share of database scanned": np.mean(share)})
            pq_rec = np.mean([vectordb.recall_at_k(pq2.search(g.vectors[i], 10).ids, t) for i, t in zip(sample, truth)])
            return pd.DataFrame(rows), pq_rec

        curve, pq_rec = tradeoff(nlist, m)
        t1, t2 = st.columns([3, 2])
        with t1:
            long = curve.melt("nprobe", var_name="Measure", value_name="Value")
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("nprobe:Q", scale=alt.Scale(type="log", base=2), title="clusters probed (nprobe, log scale)"),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                color=alt.Color("Measure:N", scale=alt.Scale(range=[ACCENT, AMBER]), legend=alt.Legend(orient="bottom")),
                tooltip=["nprobe", "Measure", alt.Tooltip("Value:Q", format=".1%")]).properties(height=280),
                width="stretch")
        with t2:
            st.markdown(f"**The speed–recall trade-off** (IVF, nlist = {nlist}, averaged over 200 sample words)")
            st.write("Probing more clusters raises recall, but also the share of the database that must be scanned. "
                     f"PQ with m = {m} reaches an average Recall@10 of **{pq_rec:.0%}** while storing "
                     f"{m} bytes instead of 200 per vector.")
        with st.expander("The same indexes in FAISS and ChromaDB"):
            st.code(f'''import faiss                                   # pip install faiss-cpu
flat = faiss.IndexFlatIP(50)                   # exact inner product
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(50), 50, {nlist}, faiss.METRIC_INNER_PRODUCT)
ivf.train(X); ivf.add(X); ivf.nprobe = {nprobe}
pq = faiss.IndexPQ(50, {m}, 8, faiss.METRIC_INNER_PRODUCT)   # m sub-vectors, 2^8 = 256 centroids
hnsw = faiss.IndexHNSWFlat(50, 32)             # graph index: 32 links per node
scores, ids = ivf.search(q.reshape(1, -1), 10)

import chromadb                                 # pip install chromadb  (HNSW inside)
from chromadb.utils import embedding_functions
client = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection("stki_docs", embedding_function=ef)
collection.add(documents=["Machine learning for NLP", "Deep learning transformers",
                          "Information retrieval search engines"], ids=["doc1", "doc2", "doc3"])
collection.query(query_texts=["how do search engines work"], n_results=2)''', language="python")
            st.caption("The ChromaDB part is the slide's example, completed with a query. This lab does not install "
                       "FAISS or ChromaDB; vectordb.py implements Flat, IVF and PQ in NumPy so each step is visible.")

# ===========================================================================
# Tab 6: evaluation
# ===========================================================================
with tab_eval:
    queries = WEB["queries"]
    st.write(f"{len(queries)} labelled queries over the current index. Precision@k = relevant results in the top k "
             f"÷ k; AP averages the precision at every relevant result; MAP and MRR average AP and 1 / (rank of the "
             f"first relevant result) over the queries. Relevant pages that are not in the index (removed as "
             f"duplicates or not crawled) are left out of the judgments. Queries are used as typed, without "
             f"spelling correction or expansion. Reranking uses {reranker_name}, depth {depth}.")
    methods = [("BM25", "BM25", False), ("Dense", "Dense", False),
               (f"Weighted (α = {alpha:.2f})", "Weighted", False), (f"RRF (k = {rrf_k})", "RRF", False),
               (f"Weighted + rerank", "Weighted", True), (f"RRF + rerank", "RRF", True)]
    per_q, summ = [], {lab: [] for lab, _, _ in methods}
    usable = 0
    same_pool = True            # do both fusions send the same documents to the reranker?
    for item in queries:
        rel = set(item["rel"]) & set(URLS)
        if not rel:
            continue
        usable += 1
        w = {t: 1.0 for t in engine.tokenize(item["q"])}
        row = {"Query": item["q"], "Relevant": len(rel)}
        pools = []
        for lab, meth, rr in methods:
            ranked = [u for u, _ in retrieve(w, item["q"], URLS, meth, rr, alpha, rrf_k, depth)["final"]]
            if not rr and meth in ("Weighted", "RRF"):
                pools.append(set(ranked[:depth]))
            vals = (engine.precision_at_k(ranked, rel, 3), engine.precision_at_k(ranked, rel, 5),
                    engine.average_precision(ranked, rel), engine.reciprocal_rank(ranked, rel))
            summ[lab].append(vals)
            row[lab] = vals[2]
        same_pool &= len(pools) == 2 and pools[0] == pools[1]
        per_q.append(row)
    if not usable:
        st.info("None of the relevant pages is in the current index. Crawl more pages in the sidebar.")
    else:
        table = pd.DataFrame([{"Method": lab, "P@3": np.mean([v[0] for v in vals]),
                               "P@5": np.mean([v[1] for v in vals]), "MAP": np.mean([v[2] for v in vals]),
                               "MRR": np.mean([v[3] for v in vals])} for lab, vals in summ.items()])
        e1, e2 = st.columns([3, 2])
        with e1:
            st.dataframe(heat(table, ["P@3", "P@5", "MAP", "MRR"]), hide_index=True, width="stretch")
            st.caption(f"Best MAP: {table.loc[table['MAP'].idxmax(), 'Method']}. Dense model: {enc.name}.")
        with e2:
            sweep = []
            for al in np.round(np.linspace(0, 1, 11), 1):
                aps = []
                for item in queries:
                    rel = set(item["rel"]) & set(URLS)
                    if rel:
                        w = {t: 1.0 for t in engine.tokenize(item["q"])}
                        ranked = [u for u, _ in retrieve(w, item["q"], URLS, "Weighted", False, al)["final"]]
                        aps.append(engine.average_precision(ranked, rel))
                sweep.append({"α": al, "MAP": np.mean(aps)})
            rrf_map = table.loc[table["Method"].str.startswith("RRF ("), "MAP"].iloc[0]
            line = alt.Chart(pd.DataFrame(sweep)).mark_line(point=True, color=ACCENT).encode(
                x=alt.X("α:Q", title="α  (0 = dense only, 1 = BM25 only)"),
                y=alt.Y("MAP:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["α", alt.Tooltip("MAP:Q", format=".3f")])
            rule = alt.Chart(pd.DataFrame({"MAP": [rrf_map], "label": [f"RRF k = {rrf_k}"]})).mark_rule(
                color=AMBER, strokeDash=[5, 3]).encode(y="MAP:Q", tooltip=["label", alt.Tooltip("MAP:Q", format=".3f")])
            st.altair_chart((line + rule).properties(height=250, title="MAP of the weighted sum as α changes"),
                            width="stretch")
            st.caption("The dashed line is RRF, which has no α to tune.")
        st.markdown("**Average precision per query**")
        aps = pd.DataFrame(per_q)
        st.dataframe(heat(aps, [lab for lab, _, _ in methods], 2), hide_index=True, width="stretch",
                     height=3 + 35 * (len(aps) + 1))
        bm_wins = [r["Query"] for r in per_q if r["BM25"] > r["Dense"] + 1e-9]
        de_wins = [r["Query"] for r in per_q if r["Dense"] > r["BM25"] + 1e-9]
        fmt = lambda qs: ", ".join(f"*{q}*" for q in qs) or "none"
        notes = [f"BM25 beats dense on {len(bm_wins)} queries ({fmt(bm_wins)}); dense beats BM25 on {len(de_wins)} "
                 f"({fmt(de_wins)}). Fusion cannot always match the better retriever on every query, but it avoids "
                 f"relying on only one."]
        if same_pool:
            notes.append(f"The two reranked rows are identical because the reranker reorders the whole top {depth}: "
                         f"fusion only decides which documents get into it, and here both fusions let in the same "
                         f"documents for every query.")
        st.caption(" ".join(notes))

st.divider()
st.caption("Crawler and document processor: web.py · indexes, query processing, BM25, fusion and metrics: engine.py · "
           "bi-encoder, token vectors, MaxSim and cross-encoder: neural.py · Flat, IVF and PQ indexes: vectordb.py. "
           "The 37 pages and 6 sites are fictional. Offline word vectors: GloVe 6B 50d (PDDL).")
