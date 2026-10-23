"""
app.py: Streamlit interface for Vector Space Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import dense
from vsm import LOG_LABEL, QUERY_ID, Settings, build, kmeans_trace, load_collections

st.set_page_config(page_title="Vector Space Lab", page_icon="🧭", layout="wide")

COLLECTIONS = load_collections()
ACCENT = "#0b7285"
TF_LABEL = {"raw": "Raw count", "binary": "Binary (0/1)", "log": "1 + log₁₀(count)"}
IDF_LABEL = {"10": "log₁₀(N/df)", "2": "log₂(N/df)", "e": "ln(N/df)"}

st.markdown(
    """
<style>
.doc{border:1px solid rgba(128,140,160,.35);border-left:3px solid rgba(11,114,133,.8);border-radius:4px;
  padding:8px 12px;margin-bottom:8px;font-size:14px}
.doc b{font-family:"Courier New",monospace}
.calc{font-family:"Courier New",monospace;font-size:14px;padding:8px 12px;border-radius:6px;
  background:rgba(128,140,160,.12);white-space:pre-wrap;line-height:1.7}
.callout{border-left:3px solid rgba(207,58,63,.7);padding:6px 12px;font-size:14px;margin:6px 0}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def fmt(x: float, n: int = 5) -> str:
    return f"{x:.{n}f}".rstrip("0").rstrip(".") if abs(x) > 0 else "0"


def heat(df: pd.DataFrame, reverse: bool = False, digits: int = 3):
    """Shade cells from light to dark teal (reverse=True: small values dark). No matplotlib needed."""
    lo, hi = float(np.nanmin(df.values)), float(np.nanmax(df.values))

    def color(v):
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
        t = 1 - t if reverse else t
        return f"background-color: rgba(11,114,133,{0.06 + 0.5 * t:.2f})"

    return df.style.format(f"{{:.{digits}f}}").map(color)


def numtable(df: pd.DataFrame, digits: int = 5):
    return df.style.format(lambda v: fmt(v, digits) if isinstance(v, (float, np.floating)) else v)


@st.cache_resource(show_spinner="Loading the sentence-transformers model (first run downloads about 90 MB)…")
def get_encoder(prefer_minilm: bool):
    return dense.load_encoder(prefer_minilm)


@st.cache_data(show_spinner=False)
def encode_cached(encoder_name: str, prefer: bool, texts: tuple[str, ...]) -> np.ndarray:
    enc, _ = get_encoder(prefer)
    return enc.encode(list(texts))


def dense_rank(prefer: bool, query: str, docs: dict[str, str]) -> tuple[list[tuple[str, float]], str]:
    enc, _ = get_encoder(prefer)
    names = list(docs)
    index = dense.VectorIndex(encode_cached(enc.name, prefer, tuple(docs[n] for n in names)))
    scores, ids = index.search(encode_cached(enc.name, prefer, (query,))[0], len(names))
    return [(names[i], float(s)) for s, i in zip(scores, ids)], index.backend


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Collection")
    name = st.selectbox("Documents", list(COLLECTIONS), key="collection")
    col = COLLECTIONS[name]
    docs: dict[str, str] = col["documents"]
    st.caption(col["note"])

    st.header("Term weighting")
    defaults = col["settings"]
    tf_scheme = st.radio("TF", list(TF_LABEL), format_func=TF_LABEL.get, key=f"tf::{name}",
                         index=list(TF_LABEL).index(defaults["tf"]))
    idf_base = st.radio("IDF", list(IDF_LABEL), format_func=IDF_LABEL.get, key=f"idf::{name}",
                        index=list(IDF_LABEL).index(defaults["idf_base"]), horizontal=True)
    q_as_doc = st.checkbox("Count the query as a document in N and df", value=defaults["query_as_document"],
                           key=f"qdoc::{name}", help="The old lecture slides do this: 3 symptoms + the query gives N = 4.")
    stop = st.checkbox("Remove stopwords", value=defaults["stopwords"], key=f"stop::{name}")
    stem = st.checkbox("Stem Indonesian words (Sastrawi)", value=defaults["stemming"], key=f"stem::{name}",
                       disabled=col["lang"] != "id")
    settings = Settings(lang=col["lang"], stopwords=stop, stemming=stem, tf=tf_scheme, idf_base=idf_base,
                        query_as_document=q_as_doc)

    st.header("Dense encoder")
    enc_choice = st.radio("Model", ["all-MiniLM-L6-v2", "Offline word vectors"], key="encoder",
                          help="MiniLM is downloaded from the Hugging Face Hub on first use. The offline option "
                               "averages GloVe word vectors and always works.")
    prefer_minilm = enc_choice == "all-MiniLM-L6-v2"

    st.header("Documents")
    for d, t in docs.items():
        show(f'<div class="doc"><b>{html.escape(d)}</b><br>{html.escape(t)}</div>')

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 4 · VECTOR SPACE MODEL & DENSE RETRIEVAL")
st.title("🧭 Vector Space Lab")
st.write("Rank documents with the **Vector Space Model** and follow every number, from term frequencies to cosine "
         "similarity. Then compare it with **dense retrieval** by a bi-encoder, and combine both in a **hybrid**.")

tab_vsm, tab_geo, tab_clu, tab_dense, tab_hyb = st.tabs(
    ["1 · VSM ranking", "2 · Cosine vs Euclidean", "3 · Document similarity & clustering",
     "4 · Dense retrieval", "5 · Sparse vs dense vs hybrid"])

# ===========================================================================
# Tab 1: VSM ranking, step by step
# ===========================================================================
with tab_vsm:
    query = st.text_input("Query", value=col["query"], key=f"q::{name}")
    res = build(docs, query, settings)
    if not res.terms:
        st.warning("No terms left after preprocessing. Try turning off stopword removal.")
        st.stop()

    log = LOG_LABEL[idf_base]
    st.latex(r"w(t,d) = TF(t,d) \times IDF(t),\qquad IDF(t) = " +
             {"10": r"\log_{10}", "2": r"\log_{2}", "e": r"\ln"}[idf_base] + r"\frac{N}{df(t)},\qquad N = " + str(res.n))
    st.caption(f"TF: {TF_LABEL[tf_scheme].lower()}. N = {res.n} "
               + ("(the documents plus the query)." if q_as_doc else "(the documents only)."))

    st.markdown("##### Table 1 · Term frequency and IDF")
    t1 = pd.DataFrame(res.counts, index=res.terms, columns=res.columns).astype(int)
    t1["df"] = res.df.astype(int)
    t1["IDF"] = [f"{log}({res.n}/{int(d)}) = {fmt(i)}" if d else "–" for d, i in zip(res.df, res.idf)]
    st.dataframe(t1, width="stretch")

    st.markdown("##### Table 2 · TF × IDF weights")
    st.dataframe(numtable(pd.DataFrame(res.weights, index=res.terms, columns=res.columns)), width="stretch")

    st.markdown("##### Table 3 · Vector length: squared weights, their sum, and its square root")
    sq = pd.DataFrame(res.weights ** 2, index=res.terms, columns=res.columns)
    sq.loc["Σ w²"] = sq.sum()
    sq.loc["√Σ w² = |v|"] = np.sqrt(sq.loc["Σ w²"])
    st.dataframe(numtable(sq, 6), width="stretch")

    st.markdown("##### Table 4 · Dot product with the query: w(t,Q) × w(t,d)")
    prod = pd.DataFrame({d: res.col(QUERY_ID) * res.col(d) for d in docs}, index=res.terms)
    prod.loc["Σ = Q · d"] = prod.sum()
    st.dataframe(numtable(prod, 6), width="stretch")

    st.markdown("##### Cosine similarity and ranking")
    lines = [f"cos(Q, {d}) = {fmt(res.dot(QUERY_ID, d), 6)} / ({fmt(res.norm(QUERY_ID), 6)} × {fmt(res.norm(d), 6)})"
             f" = {fmt(s, 6)}" for d, s in res.ranking]
    show(f'<div class="calc">{html.escape(chr(10).join(lines))}</div>')
    a, b = st.columns([2, 3])
    with a:
        st.dataframe(pd.DataFrame([{"Rank": i, "Document": d, "Cosine": round(s, 6), "Text": docs[d]}
                                   for i, (d, s) in enumerate(res.ranking, 1)]), hide_index=True, width="stretch")
    with b:
        bars = alt.Chart(pd.DataFrame(res.ranking, columns=["Document", "Cosine"])).mark_bar(color=ACCENT).encode(
            x=alt.X("Cosine:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Document:N", sort=None, title=None),
            tooltip=["Document", alt.Tooltip("Cosine:Q", format=".6f")])
        st.altair_chart(bars.properties(height=max(150, 40 * len(docs))), width="stretch")

    st.markdown("##### The vectors on two term axes")
    q_terms = [t for t in res.terms if res.col(QUERY_ID)[res.terms.index(t)] > 0]
    pick_default = (q_terms + [t for t in res.terms if t not in q_terms])[:2]
    c1, c2 = st.columns(2)
    tx = c1.selectbox("x axis term", res.terms, index=res.terms.index(pick_default[0]), key=f"ax::{name}")
    ty = c2.selectbox("y axis term", res.terms, index=res.terms.index(pick_default[-1]), key=f"ay::{name}")
    ix, iy = res.terms.index(tx), res.terms.index(ty)
    vec = pd.DataFrame([{"name": c, "x": res.weights[ix, j], "y": res.weights[iy, j], "x0": 0.0, "y0": 0.0,
                         "kind": "query" if c == QUERY_ID else "document"} for j, c in enumerate(res.columns)])
    color = alt.Color("kind:N", scale=alt.Scale(domain=["query", "document"], range=["#cf3a3f", ACCENT]), legend=None)
    arrows = alt.Chart(vec).mark_rule(strokeWidth=2.5).encode(x="x0:Q", y="y0:Q", x2="x:Q", y2="y:Q", color=color)
    tips = alt.Chart(vec).mark_point(filled=True, size=70).encode(x=alt.X("x:Q", title=f"weight of “{tx}”"),
                                                                   y=alt.Y("y:Q", title=f"weight of “{ty}”"),
                                                                   color=color, tooltip=["name", "x", "y"])
    labels = alt.Chart(vec).mark_text(dx=8, dy=-6, align="left", fontSize=13).encode(
        x="x:Q", y="y:Q", text="name:N", color=color)
    st.altair_chart((arrows + tips + labels).properties(height=320), width="stretch")
    st.caption("Only two of the dimensions are drawn. Cosine similarity compares directions: the smaller the "
               "angle between the query (red) and a document, the more relevant it is.")

    with st.expander("Strengths and weaknesses of the VSM"):
        st.markdown("**Strengths:** results are ranked; documents can match a query partially; the weights reward "
                    "terms that are rare in the collection but frequent in the document.\n\n"
                    "**Weaknesses:** every term is an independent dimension, so synonyms (*mobil*, *kendaraan*) "
                    "never match and word order is ignored.")

# ===========================================================================
# Tab 2: cosine vs Euclidean
# ===========================================================================
with tab_geo:
    st.write("Two ways to compare vectors. **Euclidean distance** measures how far apart the points are; it grows "
             "with document length. **Cosine similarity** measures the angle; a long and a short document about "
             "the same thing point the same way.")
    st.markdown("##### The lecture's example (term counts)")
    base = pd.DataFrame({"tea": [2, 2, 0, 5], "me": [0, 1, 2, 0], "two": [2, 0, 0, 7]},
                        index=["doc1", "doc2", "doc3", "doc4"])
    pts = st.data_editor(base, key="geo_points", width="stretch")
    m = pts.to_numpy(dtype=float)
    names = list(pts.index)
    norms = np.linalg.norm(m, axis=1)
    eu = pd.DataFrame([[float(np.linalg.norm(m[i] - m[j])) for j in range(len(m))] for i in range(len(m))],
                      index=names, columns=names)
    co = pd.DataFrame([[float(m[i] @ m[j] / (norms[i] * norms[j])) if norms[i] and norms[j] else 0.0
                        for j in range(len(m))] for i in range(len(m))], index=names, columns=names)
    a, b = st.columns(2)
    with a:
        st.markdown("**Euclidean distance** (smaller = closer)")
        st.dataframe(heat(eu, reverse=True), width="stretch")
    with b:
        st.markdown("**Cosine similarity** (larger = more similar)")
        st.dataframe(heat(co), width="stretch")
    if {"doc1", "doc4"} <= set(names):
        st.markdown(f"*doc1* and *doc4* use *tea* and *two* in almost the same proportion, so their cosine is "
                    f"**{co.loc['doc1', 'doc4']:.3f}**. But *doc4* is longer, so the Euclidean distance between them "
                    f"is **{eu.loc['doc1', 'doc4']:.2f}**, larger than from *doc1* to *doc3* "
                    f"({eu.loc['doc1', 'doc3']:.2f}), which has no word in common with *doc1*. That is the weakness "
                    "of Euclidean distance on raw counts.")
    if {"tea", "two"} <= set(pts.columns):
        dfp = pd.DataFrame({"name": names, "x": pts["tea"].astype(float), "y": pts["two"].astype(float), "x0": 0.0, "y0": 0.0})
        ch = (alt.Chart(dfp).mark_rule(color=ACCENT, strokeWidth=2.5).encode(x="x0:Q", y="y0:Q", x2="x:Q", y2="y:Q")
              + alt.Chart(dfp).mark_point(filled=True, color=ACCENT, size=70).encode(
                  x=alt.X("x:Q", title="tea"), y=alt.Y("y:Q", title="two"), tooltip=["name", "x", "y"])
              + alt.Chart(dfp).mark_text(dx=8, dy=-6, align="left", color=ACCENT, fontSize=13).encode(
                  x="x:Q", y="y:Q", text="name:N"))
        st.altair_chart(ch.properties(height=300), width="stretch")

    st.markdown("##### Cosine similarity calculator")
    c1, c2 = st.columns(2)
    xs = c1.text_input("Vector x (comma-separated)", "1, 1, 1, 3, 0", key="calc_x")
    ys = c2.text_input("Vector y (comma-separated)", "0, 0, 0, 1, 1", key="calc_y")
    try:
        x = np.array([float(v) for v in xs.split(",") if v.strip()])
        y = np.array([float(v) for v in ys.split(",") if v.strip()])
        if len(x) != len(y) or not len(x):
            raise ValueError("both vectors need the same number of values")
        dot, nx, ny = float(x @ y), float(np.sqrt(x @ x)), float(np.sqrt(y @ y))
        f = lambda v: fmt(float(v), 4)
        steps = [f"x · y = {' + '.join(f(a * b) for a, b in zip(x, y))} = {f(dot)}",
                 f"|x|   = √({' + '.join(f(a * a) for a in x)}) = √{f(x @ x)} = {f(nx)}",
                 f"|y|   = √({' + '.join(f(b * b) for b in y)}) = √{f(y @ y)} = {f(ny)}",
                 f"cos   = {f(dot)} / ({f(nx)} × {f(ny)}) = " + (f"{dot / (nx * ny):.4f}" if nx and ny else "undefined")]
        show(f'<div class="calc">{html.escape(chr(10).join(steps))}</div>')
    except ValueError as e:
        st.error(f"Check the vectors: {e}.")

# ===========================================================================
# Tab 3: document similarity and clustering
# ===========================================================================
with tab_clu:
    st.write("Compare every document with every other document using the TF-IDF vectors (the query is not used "
             "here), then group them with k-means.")
    r_docs = build(docs, None, Settings(**{**settings.__dict__, "query_as_document": False}))
    dnames = list(docs)
    metric = st.radio("Measure", ["Euclidean distance", "Cosine similarity"], horizontal=True, key="clu_metric")
    fn = r_docs.euclidean if metric.startswith("Euclidean") else r_docs.cosine
    mat = pd.DataFrame([[fn(a, b) for b in dnames] for a in dnames], index=dnames, columns=dnames)
    st.dataframe(heat(mat, reverse=metric.startswith("Euclidean"), digits=4), width="stretch")
    with st.expander("TF-IDF weights used here"):
        st.dataframe(numtable(pd.DataFrame(r_docs.weights, index=r_docs.terms, columns=dnames)), width="stretch")

    st.markdown("##### K-means clustering (Euclidean distance)")
    init_default = col.get("clustering", {}).get("init", dnames[:2])
    max_k = max(2, len(dnames) - 1)
    if max_k > 2:
        k = st.slider("Number of clusters", 2, max_k, min(len(init_default), max_k), key=f"k::{name}")
    else:
        k = 2
        st.caption("2 clusters (the collection has only a few documents).")
    init = st.multiselect("Starting centroids (one document per cluster)", dnames,
                          default=init_default[:k] if len(init_default) >= k else dnames[:k],
                          max_selections=k, key=f"init::{name}::{k}")
    if len(init) != k:
        st.info(f"Choose {k} documents as starting centroids.")
    else:
        iters = kmeans_trace(r_docs, dnames, init)
        for it in iters:
            label = "Iteration 1 (centroids = the chosen documents)" if it["iteration"] == 1 else f"Iteration {it['iteration']}"
            st.markdown(f"**{label}**")
            heads = [f"C{j + 1}" + (f" ({init[j]})" if it["iteration"] == 1 else "") for j in range(k)]
            tab = pd.DataFrame({h: [it["distances"][d][j] for d in dnames] for j, h in enumerate(heads)}, index=dnames)
            tab["Closest"] = [heads[it["assignment"][d]] for d in dnames]
            st.dataframe(tab.style.format({h: "{:.4f}" for h in heads}), width="stretch")
        final = iters[-1]["assignment"]
        groups = [", ".join(d for d in dnames if final[d] == j) or "(empty)" for j in range(k)]
        st.success("  ·  ".join(f"Cluster {j + 1}: {g}" for j, g in enumerate(groups))
                   + f"  (converged after {len(iters)} iteration{'s' if len(iters) > 1 else ''})")
        if name.startswith("Spreadsheet example B") and set(init) == {"D1", "D4"} and settings == Settings(
                **{**col["settings"], "lang": col["lang"]}):
            show('<div class="callout"><b>Different from SimilarityTest.xlsx.</b> The spreadsheet assigns D2 to '
                 'Cluster 2. Its distance formulas (cells L37:M40) read rows 29–35, one row below the TF-IDF table '
                 'in rows 28–34, so the term <i>mata</i> is left out. With all seven terms, D2 is 0.6275 from D1 and '
                 '0.6846 from D4, so it belongs to Cluster 1. The correct result is Cluster 1: D1, D2 and '
                 'Cluster 2: D3, D4.</div>')

# ===========================================================================
# Tab 4: dense retrieval
# ===========================================================================
with tab_dense:
    encoder, why = get_encoder(prefer_minilm)
    if why:
        st.warning(f"Using the offline fallback because {why}. On Streamlit Community Cloud with "
                   "`sentence-transformers` in requirements.txt, MiniLM downloads automatically.")
    st.write("A **bi-encoder** encodes the query and each document separately into one dense vector, "
             "then ranks by cosine similarity. Document vectors can be computed once, offline, and searched with "
             "an approximate nearest-neighbor index such as FAISS.")
    st.latex(r"\text{score}(q,d) = \cos\big(E_q(q),\,E_d(d)\big)")
    dq = st.text_input("Query", value=col["query"], key=f"dq::{name}")
    if dq.strip():
        ranking, backend = dense_rank(prefer_minilm, dq, docs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Encoder", encoder.name.split(" (")[0])
        c2.metric("Dimensions", encoder.dims)
        c3.metric("Documents indexed", len(docs))
        st.caption(f"Search backend: {backend}")
        st.dataframe(pd.DataFrame([{"Rank": i, "Document": d, "Cosine": round(s, 4), "Text": docs[d]}
                                   for i, (d, s) in enumerate(ranking, 1)]), hide_index=True, width="stretch")

        sparse_q = build(docs, dq, settings)
        nz = int((sparse_q.col(QUERY_ID) > 0).sum())
        st.markdown(f"**Sparse vs dense query vector.** The TF-IDF query vector has {len(sparse_q.terms)} dimensions "
                    f"(one per term in this collection), and {nz} of them are non-zero. The dense vector has "
                    f"{encoder.dims} dimensions, and all of them carry a value.")

        names = list(docs)
        E = encode_cached(encoder.name, prefer_minilm, tuple(docs[n] for n in names))
        sim = E @ E.T
        hm = pd.DataFrame([{"a": names[i], "b": names[j], "cos": float(sim[i, j])}
                           for i in range(len(names)) for j in range(len(names))])
        heat = alt.Chart(hm).mark_rect().encode(
            x=alt.X("a:N", sort=names, title=None), y=alt.Y("b:N", sort=names, title=None),
            color=alt.Color("cos:Q", scale=alt.Scale(scheme="teals"), title="cosine"),
            tooltip=["a", "b", alt.Tooltip("cos:Q", format=".3f")])
        st.markdown("**Document-to-document similarity** in the embedding space")
        st.altair_chart(heat.properties(height=min(420, 40 * len(names) + 40)), width="stretch")

    with st.expander("How bi-encoders are trained: DPR and contrastive learning"):
        st.markdown("**Dense Passage Retrieval** (Karpukhin et al., 2020) uses two BERT-base encoders, one for "
                    "questions and one for passages, trained so that a question lands close to its correct "
                    "passage and far from the others. On Natural Questions it reached 78.4% top-20 accuracy, "
                    "against 59.1% for BM25.")
        st.latex(r"\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\sum_i \exp(\mathrm{sim}(q,d_i)/\tau)},"
                 r"\qquad \mathrm{sim}(q,d) = E_q(q)^\top E_d(d)")
        st.markdown("*d⁺* is the relevant passage; the other passages in the batch serve as negatives (in-batch "
                    "negatives). Later models include Contriever (2022) and E5 (2023).")
        st.code('''from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(docs, normalize_embeddings=True).astype("float32")   # (n, 384)
index = faiss.IndexFlatIP(doc_embs.shape[1])     # inner product = cosine on unit vectors
index.add(doc_embs)
scores, ids = index.search(model.encode([query], normalize_embeddings=True), k=3)''', language="python")

# ===========================================================================
# Tab 5: sparse vs dense vs hybrid
# ===========================================================================
with tab_hyb:
    st.write("Sparse retrieval (TF-IDF) is good at exact keywords; dense retrieval is good at meaning. A **hybrid** "
             "normalizes both scores to [0, 1] and mixes them:")
    st.latex(r"\text{score}_{hybrid} = \alpha \cdot \text{dense} + (1-\alpha) \cdot \text{sparse}")
    labelled = col.get("labelled_queries", {})
    a, b = st.columns([3, 2])
    with a:
        if labelled:
            hq = st.selectbox("Query (labelled with its relevant document)", list(labelled), key=f"hq::{name}",
                              index=list(labelled).index(col["query"]) if col["query"] in labelled else 0)
        else:
            hq = st.text_input("Query", value=col["query"], key=f"hq::{name}")
    with b:
        alpha = st.slider("α (weight of the dense score)", 0.0, 1.0, 0.5, 0.1, key="alpha")
    relevant = set(labelled.get(hq, []))

    if hq.strip():
        sp = build(docs, hq, settings).ranking
        de, _ = dense_rank(prefer_minilm, hq, docs)
        hy = dense.hybrid_ranking(sp, de, alpha)
        c1, c2, c3 = st.columns(3)
        for c, title, rk in ((c1, "Sparse · TF-IDF", sp), (c2, "Dense · " + encoder.name.split(" (")[0], de),
                             (c3, f"Hybrid · α = {alpha:.1f}", hy)):
            with c:
                st.markdown(f"##### {title}")
                st.dataframe(pd.DataFrame([{"Rank": i, "Doc": ("✓ " if d in relevant else "") + d, "Score": round(s, 3)}
                                           for i, (d, s) in enumerate(rk[:6], 1)]), hide_index=True, width="stretch")
        if relevant:
            st.caption("✓ marks the document labelled as relevant for this query.")

    if labelled:
        st.markdown("##### Evaluation over all labelled queries")
        rows, curve = [], []
        per_query = []
        cache = {}
        for q, rel in labelled.items():
            cache[q] = (build(docs, q, settings).ranking, dense_rank(prefer_minilm, q, docs)[0])
        for method in ["Sparse", "Dense", "Hybrid"]:
            hits1, hits3, rr = [], [], []
            for q, rel in labelled.items():
                s_, d_ = cache[q]
                rk = s_ if method == "Sparse" else d_ if method == "Dense" else dense.hybrid_ranking(s_, d_, alpha)
                hits1.append(dense.hit_at_k(rk, set(rel), 1))
                hits3.append(dense.hit_at_k(rk, set(rel), 3))
                rr.append(dense.reciprocal_rank(rk, set(rel)))
            rows.append({"Method": method + (f" (α = {alpha:.1f})" if method == "Hybrid" else ""),
                         "Hit@1": np.mean(hits1), "Hit@3": np.mean(hits3), "MRR": np.mean(rr)})
        for q, rel in labelled.items():
            s_, d_ = cache[q]
            pos = lambda rk: next(i for i, (d, _) in enumerate(rk, 1) if d in rel)
            per_query.append({"Query": q, "Relevant": ", ".join(rel), "Sparse rank": pos(s_), "Dense rank": pos(d_),
                              "Hybrid rank": pos(dense.hybrid_ranking(s_, d_, alpha))})
        for al in np.round(np.arange(0, 1.01, 0.1), 1):
            rr = [dense.reciprocal_rank(dense.hybrid_ranking(*cache[q], al), set(rel)) for q, rel in labelled.items()]
            curve.append({"α": al, "MRR": float(np.mean(rr))})
        a, b = st.columns([2, 3])
        with a:
            st.dataframe(pd.DataFrame(rows).style.format({"Hit@1": "{:.2f}", "Hit@3": "{:.2f}", "MRR": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.caption("Hit@k: share of queries whose relevant document is in the top k. "
                       "MRR: mean of 1 / rank of the relevant document.")
        with b:
            line = alt.Chart(pd.DataFrame(curve)).mark_line(point=True, color=ACCENT).encode(
                x=alt.X("α:Q", title="α  (0 = sparse only, 1 = dense only)"),
                y=alt.Y("MRR:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["α", alt.Tooltip("MRR:Q", format=".3f")])
            st.altair_chart(line.properties(height=240), width="stretch")
        st.markdown("**Per query** (rank of the relevant document; lower is better)")
        st.dataframe(pd.DataFrame(per_query), hide_index=True, width="stretch")
        st.caption("Look for the two error types from the lecture: sparse retrieval misses synonyms "
                   "(*money … good grades* vs *scholarship … GPA*); dense retrieval can miss exact keywords "
                   "(*KRS*, *TOEIC*).")
    else:
        st.info("Choose *University FAQ* in the sidebar to evaluate the three methods on labelled queries.")

st.divider()
st.caption("VSM and k-means are implemented from scratch in vsm.py; the bi-encoder, FAISS/NumPy search, and hybrid "
           "fusion are in dense.py. Offline word vectors: GloVe 6B 50d (PDDL), 30,000 most frequent words.")
