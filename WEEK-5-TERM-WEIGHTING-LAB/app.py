"""
app.py: Streamlit interface for Term Weighting Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import dense
import learned_sparse as ls
from weighting import TF_VARIANTS, Collection, load_collections, substring_df, tokenize

st.set_page_config(page_title="Term Weighting Lab", page_icon="⚖️", layout="wide")

COLLECTIONS = load_collections()
ACCENT = "#0b7285"
METHODS = ["TF-IDF", "BM25", "Learned sparse", "Dense", "Hybrid"]

st.markdown(
    """
<style>
.doc{border:1px solid rgba(128,140,160,.35);border-left:3px solid rgba(11,114,133,.8);border-radius:4px;
  padding:8px 12px;margin-bottom:8px;font-size:14px}
.doc b{font-family:"Courier New",monospace}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.chip{font-family:"Courier New",monospace;font-size:13.5px;border:1px solid rgba(128,140,160,.45);
  border-radius:4px;padding:2px 8px}
.chip.orig{border-color:rgba(11,114,133,.85);background:rgba(11,114,133,.16);font-weight:700}
.chip.exp{border-color:rgba(232,160,40,.8);background:rgba(232,160,40,.16)}
.callout{border-left:3px solid rgba(207,58,63,.7);padding:6px 12px;font-size:14px;margin:6px 0}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def fmt_df(df: pd.DataFrame, digits: int = 4):
    return df.style.format(lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else v)


@st.cache_resource(show_spinner="Loading all-MiniLM-L6-v2 (first run downloads about 90 MB)…")
def get_dense(prefer: bool):
    return dense.load_dense_encoder(prefer)


@st.cache_resource(show_spinner="Loading the SPLADE model (first run downloads about 440 MB)…")
def get_sparse(prefer: bool):
    return ls.load_sparse_encoder(prefer)


@st.cache_data(show_spinner=False)
def dense_matrix(prefer: bool, texts: tuple[str, ...]) -> np.ndarray:
    return get_dense(prefer)[0].encode(list(texts))


@st.cache_data(show_spinner=False)
def sparse_vectors(prefer: bool, texts: tuple[str, ...]) -> list[dict[str, float]]:
    enc = get_sparse(prefer)[0]
    return [enc.encode(t) for t in texts]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Collection")
    name = st.selectbox("Documents", list(COLLECTIONS), key="collection")
    col = COLLECTIONS[name]
    docs: dict[str, str] = col["documents"]
    st.caption(col["note"])
    remove_sw = st.checkbox("Remove stopwords", value=True)

    st.header("BM25")
    k1 = st.slider("k₁ (term frequency saturation)", 0.0, 3.0, 1.2, 0.1, key="k1")
    b = st.slider("b (length normalization)", 0.0, 1.0, 0.75, 0.05, key="b")

    st.header("Neural models")
    dense_choice = st.radio("Dense encoder", ["all-MiniLM-L6-v2", "Offline word vectors"], key="dense_choice",
                            help="MiniLM downloads about 90 MB on first use. The offline option averages GloVe vectors.")
    sparse_choice = st.radio("Learned sparse encoder", ["Offline term expansion", "SPLADE (downloads 440 MB)"],
                             key="sparse_choice",
                             help="SPLADE needs PyTorch and transformers and about 1 GB of memory. The offline option "
                                  "imitates term expansion with GloVe neighbors; it is not a trained model.")
    hyb_norm = st.radio("Hybrid score normalization", ["max", "minmax"], key="hyb_norm",
                        format_func={"max": "BM25 ÷ max (lecture code)", "minmax": "Min-max both scores"}.get)

    st.header("Documents")
    for d, t in docs.items():
        show(f'<div class="doc"><b>{html.escape(d)}</b><br>{html.escape(t)}</div>')

prefer_dense = dense_choice == "all-MiniLM-L6-v2"
prefer_splade = sparse_choice.startswith("SPLADE")
c = Collection(docs, remove_sw)
names = list(docs)


def rank_dense(q: str) -> list[tuple[str, float]]:
    E = dense_matrix(prefer_dense, tuple(docs[n] for n in names))
    qe = dense_matrix(prefer_dense, (q,))[0]
    return sorted(zip(names, map(float, E @ qe)), key=lambda x: -x[1])


def rank_sparse(q: str) -> list[tuple[str, float]]:
    vecs = sparse_vectors(prefer_splade, tuple(docs[n] for n in names))
    qv = sparse_vectors(prefer_splade, (q,))[0]
    return sorted(((n, ls.sparse_dot(qv, v)[0]) for n, v in zip(names, vecs)), key=lambda x: -x[1])


def all_rankings(q: str, alpha: float) -> dict[str, list[tuple[str, float]]]:
    bm = c.bm25_ranking(q, k1, b)
    de = rank_dense(q)
    return {"TF-IDF": c.tfidf_ranking(q, "log"), "BM25": bm, "Learned sparse": rank_sparse(q), "Dense": de,
            "Hybrid": dense.hybrid(bm, de, alpha, hyb_norm)}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 5 · TERM WEIGHTING & LEARNED SPARSE RETRIEVAL")
st.title("⚖️ Term Weighting Lab")
st.write("Compare the ways a search engine decides how much a word counts: **TF-IDF** variants, **BM25**, the "
         "learned sparse weights of **SPLADE**, and a **hybrid** of BM25 with dense retrieval, evaluated with "
         "Precision@5 and Precision@10.")

tab_tf, tab_bm, tab_sp, tab_hy, tab_ev = st.tabs(
    ["1 · TF-IDF weighting", "2 · BM25", "3 · SPLADE (learned sparse)", "4 · Hybrid search", "5 · Evaluation"])

# ===========================================================================
# Tab 1: TF-IDF variants
# ===========================================================================
with tab_tf:
    st.latex(r"\mathrm{TF\text{-}IDF}(t,d) = TF(t,d) \times IDF(t),\qquad IDF(t) = \log_{10}\frac{N}{df(t)},\qquad N = "
             + str(c.N))
    st.write("A word that is frequent in a document (high TF) but rare in the collection (high IDF) is the most "
             "informative. The lecture lists several ways to compute TF:")
    st.dataframe(pd.DataFrame([
        {"Variant": "Raw", "Formula": "f(t,d)", "Effect": "Long documents are unfairly favored"},
        {"Variant": "Normalized", "Formula": "f(t,d) / Σ f(t',d)", "Effect": "Share of the document's words"},
        {"Variant": "Max-normalized", "Formula": "f(t,d) / max f(t',d)", "Effect": "Relative to the most frequent term"},
        {"Variant": "Log", "Formula": "1 + log₁₀ f(t,d)", "Effect": "Dampens high frequencies"},
        {"Variant": "Augmented", "Formula": "0.5 + 0.5 · f(t,d) / max f", "Effect": "Normalized to the most frequent term, never below 0.5"},
        {"Variant": "Boolean", "Formula": "1 if f(t,d) > 0", "Effect": "Presence only, not frequency"},
    ]), hide_index=True, width="stretch")

    a, b_ = st.columns([2, 3])
    with a:
        doc = st.selectbox("Document", names, index=names.index("D2") if "D2" in names else 0, key=f"tfdoc::{name}")
        show(f'<div class="doc"><b>{html.escape(doc)}</b><br>{html.escape(docs[doc])}</div>')
        st.caption(f"After tokenization: {len(c.tokens[doc])} tokens, most frequent term appears "
                   f"{max(c.tf[doc].values(), default=0)} times.")
    with b_:
        variant = st.radio("TF variant for the TF-IDF column", list(TF_VARIANTS), format_func=TF_VARIANTS.get,
                           horizontal=True, index=list(TF_VARIANTS).index("log"), key="tfvar")
    rows = c.tfidf_table(doc)
    tdf = pd.DataFrame(rows)
    tdf[f"TF-IDF ({variant})"] = tdf[variant] * tdf["IDF"]
    st.dataframe(fmt_df(tdf.rename(columns={v: v.capitalize() for v in TF_VARIANTS})), hide_index=True, width="stretch")

    st.markdown("##### How each TF variant grows with the frequency")
    st.caption("Augmented and max-normalized are drawn for a document whose most frequent term appears 10 times; "
               "normalized for a document of 20 words.")
    curve = []
    for f in range(0, 11):
        for v in ["raw", "log", "augmented", "max", "normalized", "boolean"]:
            curve.append({"f(t,d)": f, "Variant": v, "TF": 0.0 if f == 0 else
                          {"raw": f, "log": 1 + math.log10(f), "augmented": 0.5 + 0.05 * f, "max": f / 10,
                           "normalized": f / 20, "boolean": 1.0}[v]})
    st.altair_chart(alt.Chart(pd.DataFrame(curve)).mark_line(point=True).encode(
        x="f(t,d):Q", y="TF:Q", color=alt.Color("Variant:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["Variant", "f(t,d)", alt.Tooltip("TF:Q", format=".3f")]).properties(height=280), width="stretch")

    st.markdown("##### Ranking with each TF variant (cosine similarity of TF-IDF vectors)")
    q1 = st.text_input("Query", value=col["query"], key=f"tfq::{name}")
    cmp_rows = {}
    for v in TF_VARIANTS:
        r = c.tfidf_ranking(q1, v)
        cmp_rows[v.capitalize()] = [f"{d} ({s:.3f})" for d, s in r]
    st.dataframe(pd.DataFrame(cmp_rows, index=range(1, len(names) + 1)), width="stretch",
                 height=min(40 + 35 * len(names), 420))
    st.caption("Raw, normalized, and max-normalized TF always give the same cosine ranking: they only multiply a "
               "document's weights by one constant, and cosine ignores the length of the vector. Log, augmented, and "
               "boolean TF change the shape of the weights and can change the order.")

# ===========================================================================
# Tab 2: BM25
# ===========================================================================
with tab_bm:
    st.latex(r"\mathrm{BM25}(q,d) = \sum_{q_i \in q} IDF(q_i)\cdot\frac{f(q_i,d)\,(k_1+1)}"
             r"{f(q_i,d) + k_1\left(1 - b + b\cdot\frac{|d|}{avgdl}\right)}")
    st.latex(r"IDF(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)")
    st.caption(f"N = {c.N}, avgdl = {c.avgdl:.3f} tokens, k₁ = {k1}, b = {b}. "
               "Adjust k₁ and b in the sidebar; the defaults (1.2, 0.75) work well in most cases.")
    q2 = st.text_input("Query", value=col["query"], key=f"bmq::{name}")
    trace_rows = []
    for d in names:
        for r in c.bm25_trace(q2, d, k1, b):
            trace_rows.append({"Doc": d, **r})
    if not trace_rows:
        st.warning("The query has no terms left after tokenization.")
    else:
        tr = pd.DataFrame(trace_rows)
        totals = tr.groupby("Doc", sort=False)["Contribution"].sum()
        show_all = st.checkbox("Show documents that do not contain any query term", value=len(names) <= 6,
                               key=f"bmall::{name}")
        view = tr if show_all else tr[tr["Doc"].isin(totals[totals > 0].index)]
        st.dataframe(fmt_df(view), hide_index=True, width="stretch", height=min(40 + 35 * len(view), 460))

        a, b_ = st.columns(2)
        with a:
            st.markdown("**BM25 ranking**")
            bm = c.bm25_ranking(q2, k1, b)
            st.dataframe(pd.DataFrame([{"Rank": i, "Doc": d, "BM25": round(s, 4), "Text": docs[d]}
                                       for i, (d, s) in enumerate(bm, 1) if s > 0 or show_all]),
                         hide_index=True, width="stretch")
        with b_:
            st.markdown("**TF-IDF ranking** (log TF, cosine), for comparison")
            ti = c.tfidf_ranking(q2, "log")
            st.dataframe(pd.DataFrame([{"Rank": i, "Doc": d, "TF-IDF": round(s, 4)}
                                       for i, (d, s) in enumerate(ti, 1) if s > 0 or show_all]),
                         hide_index=True, width="stretch")

    st.markdown("##### What k₁ and b do")
    a, b_ = st.columns(2)
    with a:
        sat = [{"f(t,d)": f, "k₁": str(kk), "Contribution ÷ IDF": f * (kk + 1) / (f + kk) if f else 0.0}
               for kk in sorted({0.5, 1.2, 2.0, round(k1, 1)}) for f in range(0, 16)]
        st.altair_chart(alt.Chart(pd.DataFrame(sat)).mark_line().encode(
            x="f(t,d):Q", y="Contribution ÷ IDF:Q", color=alt.Color("k₁:N", scale=alt.Scale(scheme="teals")),
            tooltip=["k₁", "f(t,d)", alt.Tooltip("Contribution ÷ IDF:Q", format=".3f")]).properties(
            height=250, title="Saturation (document of average length)"), width="stretch")
        st.caption("A word that appears 10 times is not worth 10 times one occurrence: the curve flattens toward "
                   "k₁ + 1. A smaller k₁ saturates sooner; k₁ = 0 ignores frequency entirely.")
    with b_:
        ln = [{"|d| / avgdl": r_ / 10, "b": str(bb), "Contribution ÷ IDF": 2 * (k1 + 1) / (2 + k1 * (1 - bb + bb * r_ / 10))}
              for bb in sorted({0.0, 0.75, 1.0, round(b, 2)}) for r_ in range(2, 31)]
        st.altair_chart(alt.Chart(pd.DataFrame(ln)).mark_line().encode(
            x="|d| / avgdl:Q", y="Contribution ÷ IDF:Q", color=alt.Color("b:N", scale=alt.Scale(scheme="teals")),
            tooltip=["b", "|d| / avgdl", alt.Tooltip("Contribution ÷ IDF:Q", format=".3f")]).properties(
            height=250, title=f"Length normalization (term appears twice, k₁ = {k1})"), width="stretch")
        st.caption("With b > 0, the same word count is worth less in a longer document. b = 0 ignores length; "
                   "b = 1 normalizes fully.")

    qterms = c.query_terms(q2)
    diffs = [(t, c.df.get(t, 0), substring_df(t, docs)) for t in qterms]
    if diffs:
        st.markdown("##### A detail in the lecture's BM25 code")
        st.write("The slide computes document frequency with `sum(1 for d in corpus if term in d.lower())`. On a "
                 "string, `in` matches **substrings**, so *car* is also found inside *cars*, *carrots*, and *care*. "
                 "Counting whole words, as this app does, avoids that:")
        st.dataframe(pd.DataFrame([{"Term": t, "df (whole words)": w, "df (substring, slide code)": s_,
                                    "Same?": "yes" if w == s_ else "no"} for t, w, s_ in diffs]),
                     hide_index=True, width="stretch")

# ===========================================================================
# Tab 3: SPLADE
# ===========================================================================
with tab_sp:
    senc, swhy = get_sparse(prefer_splade)
    if swhy:
        st.warning(f"Using the offline term expansion because {swhy}.")
    elif not senc.real:
        st.info("Showing the **offline term expansion**: each word activates itself and its nearest GloVe "
                "neighbors. It illustrates how SPLADE expands text, but the weights are not learned. Choose "
                "*SPLADE* in the sidebar to load the real model.")
    st.write("SPLADE runs text through **BERT**'s masked-language-model head, which scores every word of the "
             "vocabulary at every position, then keeps a sparse vector:")
    st.latex(r"w_j = \max_{i \in \text{input}} \log\left(1 + \mathrm{ReLU}(\mathrm{logit}_{ij})\right),"
             r"\qquad \text{score}(q,d) = \sum_j w_j^{(q)}\, w_j^{(d)}")
    st.markdown("Input tokens → BERT encoder → MLM head → ReLU + log(1+x) → max pooling → sparse vector over the "
                f"vocabulary ({senc.vocab_size:,} dimensions here). Words that are not in the text can get a weight "
                "(**term expansion**: *car* activates *vehicle*), and because the vector is sparse it fits in an "
                "ordinary inverted index.")

    q3 = st.text_input("Text to encode (a query or a document)", value=col["query"], key=f"spq::{name}")
    if q3.strip():
        vec = sparse_vectors(prefer_splade, (q3,))[0]
        orig = set(tokenize(q3, remove_stopwords=False))
        top = sorted(vec.items(), key=lambda x: -x[1])[:25]
        st.markdown(f"**{len(vec)} non-zero dimensions** out of {senc.vocab_size:,} "
                    f"({100 * len(vec) / senc.vocab_size:.3f}%). "
                    "Teal = word from the text, amber = expansion term.")
        show('<div class="chips">' + "".join(
            f'<span class="chip {"orig" if t in orig else "exp"}">{html.escape(t)} {w:.2f}</span>' for t, w in top)
             + "</div>")
        bars = pd.DataFrame([{"Term": t, "Weight": w, "Kind": "in text" if t in orig else "expansion"} for t, w in top])
        st.altair_chart(alt.Chart(bars).mark_bar().encode(
            x="Weight:Q", y=alt.Y("Term:N", sort="-x", title=None),
            color=alt.Color("Kind:N", scale=alt.Scale(domain=["in text", "expansion"], range=[ACCENT, "#e8a028"])),
            tooltip=["Term", alt.Tooltip("Weight:Q", format=".3f"), "Kind"]).properties(height=max(200, 18 * len(top))),
            width="stretch")

        st.markdown("##### Retrieval with the sparse vectors")
        rs = rank_sparse(q3)
        dvecs = dict(zip(names, sparse_vectors(prefer_splade, tuple(docs[n] for n in names))))
        rows = []
        for i, (d, s) in enumerate(rs[:8], 1):
            _, shared = ls.sparse_dot(vec, dvecs[d])
            rows.append({"Rank": i, "Doc": d, "Score": round(s, 3),
                         "Matching terms (query × doc weight)": ", ".join(f"{t} {a * w:.2f}" for t, a, w in shared[:5]),
                         "Text": docs[d]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        nz = [len(v) for v in dvecs.values()]
        st.caption(f"Documents have {np.mean(nz):.0f} non-zero terms on average. An inverted index stores only those, "
                   "so SPLADE keeps the efficiency of BM25 while matching related words.")

    with st.expander("SPLADE++ and the code to run the real model"):
        st.markdown("SPLADE++ (Formal et al., TOIS 2024) learns the sparse representation with BERT: each of the "
                    "~30K dimensions is a learned term weight. Compared with BM25 it adds term expansion (synonyms and "
                    "related terms) and soft matching, and stays compatible with inverted indexes.")
        st.code('''import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

name = "naver/splade-cocondenser-ensembledistil"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForMaskedLM.from_pretrained(name).eval()

enc = tok("car engine maintenance", return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits                        # (1, tokens, 30522)
w = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
w = w.max(dim=1).values[0]                              # max pooling -> sparse vector
terms = {tok.convert_ids_to_tokens(int(i)): float(w[i]) for i in torch.nonzero(w).flatten()}''',
                language="python")

# ===========================================================================
# Tab 4: hybrid search
# ===========================================================================
with tab_hy:
    denc, dwhy = get_dense(prefer_dense)
    if dwhy:
        st.warning(f"Dense retrieval uses the offline fallback because {dwhy}.")
    st.write("BM25 is good at exact matches; dense retrieval is good at meaning. The lecture combines them as")
    st.latex(r"\text{hybrid} = \alpha \cdot \frac{\mathrm{BM25}}{\max \mathrm{BM25}} + (1-\alpha)\cdot \text{dense}")
    st.caption("Here α is the weight of **BM25** (as in the Session 5 slides; in Session 4, α weighted the dense "
               "score). α = 0.5 is balanced, 0.7 favors exact matches, 0.3 favors meaning.")
    labelled = col.get("labelled_queries", {})
    a, b_ = st.columns([3, 2])
    with a:
        if labelled:
            q4 = st.selectbox("Query (labelled)", list(labelled), key=f"hyq::{name}",
                              index=list(labelled).index(col["query"]) if col["query"] in labelled else 0)
        else:
            q4 = st.text_input("Query", value=col["query"], key=f"hyq::{name}")
    with b_:
        alpha = st.slider("α (weight of BM25)", 0.0, 1.0, 0.5, 0.1, key="alpha")
    relevant = set(labelled.get(q4, []))
    if q4.strip():
        R = all_rankings(q4, alpha)
        cols = st.columns(4)
        for colw, m in zip(cols, ["BM25", "Dense", "Learned sparse", "Hybrid"]):
            with colw:
                title = {"Dense": f"Dense · {denc.name.split(' (')[0]}", "Hybrid": f"Hybrid · α = {alpha:.1f}",
                         "Learned sparse": "SPLADE" if get_sparse(prefer_splade)[0].real else "Term expansion"}.get(m, m)
                st.markdown(f"##### {title}")
                st.dataframe(pd.DataFrame([{"#": i, "Doc": ("✓ " if d in relevant else "") + d, "Score": round(s, 3)}
                                           for i, (d, s) in enumerate(R[m][:10], 1)]), hide_index=True, width="stretch",
                             height=388)
        if relevant:
            p = {m: dense.precision_at_k(R[m], relevant, 5) for m in R}
            st.caption(f"✓ = relevant ({len(relevant)} documents). A score of 0 means the document was not retrieved. "
                       "Precision@5 for this query: "
                       + " · ".join(f"{m} {v:.2f}" for m, v in p.items()))
    st.code('''from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

def hybrid_search(query, docs, alpha=0.5):
    bm25 = BM25Okapi([d.split() for d in docs])
    sparse = bm25.get_scores(query.split())
    sparse_n = sparse / (sparse.max() + 1e-8)                  # BM25 / max

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_emb = model.encode(docs, normalize_embeddings=True)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    dense = doc_emb @ q_emb                                    # cosine

    hybrid = alpha * sparse_n + (1 - alpha) * dense
    return sorted(enumerate(hybrid), key=lambda x: x[1], reverse=True)''', language="python")

# ===========================================================================
# Tab 5: evaluation
# ===========================================================================
with tab_ev:
    labelled = col.get("labelled_queries", {})
    if not labelled:
        st.info("This collection has no labelled queries. Choose *Everyday articles* in the sidebar to evaluate.")
    else:
        alpha_e = st.session_state.get("alpha", 0.5)
        st.write(f"Precision@k = (relevant documents in the top k) ÷ k, averaged over {len(labelled)} labelled queries. "
                 f"MAP averages the precision at every relevant document. A document with score 0 was not retrieved, "
                 f"so it never counts as a hit. Hybrid uses α = {alpha_e:.1f} from the *Hybrid search* tab.")
        per_q, summary = [], {m: [] for m in METHODS}
        cache = {q: all_rankings(q, alpha_e) for q in labelled}
        for q, rel in labelled.items():
            rel = set(rel)
            row = {"Query": q, "Relevant": len(rel)}
            for m in METHODS:
                p5, p10 = dense.precision_at_k(cache[q][m], rel, 5), dense.precision_at_k(cache[q][m], rel, 10)
                ap = dense.average_precision(cache[q][m], rel)
                summary[m].append((p5, p10, ap))
                row[f"{m} P@5"] = p5
            per_q.append(row)
        table = pd.DataFrame([{"Method": m + (f" (α = {alpha_e:.1f})" if m == "Hybrid" else ""),
                               "P@5": np.mean([x[0] for x in v]), "P@10": np.mean([x[1] for x in v]),
                               "MAP": np.mean([x[2] for x in v])} for m, v in summary.items()])
        a, b_ = st.columns([2, 3])
        with a:
            st.dataframe(table.style.format({"P@5": "{:.3f}", "P@10": "{:.3f}", "MAP": "{:.3f}"}),
                         hide_index=True, width="stretch")
            best = table.loc[table["P@5"].idxmax(), "Method"]
            st.caption(f"Best Precision@5: {best}.")
        with b_:
            curve = []
            for al in np.round(np.arange(0, 1.01, 0.1), 1):
                p5s, p10s = [], []
                for q, rel in labelled.items():
                    h = dense.hybrid(cache[q]["BM25"], cache[q]["Dense"], al, hyb_norm)
                    p5s.append(dense.precision_at_k(h, set(rel), 5))
                    p10s.append(dense.precision_at_k(h, set(rel), 10))
                curve += [{"α": al, "Metric": "P@5", "Value": np.mean(p5s)},
                          {"α": al, "Metric": "P@10", "Value": np.mean(p10s)}]
            st.altair_chart(alt.Chart(pd.DataFrame(curve)).mark_line(point=True).encode(
                x=alt.X("α:Q", title="α  (0 = dense only, 1 = BM25 only)"),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Precision"),
                color=alt.Color("Metric:N", scale=alt.Scale(range=[ACCENT, "#e8a028"])),
                tooltip=["α", "Metric", alt.Tooltip("Value:Q", format=".3f")]).properties(height=260), width="stretch")
        st.markdown("**Precision@5 per query**")
        st.dataframe(pd.DataFrame(per_q).style.format({f"{m} P@5": "{:.2f}" for m in METHODS}),
                     hide_index=True, width="stretch")
        st.caption("*blood pressure* is an exact keyword query: BM25 finds both relevant documents (with 2 relevant "
                   "documents, 0.40 is the highest possible P@5). Queries phrased in other words (*vehicle servicing*, "
                   "*staying healthy*) need dense or expanded representations. Exact matching also has traps: for "
                   "*BM25 ranking*, S1 says *rank*, not *ranking*, and without stemming BM25 misses it; for *trouble "
                   "sleeping*, BM25 matches *engine trouble* and misses *sleep*.")

st.divider()
st.caption("TF-IDF variants and BM25 are implemented from scratch in weighting.py; SPLADE and the offline term "
           "expansion in learned_sparse.py; dense retrieval, hybrid fusion, and metrics in dense.py. "
           "Offline word vectors: GloVe 6B 50d (PDDL).")
