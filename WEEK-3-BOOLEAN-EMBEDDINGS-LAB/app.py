"""
app.py: Streamlit interface for Boolean & Embeddings Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import embeddings as emb
from boolean_model import (BooleanIndex, QueryError, and_plan, evaluate, intersect_trace,
                           load_collections, tokenize)

st.set_page_config(page_title="Boolean & Embeddings Lab", page_icon="🧮", layout="wide")

COLLECTIONS = load_collections()
CUSTOM = "Custom collection"
CASE_STUDY = {"Machine learning abstracts (Boolean vs embeddings)": "neural network text processing",
              "Car repair (synonyms, general domain)": "car repair"}

st.markdown(
    """
<style>
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 6px}
.chip{font-family:"Courier New",monospace;font-size:13.5px;border:1px solid rgba(128,140,160,.45);
  border-radius:4px;padding:2px 8px;line-height:1.5}
.chip.hit{border-color:rgba(11,114,133,.8);background:rgba(11,114,133,.16);font-weight:700}
.chip.miss{opacity:.45;text-decoration:line-through}
.chip.op{border-style:dashed;color:#7a5fd0;border-color:#7a5fd0;font-weight:700}
.bits{font-family:"Courier New",monospace;font-size:15px;letter-spacing:.12em;font-weight:700}
.doc{border:1px solid rgba(128,140,160,.35);border-left:3px solid rgba(11,114,133,.8);border-radius:4px;
  padding:8px 12px;margin-bottom:8px;font-size:14px}
.doc b{font-family:"Courier New",monospace}
mark{background:#ffe680;color:#2b2400;padding:0 2px;border-radius:2px}
.callout{border-left:3px solid rgba(207,58,63,.7);padding:6px 12px;font-size:14px;margin:6px 0}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def chips(items) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip {c}">{html.escape(t)}</span>' for t, c in items) + "</div>"


def highlight(text: str, terms: set[str]) -> str:
    return re.sub(r"[^\W_]+", lambda m: f"<mark>{m.group(0)}</mark>" if m.group(0).lower() in terms
                  else m.group(0), html.escape(text))


def bits_table(rows: dict[str, list[int]], docs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=docs).T


def style_bits(df: pd.DataFrame):
    return df.style.map(lambda v: "background-color: rgba(11,114,133,.22); font-weight:700" if v == 1 else
                        "color: rgba(128,140,160,.8)")


def set_query(key: str, q: str) -> None:
    st.session_state[key] = q


# ---------------------------------------------------------------------------
# Sidebar: collection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Collection")
    name = st.selectbox("Documents to search", list(COLLECTIONS) + [CUSTOM], key="collection")
    if name == CUSTOM:
        if "custom_docs" not in st.session_state:
            st.session_state.custom_docs = ("D1: information retrieval finds relevant documents\n"
                                            "D2: a search engine builds an inverted index\n"
                                            "D3: word embeddings capture the meaning of words")
        st.text_area("One document per line, as “ID: text”", key="custom_docs", height=160)
        docs = {}
        for i, line in enumerate(st.session_state.custom_docs.splitlines(), 1):
            if line.strip():
                did, _, txt = line.partition(":") if ":" in line else (f"D{i}", "", line)
                docs[did.strip() or f"D{i}"] = txt.strip() or did.strip()
        queries = []
        st.caption("Each line is one document. Text before the first colon is the document ID.")
    else:
        col = COLLECTIONS[name]
        docs, queries = col["documents"], col["queries"]
        st.caption(col["note"])
    st.markdown("**Documents**")
    for d, t in docs.items():
        show(f'<div class="doc"><b>{html.escape(d)}</b><br>{html.escape(t)}</div>')

if not docs:
    st.warning("The collection is empty. Add at least one document in the sidebar.")
    st.stop()

idx = BooleanIndex(docs)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 3 · BOOLEAN RETRIEVAL & WORD EMBEDDINGS")
st.title("🧮 Boolean & Embeddings Lab")
st.write("Run Boolean queries on an **incidence matrix** and an **inverted index**, step by step, then see "
         "how **word embeddings** move from exact word matching to matching by meaning.")

tab_q, tab_m, tab_i, tab_e, tab_c = st.tabs(
    ["1 · Boolean query", "2 · Incidence matrix", "3 · Inverted index", "4 · Word embeddings",
     "5 · Boolean vs embeddings"])

# ===========================================================================
# Tab 1: Boolean query
# ===========================================================================
with tab_q:
    qkey = f"query::{name}"
    if qkey not in st.session_state:
        st.session_state[qkey] = queries[0] if queries else " AND ".join(idx.terms[:2])
    st.text_input("Boolean query", key=qkey,
                  help="Use AND, OR, NOT and brackets. Words next to each other are joined with AND.")
    if queries:
        cols = st.columns(len(queries))
        for c, q in zip(cols, queries):
            c.button(q, key=f"ex::{name}::{q}", on_click=set_query, args=(qkey, q), width="stretch")
    st.caption("Precedence: brackets first, then NOT, then AND, then OR.")

    try:
        res = evaluate(st.session_state[qkey], idx)
    except QueryError as e:
        st.error(f"This query cannot be read: {e}.")
        res = None

    if res:
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**Read as** (brackets show the precedence)")
            st.code(res.bracketed.replace("(", "( ").replace(")", " )") if res.steps else res.bracketed,
                    language=None)
        with b:
            st.markdown("**Postfix order** (evaluated left to right)")
            show(chips([(t, "op" if t in ("AND", "OR", "NOT") else "hit") for t in res.postfix]))
        if res.unknown_terms:
            st.warning("Not in the index, so they match no document: "
                       + ", ".join(f"“{t}”" for t in res.unknown_terms))

        st.markdown("#### Evaluation with bit vectors")
        st.caption("Each term is a row of the incidence matrix; one bit per document, in this order: "
                   + " · ".join(idx.docs))
        term_rows = {t: idx.bits(t) for t in dict.fromkeys(x for x in res.postfix if x not in ("AND", "OR", "NOT"))}
        st.dataframe(style_bits(bits_table(term_rows, idx.docs)), width="stretch")
        if res.steps:
            st.dataframe(pd.DataFrame([{
                "Step": i, "Expression": s.expression, "Bitwise operation": s.operation,
                "Result": "".join(map(str, s.bits)), "As sets": s.sets} for i, s in enumerate(res.steps, 1)]),
                hide_index=True, width="stretch")

        st.markdown(f"#### Result: <span class='bits'>{''.join(map(str, res.bits))}</span>",
                    unsafe_allow_html=True)
        show(chips([(d, "hit" if bit else "miss") for d, bit in zip(idx.docs, res.bits)]))
        positive = {t for t in res.postfix if t not in ("AND", "OR", "NOT")}
        if res.docs:
            for d in res.docs:
                show(f'<div class="doc"><b>{html.escape(d)}</b><br>{highlight(docs[d], positive)}</div>')
            st.caption("The Boolean model only says match or no match. All these documents are treated as "
                       "equally relevant: there is no ranking.")
        else:
            st.info("No document satisfies this query.")

# ===========================================================================
# Tab 2: incidence matrix
# ===========================================================================
with tab_m:
    st.write("M[t][d] = 1 if term *t* occurs in document *d*, otherwise 0. Reading a **row** gives the documents "
             "that contain a term; reading a **column** gives the terms in a document.")
    stats = idx.size_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Terms", stats["terms"])
    c2.metric("Documents", stats["documents"])
    c3.metric("Matrix cells", stats["matrix cells"])
    c4.metric("Zeros", f"{stats['zeros'] / stats['matrix cells'] * 100:.0f}%",
              help="Share of the matrix that stores nothing useful.")
    only_q = st.checkbox("Show only the terms in the current query", value=False)
    terms = idx.terms
    if only_q and res:
        terms = [t for t in dict.fromkeys(res.postfix) if t not in ("AND", "OR", "NOT")]
    matrix = pd.DataFrame({d: [idx.bits(t)[i] for t in terms] for i, d in enumerate(idx.docs)}, index=terms)
    st.dataframe(style_bits(matrix), width="stretch", height=min(40 + 35 * len(terms), 560))
    show('<div class="callout"><b>The scale problem.</b> The matrix stores a cell for every term and every '
         f'document. Here that is {stats["matrix cells"]} cells for {stats["ones (postings)"]} ones. A real '
         'collection with 1 million terms and 1 million documents would need 10¹² cells, almost all of them 0. '
         'An inverted index stores only the ones.</div>')

# ===========================================================================
# Tab 3: inverted index
# ===========================================================================
with tab_i:
    st.write("Each term points to its **posting list**: the documents that contain it, kept in sorted order. "
             "The positional form also records the frequency and positions: ⟨id, f, [positions]⟩.")
    a, b = st.columns(2)
    with a:
        st.markdown("##### Simple inverted index")
        st.dataframe(pd.DataFrame([{"Term": t, "df": idx.df(t), "Posting list": ", ".join(idx.postings(t))}
                                   for t in idx.terms]), hide_index=True, width="stretch",
                     height=min(40 + 35 * len(idx.terms), 460))
    with b:
        st.markdown("##### Positional inverted index")
        st.dataframe(pd.DataFrame([{"Term": t, "Inverted list": ", ".join(map(str, idx.index[t]))}
                                   for t in idx.terms]), hide_index=True, width="stretch",
                     height=min(40 + 35 * len(idx.terms), 460))

    st.markdown("##### AND by merging two posting lists")
    st.caption("The lecture pseudocode: walk both sorted lists with two pointers. Cost O(|p1| + |p2|).")
    defaults = [t for t in dict.fromkeys(res.postfix) if t in idx.index][:2] if res else []
    defaults += [t for t in idx.terms if t not in defaults][: 2 - len(defaults)]
    a, b = st.columns(2)
    t1 = a.selectbox("Term 1", idx.terms, index=idx.terms.index(defaults[0]), key=f"t1::{name}")
    t2 = b.selectbox("Term 2", idx.terms, index=idx.terms.index(defaults[1 if len(defaults) > 1 else 0]),
                     key=f"t2::{name}")
    p1, p2 = idx.postings(t1), idx.postings(t2)
    inter, rows = intersect_trace(p1, p2, idx.docs)
    show(f"p1 = posting({html.escape(t1)}) = [{html.escape(', '.join(p1))}] &nbsp;&nbsp; "
         f"p2 = posting({html.escape(t2)}) = [{html.escape(', '.join(p2))}]")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(f"Result: **{', '.join(inter) or 'no documents'}** · {len(rows)} comparisons "
                f"(at most |p1| + |p2| = {len(p1) + len(p2)})")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Query optimization: shortest list first")
        st.caption("For t1 AND t2 AND …, start with the rarest term so the candidate set shrinks fastest.")
        plan_terms = st.text_input("Terms to AND together", value=" ".join(idx.terms[:3]), key=f"plan::{name}")
        if tokenize(plan_terms):
            st.dataframe(pd.DataFrame(and_plan(tokenize(plan_terms), idx)), hide_index=True, width="stretch")
    with b:
        st.markdown("##### Phrase query with positions")
        st.caption("A phrase matches when its words appear at consecutive positions in the same document.")
        two = next(((idx.texts[d], d) for d in idx.docs if len(tokenize(idx.texts[d])) > 1), ("", ""))[0]
        phrase = st.text_input("Phrase", value=" ".join(tokenize(two)[:2]), key=f"phrase::{name}")
        hits = idx.phrase(phrase)
        if tokenize(phrase):
            if hits:
                for d, starts in hits:
                    st.markdown(f"**{d}**: starts at position {', '.join(map(str, starts))}")
            else:
                st.write("No document contains this exact phrase.")

# ===========================================================================
# Tab 4: word embeddings
# ===========================================================================
with tab_e:
    st.write("In the Boolean model and TF-IDF every word is its own dimension, so *car* and *automobile* are "
             "unrelated. An **embedding** gives each word a dense vector, and words with similar meanings get "
             "vectors that point in similar directions. *“You shall know a word by the company it keeps”* "
             "(Firth, 1957).")
    st.caption("Vectors: GloVe 6B, 50 dimensions, the 30,000 most frequent English words (Pennington et al., 2014).")

    def known(word: str) -> bool:
        if not emb.has(word):
            st.warning(f"“{word}” is not in the vocabulary (out of vocabulary). FastText would build it from "
                       f"character n-grams such as {', '.join(emb.char_ngrams(word)[:6])}, …")
            return False
        return True

    a, b, c = st.columns(3)
    with a:
        st.markdown("##### Similarity")
        w1 = st.text_input("Word 1", "king", key="sim1").strip().lower()
        w2 = st.text_input("Word 2", "queen", key="sim2").strip().lower()
        if w1 and w2 and known(w1) and known(w2):
            st.metric("Cosine similarity", f"{emb.similarity(w1, w2):.3f}")
        pairs = [("car", "automobile"), ("car", "vehicle"), ("car", "banana"), ("king", "queen"),
                 ("good", "bad"), ("happy", "glad")]
        st.dataframe(pd.DataFrame([{"Pair": f"{x} · {y}", "Cosine": round(emb.similarity(x, y), 3)}
                                   for x, y in pairs]), hide_index=True, width="stretch")
        st.caption("*good* and *bad* score high: words used in the same contexts end up close, even opposites.")
    with b:
        st.markdown("##### Nearest neighbors")
        nw = st.text_input("Word", "retrieval", key="nn").strip().lower()
        if nw and known(nw):
            st.dataframe(pd.DataFrame([{"Neighbor": w, "Cosine": round(s, 3)} for w, s in emb.neighbors(nw, 10)]),
                         hide_index=True, width="stretch")
    with c:
        st.markdown("##### Analogy: a − b + c")
        x = st.text_input("a", "king", key="an_a").strip().lower()
        y = st.text_input("b (subtract)", "man", key="an_b").strip().lower()
        z = st.text_input("c (add)", "woman", key="an_c").strip().lower()
        if x and y and z and all(known(w) for w in (x, y, z)):
            st.dataframe(pd.DataFrame([{"Result": w, "Cosine": round(s, 3)} for w, s in emb.analogy(x, y, z, 6)]),
                         hide_index=True, width="stretch")
        st.caption("Try paris − france + japan, or walking − walk + swim.")

    st.markdown("##### Word map")
    words_in = st.text_input(
        "Words to plot (comma-separated)",
        "king, queen, man, woman, prince, princess, car, automobile, truck, vehicle, apple, banana, mango",
        key="map_words")
    plot_words = [w.strip().lower() for w in words_in.split(",") if w.strip()]
    missing = [w for w in plot_words if not emb.has(w)]
    plot_words = [w for w in dict.fromkeys(plot_words) if emb.has(w)]
    if missing:
        st.caption("Skipped (not in the vocabulary): " + ", ".join(missing))
    if len(plot_words) >= 3:
        xy = emb.project_2d(plot_words)
        df = pd.DataFrame({"word": plot_words, "x": xy[:, 0], "y": xy[:, 1]})
        base = alt.Chart(df).encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), tooltip=["word"])
        chart = base.mark_circle(size=90, color="#0b7285") + base.mark_text(
            dx=8, align="left", fontSize=13, color="#0b7285").encode(text="word")
        st.altair_chart(chart.properties(height=380), width="stretch")
        st.caption("50 dimensions projected to 2 with PCA. Distances are approximate, but groups of related "
                   "words stay together.")

    st.markdown("##### Sparse vs dense representation")
    sw = st.text_input("Word", "car", key="sparse_word").strip().lower()
    if sw and known(sw):
        a, b = st.columns([1, 2])
        words_all, index_all, _ = emb.load()
        with a:
            st.markdown("**Sparse (one-hot)**")
            st.metric("Dimensions", f"{len(words_all):,}")
            st.metric("Non-zero values", "1")
            st.caption(f"Position {index_all[sw]:,} is 1, every other position is 0. "
                       "Every pair of different words has similarity 0.")
        with b:
            st.markdown("**Dense (embedding)**")
            vec = emb.raw_vector(sw)
            ddf = pd.DataFrame({"Dimension": np.arange(len(vec)), "Value": vec})
            bars = alt.Chart(ddf).mark_bar(color="#0b7285").encode(x=alt.X("Dimension:O", axis=alt.Axis(labels=False)),
                                                                   y="Value:Q", tooltip=["Dimension", "Value"])
            st.altair_chart(bars.properties(height=200), width="stretch")
            st.caption("All 50 values are used. Meaning is spread across the dimensions.")

    st.markdown("##### One vector per word: the limitation")
    bank = emb.neighbors("bank", 10)
    st.write("Word2Vec and GloVe give a word a single vector, whatever the context. The neighbors of **bank** "
             f"are all about money ({', '.join(w for w, _ in bank[:8])}); the *river bank* meaning is lost. "
             "Contextual models such as BERT and Sentence-BERT (Session 4) give a different vector in each sentence.")

    with st.expander("How the vectors are learned: Word2Vec, GloVe, FastText"):
        st.markdown("**Word2Vec skip-gram** predicts the context words from the center word:")
        st.latex(r"P(w_o \mid w_i) = \frac{\exp(v'_{o}{}^{\top} v_i)}{\sum_{j} \exp(v'_{j}{}^{\top} v_i)}")
        st.markdown("**Word2Vec CBOW** predicts the center word from the average of its context:")
        st.latex(r"P(w_i \mid \text{context}) = \frac{\exp(v_i^{\top} \hat{v})}{\sum_j \exp(v_j^{\top} \hat{v})}")
        st.markdown("Both are trained by minimizing the negative log-likelihood with stochastic gradient descent:")
        st.latex(r"J = -\log P(w_o \mid w_i)")
        st.markdown("**GloVe** fits the vectors to the global co-occurrence matrix X:")
        st.latex(r"J = \sum_{i,j} f(X_{ij}) \left(w_i^{\top}\tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2")
        st.markdown("**FastText** represents a word as the sum of its character n-gram vectors, so it can build "
                    "vectors for words it never saw. That helps languages with rich morphology, such as Indonesian.")

# ===========================================================================
# Tab 5: Boolean vs embeddings
# ===========================================================================
with tab_c:
    st.write("The lecture's case study: run the same query with exact Boolean matching, TF-IDF similarity, "
             "and averaged word embeddings.")
    a, b = st.columns([2, 3])
    with a:
        cs_name = st.selectbox("Collection", list(CASE_STUDY), key="cs_collection")
    cs_docs = COLLECTIONS[cs_name]["documents"]
    with b:
        cs_q = st.text_input("Query", value=CASE_STUDY[cs_name], key=f"cs_query::{cs_name}")
    st.caption(COLLECTIONS[cs_name]["note"])

    q_terms = emb.words_of(cs_q)
    cs_idx = BooleanIndex(cs_docs)
    and_docs = evaluate(" AND ".join(q_terms), cs_idx).docs if q_terms else []
    or_docs = evaluate(" OR ".join(q_terms), cs_idx).docs if q_terms else []
    tfidf = emb.tfidf_ranking(cs_q, cs_docs)
    embr = emb.embedding_ranking(cs_q, cs_docs)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Boolean")
        st.markdown(f"`{' AND '.join(q_terms)}`")
        show(chips([(d, "hit" if d in and_docs else "miss") for d in cs_docs]))
        st.markdown(f"`{' OR '.join(q_terms)}`")
        show(chips([(d, "hit" if d in or_docs else "miss") for d in cs_docs]))
        st.caption("Exact match only. AND is often too strict, OR too loose, and neither ranks.")
    with c2:
        st.markdown("##### TF-IDF + cosine")
        st.dataframe(pd.DataFrame([{"Rank": i, "Doc": d, "Score": round(s, 3)} for i, (d, s) in enumerate(tfidf, 1)]),
                     hide_index=True, width="stretch")
        st.caption("Ranks by shared words, weighted by rarity. A document with no shared word scores 0.")
    with c3:
        st.markdown("##### Averaged word embeddings")
        st.dataframe(pd.DataFrame([{"Rank": i, "Doc": d, "Score": round(s, 3)} for i, (d, s) in enumerate(embr, 1)]),
                     hide_index=True, width="stretch")
        st.caption("Each text is the mean of its word vectors. Related words count, even without overlap.")

    st.markdown("##### Documents")
    for d, t in cs_docs.items():
        show(f'<div class="doc"><b>{html.escape(d)}</b><br>{highlight(t, set(q_terms))}</div>')

    st.markdown("##### What the embeddings think the query words mean")
    nn_rows = [{"Query word": w, "Nearest neighbors in GloVe": ", ".join(x for x, _ in emb.neighbors(w, 8))}
               for w in q_terms if emb.has(w)]
    if nn_rows:
        st.dataframe(pd.DataFrame(nn_rows), hide_index=True, width="stretch")
    if cs_name.startswith("Machine learning"):
        st.info("GloVe was trained on Wikipedia and news, so *neural* sits near *brain* and *neurons*, and "
                "*processing* near *materials* and *equipment*. The embedding ranking gives Doc 5 a score that "
                "TF-IDF cannot (TF-IDF gives it 0), but a general-purpose model does not know that *text "
                "processing* means natural language processing. Averaging word vectors also blurs a document "
                "into one point. Both problems lead to the contextual, sentence-level embeddings of Session 4.")
    else:
        st.info("None of the relevant documents contains *car* or *repair*. Boolean and TF-IDF can only find "
                "the insurance article, which shares the word *car*. Embeddings know that *automobile*, "
                "*vehicle*, *mechanics*, and *fixing* are related to the query and rank the maintenance and "
                "repair documents first.")

st.divider()
st.caption("Boolean model, incidence matrix, and inverted index are implemented from scratch in boolean_model.py. "
           "Word vectors: GloVe 6B 50d (PDDL), 30,000 most frequent words.")
