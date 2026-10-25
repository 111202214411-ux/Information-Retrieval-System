"""
app.py: Streamlit interface for IR Evaluation Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import metrics as M
import rag_eval as R
import systems as S

st.set_page_config(page_title="IR Evaluation Lab", page_icon="📏", layout="wide")

ACCENT = "#0b7285"
AMBER = "#e8a028"
RED = "#cf3a3f"
DATA = json.loads((Path(__file__).parent / "data" / "collection.json").read_text(encoding="utf-8"))
DOCS = {d["id"]: d["text"] for d in DATA["documents"]}
DOMAIN_OF = {d["id"]: d["domain"] for d in DATA["documents"]}
TOPICS = {t["id"]: t for t in DATA["topics"]}
SYS_COLORS = alt.Scale(domain=S.SYSTEMS, range=["#9aa5b1", "#5f8fbf", ACCENT, AMBER, RED])

st.markdown(
    """
<style>
.ct{border-collapse:collapse;font-size:14px;margin:4px 0 8px}
.ct th,.ct td{border:1px solid rgba(128,140,160,.45);padding:7px 12px;text-align:left;vertical-align:top}
.ct th{background:rgba(128,140,160,.12);font-weight:600}
.ct td.tp{background:rgba(11,114,133,.18)} .ct td.fp{background:rgba(232,160,40,.18)}
.ct td.fn{background:rgba(207,58,63,.14)} .ct td.tn{background:rgba(128,140,160,.08)}
.ct small{opacity:.8;font-family:"Courier New",monospace}
.need{border-left:3px solid rgba(11,114,133,.75);padding:6px 12px;margin:4px 0 10px;font-size:14.5px}
.doc{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:7px 11px;margin-bottom:7px;font-size:14px}
.doc b{font-family:"Courier New",monospace}
.g3{border-left:4px solid #0b7285}.g2{border-left:4px solid #5aa9b5}.g1{border-left:4px solid #b9d9de}
.g0{border-left:4px solid rgba(128,140,160,.35)}
.formula{font-family:"Courier New",monospace;font-size:14px;padding:6px 10px;border-radius:4px;
  background:rgba(128,140,160,.10);display:inline-block;margin:4px 0}
.sent{padding:3px 8px;border-radius:3px;margin:3px 0;font-size:14px}
.sent.ok{background:rgba(11,114,133,.14)} .sent.no{background:rgba(207,58,63,.14)}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def heat(df: pd.DataFrame, cols: list[str], digits: int = 3, extra: dict | None = None):
    def shade(col: pd.Series):
        lo, hi = col.min(), col.max()
        return [f"background-color: rgba(11,114,133,{0.06 + 0.44 * ((v - lo) / (hi - lo) if hi > lo else 0):.2f})"
                for v in col]
    return df.style.format({**{c: f"{{:.{digits}f}}" for c in cols}, **(extra or {})}).apply(shade, subset=cols)


def topic_label(tid: str) -> str:
    return f"{tid} · {TOPICS[tid]['query']}"


# ---------------------------------------------------------------------------
# Cached work
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading all-MiniLM-L6-v2 (first run downloads about 90 MB)…")
def get_encoder(prefer: bool):
    return S.load_dense_encoder(prefer)


@st.cache_data(show_spinner=False)
def encode(prefer: bool, texts: tuple[str, ...]) -> np.ndarray:
    return get_encoder(prefer)[0].encode(list(texts))


@st.cache_data(show_spinner="Running the five systems on the 15 topics…")
def get_runs(prefer: bool) -> S.Runs:
    enc = get_encoder(prefer)[0]
    return S.run_all(DOCS, {t: TOPICS[t]["query"] for t in TOPICS}, enc)     # times the document encoding too


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Relevance judgments")
    judge = st.radio("Qrels", list(DATA["qrels"]), key="judge",
                     help="Two assessors judged the same topics. Tab 5 compares them.")
    threshold = st.radio("A document counts as relevant when its grade is", [1, 2], key="threshold",
                         format_func=lambda g: f"≥ {g}  ({'marginal or better' if g == 1 else 'relevant or better'})",
                         help="Grades: 0 not relevant, 1 marginal, 2 relevant, 3 highly relevant. Binary measures "
                              "(P, R, AP, MRR) need a cut; nDCG uses the grades directly.")
    gain_kind = st.radio("nDCG gain", ["linear", "exp"], key="gain",
                         format_func={"linear": "rel (notebook code, trec_eval)",
                                      "exp": "2^rel − 1 (slide formula)"}.get)
    st.header("Dense model")
    dense_choice = st.radio("Bi-encoder", ["all-MiniLM-L6-v2", "Offline word vectors"], key="dense_choice",
                            help="MiniLM downloads about 90 MB on first use. The offline option averages the "
                                 "bundled GloVe vectors.")
    st.header("Test collection")
    st.caption(f"{len(DOCS)} documents in 3 domains, {len(TOPICS)} topics with an information need, graded "
               f"judgments from two assessors. {DATA['pool_note']}")

prefer = dense_choice == "all-MiniLM-L6-v2"
enc, enc_why = get_encoder(prefer)
if enc_why:
    st.warning(f"all-MiniLM-L6-v2 is unavailable because {enc_why}. The Dense and Hybrid systems use averaged GloVe "
               f"word vectors instead, which are much weaker but need no download.")
RUNS = get_runs(prefer)
QRELS = DATA["qrels"][judge]


def ranked_ids(system: str, tid: str) -> list[str]:
    return [d for d, _ in RUNS.rankings[system][tid]]


def per_topic(system: str, qrels_all: dict = None) -> list[dict]:
    q = qrels_all or QRELS
    return [M.evaluate(ranked_ids(system, t), q[t], threshold, gain_kind) for t in TOPICS]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 7 · EVALUATING IR MODELS: CLASSIC & MODERN METRICS")
st.title("📏 IR Evaluation Lab")
st.write("How do we know a search engine returns *relevant* results? This lab computes every measure from the "
         "lecture step by step: **precision, recall, F1 and accuracy** for unranked sets; **P@k, R@k, AP, MAP and "
         "MRR** for rankings; **nDCG** for graded relevance. It then evaluates five systems on a TREC-style test "
         "collection, measures **agreement between assessors** (Cohen's κ), and ends with the modern view: a "
         "**BEIR-style** per-domain comparison and **RAGAS-style** metrics for retrieval-augmented generation.")

tab_set, tab_rank, tab_ndcg, tab_cmp, tab_judge, tab_modern = st.tabs(
    ["1 · Precision & recall", "2 · Ranked metrics", "3 · nDCG", "4 · Comparing systems",
     "5 · Relevance judgments", "6 · BEIR & RAGAS"])

# ===========================================================================
# Tab 1: set-based evaluation
# ===========================================================================
with tab_set:
    st.markdown("Unranked retrieval returns a **set**. Each document is either retrieved or not, and relevant or "
                "not, which gives the contingency table from the lecture.")
    a, b, c = st.columns([3, 2, 2])
    tid = a.selectbox("Topic", list(TOPICS), format_func=topic_label, key="t1_topic")
    system = b.selectbox("System", S.SYSTEMS, index=2, key="t1_sys")
    is_bool = system == "Boolean AND"
    k = c.slider("Retrieved = the top k", 1, 20, 5, key="t1_k", disabled=is_bool,
                 help="A ranking becomes a set by cutting it at k. Boolean AND already returns a set.")
    show(f'<div class="need"><b>Information need:</b> {esc(TOPICS[tid]["need"])}<br>'
         f'<small>Query: <code>{esc(TOPICS[tid]["query"])}</code> · relevance is judged against the need, not the '
         f'query words.</small></div>')
    full = ranked_ids(system, tid)
    retrieved = set(full if is_bool else full[:k])
    rel = M.relevant_set(QRELS[tid], threshold)
    sc = M.set_scores(retrieved, rel, len(DOCS))

    def cell(ids: set[str], limit: int = 12) -> str:
        lst = sorted(ids)
        more = f" … +{len(lst) - limit}" if len(lst) > limit else ""
        return "<small>" + (", ".join(lst[:limit]) + more if lst else "—") + "</small>"

    l, r = st.columns([3, 2])
    with l:
        show(f"""<table class="ct">
<tr><th></th><th>Relevant</th><th>Not relevant</th><th>Total</th></tr>
<tr><th>Retrieved</th><td class="tp"><b>tp = {sc.tp}</b><br>{cell(retrieved & rel)}</td>
<td class="fp"><b>fp = {sc.fp}</b><br>{cell(retrieved - rel)}</td><td>{sc.tp + sc.fp}</td></tr>
<tr><th>Not retrieved</th><td class="fn"><b>fn = {sc.fn}</b><br>{cell(rel - retrieved)}</td>
<td class="tn"><b>tn = {sc.tn}</b><br><small>the other {sc.tn} documents</small></td><td>{sc.fn + sc.tn}</td></tr>
<tr><th>Total</th><td>{sc.tp + sc.fn}</td><td>{sc.fp + sc.tn}</td><td>{len(DOCS)}</td></tr></table>""")
        if is_bool and not retrieved:
            st.caption("Boolean AND requires every query word in the document, and no document contains all of "
                       "them, so nothing is retrieved: recall 0.")
    with r:
        beta = st.slider("β for Fβ (β > 1 weighs recall more)", 0.25, 4.0, 2.0, 0.25, key="beta")
        m1, m2 = st.columns(2)
        m1.metric("Precision", f"{sc.precision:.3f}", help="tp / (tp + fp)")
        m2.metric("Recall", f"{sc.recall:.3f}", help="tp / (tp + fn)")
        m1.metric("F1", f"{sc.f1:.3f}", help="2PR / (P + R)")
        m2.metric(f"F{beta:g}", f"{sc.f_beta(beta):.3f}", help="(1 + β²)PR / (β²P + R)")
        m1.metric("Accuracy", f"{sc.accuracy:.3f}", help="(tp + tn) / N")
        st.markdown(f'<span class="formula">P = {sc.tp}/{sc.tp + sc.fp or 1} · R = {sc.tp}/{sc.tp + sc.fn or 1}'
                    f'</span>', unsafe_allow_html=True)

    if not is_bool and full:
        curve = []
        for kk in range(1, 21):
            s2 = M.set_scores(set(full[:kk]), rel, len(DOCS))
            curve += [{"k": kk, "Measure": m_, "Value": v} for m_, v in
                      [("Precision", s2.precision), ("Recall", s2.recall), ("F1", s2.f1), ("Accuracy", s2.accuracy)]]
        base = alt.Chart(pd.DataFrame(curve)).mark_line(point=True).encode(
            x=alt.X("k:Q", title="documents retrieved (k)"), y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Precision", "Recall", "F1", "Accuracy"],
                                                         range=[ACCENT, AMBER, RED, "#9aa5b1"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["k", "Measure", alt.Tooltip("Value:Q", format=".3f")])
        rule = alt.Chart(pd.DataFrame({"k": [k]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="k:Q")
        st.altair_chart((base + rule).properties(height=380, title="The precision–recall trade-off as k grows"),
                        width="stretch")
        note = (f" {system} returns only {len(full)} documents for this query (those containing a query word), so "
                f"the set stops growing after k = {len(full)}." if len(full) < 20 else "")
        st.caption("Retrieving more documents can only raise recall, while precision usually falls. Accuracy stays "
                   "near 1 whatever k is, because almost every document is a true negative." + note)

    st.subheader("Why not accuracy?")
    st.write("The lecture's example: a collection of 27 × 17 = 459 documents, one relevant document that is not "
             "retrieved (fn) and one retrieved document that is not relevant (fp).")
    g1, g2 = st.columns([3, 2])
    with g1:
        cells = [{"x": i % 27, "y": i // 27, "kind": "tn"} for i in range(459)]
        cells[5 * 27 + 20]["kind"] = "fp: retrieved, not relevant"
        cells[12 * 27 + 8]["kind"] = "fn: relevant, not retrieved"
        for cc in cells:
            if cc["kind"] == "tn":
                cc["kind"] = "tn: neither"
        st.altair_chart(alt.Chart(pd.DataFrame(cells)).mark_rect(stroke="white", strokeWidth=1).encode(
            x=alt.X("x:O", axis=None), y=alt.Y("y:O", axis=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["tn: neither", "fp: retrieved, not relevant",
                                                              "fn: relevant, not retrieved"],
                                                      range=["#f0a58a", "#1d1d1d", "#4caf50"]),
                            legend=alt.Legend(orient="bottom", title=None))).properties(height=320), width="stretch")
    with g2:
        demo = M.SetScores(0, 1, 1, 457)
        st.markdown(f"tp = 0, fp = 1, fn = 1, tn = 457\n\n"
                    f"- **Accuracy** = (0 + 457) / 459 = **{demo.accuracy:.3f}**\n"
                    f"- **Precision** = 0 / 1 = **0**\n- **Recall** = 0 / 1 = **0**")
        nothing = M.set_scores(set(), rel, len(DOCS))
        st.markdown(f"On this collection, a system that **retrieves nothing** for *{esc(TOPICS[tid]['query'])}* "
                    f"scores accuracy {nothing.accuracy:.3f}. Relevant documents are a tiny minority, so accuracy "
                    f"rewards doing nothing; precision and recall do not.")

# ===========================================================================
# Tab 2: ranked evaluation
# ===========================================================================
with tab_rank:
    nb = DATA["notebook"]
    st.markdown("A ranking is judged position by position. The lecture notebook's example: relevant documents "
                "**D1, D3, D5**, ranking **D1 … D7**. Tick or untick *Relevant* to change it.")
    l, r = st.columns([2, 3])
    with l:
        base_df = pd.DataFrame({"Rank": range(1, len(nb["retrieved"]) + 1), "Document": nb["retrieved"],
                                "Relevant": [d in nb["relevant"] for d in nb["retrieved"]]})
        ed = st.data_editor(base_df, hide_index=True, width="stretch", key="t2_editor",
                            disabled=["Rank", "Document"],
                            column_config={"Relevant": st.column_config.CheckboxColumn("Relevant")})
        missing = st.number_input("Relevant documents that are not in the ranking", 0, 10, 0, key="t2_missing",
                                  help="They count in |Rel| (recall and AP) but can never be found.")
    rels = list(ed["Relevant"])
    docs_ = list(ed["Document"])
    rel_set = {d for d, x in zip(docs_, rels) if x}
    n_rel = len(rel_set) + missing
    rel_all = rel_set | {f"(missing {i + 1})" for i in range(missing)}
    rows, hits, terms = [], 0, []
    for i, (d, x) in enumerate(zip(docs_, rels), 1):
        hits += bool(x)
        p = hits / i
        rows.append({"k": i, "Document": d, "rel(k)": int(x), "Relevant so far": hits, "P@k": p,
                     "R@k": hits / n_rel if n_rel else 0.0, "P@k × rel(k)": p * x})
        if x:
            terms.append(f"{hits}/{i}")
    tbl = pd.DataFrame(rows)
    ap = M.average_precision(docs_, rel_all)
    with r:
        st.dataframe(heat(tbl, ["P@k", "R@k", "P@k × rel(k)"]), hide_index=True, width="stretch")
    st.markdown(f'<span class="formula">AP = (1/|Rel|) Σ P@k · rel(k) = (1/{n_rel}) × ({" + ".join(terms) or "0"}) '
                f'= <b>{ap:.4f}</b></span>', unsafe_allow_html=True)
    rr = M.reciprocal_rank(docs_, rel_all)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average precision", f"{ap:.4f}")
    c2.metric("Reciprocal rank", f"{rr:.3f}")
    c3.metric("R-precision", f"{M.r_precision(docs_, rel_all):.3f}", help="P@|Rel|")
    c4.metric("|Rel|", n_rel)
    st.caption("The notebook prints " + ", ".join(
        f"P@{kk}={M.precision_at_k(docs_, rel_all, kk):.3f}, R@{kk}={M.recall_at_k(docs_, rel_all, kk):.3f}"
        for kk in (1, 3, 5, 7)) + f" and Average Precision {ap:.4f}. With the original ticks these are its numbers "
        f"(ir_measures gives the same AP, 0.7556).")

    p1, p2 = st.columns(2)
    with p1:
        pts = M.pr_points(docs_, rel_all)
        raw = pd.DataFrame([{"Recall": rc, "Precision": pr, "Kind": "after each document"} for rc, pr in pts])
        interp = pd.DataFrame([{"Recall": rc, "Precision": pr, "Kind": "interpolated (11 points)"}
                               for rc, pr in zip(np.linspace(0, 1, 11), M.interpolated_11pt(docs_, rel_all))])
        ch = alt.Chart(pd.concat([raw, interp])).mark_line(point=True).encode(
            x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("Kind:N", scale=alt.Scale(range=["#9aa5b1", ACCENT]), legend=alt.Legend(orient="bottom")),
            strokeDash=alt.StrokeDash("Kind:N", legend=None),
            tooltip=["Kind", alt.Tooltip("Recall:Q", format=".2f"), alt.Tooltip("Precision:Q", format=".3f")])
        st.altair_chart(ch.properties(height=400, title="Precision–recall curve of this ranking"), width="stretch")
        st.caption("Interpolated precision at recall r is the best precision at any recall ≥ r, which turns the "
                   "saw-tooth into the falling curve shown in the slides.")
    with p2:
        st.markdown("**MRR over several queries**")
        st.write("The rank of the first relevant document for three queries (0 = none found):")
        mr = st.data_editor(pd.DataFrame({"Query": ["q1", "q2", "q3"], "First relevant at rank": [1, 3, 0]}),
                            hide_index=True, width="stretch", key="t2_mrr", disabled=["Query"],
                            column_config={"First relevant at rank": st.column_config.NumberColumn(min_value=0,
                                                                                                    max_value=100,
                                                                                                    step=1)})
        rrs = [1 / int(x) if x and x > 0 else 0.0 for x in mr["First relevant at rank"]]
        st.markdown(f'<span class="formula">MRR = ({" + ".join(f"{v:.3f}" for v in rrs)}) / {len(rrs)} = '
                    f'<b>{np.mean(rrs):.3f}</b></span>', unsafe_allow_html=True)
        st.caption("MRR only looks at the first relevant result, which suits question answering and navigational "
                   "search, where one good answer is enough.")
    with st.expander("About the precision–recall chart in the notebook"):
        st.write("The notebook's chart draws two hard-coded lists of precision values, labelled *BM25+Dense "
                 "(MAP=0.62)* and *BM25 only (MAP=0.35)*. They are illustrative numbers, not the output of a "
                 "system; the mean of their 11 points is 0.637 and 0.386. Tab 4 draws the same kind of chart from "
                 "real runs of BM25 and the hybrid on this lab's test collection.")

# ===========================================================================
# Tab 3: nDCG
# ===========================================================================
with tab_ndcg:
    st.markdown(r"""Binary measures treat a perfect answer and a marginal one alike. **nDCG** uses graded
relevance and discounts lower positions:
$\text{DCG@k} = \sum_{i=1}^{k} \frac{gain(rel_i)}{\log_2(i+1)}$, $\;\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$.""")
    l, r = st.columns([2, 3])
    with l:
        gdf = st.data_editor(pd.DataFrame({"Rank": range(1, 8), "Grade": DATA["notebook"]["grades"]}),
                             hide_index=True, width="stretch", key="t3_grades", disabled=["Rank"], num_rows="fixed",
                             column_config={"Grade": st.column_config.NumberColumn(min_value=0, max_value=3, step=1)})
        extra_txt = st.text_input("Grades of relevant documents that were not retrieved (comma-separated)", "",
                                  key="t3_extra", help="The notebook builds the ideal ranking from the retrieved "
                                                       "list only. Standard nDCG (trec_eval, ir_measures) uses every "
                                                       "judged document.")
        kk = st.slider("k", 1, 7, 5, key="t3_k")
    grades = [int(x or 0) for x in gdf["Grade"]]
    try:
        extra = [int(x) for x in extra_txt.replace(" ", "").split(",") if x]
    except ValueError:
        extra = []
        st.warning("Enter whole numbers separated by commas, for example 2, 1.")
    rows, cum_l, cum_e = [], 0.0, 0.0
    for i, g in enumerate(grades, 1):
        disc = math.log2(i + 1)
        cum_l += g / disc if i <= kk else 0
        cum_e += (2 ** g - 1) / disc if i <= kk else 0
        rows.append({"Rank i": i, "Grade": g, "log₂(i+1)": disc, "rel / log₂(i+1)": g / disc,
                     "(2^rel − 1) / log₂(i+1)": (2 ** g - 1) / disc, "In top k": "✓" if i <= kk else ""})
    with r:
        st.dataframe(heat(pd.DataFrame(rows), ["rel / log₂(i+1)", "(2^rel − 1) / log₂(i+1)"],
                          extra={"log₂(i+1)": "{:.3f}"}), hide_index=True, width="stretch")
    ideal = sorted(grades + extra, reverse=True)
    res = {}
    for kind in ("linear", "exp"):
        res[kind] = (M.dcg(grades, kk, kind), M.dcg(ideal, kk, kind), M.ndcg_from_grades(grades, kk, kind, extra))
    m1, m2 = st.columns(2)
    for col_, kind, title in [(m1, "linear", "gain = rel (notebook code)"), (m2, "exp", "gain = 2^rel − 1 (slide)")]:
        d, i_, n_ = res[kind]
        with col_:
            st.metric(f"nDCG@{kk}, {title}", f"{n_:.4f}")
            st.markdown(f'<span class="formula">DCG = {d:.4f} · IDCG = {i_:.4f} (ideal order {ideal[:kk]})</span>',
                        unsafe_allow_html=True)
    curve = [{"k": j, "Gain": g_, "nDCG": M.ndcg_from_grades(grades, j, kind, extra)}
             for j in range(1, len(grades) + 1) for kind, g_ in [("linear", "rel"), ("exp", "2^rel − 1")]]
    st.altair_chart(alt.Chart(pd.DataFrame(curve)).mark_line(point=True).encode(
        x="k:Q", y=alt.Y("nDCG:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Gain:N", scale=alt.Scale(range=[ACCENT, AMBER]), legend=alt.Legend(orient="bottom")),
        tooltip=["k", "Gain", alt.Tooltip("nDCG:Q", format=".4f")]).properties(height=340, title="nDCG@k for every k"),
        width="stretch")
    st.info("**The notebook uses two different gains.** Its formula cell (and the slide) say 2^rel − 1, but its "
            "`ndcg_at_k` function uses `rel` directly. For the example, the printed nDCG@5 = 0.9212 is the linear "
            "gain; the formula would give 0.9461. Both are in use: trec_eval and ir_measures default to the linear "
            "gain, while the exponential gain (Burges et al., 2005) rewards highly relevant documents more. The "
            "sidebar chooses which one the other tabs use.")

    st.subheader("Checking the slide's ir_measures example")
    ex = DATA["slide_example"]
    run_ids = [d for d, _ in ex["run"]]
    rel_ex = M.relevant_set(ex["qrels"])
    computed = {"nDCG@10": M.ndcg_at_k(run_ids, ex["qrels"], 10), "MAP": M.average_precision(run_ids, rel_ex),
                "P@5": M.precision_at_k(run_ids, rel_ex, 5), "R@5": M.recall_at_k(run_ids, rel_ex, 5)}
    e1, e2 = st.columns([2, 3])
    with e1:
        st.dataframe(pd.DataFrame([{"Rank": i + 1, "Document": d, "Score": s, "Grade": ex["qrels"].get(d, 0)}
                                   for i, (d, s) in enumerate(ex["run"])]), hide_index=True, width="stretch")
    with e2:
        st.dataframe(pd.DataFrame([{"Measure": m_, "Printed on the slide": ex["printed"][m_], "Computed": v,
                                    "Match": "✓" if abs(v - ex["printed"][m_]) < 5e-5 else "✗"}
                                   for m_, v in computed.items()]).style.format(
            {"Printed on the slide": "{:.4f}", "Computed": "{:.4f}"}), hide_index=True, width="stretch")
        st.markdown(f"MAP = (1/1 + 2/3 + 3/5) / 3 = **{computed['MAP']:.4f}**; nDCG@10 = (2/log₂2 + 1/log₂4 + "
                    f"1/log₂6) / (2/log₂2 + 1/log₂3 + 1/log₂4) = **{computed['nDCG@10']:.4f}**. Running the slide's "
                    f"code with ir_measures prints these values too, so the MAP and nDCG@10 on the slide are not "
                    f"what the code outputs. P@5 and R@5 are correct.")

# ===========================================================================
# Tab 4: comparing systems
# ===========================================================================
with tab_cmp:
    st.markdown("A **test collection** has three parts: documents, topics (information needs), and relevance "
                "judgments (qrels). Five systems are run on all 15 topics and compared, TREC style.")
    with st.expander("The 15 topics and their judgments"):
        st.dataframe(pd.DataFrame([{"Topic": t, "Domain": TOPICS[t]["domain"], "Query": TOPICS[t]["query"],
                                    "Information need": TOPICS[t]["need"],
                                    "Judged relevant": ", ".join(f"{d} ({g})" for d, g in
                                                                 sorted(QRELS[t].items(), key=lambda x: -x[1]))}
                                   for t in TOPICS]), hide_index=True, width="stretch")
    per = {s: per_topic(s) for s in S.SYSTEMS}
    cols = ["P@5", "P@10", "R@10", "R-Prec", "AP", "RR", "nDCG@10"]
    table = pd.DataFrame([{"System": s, **{("MAP" if c_ == "AP" else "MRR" if c_ == "RR" else c_):
                                            np.mean([x[c_] for x in per[s]]) for c_ in cols},
                           "Query latency (ms)": RUNS.latency_ms[s], "Index build (ms)": RUNS.index_ms[s]}
                          for s in S.SYSTEMS])
    mcols = ["P@5", "P@10", "R@10", "R-Prec", "MAP", "MRR", "nDCG@10"]
    st.dataframe(heat(table, mcols, extra={"Query latency (ms)": "{:.2f}", "Index build (ms)": "{:.1f}"}),
                 hide_index=True, width="stretch")
    ranks = table.set_index("System")[mcols].rank(ascending=False, method="min").astype(int)
    st.caption(f"Means over {len(TOPICS)} topics, judgments: {judge}, relevant = grade ≥ {threshold}, nDCG gain: "
               f"{'rel' if gain_kind == 'linear' else '2^rel − 1'}, dense model: {enc.name}. Boolean AND returns an "
               f"unranked set, listed in document-id order. Latency and index time are the other search-engine "
               f"measures from the lecture; they are wall-clock times on this machine.")

    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Precision–recall curves (11-point interpolated, averaged over topics)", S.SYSTEMS,
                                default=["BM25", "Hybrid (RRF)"], key="t4_pr")
        rows = []
        for s in chosen:
            ip = np.mean([M.interpolated_11pt(ranked_ids(s, t), M.relevant_set(QRELS[t], threshold))
                          for t in TOPICS], axis=0)
            lab = f"{s} (MAP={table.loc[table.System == s, 'MAP'].iloc[0]:.2f})"
            rows += [{"Recall": rc, "Precision": p, "System": lab} for rc, p in zip(np.linspace(0, 1, 11), ip)]
        if rows:
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
                x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1.05])),
                color=alt.Color("System:N", legend=alt.Legend(orient="bottom", columns=2, labelLimit=260)),
                tooltip=["System", "Recall", alt.Tooltip("Precision:Q", format=".3f")]).properties(height=440),
                width="stretch")
    with c2:
        st.markdown("**Rank of each system under each measure** (1 = best)")
        short_names = {"Boolean AND": "Boolean", "Hybrid (RRF)": "Hybrid"}
        st.dataframe(ranks.T.rename(columns=short_names).rename_axis("Measure").reset_index(), hide_index=True,
                     width="stretch")
        orders = {m_: tuple(ranks[m_].sort_values(kind="stable").index) for m_ in mcols}
        n_orders = len(set(orders.values()))
        winners = sorted({s_ for m_ in mcols for s_ in ranks.index[ranks[m_] == 1]})
        msg = (f"The measures produce {n_orders} different system orders" if n_orders > 1
               else "All measures produce the same system order here")
        msg += (f", and {', '.join(winners)} come{'s' if len(winners) == 1 else ''} first on at least one measure. "
                if winners else ". ")
        st.caption(msg + "The lecture's last advice is to *always use multiple metrics*: P@5 looks at the top of "
                   "the list, R@10 at coverage, MRR at the first hit, nDCG at graded order, and ties count as equal "
                   "ranks.")

    st.subheader("Per topic")
    metric_pt = st.selectbox("Measure", cols, index=4, format_func=lambda c_: {"AP": "AP", "RR": "RR"}.get(c_, c_),
                             key="t4_metric")
    pt = pd.DataFrame([{"Topic": topic_label(t), "Domain": TOPICS[t]["domain"],
                        **{s: per[s][i][metric_pt] for s in S.SYSTEMS}} for i, t in enumerate(TOPICS)])
    st.dataframe(heat(pt, S.SYSTEMS, 2), hide_index=True, width="stretch", height=3 + 35 * (len(pt) + 1))

    st.subheader("Is the difference real? A paired randomization test")
    s1, s2, s3 = st.columns(3)
    sys_a = s1.selectbox("System A", S.SYSTEMS, index=4, key="t4_a")
    sys_b = s2.selectbox("System B", S.SYSTEMS, index=2, key="t4_b")
    m_sig = s3.selectbox("Per-topic measure", cols, index=4, key="t4_sig")
    va = [x[m_sig] for x in per[sys_a]]
    vb = [x[m_sig] for x in per[sys_b]]
    p_val = M.randomization_test(va, vb)
    wins = sum(x > y + 1e-12 for x, y in zip(va, vb))
    losses = sum(y > x + 1e-12 for x, y in zip(va, vb))
    d1, d2 = st.columns([2, 3])
    with d1:
        st.metric(f"Mean {m_sig}: {sys_a} − {sys_b}", f"{np.mean(va) - np.mean(vb):+.3f}")
        st.metric("p-value (10,000 random sign flips)", f"{p_val:.4f}")
        st.write(f"{sys_a} wins on {wins} topics, loses on {losses}, ties on {len(va) - wins - losses}. "
                 + ("With p < 0.05 the difference is unlikely to be luck." if p_val < 0.05 else
                    "With p ≥ 0.05 this many topics cannot rule out luck."))
    with d2:
        diff = pd.DataFrame({"Topic": list(TOPICS), "Difference": np.array(va) - np.array(vb)})
        st.altair_chart(alt.Chart(diff).mark_bar().encode(
            x=alt.X("Topic:N", sort=None), y=alt.Y("Difference:Q", title=f"{m_sig}, A − B"),
            color=alt.condition(alt.datum.Difference >= 0, alt.value(ACCENT), alt.value(RED)),
            tooltip=["Topic", alt.Tooltip("Difference:Q", format="+.3f")]).properties(height=280), width="stretch")
        st.caption("If A and B were equally good, each topic's difference would be as likely positive as negative. "
                   "The test flips the signs at random 10,000 times and counts how often the mean difference is at "
                   "least as large as the one observed.")

    qrels_txt = "\n".join(f"{t} 0 {d} {g}" for t in TOPICS for d, g in sorted(QRELS[t].items()))
    st.subheader("Export for trec_eval or ir_measures")
    dl0, dl1, dl2, _ = st.columns([1.3, 1, 1, 1.2])
    run_sys = dl0.selectbox("Run of system", S.SYSTEMS, index=2, key="dl_sys", label_visibility="collapsed")
    tag = run_sys.replace(" ", "_").replace("(", "").replace(")", "")
    run_txt = "\n".join(f"{t} Q0 {d} {i} {sc_:.6f} {tag}" for t in TOPICS
                        for i, (d, sc_) in enumerate(RUNS.rankings[run_sys][t], 1))
    dl1.download_button("Download qrels", qrels_txt, "qrels.txt", key="dl_qrels")
    dl2.download_button(f"Download run", run_txt, f"run_{tag}.txt", key="dl_runs")
    st.caption(f"TREC formats: qrels lines are `topic 0 document grade` ({judge}); run lines are "
               f"`topic Q0 document rank score tag`. Then, for example: "
               f"`ir_measures qrels.txt run_{tag}.txt nDCG@10 AP P@5`.")

# ===========================================================================
# Tab 5: relevance judgments
# ===========================================================================
with tab_judge:
    st.markdown("Relevance is a human judgment, and humans disagree. **Cohen's κ** measures agreement beyond "
                "chance: $\\kappa = \\frac{P(A) - P(E)}{1 - P(E)}$, where P(A) is the observed agreement and P(E) "
                "the agreement expected by chance.")
    st.subheader("Cohen's κ on a 2 × 2 example")
    st.markdown("From Manning et al. (2008), Table 8.2. Change the counts:")
    k1, k2, k3, k4 = st.columns(4)
    yy = k1.number_input("Both say relevant", 0, 10000, 300, key="k_yy")
    yn = k2.number_input("Judge 1 yes, judge 2 no", 0, 10000, 20, key="k_yn")
    ny = k3.number_input("Judge 1 no, judge 2 yes", 0, 10000, 10, key="k_ny")
    nn = k4.number_input("Both say not relevant", 0, 10000, 70, key="k_nn")
    kp = M.Kappa(yy, yn, ny, nn)

    def verdict(x: float) -> str:
        return ("good agreement" if x > 0.8 else "fair agreement" if x >= 0.67 else
                "dubious, too low to draw conclusions")

    show(f'<span class="formula">P(A) = ({yy} + {nn}) / {kp.n} = {kp.p_agree:.3f} · P(E) = {kp.p_chance:.3f} · '
         f'κ = ({kp.p_agree:.3f} − {kp.p_chance:.3f}) / (1 − {kp.p_chance:.3f}) = <b>{kp.kappa:.3f}</b></span>')
    st.caption(f"κ = {kp.kappa:.3f}: {verdict(kp.kappa)} (the rule of thumb in Manning et al.: above 0.8 good, "
               f"0.67 to 0.8 fair, below 0.67 dubious). With the book's counts P(E) = 0.665 and κ = 0.776.")

    st.subheader("The two assessors of this test collection")
    pairs = [(t, d) for t in TOPICS for d in DOCS if DOMAIN_OF[d] == TOPICS[t]["domain"]]
    qa, qb = DATA["qrels"]["Judge A"], DATA["qrels"]["Judge B"]
    la = [qa[t].get(d, 0) >= threshold for t, d in pairs]
    lb = [qb[t].get(d, 0) >= threshold for t, d in pairs]
    kj = M.kappa_from_labels(la, lb)
    j1, j2 = st.columns([2, 3])
    with j1:
        show(f"""<table class="ct"><tr><th></th><th>B: relevant</th><th>B: not relevant</th></tr>
<tr><th>A: relevant</th><td class="tp">{kj.both_yes}</td><td class="fp">{kj.a_yes_b_no}</td></tr>
<tr><th>A: not relevant</th><td class="fn">{kj.a_no_b_yes}</td><td class="tn">{kj.both_no}</td></tr></table>""")
        st.metric("Cohen's κ", f"{kj.kappa:.3f}", help=f"P(A) = {kj.p_agree:.3f}, P(E) = {kj.p_chance:.3f}")
        st.caption(f"{kj.n} judgments (each topic × the 15 documents of its domain), relevant = grade ≥ "
                   f"{threshold}. {verdict(kj.kappa).capitalize()}.")
        dis = [{"Topic": t, "Document": d, "A": qa[t].get(d, 0), "B": qb[t].get(d, 0)}
               for t, d in pairs if qa[t].get(d, 0) != qb[t].get(d, 0)]
        st.markdown(f"**{len(dis)} graded disagreements**")
        st.dataframe(pd.DataFrame(dis), hide_index=True, width="stretch", height=250)
    with j2:
        per_a = {s: np.mean([x["AP"] for x in per_topic(s, qa)]) for s in S.SYSTEMS}
        per_b = {s: np.mean([x["AP"] for x in per_topic(s, qb)]) for s in S.SYSTEMS}
        bars = pd.DataFrame([{"System": s, "Judge": j, "MAP": v} for j, dd in [("Judge A", per_a), ("Judge B", per_b)]
                             for s, v in dd.items()])
        st.altair_chart(alt.Chart(bars).mark_bar().encode(
            x=alt.X("System:N", sort=S.SYSTEMS, title=None, axis=alt.Axis(labelAngle=0)), xOffset="Judge:N",
            y=alt.Y("MAP:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Judge:N", scale=alt.Scale(range=[ACCENT, AMBER]), legend=alt.Legend(orient="bottom")),
            tooltip=["System", "Judge", alt.Tooltip("MAP:Q", format=".3f")]).properties(height=400,
                                                                                         title="MAP under each assessor"),
            width="stretch")
        order_a = sorted(S.SYSTEMS, key=lambda s: -per_a[s])
        order_b = sorted(S.SYSTEMS, key=lambda s: -per_b[s])
        tau = M.kendall_tau(order_a, order_b)
        pos_b = {x: i for i, x in enumerate(order_b)}
        swaps = [(x, y) for i, x in enumerate(order_a) for y in order_a[i + 1:] if pos_b[x] > pos_b[y]]
        swap_txt = ("" if not swaps else " Swapped pairs: " + "; ".join(
            f"{x} and {y} (MAP gap {abs(per_a[x] - per_a[y]):.3f} under A, {abs(per_b[x] - per_b[y]):.3f} under B)"
            for x, y in swaps) + ".")
        st.write(f"System order by MAP. Judge A: {' > '.join(order_a)}. Judge B: {' > '.join(order_b)}. "
                 f"Kendall's τ between the two orders = **{tau:.2f}** (1 = identical).{swap_txt}")
        st.caption("The absolute scores move with the assessor, but the comparison between systems is usually "
                   "stable. That is why test collections remain useful despite disagreement (Voorhees, 2000).")

    st.subheader("Be the second assessor")
    tj = st.selectbox("Topic to judge", list(TOPICS), format_func=topic_label, key="t5_topic")
    show(f'<div class="need"><b>Information need:</b> {esc(TOPICS[tj]["need"])}</div>')
    dom_docs = [d for d in DOCS if DOMAIN_OF[d] == TOPICS[tj]["domain"]]
    mine = st.data_editor(pd.DataFrame({"Document": dom_docs, "Text": [DOCS[d] for d in dom_docs],
                                        "Your grade": [0] * len(dom_docs)}),
                          hide_index=True, width="stretch", key=f"t5_editor_{tj}", disabled=["Document", "Text"],
                          column_config={"Your grade": st.column_config.NumberColumn(min_value=0, max_value=3, step=1),
                                         "Text": st.column_config.TextColumn(width="large")})
    my_q = {d: int(g or 0) for d, g in zip(mine["Document"], mine["Your grade"]) if g}
    if not my_q:
        st.caption("Give a grade from 1 to 3 to every document you find relevant to the need, then compare.")
    else:
        km = M.kappa_from_labels([qa[tj].get(d, 0) >= threshold for d in dom_docs],
                                 [my_q.get(d, 0) >= threshold for d in dom_docs])
        y1, y2 = st.columns([1, 2])
        y1.metric("κ between you and judge A", f"{km.kappa:.3f}", help=f"P(A) = {km.p_agree:.3f}")
        y2.dataframe(pd.DataFrame([{"System": s,
                                    "AP (judge A)": M.average_precision(ranked_ids(s, tj), M.relevant_set(qa[tj], threshold)),
                                    "AP (you)": M.average_precision(ranked_ids(s, tj), M.relevant_set(my_q, threshold))}
                                   for s in S.SYSTEMS]).style.format({"AP (judge A)": "{:.3f}", "AP (you)": "{:.3f}"}),
                     hide_index=True, width="stretch")

# ===========================================================================
# Tab 6: BEIR and RAGAS
# ===========================================================================
with tab_modern:
    st.subheader("BEIR: one score per domain")
    st.write("BEIR (Thakur et al., NeurIPS 2021) evaluates retrievers **zero-shot** on 18 heterogeneous datasets "
             "(TREC-COVID, SciFact, FiQA, …) with **nDCG@10** as the main measure, to see whether a model "
             "generalizes beyond the data it was trained on. The lecture's findings: BM25 is a strong zero-shot "
             "baseline, and a BM25 + dense hybrid does best. Here the three domains of the test collection play "
             "the role of datasets (in BEIR each dataset also has its own corpus; here they share one index).")
    dom_rows = []
    for s in S.SYSTEMS:
        row = {"System": s}
        for dom in DATA["domains"].values():
            ts = [t for t in TOPICS if TOPICS[t]["domain"] == dom]
            row[dom] = np.mean([M.ndcg_at_k(ranked_ids(s, t), QRELS[t], 10, gain_kind) for t in ts])
        row["Average"] = np.mean([row[dm] for dm in DATA["domains"].values()])
        dom_rows.append(row)
    bt = pd.DataFrame(dom_rows)
    dcols = list(DATA["domains"].values()) + ["Average"]
    b1, b2 = st.columns([3, 2])
    with b1:
        st.dataframe(heat(bt, dcols), hide_index=True, width="stretch")
        avg = bt.set_index("System")["Average"]
        lead = avg.idxmax()
        cmp_ = ("BM25 beats Dense" if avg["BM25"] > avg["Dense"] + 1e-9 else
                "Dense beats BM25" if avg["Dense"] > avg["BM25"] + 1e-9 else "BM25 and Dense tie")
        per_dom = [dm for dm in dcols[:-1] if bt.set_index("System").loc["BM25", dm] >
                   bt.set_index("System").loc["Dense", dm] + 1e-9]
        st.caption(f"nDCG@10 per domain ({'rel' if gain_kind == 'linear' else '2^rel − 1'} gain, {judge}), dense "
                   f"model: {enc.name}. Here {lead} has the best average, and on average {cmp_}"
                   + (f" (BM25 wins in: {', '.join(per_dom)})." if per_dom else ".")
                   + " Whether BM25 or the dense model wins depends on the model and the domain, which is exactly "
                     "what BEIR was built to measure.")
    with b2:
        long = bt.melt("System", dcols[:-1], "Domain", "nDCG@10")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("Domain:N", title=None, sort=dcols[:-1], axis=alt.Axis(labelAngle=0, labelLimit=140)),
            xOffset=alt.XOffset("System:N", sort=S.SYSTEMS),
            y=alt.Y("nDCG@10:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("System:N", scale=SYS_COLORS, legend=alt.Legend(orient="bottom", columns=3)),
            tooltip=["System", "Domain", alt.Tooltip("nDCG@10:Q", format=".3f")]).properties(height=400),
            width="stretch")

    st.subheader("RAGAS: evaluating a retrieval-augmented answer")
    st.write("In RAG, a retriever finds contexts and a language model writes an answer from them. RAGAS (Es et al., "
             "EACL 2024) scores both parts, mostly **without reference answers**: *faithfulness* (is the answer "
             "supported by the contexts?), *answer relevance* (does it address the question?), *context "
             "precision* and *context recall* (did retrieval find the right material?).")
    st.warning("RAGAS uses an LLM as the judge. This tab has no LLM, so it uses **simple stand-ins**: word overlap "
               "for support, embedding similarity for answer relevance, and the qrels for context relevance. They "
               "show what each metric measures, not the scores the RAGAS library would give.")
    r1, r2, r3 = st.columns([3, 2, 1])
    tr = r1.selectbox("Question (the topic's information need)", list(TOPICS),
                      format_func=lambda t: f"{t} · {TOPICS[t]['need']}", key="t6_topic")
    sr = r2.selectbox("Retriever", S.SYSTEMS[1:], index=3, key="t6_sys")
    kr = r3.number_input("Contexts", 1, 5, 3, key="t6_k")
    ctx_ids = ranked_ids(sr, tr)[:kr]
    contexts = [DOCS[d] for d in ctx_ids]
    question = TOPICS[tr]["need"]
    dom = TOPICS[tr]["domain"]
    other = next(DOCS[d] for d in DOCS if DOMAIN_OF[d] != dom)
    presets = R.preset_answers(contexts, dom, R.sentences(other)[0])
    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown(f"**Retrieved contexts** (query `{esc(TOPICS[tr]['query'])}`)")
        for d in ctx_ids:
            g = QRELS[tr].get(d, 0)
            show(f'<div class="doc g{g}"><b>{d}</b> · grade {g}<br>{esc(DOCS[d])}</div>')
    with cr:
        choice = st.radio("Answer", list(presets), key=f"t6_choice_{tr}_{sr}_{kr}")
        answer = st.text_area("Generated answer (edit it)", presets[choice], height=110,
                              key=f"t6_answer_{tr}_{sr}_{kr}_{choice}")
    faith, sup = R.faithfulness(answer, contexts)
    arel = R.answer_relevance(question, answer, lambda t: encode(prefer, tuple(t)))
    flags = [QRELS[tr].get(d, 0) >= threshold for d in ctx_ids]
    cprec = R.context_precision(flags)
    reference = " ".join(DOCS[d] for d, g in sorted(QRELS[tr].items(), key=lambda x: -x[1]) if g >= 2)
    crec, rsup = R.context_recall(reference, contexts)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Faithfulness", f"{faith:.2f}", help="Share of answer sentences supported by the contexts")
    q2.metric("Answer relevance", f"{arel:.2f}", help="cos(question, answer), clipped to [0, 1]")
    q3.metric("Context precision", f"{cprec:.2f}", help="Σ P@k·v_k / relevant contexts, relevance from the qrels")
    q4.metric("Context recall", f"{crec:.2f}", help="Share of reference sentences supported by the contexts")
    f1_, f2_ = st.columns(2)
    with f1_:
        st.markdown("**Faithfulness, sentence by sentence** (supported when ≥ 60% of its content words occur in the "
                    "contexts)")
        for s_ in sup:
            ok = s_.ratio >= 0.6
            missing_w = [w for w in s_.words if w not in s_.found]
            show(f'<div class="sent {"ok" if ok else "no"}">{"✓" if ok else "✗"} {esc(s_.sentence)}<br>'
                 f'<small>{len(s_.found)}/{len(s_.words)} words found'
                 f'{" · not in the contexts: " + esc(", ".join(missing_w)) if missing_w else ""}</small></div>')
    with f2_:
        st.markdown("**Context recall** (reference = the documents judged relevant, grade ≥ 2)")
        for s_ in rsup:
            ok = s_.ratio >= 0.6
            show(f'<div class="sent {"ok" if ok else "no"}">{"✓" if ok else "✗"} {esc(s_.sentence)}<br>'
                 f'<small>{len(s_.found)}/{len(s_.words)} words in the contexts</small></div>')
    st.caption("Try the three answers: copying from the contexts is faithful even when the top context is not the "
               "right one (faithful ≠ correct); an invented claim halves faithfulness; an off-topic answer can be "
               "faithful to nothing and irrelevant to the question. Context precision and recall depend only on "
               "the retriever, so change it and watch them move.")
    with st.expander("How the real RAGAS is used"):
        st.markdown("`pip install ragas`, then pass a dataset of questions, retrieved contexts, answers and "
                    "(optionally) reference answers to its `evaluate` function together with an LLM and an "
                    "embedding model. The LLM splits the answer into claims and checks each against the contexts "
                    "(faithfulness), writes questions that the answer would fit (answer relevance), and judges each "
                    "context (context precision and recall). The exact API changes between versions, so follow the "
                    "documentation of the version you install. A related framework is ARES (Saad-Falcon et al., "
                    "NAACL 2024), which trains small judge models instead of prompting an LLM.")

st.divider()
st.caption("Measures: metrics.py (checked against ir_measures) · systems: systems.py · RAGAS stand-ins: rag_eval.py. "
           "The test collection was written for this lab. Offline word vectors: GloVe 6B 50d (PDDL).")
