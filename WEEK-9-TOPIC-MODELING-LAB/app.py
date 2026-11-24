"""
app.py: Streamlit interface for Topic Modeling Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import html
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import clustering as CL
import corpora
import embeddings as E
import topics as TP

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Topic Modeling Lab", page_icon="🧭", layout="wide")

ACCENT = "#0b7285"
AMBER = "#e8a028"
PALETTE = ["#0b7285", "#e8a028", "#cf3a3f", "#5f3dc4", "#2b8a3e", "#c2255c", "#1971c2", "#8f6b00", "#495057",
           "#0ca678", "#d9480f", "#7048e8"]

st.markdown(
    """
<style>
.formula{font-family:"Courier New",monospace;font-size:14px;padding:6px 10px;border-radius:4px;
  background:rgba(128,140,160,.10);display:inline-block;margin:4px 0}
.callout{border-left:3px solid rgba(232,160,40,.9);padding:6px 12px;margin:6px 0 10px;font-size:14.5px;
  background:rgba(232,160,40,.07)}
.pipe{display:flex;gap:4px;align-items:stretch;margin:4px 0 12px}
.stage{flex:1 1 0;min-width:0;border:1px solid rgba(128,140,160,.4);border-top:3px solid rgba(11,114,133,.85);
  border-radius:4px;padding:5px 8px;font-size:12.5px;line-height:1.35}
.stage b{display:block;font-size:13.5px}
.arrow{align-self:center;opacity:.55}
.topic{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:6px 10px;margin-bottom:6px;font-size:14px}
.topic b{font-family:"Courier New",monospace}
</style>
""",
    unsafe_allow_html=True,
)


def show(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def heat(df: pd.DataFrame, cols: list[str], digits: int = 3, extra: dict | None = None, na_rep: str = "—"):
    def shade(col: pd.Series):
        v = col.astype(float)
        lo, hi = v.min(), v.max()
        return [f"background-color: rgba(11,114,133,{0.06 + 0.44 * ((x - lo) / (hi - lo) if hi > lo else 0):.2f})"
                if pd.notna(x) else "" for x in v]
    return df.style.format({**{c: f"{{:.{digits}f}}" for c in cols}, **(extra or {})}, na_rep=na_rep
                           ).apply(shade, subset=cols)


def topic_color(domain: list) -> alt.Scale:
    dom = [str(d) for d in domain]
    rng = [("#c9ced6" if d in ("-1", "outlier") else PALETTE[i % len(PALETTE)]) for i, d in enumerate(dom)]
    return alt.Scale(domain=dom, range=rng)


def scatter(xy: np.ndarray, labels, ids, texts, title: str, height: int = 360, legend_title: str = "Cluster"):
    lab = [str(x) for x in labels]
    dom = sorted(set(lab), key=lambda s: (s == "-1", int(s) if s.lstrip("-").isdigit() else 0, s))
    data = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], legend_title: lab, "Doc": ids,
                         "Text": [t[:90] for t in texts]})
    return alt.Chart(data).mark_circle(size=60 if len(data) < 100 else 28, opacity=0.8).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        color=alt.Color(f"{legend_title}:N", scale=topic_color(dom), sort=dom,
                        legend=alt.Legend(orient="bottom", columns=6, labelLimit=160)),
        tooltip=["Doc", legend_title, "Text"]).properties(height=height, title=title)


def crosstab(pred, truth, kind: str, pred_name: str = "Cluster") -> pd.DataFrame:
    t = pd.crosstab(pd.Series([str(p) for p in pred], name=pred_name), pd.Series(truth, name=kind))
    return t.reset_index()


SHORT = {"Health & nutrition": "Health", "Personal finance": "Finance", "Search technology": "Search"}


def show_crosstab(t: pd.DataFrame):
    """Short class names so the table fits in half the page."""
    t = t.rename(columns=SHORT)
    st.dataframe(t, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Cached work
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_corpus(name: str) -> corpora.Corpus:
    return corpora.load(name)


@st.cache_data(show_spinner=False)
def get_tfidf(name: str) -> np.ndarray:
    c = get_corpus(name)
    if name == corpora.NAMES[0]:
        vec = TfidfVectorizer(max_features=100)                      # the slide's settings
    else:
        vec = TfidfVectorizer(stop_words=sorted(c.stopwords) or None, min_df=1)
    return vec.fit_transform(c.clean).toarray()          # float64, as TfidfVectorizer returns (float32 changes K-Means)


@st.cache_resource(show_spinner="Loading the sentence-transformer (first run downloads it)…")
def get_encoder(choice: str):
    return E.load(choice)


@st.cache_data(show_spinner="Encoding the documents…")
def get_embeddings(name: str, choice: str) -> np.ndarray:
    c = get_corpus(name)
    return E.embed(get_encoder(choice)[0], c.raw, c.clean, c.stopwords)


@st.cache_data(show_spinner="Reducing the embeddings (UMAP compiles on its first run; that can take a minute)…")
def get_reduced(name: str, choice: str, reducer: str) -> np.ndarray:
    return TP.reduce(get_embeddings(name, choice), reducer)


@st.cache_data(show_spinner=False)
def get_map(name: str, choice: str, reducer: str) -> np.ndarray:
    return TP.map_2d(get_reduced(name, choice, reducer))


@st.cache_data(show_spinner=False)
def get_steps(name: str, k: int, seed: int):
    return CL.kmeans_steps(get_tfidf(name), k, seed)


@st.cache_data(show_spinner="Running K-Means for every K…")
def get_k_choice(name: str, kmax: int):
    return CL.choose_k(sparse.csr_matrix(get_tfidf(name)), list(range(2, kmax + 1)))


@st.cache_data(show_spinner=False)
def get_kmeans_labels(name: str, k: int, space: str, choice: str = "") -> np.ndarray:
    # K-Means on the sparse TF-IDF matrix, as the slide's code does (dense input can end in another clustering)
    X = sparse.csr_matrix(get_tfidf(name)) if space == "tfidf" else get_embeddings(name, choice)
    return CL.kmeans(X, k).labels_


@st.cache_resource(show_spinner="Fitting LDA…")
def get_lda(name: str, k: int, alpha: float | None, beta: float | None):
    c = get_corpus(name)
    return TP.lda(c.clean, k, c.stopwords, alpha=alpha, beta=beta)


@st.cache_resource(show_spinner=False)
def get_bertopic(name: str, choice: str, reducer: str, min_size: int):
    c = get_corpus(name)
    return TP.bertopic(get_embeddings(name, choice), c.clean, min_size, reducer, c.stopwords,
                       reduced=get_reduced(name, choice, reducer))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Collection")
    cname = st.radio("Documents", corpora.NAMES, index=1, key="corpus")
    C = get_corpus(cname)
    st.caption({corpora.NAMES[0]: "The 8 documents of the P09 K-Means slide.",
                corpora.NAMES[1]: "45 short English documents written for the Session 7 lab, 15 each about search "
                                  "technology, health & nutrition and personal finance. The domains are the ground "
                                  "truth.",
                corpora.NAMES[2]: "The course's 903 Indonesian tweets about COVID-19, preprocessed by the course "
                                  "pipeline. They are labelled with sentiment, not topics, so there is no topic "
                                  "ground truth."}[cname])
    kmax = min(10, len(C.raw) - 1)
    K = st.slider("K, the number of clusters / topics (K-Means, LDA)", 2, kmax, min(C.default_k, kmax),
                  key=f"K_{cname}")
    st.header("BERTopic")
    enc_choice = st.radio("Embeddings", list(E.MODELS) + [E.LSA], key="enc",
                          help="all-MiniLM-L6-v2 is the model in the slide (English, about 90 MB). The multilingual "
                               "model knows Indonesian (about 470 MB). LSA needs no download.")
    reducer = st.radio("Dimensionality reduction", ["UMAP", "PCA"], key="reducer",
                       format_func={"UMAP": "UMAP (as in BERTopic)", "PCA": "PCA (fast)"}.get,
                       help="BERTopic reduces the embeddings to 5 dimensions with UMAP before clustering. UMAP "
                            "compiles code on its first run, which can take about a minute.")
    min_size = st.slider("HDBSCAN min_cluster_size", 2, 30, C.default_min_size, key=f"min_size_{cname}",
                         help="BERTopic's min_topic_size: the smallest group of documents that counts as a topic.")

ENC, ENC_WHY = get_encoder(enc_choice)
if ENC_WHY:
    st.warning(f"{enc_choice} is unavailable because {ENC_WHY}. BERTopic uses LSA embeddings (TF-IDF compressed "
               f"with truncated SVD) instead.")
EMB_NAME = ENC.name
N = len(C.raw)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 9 · DOCUMENT CLUSTERING & NEURAL TOPIC MODELING")
st.title("🧭 Topic Modeling Lab")
st.write("Classification needs labels; **clustering does not**. It groups documents by similarity alone, and "
         "**topic modeling** goes one step further by describing each group with its characteristic words. This lab "
         "runs **K-Means** one iteration at a time, shows how to **choose K** (elbow and silhouette), fits the "
         "classic topic model **LDA**, builds **BERTopic** step by step (sentence embeddings → UMAP → HDBSCAN → "
         "c-TF-IDF), and compares them on coherence, diversity, separation and, when known, the true categories.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", N)
m2.metric("Ground truth", C.label_kind or "none")
m3.metric("K (K-Means, LDA)", K)
m4.metric("Embeddings", EMB_NAME if len(EMB_NAME) < 24 else EMB_NAME[:22] + "…", help=EMB_NAME)

tab_km, tab_k, tab_lda, tab_bt, tab_cmp = st.tabs(
    ["1 · K-Means step by step", "2 · Choosing K", "3 · LDA", "4 · BERTopic", "5 · Compare"])

X_TFIDF = get_tfidf(cname)
PCA_TFIDF = PCA(n_components=2, random_state=0).fit(X_TFIDF)
XY_TFIDF = PCA_TFIDF.transform(X_TFIDF)

# ===========================================================================
# Tab 1: K-Means step by step
# ===========================================================================
with tab_km:
    st.markdown(r"""K-Means minimises $J = \sum_{i=1}^{K} \sum_{x \in C_i} \lVert x - \mu_i \rVert^2$: start from
$K$ centroids, **assign** every document to its nearest centroid, **update** each centroid to the mean of its
documents, $\mu_i = \frac{1}{|C_i|}\sum_{x \in C_i} x$, and repeat until nothing changes. Documents are TF-IDF
vectors, as in the slide's code.""")
    s1, s2 = st.columns([1, 3])
    seed = s1.number_input("Random initial centroids (seed)", 0, 999, 0, key="km_seed",
                           help="The starting centroids are K documents drawn at random. Different seeds can end "
                                "in different clusterings.")
    steps = get_steps(cname, K, int(seed))
    it = s2.slider("Iteration", 1, len(steps), len(steps), key=f"km_it_{cname}_{K}_{seed}") if len(steps) > 1 else 1
    step = steps[it - 1]
    final = steps[-1]
    sk = CL.kmeans(sparse.csr_matrix(X_TFIDF), K)
    a1, a2, a3 = st.columns(3)
    a1.metric(f"J after iteration {it}", f"{step.J:.4f}")
    a2.metric("Documents that changed cluster", step.changed if it > 1 else "— (first assignment)")
    a3.metric("Converged after", f"{len(steps)} iterations")
    c1, c2 = st.columns([3, 2])
    with c1:
        cxy = PCA_TFIDF.transform(step.centroids)
        pts = pd.DataFrame({"x": XY_TFIDF[:, 0], "y": XY_TFIDF[:, 1], "Cluster": [str(l) for l in step.labels],
                            "Doc": C.ids, "Text": [t[:90] for t in C.raw]})
        cen = pd.DataFrame({"x": cxy[:, 0], "y": cxy[:, 1], "Cluster": [str(i) for i in range(K)]})
        dom = [str(i) for i in range(K)]
        base_pts = alt.Chart(pts).mark_circle(size=70 if N < 100 else 26, opacity=0.8).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("Cluster:N", scale=topic_color(dom), legend=alt.Legend(orient="bottom")),
            tooltip=["Doc", "Cluster", "Text"])
        cents = alt.Chart(cen).mark_point(shape="cross", size=420, filled=True, stroke="#1d1d1d",
                                          strokeWidth=1.2).encode(
            x="x:Q", y="y:Q", color=alt.Color("Cluster:N", scale=topic_color(dom), legend=None))
        labels_layer = alt.Chart(pts).mark_text(dx=9, dy=-7, fontSize=11, color="#555").encode(
            x="x:Q", y="y:Q", text="Doc:N") if N <= 50 else None
        layers = base_pts + cents + labels_layer if labels_layer is not None else base_pts + cents
        st.altair_chart(layers.properties(height=420, title=f"Iteration {it}: documents and centroids (✚)"),
                        width="stretch")
        st.caption("A 2-D PCA view of the TF-IDF vectors. The centroids shown are the ones this assignment used.")
    with c2:
        jdf = pd.DataFrame({"Iteration": [s.iteration for s in steps], "J": [s.J for s in steps]})
        st.altair_chart(alt.Chart(jdf).mark_line(point=True, color=ACCENT).encode(
            x=alt.X("Iteration:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("J:Q", scale=alt.Scale(zero=False)),
            tooltip=["Iteration", alt.Tooltip("J:Q", format=".4f")]).properties(height=200, title="J never increases"),
            width="stretch")
        if final.J > sk.inertia_ + 1e-6:
            show(f'<div class="callout"><b>A local minimum.</b> This run ends at J = {final.J:.4f}, but scikit-learn\'s '
                 f'<code>KMeans(n_init=10, random_state=42)</code> reaches J = {sk.inertia_:.4f}. K-Means only '
                 f'improves on its starting point; <code>n_init=10</code> runs it 10 times and keeps the best. Try '
                 f'other seeds.</div>')
        else:
            st.success(f"This run reaches J = {final.J:.4f}, the same as scikit-learn's best of 10 starts "
                       f"({sk.inertia_:.4f}).")
    st.markdown(f"**Distances to the centroids at iteration {it}** (Euclidean, on TF-IDF vectors)")
    dist = pd.DataFrame(step.distances, columns=[f"to centroid {j}" for j in range(K)])
    dist.insert(0, "Document", [t[:70] for t in C.raw])
    dist.insert(0, "Doc", C.ids)
    dist["Assigned"] = step.labels
    show_rows = dist if N <= 60 else dist.head(40)
    st.dataframe(heat(show_rows, [f"to centroid {j}" for j in range(K)], 4), hide_index=True, width="stretch",
                 height=min(560, 38 + 35 * len(show_rows)))
    if N > 60:
        st.caption(f"First 40 of {N} documents.")
    if cname == corpora.NAMES[0] and K == 3:
        sil = silhouette_score(X_TFIDF, sk.labels_)
        show(f'<div class="callout"><b>The slide\'s code</b>, <code>KMeans(n_clusters=3, random_state=42, n_init=10)'
             f'</code> on <code>TfidfVectorizer(max_features=100)</code>, gives labels <code>{[int(x) for x in sk.labels_]}</code>: '
             f'machine learning (d1, d4, d7), information retrieval (d2, d5, d8) and text analysis (d3, d6), with '
             f'<code>silhouette_score</code> = {sil:.4f}. A low score: the documents share very few words, so even '
             f'documents in the same cluster are far apart.</div>')

# ===========================================================================
# Tab 2: choosing K
# ===========================================================================
with tab_k:
    st.markdown(r"""K is a hyperparameter. Two common guides from the slide: the **elbow method** (plot $J$ against
$K$ and pick the bend, after which adding clusters helps little) and the **silhouette score**
$s(x) = \frac{b - a}{\max(a, b)}$, where $a$ is the mean distance to the document's own cluster and $b$ to the
nearest other cluster; it runs from −1 (wrong cluster) to 1 (well separated).""")
    kc = get_k_choice(cname, kmax)
    kdf = pd.DataFrame([{"K": c.k, "J (inertia)": c.inertia, "Silhouette": c.silhouette} for c in kc])
    best_sil = int(kdf.loc[kdf["Silhouette"].idxmax(), "K"])
    g1, g2 = st.columns(2)
    rule = alt.Chart(pd.DataFrame({"K": [K]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="K:Q")
    with g1:
        st.altair_chart((alt.Chart(kdf).mark_line(point=True, color=ACCENT).encode(
            x=alt.X("K:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("J (inertia):Q", scale=alt.Scale(zero=False)),
            tooltip=["K", alt.Tooltip("J (inertia):Q", format=".4f")]) + rule).properties(
            height=300, title="Elbow: J against K"), width="stretch")
    with g2:
        st.altair_chart((alt.Chart(kdf).mark_line(point=True, color=AMBER).encode(
            x=alt.X("K:Q", axis=alt.Axis(tickMinStep=1)), y="Silhouette:Q",
            tooltip=["K", alt.Tooltip("Silhouette:Q", format=".4f")]) + rule).properties(
            height=300, title="Silhouette against K"), width="stretch")
    st.caption(f"Highest silhouette at K = {best_sil} ({kdf['Silhouette'].max():.4f}). The dashed line is the K in the "
               f"sidebar. On short texts the elbow is often not sharp and silhouettes stay low: TF-IDF vectors of "
               f"different documents share few words, so they are all far apart. Each point is the best of 10 "
               f"K-Means starts (random_state = 42).")

    labels_k = get_kmeans_labels(cname, K, "tfidf")
    sil_s = CL.silhouettes(X_TFIDF, labels_k)
    sdf = pd.DataFrame({"Doc": C.ids, "Cluster": [str(l) for l in labels_k], "Silhouette": sil_s,
                        "Text": [t[:80] for t in C.raw]})
    sdf = sdf.sort_values(["Cluster", "Silhouette"], ascending=[True, False]).reset_index(drop=True)
    sdf["order"] = range(len(sdf))
    h1, h2 = st.columns([3, 2])
    with h1:
        dom = [str(i) for i in range(K)]
        st.altair_chart(alt.Chart(sdf).mark_bar().encode(
            y=alt.Y("order:O", axis=None),
            x=alt.X("Silhouette:Q", scale=alt.Scale(domain=[min(-0.05, float(sil_s.min()) * 1.1),
                                                           max(0.1, float(sil_s.max()) * 1.1)])),
            color=alt.Color("Cluster:N", scale=topic_color(dom), legend=alt.Legend(orient="bottom")),
            tooltip=["Doc", "Cluster", alt.Tooltip("Silhouette:Q", format=".3f"), "Text"]).properties(
            height=min(560, max(240, 6 * len(sdf) + 60)),
            title=f"Silhouette of every document, K = {K} (mean {sil_s.mean():.4f})"), width="stretch")
        st.caption("Bars below 0 are documents closer to another cluster than to their own.")
    with h2:
        ct = TP.c_tfidf(C.clean, labels_k, C.stopwords)
        for cl in ct.clusters:
            n_cl = int((labels_k == cl).sum())
            words = ", ".join(w for w, _ in ct.top(cl, 6))
            show(f'<div class="topic"><b>Cluster {cl}</b> · {n_cl} documents<br>{esc(words)}</div>')
        st.caption("Top words of each cluster by c-TF-IDF (tab 4 explains the formula).")
        if C.labels:
            show_crosstab(crosstab(labels_k, C.labels, C.label_kind))
            st.caption(f"Clusters against the {C.label_kind}: ARI {adjusted_rand_score(C.labels, labels_k):.3f}, "
                       f"NMI {normalized_mutual_info_score(C.labels, labels_k):.3f} (1 = identical grouping, 0 = no "
                       f"better than chance).")

# ===========================================================================
# Tab 3: LDA
# ===========================================================================
with tab_lda:
    st.markdown("""**Latent Dirichlet Allocation** (Blei, Ng & Jordan, 2003) is the classic topic model. It
imagines that each document was written by picking a **mixture of topics** (a Dirichlet distribution with
parameter α) and then, word by word, a topic from that mixture and a word from that **topic's word distribution**
(Dirichlet with parameter β). Fitting reverses the story: from word counts alone it estimates the topics and each
document's mixture. Unlike K-Means, a document can belong to several topics at once.""")
    l1, l2, _ = st.columns([1, 1, 2])
    alpha_txt = l1.selectbox("α (document–topic prior)", ["1/K (default)", "0.1", "0.5", "1.0"], key="lda_alpha",
                             help="Small α: each document uses few topics. Large α: documents mix many topics.")
    beta_txt = l2.selectbox("β (topic–word prior)", ["1/K (default)", "0.01", "0.1", "1.0"], key="lda_beta",
                            help="Small β: each topic uses few words.")
    alpha = None if alpha_txt.startswith("1/K") else float(alpha_txt)
    beta = None if beta_txt.startswith("1/K") else float(beta_txt)
    L = get_lda(cname, K, alpha, beta)
    tops = {t: [w for w, _ in L.top(t, 10)] for t in range(K)}
    coh = TP.coherence(tops, [" ".join(set(t.split()) - C.stopwords) for t in C.clean])
    rows = [{"Topic": f"Topic {t}", "Word": w, "P(word | topic)": p, "rank": r}
            for t in range(K) for r, (w, p) in enumerate(L.top(t, 8))]
    st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar(color=ACCENT).encode(
        x=alt.X("P(word | topic):Q", title=None), y=alt.Y("Word:N", sort=alt.EncodingSortField("rank"), title=None),
        facet=alt.Facet("Topic:N", columns=min(K, 4), title=None),
        tooltip=["Topic", "Word", alt.Tooltip("P(word | topic):Q", format=".4f")]
    ).properties(width=170, height=160).resolve_scale(y="independent"), width="content")
    t2, t3 = st.columns([3, 2])
    with t2:
        st.dataframe(pd.DataFrame([{"Topic": t, "Documents (main topic)": int((L.labels == t).sum()),
                                    "NPMI coherence": coh[t], "Top words": ", ".join(tops[t][:6])} for t in range(K)]
                                  ).style.format({"NPMI coherence": "{:.3f}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        st.caption(f"Topic diversity {TP.diversity(tops):.3f} (share of distinct words among the top 10 of all "
                   f"topics). NPMI from −1 (the words never appear together) to 1 (always together).")
    with t3:
        if C.labels:
            show_crosstab(crosstab(L.labels, C.labels, C.label_kind, "Topic"))
            st.caption(f"Main topics against the {C.label_kind}: ARI {adjusted_rand_score(C.labels, L.labels):.3f}, "
                       f"NMI {normalized_mutual_info_score(C.labels, L.labels):.3f}.")
    st.markdown("**Topic mixture of some documents**")
    spread = sorted({int(i) for i in np.linspace(0, N - 1, min(N, 8))})
    pick = st.multiselect("Documents", list(range(N)), default=spread, key=f"lda_docs_{cname}",
                          format_func=lambda i: f"{C.ids[i]}: {C.raw[i][:60]}")
    if pick:
        mix = pd.DataFrame([{"Doc": f"{C.ids[i]}: {C.raw[i][:45]}", "Topic": f"Topic {t}", "Share": L.doc_topic[i, t]}
                            for i in pick for t in range(K)])
        st.altair_chart(alt.Chart(mix).mark_bar().encode(
            y=alt.Y("Doc:N", sort=None, title=None, axis=alt.Axis(labelLimit=330)),
            x=alt.X("Share:Q", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("Topic:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(orient="bottom")),
            tooltip=["Doc", "Topic", alt.Tooltip("Share:Q", format=".2f")]).properties(height=34 * len(pick) + 70),
            width="stretch")
    if N < 100:
        st.caption("LDA learns from word co-occurrence counts, so it needs many, longer documents. On a few dozen "
                   "one-sentence documents its topics are noisy, which is one of the reasons embedding-based methods "
                   "such as BERTopic took over for short texts.")

# ===========================================================================
# Tab 4: BERTopic
# ===========================================================================
with tab_bt:
    emb = get_embeddings(cname, enc_choice)
    BT = get_bertopic(cname, enc_choice, reducer, min_size)
    n_out = int((BT.labels == -1).sum())
    stages = [("1 · Embeddings", f"{N} × {emb.shape[1]}<br>{esc(EMB_NAME[:24])}"),
              ("2 · Reduce", f"{reducer} → {BT.reduced.shape[1]} dimensions"),
              ("3 · HDBSCAN", f"{len(BT.topics)} topics<br>{n_out} outliers"),
              ("4 · c-TF-IDF", f"{len(BT.ctfidf.vocab):,} terms<br>top words per topic")]
    show('<div class="pipe">' + '<span class="arrow">→</span>'.join(
        f'<div class="stage"><b>{a}</b>{b}</div>' for a, b in stages) + "</div>")
    st.markdown("""**BERTopic** (Grootendorst, 2022): embed every document with a sentence-transformer, reduce the
embeddings with **UMAP** (5 dimensions), cluster them with **HDBSCAN**, which finds the number of clusters itself
and leaves documents that fit nowhere as **outliers (topic −1)**, then describe each cluster with **c-TF-IDF**.""")
    XY = get_map(cname, enc_choice, reducer)
    b1, b2 = st.columns([3, 2])
    with b1:
        st.altair_chart(scatter(XY, BT.labels, C.ids, C.raw, f"BERTopic topics, 2-D view of the {reducer} space",
                                420, "Topic"), width="stretch")
    with b2:
        info = []
        for t in sorted(set(BT.labels.tolist())):
            words = [w for w, _ in BT.ctfidf.top(t, 10)]
            info.append({"Topic": t, "Count": int((BT.labels == t).sum()),
                         "Name": f"{t}_" + "_".join(words[:4]), "Representation": ", ".join(words)})
        st.dataframe(pd.DataFrame(info), hide_index=True, width="stretch", height=min(420, 38 + 35 * len(info)))
        st.caption("The same columns as BERTopic's get_topic_info(). Change min_cluster_size in the sidebar: larger "
                   "values give fewer, bigger topics and more outliers.")
    if not BT.topics:
        st.info("HDBSCAN found no cluster: every document is an outlier. Lower min_cluster_size in the sidebar.")
    else:
        st.subheader("c-TF-IDF, computed by hand")
        st.markdown(r"""$w_{t,c} = tf_{t,c} \cdot \log\left(1 + \frac{A}{tf_t}\right)$: $tf_{t,c}$ is the
frequency of term $t$ in cluster $c$ (all its documents joined into one), $tf_t$ its frequency in the whole
corpus, and $A$ the average number of words per cluster. A term scores high when it is frequent in the cluster
and rare elsewhere.""")
        tsel = st.selectbox("Topic", BT.topics, key=f"bt_topic_{cname}_{enc_choice}_{reducer}_{min_size}",
                            format_func=lambda t: f"Topic {t}: " + ", ".join(w for w, _ in BT.ctfidf.top(t, 4)))
        ct = BT.ctfidf
        i = ct.clusters.index(tsel)
        order = [j for j in np.argsort(-ct.weights[i])[:12] if ct.tf[i, j] > 0]
        ctab = pd.DataFrame([{"Term": ct.vocab[j], "tf(t, c)": int(ct.tf[i, j]), "tf(t)": int(ct.tf_t[j]),
                              "A / tf(t)": ct.A / ct.tf_t[j], "log(1 + A / tf(t))": np.log(1 + ct.A / ct.tf_t[j]),
                              "w(t, c)": ct.weights[i, j]} for j in order])
        d1, d2 = st.columns([3, 2])
        with d1:
            st.dataframe(heat(ctab, ["w(t, c)"], 3, {"A / tf(t)": "{:.2f}", "log(1 + A / tf(t))": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.caption(f"A = {ct.A:.1f} words per cluster ({len(ct.clusters)} clusters, the outlier group "
                       f"included). BERTopic also L1-normalizes tf(t, c) within each cluster, which does not change "
                       f"the order of the words.")
        with d2:
            idx = np.flatnonzero(BT.labels == tsel)
            cen = BT.reduced[idx].mean(axis=0)
            rep = idx[np.argsort(((BT.reduced[idx] - cen) ** 2).sum(axis=1))[:4]]
            st.markdown("**Representative documents** (closest to the topic's centre)")
            for r in rep:
                show(f'<div class="topic"><b>{esc(C.ids[r])}</b> {esc(C.raw[r][:220])}</div>')
        if C.labels:
            show_crosstab(crosstab(BT.labels, C.labels, C.label_kind, "Topic"))
            st.caption(f"Topics against the {C.label_kind}: ARI {adjusted_rand_score(C.labels, BT.labels):.3f}, NMI "
                       f"{normalized_mutual_info_score(C.labels, BT.labels):.3f} (outliers counted as one more group).")
    with st.expander("The same pipeline with the bertopic package"):
        st.code('''# pip install bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom")
topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model,
                       hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(docs)
print(topic_model.get_topic_info())
print(topic_model.get_topic(0))        # top words of topic 0''', language="python")
        st.caption("These are BERTopic's default settings, which this tab follows (scikit-learn's HDBSCAN stands in "
                   "for the hdbscan package). The slide uses min_topic_size=2 for its six example documents.")

# ===========================================================================
# Tab 5: compare
# ===========================================================================
with tab_cmp:
    st.markdown("Every method on the same documents. Topic words come from c-TF-IDF for the clustering methods and "
                "from the topic–word distribution for LDA; coherence uses the top 10 words.")
    emb = get_embeddings(cname, enc_choice)
    texts_sw = [" ".join(set(t.split()) - C.stopwords) for t in C.clean]
    results = {
        f"K-Means on TF-IDF (K = {K})": get_kmeans_labels(cname, K, "tfidf"),
        f"K-Means on embeddings (K = {K})": get_kmeans_labels(cname, K, "emb", enc_choice),
        f"LDA (K = {K}, main topic)": get_lda(cname, K, None, None).labels,
        f"BERTopic (min_cluster_size = {min_size})": BT.labels,
    }
    rows, words_by_method = [], {}
    for name, lab in results.items():
        lab = np.asarray(lab)
        if name.startswith("LDA"):
            Lm = get_lda(cname, K, None, None)
            tw = {t: [w for w, _ in Lm.top(t, 10)] for t in range(K)}
        else:
            ctm = TP.c_tfidf(C.clean, lab, C.stopwords)
            tw = {t: [w for w, _ in ctm.top(t, 10)] for t in ctm.clusters if t != -1}
        words_by_method[name] = tw
        coh = TP.coherence(tw, texts_sw)
        inl = lab != -1
        n_clusters = len(set(lab[inl].tolist()))
        sil = (silhouette_score(emb[inl], lab[inl], metric="cosine")
               if 1 < n_clusters < inl.sum() else np.nan)
        row = {"Method": name, "Topics": n_clusters, "Outliers": int((~inl).sum()),
               "Silhouette (embedding space)": sil, "NPMI coherence": np.nanmean(list(coh.values())) if coh else np.nan,
               "Topic diversity": TP.diversity(tw)}
        if C.labels:
            row[f"ARI vs {C.label_kind}"] = adjusted_rand_score(C.labels, lab)
            row[f"NMI vs {C.label_kind}"] = normalized_mutual_info_score(C.labels, lab)
        rows.append(row)
    comp = pd.DataFrame(rows)
    num_cols = [c for c in comp.columns if c not in ("Method", "Topics", "Outliers")]
    st.dataframe(heat(comp, num_cols, 3), hide_index=True, width="stretch")
    st.caption(f"Silhouette: cosine, on the {EMB_NAME} embeddings, outliers left out, so all methods are judged in "
               f"one space. NPMI: document co-occurrence in this corpus. "
               + (f"ARI and NMI compare with the {C.label_kind} labels" if C.labels else "")
               + ("; for the tweets these are sentiment labels, which topics are not expected to follow."
                  if C.label_kind == "sentiment" else "." if C.labels else ""))

    XY = get_map(cname, enc_choice, reducer)
    v1, v2 = st.columns(2)
    with v1:
        st.altair_chart(scatter(XY, results[f"K-Means on TF-IDF (K = {K})"], C.ids, C.raw,
                                f"K-Means (TF-IDF, K = {K}): centroid-based, K fixed", 380), width="stretch")
    with v2:
        st.altair_chart(scatter(XY, BT.labels, C.ids, C.raw,
                                "BERTopic (HDBSCAN): density-based, K found automatically", 380, "Topic"),
                        width="stretch")
    st.caption(f"Both plots use the same 2-D view of the {reducer}-reduced embeddings, as in the slide's comparison. "
               "Grey points are BERTopic outliers.")
    if C.labels:
        v3, v4 = st.columns(2)
        with v3:
            st.altair_chart(scatter(XY, C.labels, C.ids, C.raw, f"The true {C.label_kind}s", 340, C.label_kind.title()),
                            width="stretch")
        with v4:
            st.altair_chart(scatter(XY, results[f"LDA (K = {K}, main topic)"], C.ids, C.raw,
                                    f"LDA, main topic (K = {K})", 340, "Topic"), width="stretch")

    st.markdown("**Topic words side by side**")
    cols = st.columns(len(words_by_method))
    for col_, (name, tw) in zip(cols, words_by_method.items()):
        with col_:
            st.markdown(f"*{name}*")
            for t, ws in list(tw.items())[:12]:
                show(f'<div class="topic"><b>{t}</b> {esc(", ".join(ws[:6]))}</div>')

st.divider()
st.caption("K-Means: clustering.py · LDA, c-TF-IDF, BERTopic pipeline and quality measures: topics.py · embeddings: "
           "embeddings.py · collections: corpora.py. Tweets: STKI course dataset, Universitas Dian Nuswantoro.")
