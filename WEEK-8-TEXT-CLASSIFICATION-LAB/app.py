"""
app.py: Streamlit interface for Text Classification Lab.

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
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import classifiers as C
import handcalc as H
import neural as N
import textprep as T

st.set_page_config(page_title="Text Classification Lab", page_icon="🏷️", layout="wide")

ACCENT = "#0b7285"
AMBER = "#e8a028"
RED = "#cf3a3f"
LABEL_COLORS = alt.Scale(domain=["negative", "neutral", "positive"], range=[RED, "#9aa5b1", ACCENT])

st.markdown(
    """
<style>
.formula{font-family:"Courier New",monospace;font-size:14px;padding:6px 10px;border-radius:4px;
  background:rgba(128,140,160,.10);display:inline-block;margin:4px 0}
.callout{border-left:3px solid rgba(232,160,40,.9);padding:6px 12px;margin:6px 0 10px;font-size:14.5px;
  background:rgba(232,160,40,.07)}
.result{border:1px solid rgba(128,140,160,.35);border-radius:4px;padding:8px 12px;font-size:14px;height:100%}
.result.win{border-color:rgba(11,114,133,.85);background:rgba(11,114,133,.08)}
.result b.big{font-size:20px;display:block;margin:2px 0}
.steps td{padding:3px 10px;vertical-align:top;font-size:14px;border-bottom:1px solid rgba(128,140,160,.2)}
.steps td:first-child{white-space:nowrap;opacity:.75}
.steps code{white-space:pre-wrap}
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
        lo, hi = col.min(), col.max()
        return [f"background-color: rgba(11,114,133,{0.06 + 0.44 * ((v - lo) / (hi - lo) if hi > lo else 0):.2f})"
                for v in col]
    return df.style.format({**{c: f"{{:.{digits}f}}" for c in cols}, **(extra or {})}, na_rep=na_rep
                           ).apply(shade, subset=cols)


def confusion_chart(cm: np.ndarray, title: str, height: int = 300):
    names = [C.LABELS[c] for c in C.CLASSES]
    rows = [{"Actual": names[i], "Predicted": names[j], "Tweets": int(cm[i, j])} for i in range(3) for j in range(3)]
    base = alt.Chart(pd.DataFrame(rows)).encode(x=alt.X("Predicted:N", sort=names, axis=alt.Axis(labelAngle=0)),
                                                y=alt.Y("Actual:N", sort=names))
    rect = base.mark_rect().encode(color=alt.Color("Tweets:Q", scale=alt.Scale(scheme="teals"), legend=None))
    text = base.mark_text(fontSize=15).encode(text="Tweets:Q", color=alt.condition(
        alt.datum.Tweets > cm.max() * 0.55, alt.value("white"), alt.value("#1d1d1d")))
    return (rect + text).properties(height=height, title=title)


def metric_row(name: str, f: C.Fitted) -> dict:
    return {"Model": name, "Train accuracy": f.train.accuracy, "Test accuracy": f.test.accuracy,
            "Test precision (weighted)": f.test.precision_w, "Test recall (weighted)": f.test.recall_w,
            "Test F1 (weighted)": f.test.f1_w, "Test F1 (macro)": f.test.f1_macro}


# ---------------------------------------------------------------------------
# Cached work
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    df = C.load()
    return df, C.split(df)


@st.cache_resource(show_spinner="Grid search over 24 Naive Bayes settings, as in the notebook…")
def get_nb():
    return C.naive_bayes_grid(get_data()[1])


@st.cache_data(show_spinner="Cross-validating KNN for each k…")
def get_k_curve(metric: str, weights: str) -> pd.DataFrame:
    return C.knn_k_curve(get_data()[1], metric, weights)


@st.cache_resource(show_spinner=False)
def get_knn(k: int, metric: str, weights: str):
    return C.knn(get_data()[1], k, metric, weights)


@st.cache_resource(show_spinner=False)
def get_baselines():
    s = get_data()[1]
    return C.notebook_knn(s), C.majority(s)


@st.cache_resource(show_spinner="Loading the sentence-transformer (first run downloads it)…")
def get_encoder(choice: str):
    if choice == N.LSA:
        return N.lsa_encoder(list(get_data()[1].train_x)), None
    enc, why = N.load_transformer(choice)
    if enc is None:
        return N.lsa_encoder(list(get_data()[1].train_x)), why
    return enc, None


@st.cache_data(show_spinner="Encoding the 903 tweets…")
def get_embeddings(choice: str):
    enc = get_encoder(choice)[0]
    s = get_data()[1]
    src_tr = s.train_raw if enc.uses_raw else s.train_x
    src_te = s.test_raw if enc.uses_raw else s.test_x
    return enc.encode(list(src_tr)), enc.encode(list(src_te))


@st.cache_data(show_spinner="Training on 8 to 64 examples per class, five random samples each…")
def get_fewshot(choice: str) -> pd.DataFrame:
    s = get_data()[1]
    tr, te = get_embeddings(choice)
    ytr, yte = s.train_y.values, s.test_y.values
    rows = []
    for n in (8, 16, 32, 64):
        for seed in range(5):
            idx = N.sample_per_class(ytr, n, seed)
            x_few = s.train_x.iloc[idx]
            nb = Pipeline([("tfidf", TfidfVectorizer()), ("m", MultinomialNB(alpha=0.1))]).fit(x_few, ytr[idx])
            kn = C.knn_pipeline(3).fit(x_few, ytr[idx])
            emb, _ = N.head_scores(tr[idx], ytr[idx], te, yte, seed)
            for name, acc in [("Naive Bayes", (nb.predict(s.test_x) == yte).mean()),
                              ("KNN (k = 3)", (kn.predict(s.test_x) == yte).mean()),
                              ("Embeddings + logistic regression", emb.accuracy)]:
                rows.append({"Examples per class": n, "Method": name, "seed": seed, "Accuracy": acc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
df, SPLIT = get_data()
with st.sidebar:
    st.header("Sentence embeddings")
    enc_choice = st.radio("Encoder for tabs 4 and 5", list(N.MODELS) + [N.LSA], key="enc_choice",
                          help="all-MiniLM-L6-v2 is the model in the slide's SetFit code (English, about 90 MB). "
                               "The multilingual model knows Indonesian (about 470 MB). LSA needs no download.")
    st.caption({"all-MiniLM-L6-v2": "English model, as in the slide. Indonesian tweets are split into English "
                                    "word pieces, so expect weak embeddings.",
                "paraphrase-multilingual-MiniLM-L12-v2": "Covers Indonesian. Larger download and memory; "
                                                         "SetFit training is about three times slower.",
                N.LSA: "TF-IDF compressed to 100 dimensions with truncated SVD. A classical dense representation, "
                       "not a transformer, so SetFit is not available."}[enc_choice])
    st.header("Dataset")
    vc = df["sentiment"].value_counts()
    st.caption(f"{len(df)} Indonesian tweets about COVID-19 from the course (Dataset_Sentimen_Emosi.csv), "
               f"{vc[1]} positive, {vc[0]} neutral, {vc[-1]} negative. Split as in the notebook: "
               f"{len(SPLIT.train_x)} training and {len(SPLIT.test_x)} test tweets (test_size = 0.3, "
               f"random_state = 42).")

ENC, ENC_WHY = get_encoder(enc_choice)
if ENC_WHY:
    st.warning(f"{enc_choice} is unavailable because {ENC_WHY}. Tabs 4 and 5 use the offline LSA embeddings "
               f"instead, and SetFit is disabled.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 8 · TEXT CLASSIFICATION: NAIVE BAYES, KNN & TRANSFORMERS")
st.title("🏷️ Text Classification Lab")
st.write("Text classification assigns a document to one of a set of known classes: spam or not, a topic, a "
         "sentiment. It is **supervised**: the classes are learned from labelled examples. This lab works through "
         "the lecture's two classic classifiers **by hand** (Naive Bayes and K-Nearest Neighbor), reruns the "
         "notebook's **sentiment experiment on 903 Indonesian tweets**, then moves to **sentence-transformer "
         "embeddings and SetFit few-shot learning**, and lets you classify your own text.")

tab_nb, tab_knn, tab_data, tab_tf, tab_cmp = st.tabs(
    ["1 · Naive Bayes by hand", "2 · KNN by hand", "3 · Sentiment dataset", "4 · Transformers & SetFit",
     "5 · Compare & try"])

# ===========================================================================
# Tab 1: Naive Bayes by hand
# ===========================================================================
NB_PRESETS = {
    "Slide example: sport or politics (D6)": (
        [("Sepakbola indah menyerang", "Olahraga"), ("Presiden menaikan harga BBM", "Politik"),
         ("Partai politik Indonesia berburu suara", "Politik"), ("Manchester United Juara Liga Inggris", "Olahraga"),
         ("Timnas Indonesia gagal juara AFC", "Olahraga")], "PSIS berburu juara Liga Indonesia", False),
    "Quiz: three classes (D7)": (
        [("Timnas Indonesia gagal juara AFC", "Olahraga"), ("PSIS berburu juara Liga Indonesia", "Olahraga"),
         ("Partai politik Indonesia berburu suara", "Politik"), ("Hasil putusan Sidang Elit Dewan", "Politik"),
         ("Harga cabai naik dipasar tradisional", "Ekonomi"), ("Upah minimum regional diprediksi naik", "Ekonomi")],
        "Demo buruh menaikan upah minimum", True),
}


def tokens_of(text: str, stem: bool) -> list[str]:
    import re
    t = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    if stem and T.stemmer() is not None:
        t = T.stemmer().stem(t)
    return t.split()


with tab_nb:
    st.markdown(r"""Multinomial Naive Bayes picks the class with the highest
$P(c \mid d) \propto P(c)\,\prod_i P(w_i \mid c)$, where
$P(w \mid c) = \frac{count(w, c) + \alpha}{count(c) + \alpha\,|V|}$: $count(w, c)$ is how often $w$ occurs in the
documents of class $c$, $count(c)$ the number of words in those documents, $|V|$ the vocabulary size, and
$\alpha = 1$ Laplace smoothing, so that an unseen word does not make the whole product 0.""")
    p1, p2 = st.columns([2, 3])
    preset = p1.radio("Example", list(NB_PRESETS), key="nb_preset")
    docs0, test0, stem0 = NB_PRESETS[preset]
    with p2:
        o1, o2, o3 = st.columns(3)
        stem = o1.checkbox("Stem with Sastrawi", value=stem0, key=f"nb_stem_{preset}",
                           help="The quiz asks for tokenization and stemming; the slide example uses the words as "
                                "written.")
        alpha = o2.number_input("α (smoothing)", 0.01, 5.0, 1.0, 0.1, key="nb_alpha")
        unseen = o3.radio("Words not in the vocabulary", ["laplace", "ignore"], key="nb_unseen",
                          format_func={"laplace": "smooth them (slide)", "ignore": "skip them (scikit-learn)"}.get)
    e1, e2 = st.columns([3, 2])
    with e1:
        ed = st.data_editor(pd.DataFrame([{"Document": f"D{i + 1}", "Text": t, "Class": c}
                                          for i, (t, c) in enumerate(docs0)]),
                            hide_index=True, width="stretch", num_rows="dynamic", key=f"nb_docs_{preset}",
                            disabled=["Document"])
    with e2:
        test_text = st.text_input(f"Test document D{len(docs0) + 1}", test0, key=f"nb_test_{preset}")
        st.caption("Edit the training table (add rows, change classes) or the test document; everything below "
                   "is recomputed.")
    train = [(tokens_of(str(t), stem), str(c).strip()) for t, c in zip(ed["Text"], ed["Class"])
             if isinstance(t, str) and t.strip() and isinstance(c, str) and c.strip()]
    test_toks = tokens_of(test_text, stem)
    if len({c for _, c in train}) < 2 or not test_toks:
        st.info("Give at least two classes of training documents and a test document.")
    else:
        model = H.nb_train(train, alpha)
        scores = H.nb_classify(model, test_toks, unseen)
        post = H.nb_posterior(scores)
        winner = max(scores, key=lambda s_: s_.log_score)

        st.markdown(f"**Step 1 · Training.** Vocabulary |V| = **{len(model.vocab)}** distinct words.")
        vocab_df = pd.DataFrame([{"Word": w, **{c: model.counts[c][w] for c in model.classes}}
                                 for w in model.vocab])
        v1, v2 = st.columns([3, 2])
        with v1:
            st.dataframe(vocab_df, hide_index=True, width="stretch", height=min(740, 38 + 35 * len(vocab_df)))
        with v2:
            st.dataframe(pd.DataFrame([{"Class": c, "Documents": model.n_docs[c],
                                        "P(c)": f"{model.n_docs[c]}/{sum(model.n_docs.values())} = {model.prior[c]:.3f}",
                                        "count(c) (words)": model.tokens[c]} for c in model.classes]),
                         hide_index=True, width="stretch")
            st.caption("count(c) counts every word occurrence in the class, repeats included.")

        st.markdown(f"**Step 2 · Likelihood of each test word** P(w | c) = (count(w, c) + {alpha:g}) / "
                    f"(count(c) + {alpha:g} × {len(model.vocab)})")
        rows = []
        for i, w in enumerate([t[0] for t in scores[0].terms]):
            row = {"Word": w + ("" if w in model.vocab else "  (not in V)")}
            for s_ in scores:
                _, cnt, p = s_.terms[i]
                row[s_.cls] = f"({cnt}+{alpha:g})/({model.tokens[s_.cls]}+{alpha * len(model.vocab):g}) = {p:.5f}"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No test word is in the vocabulary, so only the priors decide.")

        st.markdown("**Step 3 · Score of each class** P(c) × Π P(w | c), and its logarithm (what code computes, "
                    "to avoid numbers too small for the computer)")
        cols = st.columns(len(scores))
        for col_, s_ in zip(cols, scores):
            with col_:
                prod = " × ".join(f"{p:.5f}" for _, _, p in s_.terms)
                show(f'<div class="result{" win" if s_ is winner else ""}"><b>{esc(s_.cls)}</b>'
                     f'<b class="big">{s_.product:.6e}</b><small>{s_.prior:.3f}{" × " + prod if prod else ""}'
                     f'</small><br><small>log score {s_.log_score:.4f} · normalized P(c | d) = {post[s_.cls]:.3f}'
                     f'</small></div>')
        st.success(f"The test document is classified as **{winner.cls}**.")

        if preset.startswith("Slide") and not stem and alpha == 1.0 and unseen == "laplace":
            show('<div class="callout"><b>A correction to the slide.</b> The slide uses count(Olahraga) = 12, the '
                 'number of <i>distinct</i> words in the sport documents. The formula needs the number of words: '
                 '<i>juara</i> occurs twice (D4 and D5), so count(Olahraga) = 13 and every sport likelihood has '
                 'denominator 13 + 20 = 33, not 32. The slide\'s 2,14577 × 10⁻⁷ becomes <b>1,8398 × 10⁻⁷</b>; '
                 'Politik stays 7,80 × 10⁻⁸, so the answer is still Olahraga.</div>')

# ===========================================================================
# Tab 2: KNN by hand
# ===========================================================================
KNN_PRESETS = {
    "Slide case study (D5)": (
        [("D1", "partai golkar demokrat tanding kampanye 2009", "Politik"),
         ("D2", "tanding pertama persema persebaya malang", "Olahraga"),
         ("D3", "besar wasit tanding sepakbola adil", "Olahraga"),
         ("D4", "partai demokrat menang pemilu 2009 figur sby", "Politik")],
        "tanding sepakbola persebaya kampanye pemilu 2009 tunda", False),
    "Quiz (D5: Presiden menaikan gaji Timnas Indonesia)": (
        [("D1", "Sepakbola PSIS tahun ini semakin indah.", "Olahraga"),
         ("D2", "Presiden Indonesia menaikan harga BBM.", "Politik"),
         ("D3", "Partai politik Indonesia berburu suara.", "Politik"),
         ("D4", "Timnas Indonesia gagal juara AFC", "Olahraga")],
        "Presiden menaikan gaji Timnas Indonesia", True),
}

with tab_knn:
    st.markdown(r"""K-Nearest Neighbor stores the training documents and classifies a new one by the **majority
class of its k most similar** documents. The slide's recipe: weight terms with TF-IDF,
$w_{t,d} = tf_{t,d} \cdot \log_{10}(N / df_t)$, compare with cosine similarity
$\cos(d_i, d_j) = \frac{\sum_k d_{ik} d_{jk}}{\sqrt{\sum_k d_{ik}^2}\sqrt{\sum_k d_{jk}^2}}$, rank, and vote.""")
    k1, k2 = st.columns([2, 3])
    kpreset = k1.radio("Example", list(KNN_PRESETS), key="knn_preset")
    kdocs, ktest, kprep0 = KNN_PRESETS[kpreset]
    with k2:
        q1, q2, q3 = st.columns(3)
        kprep = q1.checkbox("Remove stopwords and stem", value=kprep0, key=f"knn_prep_{kpreset}",
                            help="NLTK Indonesian stopwords and the Sastrawi stemmer. The slide case study is "
                                 "already preprocessed.")
        metric = q2.radio("Measure", ["cosine", "euclidean", "manhattan"], key="knn_metric",
                          format_func={"cosine": "cosine similarity", "euclidean": "Euclidean distance",
                                       "manhattan": "Manhattan distance"}.get)
    ked = st.data_editor(pd.DataFrame([{"Document": d, "Text": t, "Class": c} for d, t, c in kdocs]),
                         hide_index=True, width="stretch", num_rows="dynamic", key=f"knn_docs_{kpreset}")
    ktest_text = st.text_input("Test document", ktest, key=f"knn_test_{kpreset}")

    def knn_tokens(text: str) -> list[str]:
        if not kprep:
            return tokens_of(text, False)
        sw = T.stopwords()
        return tokens_of(" ".join(w for w in tokens_of(text, False) if w not in sw), True)

    rows_ok = [(str(d), str(t), str(c)) for d, t, c in zip(ked["Document"], ked["Text"], ked["Class"])
               if isinstance(t, str) and t.strip() and isinstance(c, str) and c.strip() and isinstance(d, str)]
    test_name = f"D{len(rows_ok) + 1}" if "D" + str(len(rows_ok) + 1) not in {r[0] for r in rows_ok} else "Test"
    if len(rows_ok) < 2 or not knn_tokens(ktest_text):
        st.info("Give at least two training documents and a test document.")
    else:
        docs_tok = {d: knn_tokens(t) for d, t, _ in rows_ok}
        docs_tok[test_name] = knn_tokens(ktest_text)
        labels = {d: c for d, _, c in rows_ok}
        tab = H.tfidf_table(docs_tok)
        if kprep:
            st.caption("After preprocessing: " + " · ".join(f"**{d}** {' '.join(docs_tok[d])}" for d in docs_tok))
        kk = q3.slider("k", 1, len(rows_ok), min(3, len(rows_ok)), key=f"knn_k_{kpreset}_{len(rows_ok)}")

        st.markdown(f"**Step 1 · TF-IDF weights** (N = {len(tab.names)} documents, the test document included, as "
                    f"on the slide)")
        trows = []
        for t_ in tab.terms:
            r = {"Term": t_}
            r.update({f"tf {d}": str(tab.tf[d][t_]) if tab.tf[d][t_] else "" for d in tab.names})
            r.update({"df": tab.df[t_], "idf = log10(N/df)": tab.idf[t_]})
            r.update({f"W {d}": tab.weights[d][t_] for d in tab.names})
            trows.append(r)
        wcols = [f"W {d}" for d in tab.names]
        st.dataframe(heat(pd.DataFrame(trows), wcols, 5, {"idf = log10(N/df)": "{:.5f}"}), hide_index=True,
                     width="stretch", height=min(740, 38 + 35 * len(trows)))

        ranked = H.knn_scores(tab, test_name, labels, metric)
        q_len = math.sqrt(sum(v * v for v in tab.weights[test_name].values()))
        st.markdown(f"**Step 2 · Similarity to {test_name}** (‖{test_name}‖ = {q_len:.5f})")
        score_name = {"cosine": "cos", "euclidean": "Euclidean distance", "manhattan": "Manhattan distance"}[metric]
        srows = []
        for i, n in enumerate(ranked, 1):
            r = {"Rank": i, "Document": n.name, "Class": n.label, f"{test_name} · D (dot product)": n.dot,
                 "‖D‖": n.length, score_name: n.score, "Among the k nearest": "✓" if i <= kk else ""}
            if metric == "cosine":
                r["Formula"] = f"{n.dot:.5f} / ({q_len:.5f} × {n.length:.5f})"
            srows.append(r)
        st.dataframe(pd.DataFrame(srows).style.format({f"{test_name} · D (dot product)": "{:.6f}", "‖D‖": "{:.6f}",
                                                       score_name: "{:.5f}"}), hide_index=True, width="stretch")
        votes, best = H.vote(ranked, kk)
        st.markdown(f"**Step 3 · Vote of the {kk} nearest:** " +
                    ", ".join(f"{c} {v}" for c, v in sorted(votes.items(), key=lambda x: -x[1])))
        if len(best) == 1:
            st.success(f"{test_name} is classified as **{best[0]}**.")
        else:
            st.warning(f"Tie between {' and '.join(best)}. Odd k avoids ties between two classes; "
                       f"scikit-learn would pick the class that comes first in its sorted list of classes.")
        if metric != "cosine":
            st.caption("With a distance, smaller means closer. Euclidean distance also grows with document length, "
                       "which is why text retrieval usually prefers the cosine.")
        if kpreset.startswith("Slide") and not kprep:
            show('<div class="callout"><b>The slide checks out.</b> cos(D5, D1) = 0,1975, cos(D5, D2) = cos(D5, D3) = '
                 '0,1208, cos(D5, D4) = 0,13542, and with k = 3 the vote is Politik 2, Olahraga 1. One typo: the '
                 'weight table shows <i>golkar</i> in D1 as 0,65897, but it is 0,69897 (log10 5), and the length '
                 'calculation correctly uses its square 0,488559.</div>')

# ===========================================================================
# Tab 3: the sentiment dataset
# ===========================================================================
with tab_data:
    st.markdown("The lecture notebook's experiment: **sentiment of Indonesian tweets about COVID-19** (negative, "
                "neutral, positive), preprocessed with ekphrasis, a slang dictionary, NLTK stopwords and Sastrawi, "
                "then classified from TF-IDF features.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tweets", len(df))
    m2.metric("Training / test", f"{len(SPLIT.train_x)} / {len(SPLIT.test_x)}")
    m3.metric("Distinct words after preprocessing", f"{len({w for t in df['processed'] for w in t.split()}):,}")
    m4.metric("Always guessing 'positive'", f"{(SPLIT.test_y == 1).mean():.3f}", help="Test accuracy of the "
              "majority class: the baseline every classifier has to beat.")
    d1, d2 = st.columns([2, 3])
    with d1:
        dist = df["sentiment"].map(C.LABELS).value_counts().reindex(["negative", "neutral", "positive"]).reset_index()
        dist.columns = ["Sentiment", "Tweets"]
        st.altair_chart(alt.Chart(dist).mark_bar().encode(
            x=alt.X("Sentiment:N", sort=None, axis=alt.Axis(labelAngle=0)), y="Tweets:Q",
            color=alt.Color("Sentiment:N", scale=LABEL_COLORS, legend=None), tooltip=["Sentiment", "Tweets"]
        ).properties(height=260, title="Class distribution"), width="stretch")
    with d2:
        from collections import Counter
        top = Counter(w for t in df["processed"] for w in t.split()).most_common(12)
        st.dataframe(pd.DataFrame(top, columns=["Most frequent feature", "Occurrences"]), hide_index=True,
                     width="stretch", height=280)
    st.caption("Two of the most frequent features are not words: *number* and *allcaps* are ekphrasis annotations "
               "(<number>, <allcaps>…</allcaps>) that became ordinary tokens when punctuation was removed. They "
               "carry some signal (news-style tweets use numbers and capitals) but are worth knowing about.")
    with st.expander("Browse the tweets and their preprocessed text"):
        st.dataframe(df.assign(sentiment=df["sentiment"].map(C.LABELS))[["tweet", "sentiment", "processed"]],
                     hide_index=True, width="stretch", height=300)

    train_set = set(SPLIT.train_x)
    dup_mask = np.array([t in train_set for t in SPLIT.test_x])
    conflict = int((df.groupby("processed")["sentiment"].nunique() > 1).sum())
    show(f'<div class="callout"><b>Duplicates after preprocessing.</b> All {len(df)} raw tweets are different, but '
         f'{int(df["processed"].duplicated().sum())} preprocessed texts repeat an earlier one (retweets and copied '
         f'news headlines that differ only in links, numbers or punctuation). {int(dup_mask.sum())} of the '
         f'{len(SPLIT.test_x)} test tweets have an identical twin in the training set, which makes them easy to '
         f'classify, and {conflict} repeated texts carry different labels, which no classifier can get right every '
         f'time.</div>')

    st.subheader("Naive Bayes, tuned as in the notebook")
    nbf, grid_tab = get_nb()
    g1, g2 = st.columns([3, 2])
    with g1:
        base = alt.Chart(grid_tab).encode(x=alt.X("α:O", title="α (smoothing)"), y=alt.Y("n-grams:O"))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("CV accuracy:Q", scale=alt.Scale(scheme="teals"),
                                                                 legend=None)) +
                         base.mark_text(fontSize=13).encode(text=alt.Text("CV accuracy:Q", format=".3f"))
                         ).properties(height=260, title="GridSearchCV: mean 3-fold accuracy on the training set"),
                        width="stretch")
    with g2:
        st.markdown(f"**{nbf.note}**")
        st.dataframe(pd.DataFrame([{"Set": "Training", "Accuracy": nbf.train.accuracy, "F1 (weighted)": nbf.train.f1_w},
                                   {"Set": "Test", "Accuracy": nbf.test.accuracy, "F1 (weighted)": nbf.test.f1_w}]
                                  ).style.format({"Accuracy": "{:.4f}", "F1 (weighted)": "{:.4f}"}),
                     hide_index=True, width="stretch")
        nb_pred = nbf.model.predict(SPLIT.test_x)
        acc_dup = (nb_pred[dup_mask] == SPLIT.test_y.values[dup_mask]).mean()
        acc_new = (nb_pred[~dup_mask] == SPLIT.test_y.values[~dup_mask]).mean()
        st.caption(f"Accuracy on the {int(dup_mask.sum())} test tweets with a training twin: {acc_dup:.3f}; on the "
                   f"other {int((~dup_mask).sum())}: {acc_new:.3f}.")
        st.caption(f"The notebook prints the same numbers: training 0.9763, test 0.5904. The gap is overfitting: "
                   f"the model nearly memorises the training tweets. On the test set it beats the majority "
                   f"baseline ({(SPLIT.test_y == 1).mean():.3f}) by only {nbf.test.accuracy - (SPLIT.test_y == 1).mean():.3f}.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.altair_chart(confusion_chart(nbf.test.confusion, "Naive Bayes, test set"), width="stretch")
    with c2:
        st.dataframe(heat(nbf.test.per_class, ["Precision", "Recall", "F1"], extra={"Support": "{:.0f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Weighted F1 {nbf.test.f1_w:.3f} (each class weighted by its size, as in the notebook) versus "
                   f"macro F1 {nbf.test.f1_macro:.3f} (every class counts the same). Neutral and negative tweets "
                   f"are where the model fails; the weighted average hides it because positive tweets are the "
                   f"majority.")

    st.subheader("KNN on TF-IDF vectors (the correct way)")
    kc1, kc2 = st.columns(2)
    kmetric = kc1.radio("Measure", ["cosine", "euclidean"], horizontal=True, key="ds_metric")
    kweights = kc2.radio("Votes", ["uniform", "distance"], horizontal=True, key="ds_weights",
                         format_func={"uniform": "one vote each", "distance": "weighted by closeness"}.get)
    curve = get_k_curve(kmetric, kweights)
    best_k = int(curve.loc[curve["CV accuracy (training set)"].idxmax(), "k"])
    long = curve.melt("k", var_name="Measured on", value_name="Accuracy")
    ch = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("k:Q", title="k (neighbours)"), y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.3, 0.7])),
        color=alt.Color("Measured on:N", scale=alt.Scale(range=[ACCENT, AMBER]), legend=alt.Legend(orient="bottom")),
        tooltip=["k", "Measured on", alt.Tooltip("Accuracy:Q", format=".4f")])
    rule = alt.Chart(pd.DataFrame({"k": [best_k]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="k:Q")
    kk1, kk2 = st.columns([3, 2])
    with kk1:
        st.altair_chart((ch + rule).properties(height=320, title="Choosing k"), width="stretch")
    with kk2:
        kf = get_knn(best_k, kmetric, kweights)
        st.metric(f"Test accuracy with k = {best_k}", f"{kf.test.accuracy:.4f}",
                  delta=f"{kf.test.accuracy - nbf.test.accuracy:+.4f} vs Naive Bayes")
        st.caption(f"k = {best_k} has the best 5-fold cross-validation accuracy on the training set (dashed line). "
                   f"Choosing k on the test curve would be cheating: it would use the test labels to tune the model. "
                   f"Macro F1 {kf.test.f1_macro:.3f}.")
        st.altair_chart(confusion_chart(kf.test.confusion, f"KNN, k = {best_k}", 240), width="stretch")

    st.subheader("Why the notebook's KNN scores 0.36")
    nk, maj = get_baselines()
    st.markdown("The notebook passes the text through `LabelEncoder` before KNN:")
    st.code("label_encoder = LabelEncoder()\ntrain_x_encoded = label_encoder.fit_transform(train_x)   # one integer per tweet\n"
            "test_x_encoded = label_encoder.fit_transform(test_x)     # fitted again, on the test set\n"
            "model = KNeighborsClassifier(n_neighbors=3).fit(train_x_encoded.reshape(-1, 1), train_label)",
            language="python")
    sorted_tr = sorted(SPLIT.train_x)
    ex = pd.DataFrame({"Integer": range(5), "Tweet (preprocessed), in alphabetical order": sorted_tr[:5]})
    w1, w2 = st.columns([3, 2])
    with w1:
        st.dataframe(ex, hide_index=True, width="stretch")
    with w2:
        st.metric("Notebook's KNN, test accuracy", f"{nk.test.accuracy:.4f}",
                  delta=f"{nk.test.accuracy - kf.test.accuracy:+.4f} vs KNN on TF-IDF")
    st.markdown(f"`LabelEncoder` gives each distinct tweet its **position in alphabetical order**, so every tweet "
                f"becomes a single number and the \"nearest neighbours\" are the tweets that start with similar "
                f"letters. The test set is also encoded separately, so test integer 5 and training integer 5 are "
                f"unrelated tweets. Its test accuracy, {nk.test.accuracy:.3f}, is below the {maj.test.accuracy:.3f} of "
                f"always answering *positive*. `LabelEncoder` is meant for the **labels** (y), not the text; KNN "
                f"needs the TF-IDF vectors, as in the P08 slide (`KNeighborsClassifier(n_neighbors=3, "
                f"metric='cosine')` on `X_train`). With k = 3 that gives {get_knn(3, 'cosine', 'uniform').test.accuracy:.4f}.")

    st.subheader("Why was this tweet classified this way?")
    opts = list(range(len(SPLIT.test_x)))
    pick = st.selectbox("Test tweet", opts, format_func=lambda i: f"[{C.LABELS[int(SPLIT.test_y.iloc[i])]}] "
                                                                 f"{SPLIT.test_raw.iloc[i][:110]}", key="ds_pick")
    text_p = SPLIT.test_x.iloc[pick]
    show(f'<div class="callout"><b>Tweet:</b> {esc(SPLIT.test_raw.iloc[pick])}<br><b>Preprocessed:</b> '
         f'<code>{esc(text_p)}</code><br><b>True label:</b> {C.LABELS[int(SPLIT.test_y.iloc[pick])]}</div>')
    nb_pipe = nbf.model.best_estimator_
    contrib, log_prior, post = C.nb_explain(nb_pipe, text_p)
    x1, x2 = st.columns(2)
    with x1:
        pred_nb = C.LABELS[max(post, key=post.get)]
        st.markdown(f"**Naive Bayes → {pred_nb}** · " + " · ".join(f"{C.LABELS[c]} {p:.2f}" for c, p in post.items()))
        if len(contrib):
            st.dataframe(heat(contrib.head(12), ["negative", "neutral", "positive"], 3, {"TF-IDF": "{:.3f}"}),
                         hide_index=True, width="stretch")
        st.caption("Each cell is TF-IDF weight × log P(w | c): closer to 0 is better for that class. The class "
                   "score adds these to log P(c) = " + ", ".join(f"{C.LABELS[c]} {v:.2f}" for c, v in log_prior.items())
                   + ". MultinomialNB here reads TF-IDF weights instead of raw counts, as in the notebook.")
    with x2:
        nbrs = C.knn_neighbors(kf.model, text_p, SPLIT.train_x, SPLIT.train_y, SPLIT.train_raw)
        st.markdown(f"**KNN (k = {best_k}) → {C.LABELS[int(kf.model.predict([text_p])[0])]}**")
        st.dataframe(nbrs[["Rank", "Cosine similarity", "Label", "Tweet"]].style.format(
            {"Cosine similarity": "{:.3f}"}), hide_index=True, width="stretch")

# ===========================================================================
# Tab 4: transformers and SetFit
# ===========================================================================
with tab_tf:
    st.subheader("BERT fine-tuning")
    b1, b2 = st.columns([3, 2])
    with b1:
        st.markdown("""**[CLS] tweet [SEP] → BERT → [CLS] vector → linear layer → softmax → class.** A pre-trained
BERT already understands language; fine-tuning adds a small classification layer and trains the whole network on
the labelled data (Devlin et al., 2019). It gives the best accuracy, but the lecture lists its costs: a GPU, 1,000+
labelled examples, 30 to 60 minutes of training. That does not fit in a free cloud app, so this lab **shows the
code instead of running it**, and runs the cheaper alternatives below.""")
    with b2:
        st.code('''# pip install transformers datasets  (GPU recommended)
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
name = "indobenchmark/indobert-base-p1"        # an Indonesian BERT
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)
ds = Dataset.from_dict({"text": tweets, "label": [y + 1 for y in sentiment]})  # -1,0,1 -> 0,1,2
ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                          max_length=128), batched=True)
args = TrainingArguments("out", num_train_epochs=3, learning_rate=2e-5,
                         per_device_train_batch_size=16)
Trainer(model=model, args=args, train_dataset=ds).train()''', language="python")

    st.subheader("Sentence embeddings + logistic regression")
    tr_emb, te_emb = get_embeddings(enc_choice)
    full_res, full_head = N.head_scores(tr_emb, SPLIT.train_y.values, te_emb, SPLIT.test_y.values)
    e1, e2, e3 = st.columns(3)
    e1.metric("Encoder", ENC.name if len(ENC.name) < 28 else ENC.name[:26] + "…", help=ENC.name)
    e2.metric("Test accuracy (632 training tweets)", f"{full_res.accuracy:.4f}",
              delta=f"{full_res.accuracy - get_nb()[0].test.accuracy:+.4f} vs Naive Bayes")
    e3.metric("Test F1 (macro)", f"{full_res.f1_macro:.4f}")
    st.caption(f"Each tweet becomes one {ENC.dims}-dimensional vector "
               f"({'from the raw tweet' if ENC.uses_raw else 'from the preprocessed text'}); a logistic regression "
               f"learns the classes from those vectors. This is SetFit without its fine-tuning step.")

    st.subheader("Few-shot: how many labelled tweets does each method need?")
    fs = get_fewshot(enc_choice)
    agg = fs.groupby(["Examples per class", "Method"], as_index=False)["Accuracy"].agg(["mean", "std"])
    agg.columns = ["Examples per class", "Method", "Accuracy", "sd"]
    agg["low"], agg["high"] = agg["Accuracy"] - agg["sd"], agg["Accuracy"] + agg["sd"]
    meth_scale = alt.Scale(domain=["Naive Bayes", "KNN (k = 3)", "Embeddings + logistic regression"],
                           range=[ACCENT, "#9aa5b1", AMBER])
    line = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X("Examples per class:Q", scale=alt.Scale(type="log", base=2), axis=alt.Axis(values=[8, 16, 32, 64])),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.25, 0.7], clamp=True), title="Test accuracy"),
        color=alt.Color("Method:N", scale=meth_scale, legend=alt.Legend(orient="bottom", labelLimit=300)),
        tooltip=["Method", "Examples per class", alt.Tooltip("Accuracy:Q", format=".3f"), alt.Tooltip("sd:Q", format=".3f")])
    band = alt.Chart(agg).mark_area(opacity=0.15, clip=True).encode(
        x=alt.X("Examples per class:Q", scale=alt.Scale(type="log", base=2)),
        y=alt.Y("low:Q", scale=alt.Scale(domain=[0.25, 0.7], clamp=True)), y2="high:Q",
        color=alt.Color("Method:N", scale=meth_scale, legend=None))
    base_rule = alt.Chart(pd.DataFrame({"y": [(SPLIT.test_y == 1).mean()]})).mark_rule(
        strokeDash=[4, 3], color="gray").encode(y="y:Q")
    f1_, f2_ = st.columns([3, 2])
    with f1_:
        st.altair_chart((band + line + base_rule).resolve_scale(color="independent").properties(height=400),
                        width="stretch")
    with f2_:
        last = agg[agg["Examples per class"] == 64].set_index("Method")["Accuracy"]
        first = agg[agg["Examples per class"] == 8].set_index("Method")["Accuracy"]
        st.markdown(f"Each point is the mean test accuracy over five random samples of *n* tweets per class "
                    f"(shaded: ± one standard deviation); the dashed line is the majority baseline. With 8 per "
                    f"class: " + ", ".join(f"{m} {v:.3f}" for m, v in first.items()) + ". With 64: " +
                    ", ".join(f"{m} {v:.3f}" for m, v in last.items()) + ".")
        st.caption("With so few examples every method is close to guessing on this data: the classes are "
                   "imbalanced and many tweets are ambiguous. Naive Bayes and KNN only know the words of the "
                   "sampled tweets; the embeddings bring knowledge from pre-training (or, for LSA, from all "
                   "training tweets without their labels).")

    st.subheader("SetFit: contrastive fine-tuning on a few examples")
    st.markdown("SetFit (Tunstall et al., 2022) first **fine-tunes the sentence-transformer on pairs** built from "
                "the few labelled examples (same class → similarity 1, different class → 0), then trains the "
                "logistic-regression head on the new embeddings. Below, both steps run here, on the CPU.")
    is_tf = ENC.model is not None
    s1, s2, s3, s4 = st.columns(4)
    n_shot = s1.select_slider("Examples per class", [8, 16, 32], 8, key="sf_n", disabled=not is_tf)
    iters = s2.select_slider("Iterations (pairs per example ÷ 2)", [5, 10, 20], 10, key="sf_iter",
                             disabled=not is_tf)
    seed = s3.number_input("Random seed", 0, 99, 0, key="sf_seed", disabled=not is_tf)
    n_pairs = n_shot * 3 * iters * 2
    s4.metric("Training pairs", n_pairs, help=f"{n_shot} × 3 classes × {iters} iterations × 2 (one positive and "
                                              f"one negative pair each)")
    if not is_tf:
        st.info("SetFit fine-tunes a transformer. Choose all-MiniLM-L6-v2 or the multilingual model in the sidebar.")
    elif st.button("Train SetFit", type="primary", key="sf_train"):
        idx = N.sample_per_class(SPLIT.train_y.values, n_shot, int(seed))
        texts = [SPLIT.train_raw.iloc[i] for i in idx]
        ys = [int(SPLIT.train_y.iloc[i]) for i in idx]
        bar = st.progress(0.0, "starting…")
        ft, npairs, steps = N.setfit_finetune(ENC.model, texts, ys, iterations=iters, seed=int(seed),
                                              progress=lambda f, m: bar.progress(min(f, 1.0), m))
        bar.empty()
        ft_enc = N.transformer_encoder(f"SetFit ({ENC.name})", ft)
        before, _ = N.head_scores(tr_emb[idx], np.array(ys), te_emb, SPLIT.test_y.values, int(seed))
        ft_te = ft_enc.encode(list(SPLIT.test_raw))
        ft_tr = ft_enc.encode(texts)
        after, head = N.head_scores(ft_tr, np.array(ys), ft_te, SPLIT.test_y.values, int(seed))
        st.session_state["setfit"] = {"encoder": ft_enc, "head": head, "before": before, "after": after,
                                      "n": n_shot, "iters": iters, "seed": int(seed), "pairs": npairs,
                                      "steps": steps, "base": ENC.name, "test_emb": ft_te}
    sf = st.session_state.get("setfit")
    if sf and sf["base"] == ENC.name:
        r1, r2, r3 = st.columns(3)
        r1.metric(f"Frozen embeddings + head ({sf['n']} per class)", f"{sf['before'].accuracy:.4f}")
        r2.metric("SetFit (fine-tuned + head)", f"{sf['after'].accuracy:.4f}",
                  delta=f"{sf['after'].accuracy - sf['before'].accuracy:+.4f}")
        r3.metric("Macro F1, SetFit", f"{sf['after'].f1_macro:.4f}")
        pca_rows = []
        for name, emb in [("Before fine-tuning", te_emb), ("After SetFit fine-tuning", sf["test_emb"])]:
            xy = PCA(2, random_state=0).fit_transform(emb)
            pca_rows += [{"View": name, "x": a, "y": b, "Sentiment": C.LABELS[int(lbl)]}
                         for (a, b), lbl in zip(xy, SPLIT.test_y.values)]
        st.altair_chart(alt.Chart(pd.DataFrame(pca_rows)).mark_circle(size=34, opacity=0.7).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("Sentiment:N", scale=LABEL_COLORS, legend=alt.Legend(orient="bottom")),
            column=alt.Column("View:N", title=None, sort=["Before fine-tuning", "After SetFit fine-tuning"])
        ).properties(height=300, width=420), width="content")
        st.caption(f"The {len(SPLIT.test_x)} test tweets projected to 2-D with PCA. Fine-tuning on {sf['pairs']} "
                   f"pairs ({sf['steps']} steps) pulls tweets of the same class together. Results vary with the "
                   f"seed: with 8 examples per class, which 24 tweets you draw matters a lot.")
    elif is_tf:
        st.caption("Press **Train SetFit**. With 8 examples per class and 10 iterations, all-MiniLM-L6-v2 trains in "
                   "well under a minute on a typical CPU.")

# ===========================================================================
# Tab 5: compare and try
# ===========================================================================
EXAMPLES = ["Alhamdulillah, semoga pandemi covid-19 cepat berlalu dan kita semua tetap sehat. Jaga jarak ya!",
            "Kesal banget!!! Bantuan sosial covid nggak pernah sampai ke warga, pejabatnya korupsi terus",
            "Pemerintah umumkan jadwal vaksinasi covid tahap kedua mulai Senin 12 April",
            "PSBB diperpanjang lagi, gak bisa kerja, mau makan apa kita"]

with tab_cmp:
    st.subheader("All methods on the same test set")
    nbf, _ = get_nb()
    nk, maj = get_baselines()
    curve = get_k_curve("cosine", "uniform")
    kbest = int(curve.loc[curve["CV accuracy (training set)"].idxmax(), "k"])
    kf = get_knn(kbest, "cosine", "uniform")
    tr_emb, te_emb = get_embeddings(enc_choice)
    full_res, full_head = N.head_scores(tr_emb, SPLIT.train_y.values, te_emb, SPLIT.test_y.values)
    rows = [{"Method": "Majority class (always positive)", "Labelled tweets": len(SPLIT.train_x),
             "Accuracy": maj.test.accuracy, "F1 (weighted)": maj.test.f1_w, "F1 (macro)": maj.test.f1_macro},
            {"Method": "KNN as in the notebook (LabelEncoder)", "Labelled tweets": len(SPLIT.train_x),
             "Accuracy": nk.test.accuracy, "F1 (weighted)": nk.test.f1_w, "F1 (macro)": nk.test.f1_macro},
            {"Method": "Naive Bayes (TF-IDF, α = 0.1)", "Labelled tweets": len(SPLIT.train_x),
             "Accuracy": nbf.test.accuracy, "F1 (weighted)": nbf.test.f1_w, "F1 (macro)": nbf.test.f1_macro},
            {"Method": f"KNN (TF-IDF, cosine, k = {kbest})", "Labelled tweets": len(SPLIT.train_x),
             "Accuracy": kf.test.accuracy, "F1 (weighted)": kf.test.f1_w, "F1 (macro)": kf.test.f1_macro},
            {"Method": f"Embeddings + logistic regression ({ENC.name})", "Labelled tweets": len(SPLIT.train_x),
             "Accuracy": full_res.accuracy, "F1 (weighted)": full_res.f1_weighted, "F1 (macro)": full_res.f1_macro}]
    sf = st.session_state.get("setfit")
    if sf and sf["base"] == ENC.name:
        rows.append({"Method": f"SetFit ({ENC.name}, {sf['n']} per class)", "Labelled tweets": sf["n"] * 3,
                     "Accuracy": sf["after"].accuracy, "F1 (weighted)": sf["after"].f1_weighted,
                     "F1 (macro)": sf["after"].f1_macro})
    comp = pd.DataFrame(rows)
    cc1, cc2 = st.columns([4, 2])
    with cc1:
        st.dataframe(heat(comp, ["Accuracy", "F1 (weighted)", "F1 (macro)"], 4), hide_index=True, width="stretch")
        st.caption("Train SetFit in tab 4 to add its row. All rows use the same 271 test tweets.")
    with cc2:
        st.altair_chart(alt.Chart(comp).mark_bar(color=ACCENT).encode(
            y=alt.Y("Method:N", sort=None, title=None, axis=alt.Axis(labelLimit=230)),
            x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Method", alt.Tooltip("Accuracy:Q", format=".4f")]).properties(height=36 * len(comp) + 40),
            width="stretch")
    show('<div class="callout">The slide\'s comparison chart (Naive Bayes 78%, KNN 75%, SVM 82%, BERT 92%, SetFit 89%) '
         'is an illustration, not a result on this dataset. Here the best methods reach about 0.60 accuracy, only a '
         'few points above always answering <i>positive</i>: 903 short, informal, often ambiguous tweets are a hard '
         'task, and the labels themselves are subjective.</div>')

    st.subheader("Classify your own tweet")
    pick_ex = st.selectbox("Example", EXAMPLES + ["(type your own below)"], key="try_pick")
    user_text = st.text_area("Tweet", pick_ex if pick_ex in EXAMPLES else "", height=90,
                             key=f"try_text_{EXAMPLES.index(pick_ex) if pick_ex in EXAMPLES else 'own'}")
    if user_text.strip():
        stp = T.preprocess(user_text)
        show('<table class="steps">' + "".join(
            f"<tr><td>{a}</td><td><code>{esc(b)}</code></td></tr>" for a, b in [
                ("1 · ekphrasis-style cleaning", stp.step01), ("2 · tokens", " · ".join(stp.tokens)),
                ("3 · slang dictionary", " · ".join(stp.normalized)), ("4 · without stopwords", " · ".join(stp.no_stopwords)),
                ("5 · Sastrawi stemming", stp.processed)]) + "</table>")
        if not stp.processed.strip():
            st.info("Nothing is left after preprocessing.")
        else:
            res = []
            nb_p = nbf.model.best_estimator_.predict_proba([stp.processed])[0]
            res.append(("Naive Bayes", dict(zip(nbf.model.best_estimator_.classes_, nb_p))))
            kn_p = kf.model.predict_proba([stp.processed])[0]
            res.append((f"KNN (k = {kbest}, share of votes)", dict(zip(kf.model.classes_, kn_p))))
            src = user_text if ENC.uses_raw else stp.processed
            emb_p = full_head.predict_proba(ENC.encode([src]))[0]
            res.append(("Embeddings + logistic regression", dict(zip(full_head.classes_, emb_p))))
            if sf and sf["base"] == ENC.name:
                sp = sf["head"].predict_proba(sf["encoder"].encode([user_text]))[0]
                res.append(("SetFit", dict(zip(sf["head"].classes_, sp))))
            prob_rows = [{"Method": m, "Sentiment": C.LABELS[int(c)], "Probability": float(p)}
                         for m, d in res for c, p in d.items()]
            pc1, pc2 = st.columns([3, 2])
            with pc1:
                st.altair_chart(alt.Chart(pd.DataFrame(prob_rows)).mark_bar().encode(
                    y=alt.Y("Method:N", sort=[m for m, _ in res], title=None, axis=alt.Axis(labelLimit=220)),
                    x=alt.X("Probability:Q", stack="normalize", axis=alt.Axis(format="%")),
                    color=alt.Color("Sentiment:N", scale=LABEL_COLORS, legend=alt.Legend(orient="bottom")),
                    order=alt.Order("Sentiment:N", sort="ascending"),
                    tooltip=["Method", "Sentiment", alt.Tooltip("Probability:Q", format=".3f")]
                ).properties(height=60 * len(res) + 80), width="stretch")
            with pc2:
                for m, d in res:
                    best = max(d, key=d.get)
                    st.markdown(f"**{m}** → {C.LABELS[int(best)]} ({d[best]:.0%})")
                st.caption("Preprocessing re-implements the course's ekphrasis + slang dictionary + stopwords + "
                           "Sastrawi pipeline; it reproduces 93% of the dataset's preprocessed tweets exactly.")

st.divider()
st.caption("Hand calculations: handcalc.py · dataset experiment: classifiers.py · preprocessing: textprep.py · "
           "embeddings and SetFit: neural.py. Dataset, slang dictionary and slides: STKI course, Universitas Dian "
           "Nuswantoro. Stopwords: NLTK Indonesian list.")
