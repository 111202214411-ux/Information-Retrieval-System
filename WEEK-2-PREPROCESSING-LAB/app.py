"""
app.py: Streamlit interface for Preprocessing Lab.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json

import altair as alt
import pandas as pd
import streamlit as st

from preprocessing import (DATA_DIR, LANGUAGES, STEMMERS, STOPLISTS, Options, lemmatize,
                           load_stoplist, naive_stem_id, preprocess, stem, term_frequencies,
                           vocabulary_sizes)
from subword import (PRESET_CORPORA, PRETRAINED, UNK, bpe_train, load_pretrained, parse_corpus,
                     word_frequencies, word_level_tokenize, wordpiece_train)

st.set_page_config(page_title="Preprocessing Lab", page_icon="🧪", layout="wide")

SAMPLES = json.loads((DATA_DIR / "samples.json").read_text(encoding="utf-8"))
DOCS = {d["name"]: d for d in SAMPLES["documents"]}
CUSTOM = "Custom text"

# ---------------------------------------------------------------------------
# Styles: translucent colors so chips read in both light and dark themes
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 2px}
.chip{font-family:"Courier New",monospace;font-size:13.5px;border:1px solid rgba(128,140,160,.45);
  border-radius:4px;padding:2px 8px;line-height:1.5}
.chip.removed{opacity:.45;text-decoration:line-through}
.chip.kept{border-color:currentColor;font-weight:700}
.chip.phrase{background:rgba(11,114,133,.16);border-color:rgba(11,114,133,.7);font-weight:700}
.chip.changed{background:rgba(232,160,40,.16);border-color:rgba(232,160,40,.7)}
.chip.junk{background:rgba(207,58,63,.12);border-color:rgba(207,58,63,.55);text-decoration:line-through}
.chip.unk{background:rgba(207,58,63,.14);border-color:rgba(207,58,63,.7);font-weight:700}
.step{display:flex;align-items:baseline;gap:10px;margin:18px 0 2px}
.step .n{font-family:"Courier New",monospace;font-weight:700;font-size:13px;border:1.5px solid currentColor;
  border-radius:50%;width:24px;height:24px;display:inline-flex;align-items:center;justify-content:center;flex:none}
.step .t{font-weight:700;font-size:16px}
.step .d{font-size:13px;opacity:.7}
.mono{font-family:"Courier New",monospace;font-size:14px;padding:8px 12px;border-radius:6px;
  background:rgba(128,140,160,.12);white-space:pre-wrap;word-break:break-word}
.def{border-left:3px solid rgba(11,114,133,.7);padding:4px 0 4px 12px;font-size:14px}
</style>
""",
    unsafe_allow_html=True,
)


def chips(items, cls: str = "") -> str:
    if not items:
        return '<div class="chips"><span class="chip removed">(none)</span></div>'
    return '<div class="chips">' + "".join(
        f'<span class="chip {c}">{html.escape(t)}</span>' for t, c in
        ((i, cls) if isinstance(i, str) else i for i in items)) + "</div>"


def step(n: int | str, title: str, desc: str) -> None:
    st.markdown(f'<div class="step"><span class="n">{n}</span><span class="t">{title}</span>'
                f'<span class="d">{desc}</span></div>', unsafe_allow_html=True)


def show(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Downloading tokenizer from the Hugging Face Hub…")
def get_pretrained(name: str):
    return load_pretrained(name)


# ---------------------------------------------------------------------------
# Session state and callbacks
# ---------------------------------------------------------------------------
if "sample" not in st.session_state:
    st.session_state.sample = "Santiago (Indonesian)"
    st.session_state.doc_text = DOCS["Santiago (Indonesian)"]["text"]
    st.session_state.lang = "id"


def on_sample_change() -> None:
    doc = DOCS.get(st.session_state.sample)
    if doc:
        st.session_state.doc_text = doc["text"]
        st.session_state.lang = doc["lang"]


def switch_to_indonesian() -> None:
    st.session_state.lang = "id"


# ---------------------------------------------------------------------------
# Sidebar: pipeline settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline settings")
    lang = st.radio("Language", list(LANGUAGES), format_func=LANGUAGES.get, key="lang", horizontal=True)

    st.markdown("**1 · Parsing & cleaning**")
    rm_html = st.checkbox("Remove HTML tags and entities", True)
    rm_urls = st.checkbox("Remove URLs", True)
    rm_emails = st.checkbox("Remove email addresses", True)

    st.markdown("**2 · Case folding**")
    folding = st.checkbox("Convert to lowercase", True)

    st.markdown("**3 · Tokenization**")
    rm_numbers = st.checkbox("Treat numbers as delimiters", True)
    rm_punct = st.checkbox("Treat punctuation as delimiters", True)

    st.markdown("**4 · Filtering**")
    method_label = st.radio("Method", ["Stoplist", "Wordlist", "None"], horizontal=True,
                            help="Stoplist removes words that are on a list of non-descriptive words. "
                                 "Wordlist keeps only words that are on a list of descriptive words.")
    stoplist, wordlist = list(STOPLISTS[lang])[0], frozenset()
    if method_label == "Stoplist":
        stoplist = st.selectbox(
            "Stopword list", list(STOPLISTS[lang]), key=f"stoplist_{lang}",
            format_func=lambda k: f"{STOPLISTS[lang][k]} · {len(load_stoplist(lang, k))} words")
    elif method_label == "Wordlist":
        key = f"wordlist_{lang}"
        if key not in st.session_state:
            st.session_state[key] = ", ".join(SAMPLES["wordlists"][lang])
        wordlist = frozenset(w.strip().lower() for w in st.session_state[key].replace("\n", ",").split(",")
                             if w.strip())
        st.text_area("Words to keep (comma-separated)", key=key, height=80)

    st.markdown("**5 · Phrase detection**")
    detect = st.checkbox("Join phrases from a thesaurus", False)
    phrases: tuple[str, ...] = ()
    if detect:
        key = f"phrases_{lang}"
        if key not in st.session_state:
            st.session_state[key] = "\n".join(SAMPLES["phrases"][lang])
        st.text_area("Thesaurus (one phrase per line)", key=key, height=110)
        phrases = tuple(p for p in st.session_state[key].splitlines() if p.strip())

    st.markdown("**6 · Stemming / lemmatization**")
    norm_label = st.radio("Normalization", ["Stemming", "Lemmatization", "None"], horizontal=True)
    stemmer = list(STEMMERS[lang])[0]
    if norm_label == "Stemming":
        stemmer = st.selectbox("Stemmer", list(STEMMERS[lang]), format_func=STEMMERS[lang].get,
                               key=f"stemmer_{lang}")

opt = Options(
    lang=lang, remove_html=rm_html, remove_urls=rm_urls, remove_emails=rm_emails,
    case_folding=folding, remove_numbers=rm_numbers, remove_punctuation=rm_punct,
    filter_method=method_label.lower(), stoplist=stoplist, wordlist=wordlist,
    detect_phrases=detect, phrases=phrases,
    normalization={"Stemming": "stem", "Lemmatization": "lemma", "None": "none"}[norm_label],
    stemmer=stemmer,
)
norm_name = (STEMMERS[lang][stemmer] if opt.normalization == "stem"
             else "simplemma" if opt.normalization == "lemma" else "none")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.caption("INFORMATION RETRIEVAL · SESSION 2 · DOCUMENT PREPROCESSING")
st.title("🧪 Preprocessing Lab")
st.write(
    "Turn raw text into **index terms**, one step at a time. Adjust each step in the sidebar and see "
    "what it does to tokens, types, and terms, then compare classic stemming with the **subword "
    "tokenization** (BPE, WordPiece) behind BERT and GPT."
)

tab_pipe, tab_stem, tab_sub, tab_case = st.tabs(
    ["1 · Pipeline", "2 · Stemming vs lemmatization", "3 · Subword tokenization", "4 · Case study"])

# ===========================================================================
# Tab 1: pipeline
# ===========================================================================
with tab_pipe:
    left, right = st.columns([1, 2])
    with left:
        st.selectbox("Sample document", list(DOCS) + [CUSTOM], key="sample", on_change=on_sample_change)
    with right:
        note = DOCS.get(st.session_state.sample, {}).get("note", "Type or paste any text below.")
        st.markdown(f"<div style='padding-top:34px;font-size:14px;opacity:.75'>{html.escape(note)}</div>",
                    unsafe_allow_html=True)
    st.text_area("Document", key="doc_text", height=90)

    r = preprocess(st.session_state.doc_text, opt)
    n_tokens, n_types, n_terms = len(r.tokens), len(r.types), len(r.terms)
    reduction = (1 - n_terms / n_types) * 100 if n_types else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens", n_tokens, help="Every word occurrence, duplicates included.")
    c2.metric("Types", n_types, help="Distinct tokens.")
    c3.metric("Terms", n_terms, help="Distinct tokens after filtering and normalization: the index terms.")
    c4.metric("Vocabulary reduction", f"{reduction:.0f}%", help="1 − |terms| / |types|")

    step(1, "Parsing & cleaning", "remove parts that carry no content: markup, links, addresses")
    removed = [(item, "junk") for items in r.removed.values() for item in items]
    if removed:
        show(chips(removed))
    show(f'<div class="mono">{html.escape(r.cleaned) or "(empty)"}</div>')

    step(2, "Case folding", "Bermain, bermain, BERMAIN → bermain" if folding else "turned off")
    show(f'<div class="mono">{html.escape(r.folded) or "(empty)"}</div>')

    step(3, "Tokenization", f"{n_tokens} tokens · split on spaces"
         + (", numbers" if rm_numbers else "") + (", and punctuation" if rm_punct else ""))
    show(chips(r.tokens))

    desc = {"stoplist": f"stoplist · {STOPLISTS[lang][stoplist]} ({len(load_stoplist(lang, stoplist))} words)",
            "wordlist": f"wordlist · keep {len(wordlist)} listed words", "none": "turned off"}[opt.filter_method]
    step(4, "Filtering", desc)
    show(chips([(t.token, "kept" if t.kept else "removed") for t in r.trace]))

    step(5, "Phrase detection", "thesaurus lookup on adjacent words" if detect else "turned off")
    if detect:
        show(chips([(u.replace("_", " "), "phrase" if "_" in u else "") for u in r.with_phrases]))

    step(6, "Stemming / lemmatization", norm_name)
    show(chips([(f"{u} → {t}" if u != t else t, "changed" if u != t else "")
                for u, t in zip(r.with_phrases, r.terms_seq)]))

    st.markdown("#### Index terms")
    show(chips(r.terms, "kept"))

    st.markdown("#### What happened to each token")
    st.caption("The same table as in the lecture slides. “–” means the token was removed; “↑” means the token "
               "is part of the phrase above.")
    trace_df = pd.DataFrame([{
        "Token": t.token,
        "Filtering": t.token if t.kept else "–",
        **({"Phrase": (t.phrase or "").replace("_", " ")} if detect else {}),
        "Stemming / lemma": t.term if t.kept else "–",
    } for t in r.trace])
    st.dataframe(trace_df, hide_index=True, width="stretch", height=min(38 + 35 * len(trace_df), 420))

    st.markdown("#### Tokens, types, and terms")
    a, b, c = st.columns(3)
    with a:
        show('<div class="def"><b>Token</b>: every word cut from the text, duplicates included.</div>')
        show(chips(r.tokens))
    with b:
        show('<div class="def"><b>Type</b>: a token written once, even if it appears several times.</div>')
        show(chips(r.types))
    with c:
        show('<div class="def"><b>Term</b>: a type after normalization (filtering, stemming, …).</div>')
        show(chips(r.terms, "kept"))

    st.markdown("#### The pipeline as set operations")
    st.latex(rf"T(d)=\{{t_1,\dots,t_n\}},\quad |T(d)| = {n_tokens}")
    st.latex(rf"T'(d)=T(d)-S,\quad |T'(d)| = {len(r.filtered)}")
    st.latex(rf"T''(d)=\{{\,\mathrm{{stem}}(t)\mid t\in T'(d)\,\}},\quad |T''(d)| = {len(r.terms_seq)}")
    st.latex(rf"|V| = {n_types} \;\rightarrow\; |V'| = {n_terms}")

    with st.expander("Why preprocessing matters"):
        st.markdown(
            "- Raw text contains noise: HTML tags, special characters, numbers.\n"
            "- The same word can be written differently: *Bermain*, *bermain*, *BERMAIN*.\n"
            "- Stopwords such as *yang*, *dan*, *di*, *ke* carry little meaning.\n"
            "- Morphological variants (*memainkan*, *dimainkan*, *permainan*) share one root: *main*.\n"
            "- Preprocessing turns raw text into a consistent representation, and its quality "
            "sets the quality of retrieval.")

# ===========================================================================
# Tab 2: stemming vs lemmatization
# ===========================================================================
with tab_stem:
    st.write("Compare stemmers and a lemmatizer on the same words. **Algorithmic** stemmers strip "
             "anything that looks like an affix; **dictionary-based** methods check the result against "
             "a list of real root words.")
    defaults = {
        "id": "makan memutuskan mencari pembelajaran pelaksanaan permainan dimainkan menceritakan "
              "perhatiannya keringanan",
        "en": "applied words texts running happiness generously organization saw seen mice better studies",
    }
    words_key = f"stem_words_{lang}"
    if words_key not in st.session_state:
        st.session_state[words_key] = defaults[lang]
    words = st.text_input("Words (separated by spaces)", key=words_key).lower().split()

    if lang == "id":
        cols = {"Naive (no dictionary)": lambda w: naive_stem_id(w),
                "Sastrawi (Nazief–Adriani)": lambda w: stem(w, "id", "sastrawi"),
                "Lemma (simplemma)": lambda w: lemmatize(w, "id")}
    else:
        cols = {"Porter": lambda w: stem(w, "en", "porter"),
                "Snowball": lambda w: stem(w, "en", "snowball"),
                "Lancaster": lambda w: stem(w, "en", "lancaster"),
                "Lemma (simplemma)": lambda w: lemmatize(w, "en")}
    table = pd.DataFrame([{"Word": w, **{k: f(w) for k, f in cols.items()}} for w in words])

    if not table.empty:
        lemma_col = "Lemma (simplemma)"

        def mark(row):
            return ["" if c in ("Word", lemma_col) or row[c] == row[lemma_col]
                    else "background-color: rgba(232,160,40,.22)" for c in row.index]

        st.dataframe(table.style.apply(mark, axis=1), hide_index=True, width="stretch")
        st.caption("Highlighted cells differ from the lemma. A difference is not always an error: a stem only "
                   "has to be the same for related words, it does not have to be a real word.")

    a, b = st.columns(2)
    with a:
        st.markdown("##### Algorithmic stemming")
        if lang == "id":
            st.markdown(
                "Detects prefixes and suffixes by pattern and removes them.\n\n"
                "- **Pro:** fast, needs no word list.\n"
                f"- **Con:** removes letters that only look like affixes: *makan* → *{naive_stem_id('makan')}* "
                f"(the *-an* belongs to the root), *mencari* → *{naive_stem_id('mencari')}*, "
                f"*memutuskan* → *{naive_stem_id('memutuskan')}*.\n\n"
                "Sastrawi also strips affixes, but only accepts a result that is in its root-word dictionary "
                "(the Nazief–Adriani approach), so *makan* stays *makan*.")
        else:
            st.markdown(
                "Porter (1980) removes English suffixes with ordered rules; Snowball is its refined "
                "successor, and Lancaster is more aggressive.\n\n"
                "- **Pro:** fast, no dictionary.\n"
                f"- **Con:** stems are often not words (*happiness* → *{stem('happiness', 'en', 'porter')}*), "
                f"and irregular forms are missed (*saw* → *{stem('saw', 'en', 'porter')}*).")
    with b:
        st.markdown("##### Lemmatization")
        st.markdown(
            "Uses a vocabulary and morphological analysis to return the dictionary form (lemma).\n\n"
            "- **Pro:** handles irregular forms and returns real words"
            + (f": *saw*, *seen* → *{lemmatize('saw', 'en')}*, *mice* → *{lemmatize('mice', 'en')}*.\n"
               if lang == "en" else
               f": *pembelajaran* → *{lemmatize('pembelajaran', 'id')}* "
               f"(Sastrawi: *{stem('pembelajaran', 'id', 'sastrawi')}*).\n")
            + "- **Con:** needs a dictionary for each language and is slower.")

# ===========================================================================
# Tab 3: subword tokenization
# ===========================================================================
with tab_sub:
    st.write("Subword tokenizers learn their vocabulary from data instead of using hand-written rules. "
             "Both start from single characters and merge pairs step by step. **BPE** merges the most "
             "frequent pair; **WordPiece** merges the pair with the highest likelihood score "
             r"$\frac{freq(ab)}{freq(a)\,freq(b)}$ and marks word-internal pieces with `##`.")

    FROM_PIPELINE = "Words from the Pipeline tab document"
    left, right = st.columns([2, 1])
    with left:
        preset = st.selectbox("Training corpus", list(PRESET_CORPORA) + [FROM_PIPELINE], key="sub_preset")
    with right:
        n_merges = st.slider("Number of merges", 0, 40, 10, key="n_merges")

    if preset == FROM_PIPELINE:
        corpus_spec = "\n".join(f"{w} {c}" for w, c in word_frequencies(st.session_state.doc_text).items())
    else:
        corpus_spec = "\n".join(f"{w} {c}" for w, c in PRESET_CORPORA[preset].items())
    spec_key = "corpus_spec::" + hashlib.md5(corpus_spec.encode()).hexdigest()
    with st.expander("Training words and their frequencies (editable)", expanded=False):
        spec = st.text_area("One “word count” per line", value=corpus_spec, key=spec_key, height=160)
    freqs = parse_corpus(spec)

    if not freqs:
        st.info("The training corpus is empty. Add some words above.")
    else:
        bpe, wp = bpe_train(freqs, n_merges), wordpiece_train(freqs, n_merges)

        a, b = st.columns(2)
        for col, model, label, score_name in ((a, bpe, "BPE", "frequency"), (b, wp, "WordPiece", "score")):
            with col:
                st.markdown(f"##### {label} merges")
                st.caption(f"Base vocabulary: {len(model.base_vocab)} symbols → "
                           f"{len(model.vocab)} after {len(model.merges)} merges")
                df = pd.DataFrame([{"Step": m.step, "Pair": f"{m.pair[0]} + {m.pair[1]}",
                                    score_name.capitalize(): round(m.score, 4) if label == "WordPiece" else m.score,
                                    "New token": m.new_token, "|V|": m.vocab_size} for m in model.merges])
                if df.empty:
                    st.write("No merges yet: every word is still split into characters.")
                else:
                    st.dataframe(df, hide_index=True, width="stretch", height=min(38 + 35 * len(df), 390))

        st.markdown("##### Training words after the last merge")
        seg = pd.DataFrame([{"Word": w, "Count": c, "BPE": " · ".join(bpe.splits[w]),
                             "WordPiece": " · ".join(wp.splits[w])} for w, c in freqs.items()])
        st.dataframe(seg, hide_index=True, width="stretch", height=min(38 + 35 * len(seg), 320))
        st.caption(f"BPE marks the end of a word with `</w>`, as in the original paper. "
                   "WordPiece marks pieces that continue a word with `##`.")

        st.markdown("##### Tokenize new words (out-of-vocabulary test)")
        test_defaults = {"Sennrich": "lowest newer wider slow", "Hugging": "hugs bugs mug thug",
                         "Indonesian": "mainan pengajar dipelajari bermainlah"}
        default_test = next((v for k, v in test_defaults.items() if preset.startswith(k)), "")
        test_words = st.text_input("Words to tokenize", value=default_test, key=f"test_{preset}").split()
        rows = []
        for w in test_words:
            wl, bp, wpt = word_level_tokenize(w, freqs), bpe.tokenize(w), wp.tokenize(w)
            rows.append({"Word": w, "Word-level": " · ".join(wl), "BPE": " · ".join(bp),
                         "WordPiece": " · ".join(wpt)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Word-level tokenization turns every unseen word into `{UNK}` (the OOV problem). "
                       "Subword tokenizers can still build it from known pieces. WordPiece falls back to "
                       f"`{UNK}` only when a character was never seen in training.")

    with st.expander("Pretrained tokenizers: BERT, GPT-2, IndoBERT (downloads from the Hugging Face Hub)"):
        sample_text = {"en": "unhappiness is unplayable", "id": "pembelajaran mesin klasifikasi"}[lang]
        text = st.text_input("Text", value=sample_text, key=f"pretrained_text_{lang}")
        if st.button("Tokenize with pretrained models"):
            for name, (model_name, algo) in PRETRAINED.items():
                try:
                    tok = get_pretrained(name)
                    enc = tok.encode(text, add_special_tokens=False)
                    st.markdown(f"**{model_name}** · {algo} · vocabulary {tok.get_vocab_size():,}")
                    show(chips(enc.tokens))
                except Exception as e:  # no internet, Hub down, or rate-limited
                    st.warning(f"{model_name}: could not load the tokenizer from the Hugging Face Hub "
                               f"({type(e).__name__}). This feature needs an internet connection.")
            st.caption("GPT-2 shows a space before a word as `Ġ`.")

    st.markdown("##### Classic vs modern preprocessing")
    st.table(pd.DataFrame({
        "Classic (NLTK, Sastrawi)": ["Word-level tokenization", "Rule-based stemming", "Manual stopword list",
                                     "Language-specific", "Suits TF-IDF and BM25"],
        "Modern (subword)": ["Subword tokenization", "Learned from data", "No explicit stopwords",
                             "Works across languages", "Required by BERT and GPT"],
    }))

# ===========================================================================
# Tab 4: case study
# ===========================================================================
with tab_case:
    LECTURE = "Student feedback (lecture case study, 4 documents)"
    UPLOAD = "Upload your own .txt files"
    source = st.radio("Corpus", [LECTURE, UPLOAD], horizontal=True)

    docs: list[tuple[str, str]] = []
    if source == LECTURE:
        docs = [(d["id"], d["text"]) for d in SAMPLES["case_study"]]
        if lang != "id":
            st.warning("These documents are in Indonesian, but the pipeline is set to English.")
            st.button("Switch the pipeline to Indonesian", on_click=switch_to_indonesian)
    else:
        files = st.file_uploader("Text files (each file is one document)", type=["txt"],
                                 accept_multiple_files=True)
        per_line = st.checkbox("Treat each line as a separate document")
        for f in files or []:
            content = f.getvalue().decode("utf-8", errors="replace")
            if per_line:
                docs += [(f"{f.name}:{i}", line) for i, line in enumerate(content.splitlines(), 1) if line.strip()]
            else:
                docs.append((f.name, content))
        if not docs:
            st.info("Upload one or more .txt files. They are processed with the settings in the sidebar.")

    if docs:
        results = [preprocess(t, opt) for _, t in docs]
        st.markdown("##### Document by document")
        view = pd.DataFrame([{
            "Doc": d, "Text": t, "Tokenization": " ".join(r.tokens),
            "Filtering": " ".join(r.with_phrases), "Stemming / lemma": " ".join(r.terms_seq),
        } for (d, t), r in zip(docs, results)])
        if len(view) <= 8:
            st.table(view.set_index("Doc"))
        else:
            st.dataframe(view, hide_index=True, width="stretch")

        st.markdown("##### Vocabulary reduction across the corpus")
        sizes = vocabulary_sizes([t for _, t in docs], opt)
        a, b = st.columns([3, 2])
        with a:
            chart_df = pd.DataFrame({"Stage": [f"{i} · {k} ({v})" for i, (k, v) in enumerate(sizes.items(), 1)],
                                     "Vocabulary size |V|": list(sizes.values())})
            bars = alt.Chart(chart_df).mark_bar(color="#0b7285", cornerRadiusEnd=3).encode(
                x=alt.X("Vocabulary size |V|:Q"),
                y=alt.Y("Stage:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
                tooltip=["Stage", "Vocabulary size |V|"])
            st.altair_chart(bars.properties(height=220), width="stretch")
        with b:
            first, last = list(sizes.values())[0], list(sizes.values())[-1]
            st.metric("Documents", len(docs))
            st.metric("Tokens", sum(len(r.tokens) for r in results))
            st.metric("Vocabulary", f"{first} → {last}",
                      delta=f"-{(1 - last / first) * 100:.0f}%" if first else None, delta_color="inverse")

        st.markdown("##### Term frequencies")
        tf = pd.DataFrame(term_frequencies(results), columns=["Term", "Frequency (tf)", "Documents (df)"])
        st.dataframe(tf, hide_index=True, width="stretch", height=300)
        st.download_button("Download terms as CSV", tf.to_csv(index=False).encode("utf-8"),
                           file_name="terms.csv", mime="text/csv")
        st.caption("These terms are what the next step of an IR system (indexing) builds the inverted index from.")

st.divider()
st.caption(
    "Stopword lists: NLTK (English), Tala (Indonesian, via NLTK), and Sastrawi. Stemmers: Sastrawi and NLTK. "
    "Lemmatizer: simplemma. BPE and WordPiece are implemented from scratch in subword.py."
)
