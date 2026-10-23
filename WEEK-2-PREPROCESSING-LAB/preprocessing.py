"""
preprocessing.py: the classic document preprocessing pipeline.

    raw text
      -> 1. parsing & cleaning   (HTML tags, entities, URLs, email addresses)
      -> 2. case folding         (lowercase)
      -> 3. tokenization         (split into words; drop numbers and punctuation)
      -> 4. filtering            (stoplist or wordlist)
      -> 5. phrase detection     (thesaurus lookup, optional)
      -> 6. stemming / lemmatization
      -> index terms

This module does not depend on Streamlit. Try it from the terminal:
    python preprocessing.py "Pembelajaran mesin digunakan untuk klasifikasi dokumen"
    python preprocessing.py "They are applied to the words in the texts." --lang en
"""

from __future__ import annotations

import argparse
import html
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
LANGUAGES = {"id": "Indonesian", "en": "English"}

STOPLISTS = {
    "id": {"tala": "Tala (NLTK)", "sastrawi": "Sastrawi"},
    "en": {"nltk": "NLTK"},
}
STEMMERS = {
    "id": {"sastrawi": "Sastrawi (Nazief–Adriani)", "naive": "Naive affix stripping (no dictionary)"},
    "en": {"porter": "Porter", "snowball": "Snowball (Porter2)", "lancaster": "Lancaster"},
}


# ---------------------------------------------------------------------------
# Resources (loaded once)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_stoplist(lang: str, name: str) -> frozenset[str]:
    """Read a stopword file from data/stopwords/<lang>_<name>.txt (lines starting with # are comments)."""
    path = DATA_DIR / "stopwords" / f"{lang}_{name}.txt"
    words = [w.strip() for w in path.read_text(encoding="utf-8").splitlines()]
    return frozenset(w for w in words if w and not w.startswith("#"))


@lru_cache(maxsize=None)
def _sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=None)
def _nltk_stemmer(name: str):
    from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer
    return {"porter": PorterStemmer, "lancaster": LancasterStemmer,
            "snowball": lambda: SnowballStemmer("english")}[name]()


# ---------------------------------------------------------------------------
# Stemming and lemmatization
# ---------------------------------------------------------------------------

_PARTICLES = ("lah", "kah", "tah", "pun")
_POSSESSIVES = ("nya", "ku", "mu")
_SUFFIXES = ("kan", "an", "i")
_PREFIXES = ("meng", "meny", "mem", "men", "me", "peng", "peny", "pem", "pen", "per", "pe",
             "ber", "be", "ter", "te", "di", "ke", "se")


def naive_stem_id(word: str) -> str:
    """
    Strip Indonesian affixes by pattern only, with no root-word dictionary.

    Fast, but it cannot tell a real affix from letters that only look like one,
    e.g. 'makan' -> 'mak' (the '-an' is part of the root) and
    'memutuskan' -> 'utus' (the 'p' of 'putus' melts into 'mem-').
    """
    w = word
    for group in (_PARTICLES, _POSSESSIVES, _SUFFIXES):
        for affix in group:
            if w.endswith(affix) and len(w) - len(affix) >= 3:
                w = w[: -len(affix)]
                break
    for _ in range(2):                      # at most two prefixes, e.g. mem-per-
        for affix in _PREFIXES:
            if w.startswith(affix) and len(w) - len(affix) >= 3:
                w = w[len(affix):]
                break
        else:
            break
    return w


def stem(word: str, lang: str, stemmer: str) -> str:
    if lang == "id":
        return _sastrawi().stem(word) if stemmer == "sastrawi" else naive_stem_id(word)
    return _nltk_stemmer(stemmer).stem(word)


def lemmatize(word: str, lang: str) -> str:
    """Dictionary-based lemmatization with simplemma (works offline for Indonesian and English)."""
    import simplemma
    return simplemma.lemmatize(word, lang=lang)


def normalize(word: str, lang: str, method: str, stemmer: str) -> str:
    """Apply stemming or lemmatization to a token. Phrases (joined with '_') are normalized word by word."""
    if method == "none":
        return word
    fn = (lambda w: stem(w, lang, stemmer)) if method == "stem" else (lambda w: lemmatize(w, lang))
    return "_".join(fn(part) for part in word.split("_"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


@dataclass
class Options:
    lang: str = "id"
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    case_folding: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    filter_method: str = "stoplist"            # "stoplist" | "wordlist" | "none"
    stoplist: str = "tala"
    wordlist: frozenset[str] = frozenset()
    detect_phrases: bool = False
    phrases: tuple[str, ...] = ()
    normalization: str = "stem"                # "stem" | "lemma" | "none"
    stemmer: str = "sastrawi"


@dataclass
class TokenTrace:
    """One row of the lecture-style table: what happened to each token."""
    token: str
    kept: bool
    phrase: str | None
    term: str | None


@dataclass
class Result:
    raw: str
    cleaned: str
    removed: dict[str, list[str]]
    folded: str
    tokens: list[str]
    filtered: list[str]
    with_phrases: list[str]
    terms_seq: list[str]                       # normalized tokens, in order (duplicates kept)
    trace: list[TokenTrace] = field(default_factory=list)

    @property
    def types(self) -> list[str]:
        """Unique tokens, first occurrence order."""
        return list(dict.fromkeys(self.tokens))

    @property
    def terms(self) -> list[str]:
        """Unique normalized tokens after filtering: the index terms."""
        return list(dict.fromkeys(self.terms_seq))


def clean(text: str, opt: Options) -> tuple[str, dict[str, list[str]]]:
    """Step 1: remove markup and other non-content parts of a document."""
    removed: dict[str, list[str]] = {}
    if opt.remove_html:
        removed["HTML tags"] = _HTML_TAG.findall(text)
        text = _HTML_TAG.sub(" ", text)
        unescaped = html.unescape(text)
        if unescaped != text:
            removed["HTML entities"] = re.findall(r"&[#\w]+;", text)
        text = unescaped
    if opt.remove_urls:
        removed["URLs"] = _URL.findall(text)
        text = _URL.sub(" ", text)
    if opt.remove_emails:
        removed["Email addresses"] = _EMAIL.findall(text)
        text = _EMAIL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, {k: v for k, v in removed.items() if v}


def tokenize(text: str, opt: Options) -> list[str]:
    """Step 3: split text into tokens. Numbers and punctuation act as delimiters when removed."""
    if opt.remove_numbers and opt.remove_punctuation:
        pattern = r"[^\W\d_]+"                 # letters only
    elif opt.remove_punctuation:
        pattern = r"[^\W_]+"                   # letters and digits
    elif opt.remove_numbers:
        pattern = r"[^\W\d_]+|[^\w\s]"         # letters, punctuation marks as tokens
    else:
        pattern = r"[^\W_]+|[^\w\s]"
    return re.findall(pattern, text)


def _is_kept(token: str, opt: Options) -> bool:
    if opt.filter_method == "stoplist":
        return token not in load_stoplist(opt.lang, opt.stoplist)
    if opt.filter_method == "wordlist":
        return token in opt.wordlist
    return True


def _merge_phrases(kept: list[tuple[int, str]], phrases: tuple[str, ...]) -> list[tuple[list[int], str]]:
    """
    Step 5: join consecutive kept tokens that form a phrase from the thesaurus.
    Tokens must have been adjacent in the original text, so removing a stopword
    between two words never creates a false phrase.
    """
    table = sorted({tuple(p.lower().split()) for p in phrases if p.strip()}, key=len, reverse=True)
    out, i = [], 0
    while i < len(kept):
        for ph in table:
            n = len(ph)
            window = kept[i:i + n]
            if (len(window) == n and tuple(t.lower() for _, t in window) == ph
                    and all(window[k + 1][0] == window[k][0] + 1 for k in range(n - 1))):
                out.append(([idx for idx, _ in window], "_".join(t for _, t in window)))
                i += n
                break
        else:
            out.append(([kept[i][0]], kept[i][1]))
            i += 1
    return out


def preprocess(text: str, opt: Options) -> Result:
    cleaned, removed = clean(text, opt)
    folded = cleaned.lower() if opt.case_folding else cleaned
    tokens = tokenize(folded, opt)

    kept = [(i, t) for i, t in enumerate(tokens) if _is_kept(t, opt)]
    filtered = [t for _, t in kept]

    units = _merge_phrases(kept, opt.phrases) if opt.detect_phrases else [([i], t) for i, t in kept]
    with_phrases = [u for _, u in units]
    terms_seq = [normalize(u, opt.lang, opt.normalization, opt.stemmer) for u in with_phrases]

    # lecture-style trace: one row per original token
    unit_of = {idx: n for n, (idxs, _) in enumerate(units) for idx in idxs}
    trace, shown = [], set()
    for i, t in enumerate(tokens):
        n = unit_of.get(i)
        if n is None:
            trace.append(TokenTrace(t, False, None, None))
            continue
        idxs, unit = units[n]
        term = "↑" if n in shown else terms_seq[n]   # second word of a phrase points up
        shown.add(n)
        trace.append(TokenTrace(t, True, unit if len(idxs) > 1 else None, term))

    return Result(text, cleaned, removed, folded, tokens, filtered, with_phrases, terms_seq, trace)


def vocabulary_sizes(texts: list[str], opt: Options) -> dict[str, int]:
    """|V| after each stage for a whole corpus, to show vocabulary reduction (|V'| << |V|)."""
    raw_opt = Options(**{**opt.__dict__, "case_folding": False})
    names = ["Tokens as written", "Case folding", "Filtering", "Stemming / lemma"]
    stages = {n: set() for n in names}
    for text in texts:
        r_raw = preprocess(text, raw_opt)
        r = preprocess(text, opt)
        stages[names[0]].update(r_raw.tokens)
        stages[names[1]].update(r.tokens)
        stages[names[2]].update(r.with_phrases)
        stages[names[3]].update(r.terms_seq)
    return {k: len(v) for k, v in stages.items()}


def term_frequencies(results: list[Result]) -> list[tuple[str, int, int]]:
    """(term, total frequency, document frequency) across a corpus, most frequent first."""
    tf, df = Counter(), Counter()
    for r in results:
        tf.update(r.terms_seq)
        df.update(set(r.terms_seq))
    return sorted(((t, tf[t], df[t]) for t in tf), key=lambda x: (-x[1], x[0]))


def default_options(lang: str) -> Options:
    return Options(lang=lang, stoplist="tala" if lang == "id" else "nltk",
                   stemmer="sastrawi" if lang == "id" else "porter")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the preprocessing pipeline on a piece of text.")
    ap.add_argument("text", nargs="?", default="Pembelajaran mesin digunakan untuk klasifikasi dokumen")
    ap.add_argument("--lang", choices=list(LANGUAGES), default="id")
    ap.add_argument("--lemma", action="store_true", help="use lemmatization instead of stemming")
    args = ap.parse_args()

    opt = default_options(args.lang)
    if args.lemma:
        opt.normalization = "lemma"
    r = preprocess(args.text, opt)
    print("Raw        :", r.raw)
    print("Cleaned    :", r.cleaned)
    print("Folded     :", r.folded)
    print("Tokens     :", r.tokens)
    print("Filtered   :", r.filtered)
    print("Terms      :", r.terms_seq)
    print(f"|tokens| = {len(r.tokens)}, |types| = {len(r.types)}, |terms| = {len(r.terms)}")
