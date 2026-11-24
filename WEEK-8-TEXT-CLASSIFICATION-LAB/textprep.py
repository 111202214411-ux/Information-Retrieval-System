"""
textprep.py: the course's preprocessing pipeline for Indonesian tweets, so new text can be classified.

The lecture notebook cleans each tweet in two stages:

  step01   ekphrasis (normalize numbers, dates, times and percentages; annotate ALL-CAPS words, elongated words,
           repeated punctuation and hashtags; turn emoticons into words), then remove "RT", lowercase, split on
           hyphens, drop non-ASCII characters, URLs, single letters and punctuation
  step02   tokenize, replace informal words with the "kamus perbaikan kata" (661 pairs), remove the NLTK
           Indonesian stopwords (757 words), then stem with Sastrawi

ekphrasis is a large dependency that downloads word statistics on first use, so `step01()` below re-implements
the parts of it that this dataset uses with regular expressions. The annotations become ordinary words:
"<number>" becomes "number" and "<allcaps> PSBB </allcaps>" becomes "allcaps psbb allcaps". `python textprep.py`
measures how often this re-implementation reproduces the course's own `ProcessedText` column exactly.

This module does not depend on Streamlit.
"""

from __future__ import annotations

import csv
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

EMOTICONS = [  # longest first; the words are the ones ekphrasis uses
    (">:)", "devil"), (">:(", "angry"), (":')", "happy"), (":-)", "happy"), (":)", "happy"), ("(:", "happy"),
    (":-(", "sad"), (":(", "sad"), (":'(", "sad"), (";)", "wink"), (";-)", "wink"), (":d", "laugh"),
    (":-d", "laugh"), ("xd", "laugh"), (":p", "tong"), (":-p", "tong"), ("<3", "heart"), (":*", "kiss"),
    (":o", "surprise"),
]


@lru_cache(maxsize=1)
def kamus() -> dict[str, str]:
    with open(DATA_DIR / "kamus_perbaikan_kata.csv", encoding="utf-8") as f:
        return {r["informal"]: r["standard"] for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    return frozenset(w.strip() for w in (DATA_DIR / "stopwords_id_nltk.txt").read_text(encoding="utf-8").split())


@lru_cache(maxsize=1)
def stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except ImportError:          # PySastrawi missing: skip stemming rather than fail
        return None


MONTHS = ("januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
          "january|february|march|may|june|july|august|october|december|jan|feb|mar|apr|jun|jul|agu|agt|aug|"
          "sep|sept|okt|oct|nov|des|dec")


def _ekphrasis_like(text: str) -> str:
    t = text
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " <user> ", t)
    for emo, word in EMOTICONS:
        t = re.sub(re.escape(emo), f" <{word}> ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " <date> ", t)
    t = re.sub(rf"\b\d{{1,2}}\s+(?:{MONTHS})\b(?:\s+\d{{4}})?", " <date> ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{1,2}[:.]\d{2}\s?(?:wib|wita|wit)?\b", " <time> ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d+(?:[.,]\d+)?\s?%", " <percent> ", t)
    t = re.sub(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?![A-Za-z0-9])", " <number> ", t)
    t = re.sub(r"#(\w+)", r" <hashtag> \1 </hashtag> ", t)
    t = re.sub(r"[!?.,]{2,}", " <repeated> ", t)                           # !!!  ...  ?!..
    # a run of ALL-CAPS words is annotated once, as ekphrasis does: <allcaps> AP II </allcaps>
    # (a single all-caps word needs 3+ letters: RS and RI stay plain, WIB and NKRI are annotated)
    t = re.sub(r"(?<![\w<])([A-Z]{2,}(?:[ \t]+[A-Z]{2,})+|[A-Z]{3,})(?![\w>])", r" <allcaps> \1 </allcaps> ", t)

    def elong(m: re.Match) -> str:
        w = m.group(0)
        return re.sub(r"([a-zA-Z])\1{2,}", r"\1", w) + " <elongated>" if re.search(r"([a-zA-Z])\1\1", w) else w

    t = re.sub(r"(?<![<\w])[A-Za-z]+(?![\w>])", elong, t)
    t = re.sub(r"&\w+;", " ", t)                                           # HTML entities such as &amp;
    t = re.sub(r"[^\w\s<>]", " ", t)          # the ekphrasis tokenizer splits words at punctuation and emoji
    return " ".join(t.split()).lower()


def step01(text: str) -> str:
    """Cleaning stage, mirroring the notebook's order of functions."""
    proc = _ekphrasis_like(text)
    proc = proc.replace("RT", " ").lower()
    proc = proc.replace("-", " ")
    proc = re.sub(r"\s+", " ", proc)
    proc = proc.encode("ascii", "replace").decode("ascii")
    proc = proc.replace("http://", " ").replace("https://", " ")
    proc = re.sub(r"\b[a-zA-Z]\b", "", proc)
    proc = re.sub("[{}]".format(re.escape(string.punctuation.replace("_", ""))), "", proc)
    return " ".join(proc.split())


@dataclass
class Steps:
    raw: str
    step01: str
    tokens: list[str]
    normalized: list[str]
    no_stopwords: list[str]
    processed: str


def preprocess(text: str) -> Steps:
    s1 = step01(text)
    tokens = s1.split()
    k = kamus()
    normalized = [k.get(t, t) for t in tokens]
    sw = stopwords()
    kept = [t for t in normalized if t not in sw]
    st = stemmer()
    joined = " ".join(kept)
    processed = st.stem(joined) if st is not None else joined
    return Steps(text, s1, tokens, normalized, kept, processed)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "tweets_sentiment.csv")
    s1 = [step01(t) for t in df["tweet"]]
    full = [preprocess(t).processed for t in df["tweet"]]
    m1 = sum(a == b for a, b in zip(s1, df["step01"]))
    m2 = sum(a == b for a, b in zip(full, df["processed"]))
    print(f"step01 identical: {m1}/{len(df)} ({m1 / len(df):.1%}); final text identical: {m2}/{len(df)} ({m2 / len(df):.1%})")
    tok_a = [set(a.split()) for a in full]
    tok_b = [set(b.split()) for b in df["processed"]]
    jac = sum(len(a & b) / len(a | b) if a | b else 1 for a, b in zip(tok_a, tok_b)) / len(df)
    print(f"mean token-set Jaccard with the course's ProcessedText: {jac:.3f}")
    shown = 0
    for a, b, t in zip(s1, df["step01"], df["tweet"]):
        if a != b and shown < 3:
            print(" RAW ", t[:150]); print(" COURSE", b[:150]); print(" OURS  ", a[:150]); print()
            shown += 1
