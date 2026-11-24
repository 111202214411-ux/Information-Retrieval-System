"""
corpora.py: the three document collections of the lab.

  Slide example        the 8 documents of the P09 K-Means slide (no labels)
  STKI collection      45 short English documents in 3 domains (search technology, health & nutrition, personal
                       finance), written for the Session 7 lab; the domains serve as ground truth for clustering
  COVID-19 tweets      903 Indonesian tweets from the course dataset, already preprocessed by the course pipeline
                       (ekphrasis, slang dictionary, NLTK stopwords, Sastrawi); labelled with sentiment, not topic

Each collection has `raw` texts (read by the sentence-transformers) and `clean` texts (read by TF-IDF, K-Means, LDA and
c-TF-IDF), plus the stopwords those term-based methods should ignore.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_DIR = Path(__file__).parent / "data"

SLIDE_DOCS = ["machine learning algorithm neural network deep",
              "information retrieval search query document",
              "natural language processing text analysis",
              "neural network deep learning classification",
              "search engine web crawling indexing",
              "text mining sentiment analysis opinion",
              "reinforcement learning reward policy agent",
              "document ranking relevance scoring evaluation"]

# ekphrasis annotations that the course's preprocessing turned into words; not topical
EKPHRASIS_TAGS = {"number", "allcaps", "repeated", "elongated", "date", "time", "percent", "hashtag", "censored",
                  "emphasis", "user", "url", "happy", "sad", "devil", "tong", "wink", "laugh", "annoyed"}

NAMES = ["Slide example (8 documents)", "STKI collection (45 documents, 3 domains)",
         "COVID-19 tweets (903, Indonesian)"]


@dataclass
class Corpus:
    name: str
    ids: list[str]
    raw: list[str]
    clean: list[str]
    labels: list[str] | None          # ground truth, when there is one
    label_kind: str                   # what the labels are
    stopwords: frozenset[str] = field(default_factory=frozenset)
    default_k: int = 3
    default_min_size: int = 3
    language: str = "en"


def load(name: str) -> Corpus:
    if name == NAMES[0]:
        return Corpus(name, [f"d{i + 1}" for i in range(8)], SLIDE_DOCS, SLIDE_DOCS, None, "", frozenset(),
                      3, 2)
    if name == NAMES[1]:
        data = json.loads((DATA_DIR / "stki_collection.json").read_text(encoding="utf-8"))
        docs = data["documents"]
        return Corpus(name, [d["id"] for d in docs], [d["text"] for d in docs], [d["text"].lower() for d in docs],
                      [d["domain"] for d in docs], "domain", frozenset(ENGLISH_STOP_WORDS), 3, 3)
    df = pd.read_csv(DATA_DIR / "tweets_sentiment.csv")
    names = {-1: "negative", 0: "neutral", 1: "positive"}
    return Corpus(name, [f"t{i}" for i in df["id"]], list(df["tweet"]), list(df["processed"].astype(str)),
                  [names[s] for s in df["sentiment"]], "sentiment", frozenset(EKPHRASIS_TAGS), 6, 10, "id")


if __name__ == "__main__":
    for n in NAMES:
        c = load(n)
        print(f"{n}: {len(c.raw)} documents, labels: {c.label_kind or 'none'}, first: {c.clean[0][:60]}")
