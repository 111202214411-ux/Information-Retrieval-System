"""
embeddings.py: document embeddings for BERTopic, with an offline fallback.

  all-MiniLM-L6-v2                         the model in the slide's BERTopic code (English, 384 d, about 90 MB)
  paraphrase-multilingual-MiniLM-L12-v2    50+ languages including Indonesian (384 d, about 470 MB)
  LSA (offline)                            TF-IDF reduced with truncated SVD; no download, no PyTorch

Transformers read the raw text (URLs and @mentions removed); LSA reads the cleaned text.
This module does not depend on Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
LSA = "LSA (offline)"


@dataclass
class Encoder:
    name: str
    encode: Callable[[list[str]], np.ndarray] | None   # None for LSA, which is fitted per corpus
    real: bool


def _clean(t: str) -> str:
    return " ".join(re.sub(r"https?://\S+|@\w+", " ", t).split())


def load(choice: str) -> tuple[Encoder, str | None]:
    """Return (encoder, reason the transformer could not be used or None)."""
    if choice == LSA:
        return Encoder(LSA, None, False), None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[choice], device="cpu")
    except ImportError:
        return Encoder(LSA, None, False), "sentence-transformers is not installed"
    except Exception as e:                                       # no internet, Hub unavailable, out of memory, …
        return Encoder(LSA, None, False), f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode([_clean(t) for t in texts], normalize_embeddings=True, batch_size=64,
                                       show_progress_bar=False), dtype=np.float32)

    return Encoder(choice, encode, True), None


def lsa(texts: list[str], stopwords=frozenset(), dims: int = 100, seed: int = 0) -> np.ndarray:
    X = TfidfVectorizer(stop_words=sorted(stopwords) or None).fit_transform(texts)
    n = max(2, min(dims, X.shape[0] - 1, X.shape[1] - 1))
    Z = TruncatedSVD(n_components=n, random_state=seed).fit_transform(X)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    return (Z / np.where(norm == 0, 1, norm)).astype(np.float32)


def embed(encoder: Encoder, raw: list[str], clean: list[str], stopwords=frozenset()) -> np.ndarray:
    return encoder.encode(raw) if encoder.real else lsa(clean, stopwords)
