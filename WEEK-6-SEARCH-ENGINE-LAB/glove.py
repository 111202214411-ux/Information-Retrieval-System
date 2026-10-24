"""
glove.py: the bundled GloVe word vectors (6B, 50 dimensions, the 30,000 most frequent words).

They are used in four places:
  * offline fallback for the dense encoder (average of the word vectors of a text),
  * offline fallback for the token encoder of the late interaction tab (one vector per word),
  * query expansion (nearest neighbours of a query word that also occur in the index),
  * the vector database tab, where the 30,000 vectors are the database being searched.

GloVe: Pennington, Socher & Manning (2014), released under the Open Data Commons PDDL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Glove:
    words: list[str]
    vectors: np.ndarray            # unit-length rows, float32
    index: dict[str, int]

    def has(self, w: str) -> bool:
        return w in self.index

    def vec(self, w: str) -> np.ndarray:
        return self.vectors[self.index[w]]

    def text_vector(self, text: str, stopwords: set[str] = frozenset()) -> np.ndarray:
        ids = [self.index[w] for w in re.findall(r"[a-z]+", text.lower()) if w in self.index and w not in stopwords]
        if not ids:
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        v = self.vectors[ids].mean(axis=0)
        return v / np.linalg.norm(v)


@lru_cache(maxsize=1)
def load() -> Glove:
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    words = [str(w) for w in data["words"]]
    vecs = data["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return Glove(words, vecs, {w: i for i, w in enumerate(words)})
