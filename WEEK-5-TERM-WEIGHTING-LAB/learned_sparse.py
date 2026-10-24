"""
learned_sparse.py: learned sparse representations (SPLADE) and an offline stand-in.

SPLADE (Formal et al.): run the text through BERT's masked-language-model head, which gives a score
for every word of the 30,522-word BERT vocabulary at every input position. Then

    w_j = max_i log(1 + ReLU(logit_ij))          (max pooling over input positions i)

Most w_j are 0, so the result is a sparse vector over the vocabulary. Words that are not in the
text can still get a weight (term expansion: "car" activates "vehicle"), and the vectors fit in an
ordinary inverted index. The score is the dot product of the query and document vectors.

Encoders, in order of preference:
  1. naver/splade-cocondenser-ensembledistil (BERT-base, about 440 MB), via `transformers`.
  2. Offline stand-in: each word activates itself and its nearest GloVe neighbors (cosine above a
     threshold), and the same log(1 + ReLU(x)) and max pooling are applied. It shows the idea of
     term expansion but it is NOT a trained SPLADE model.

This module does not depend on Streamlit. Try it from the terminal:
    python learned_sparse.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from weighting import STOPWORDS

DATA_DIR = Path(__file__).parent / "data"
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


@dataclass
class SparseEncoder:
    name: str
    vocab_size: int
    encode: Callable[[str], dict[str, float]]     # text -> {term: weight > 0}
    real: bool                                    # True for a trained SPLADE model


def _splade_encoder(model_name: str) -> SparseEncoder:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    id2tok = {i: t for t, i in tok.get_vocab().items()}

    def encode(text: str) -> dict[str, float]:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits                                   # (1, L, V)
        act = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
        w = act.max(dim=1).values[0]                                        # max pooling -> (V,)
        idx = torch.nonzero(w > 0).flatten().tolist()
        out = {id2tok[i]: float(w[i]) for i in idx}
        for special in tok.all_special_tokens:
            out.pop(special, None)
        return out

    return SparseEncoder(f"SPLADE ({model_name.split('/')[-1]})", len(id2tok), encode, real=True)


@lru_cache(maxsize=1)
def _glove():
    data = np.load(DATA_DIR / "glove_50d_30k.npz")
    words = [str(w) for w in data["words"]]
    vecs = data["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return words, {w: i for i, w in enumerate(words)}, vecs


def _expansion_encoder(top_k: int = 5, threshold: float = 0.6) -> SparseEncoder:
    words, index, vecs = _glove()

    @lru_cache(maxsize=4096)
    def neighbors(word: str) -> tuple[tuple[str, float], ...]:
        if word not in index:
            return ()
        sims = vecs @ vecs[index[word]]
        best = np.argsort(-sims)[1:top_k + 1]
        return tuple((words[i], float(sims[i])) for i in best if sims[i] >= threshold)

    def encode(text: str) -> dict[str, float]:
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
        logits: dict[str, float] = {}
        for t in tokens:
            # each occurrence "activates" the word itself (logit 2) and its neighbors (logit 2·cos)
            for term, act in ((t, 2.0),) + tuple((w, 2.0 * s) for w, s in neighbors(t)):
                logits[term] = max(logits.get(term, 0.0), act)            # max pooling
        return {t: math.log1p(max(x, 0.0)) for t, x in logits.items() if x > 0}

    return SparseEncoder("Offline term expansion (GloVe neighbors, not a trained SPLADE)", len(words),
                         encode, real=False)


def load_sparse_encoder(prefer_splade: bool) -> tuple[SparseEncoder, str | None]:
    """Return (encoder, reason SPLADE could not be used, or None)."""
    if prefer_splade:
        try:
            return _splade_encoder(SPLADE_MODEL), None
        except ImportError:
            reason = "PyTorch / transformers are not installed"
        except Exception as e:                    # no internet, Hub unavailable, out of memory, …
            reason = f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
        return _expansion_encoder(), reason
    return _expansion_encoder(), None


def sparse_dot(q: dict[str, float], d: dict[str, float]) -> tuple[float, list[tuple[str, float, float]]]:
    """Dot product over shared terms, and the matching terms (term, query weight, document weight)."""
    shared = [(t, q[t], d[t]) for t in q if t in d]
    return sum(a * b for _, a, b in shared), sorted(shared, key=lambda x: -x[1] * x[2])


if __name__ == "__main__":
    enc, why = load_sparse_encoder(prefer_splade=False)
    print(enc.name)
    for text in ["car engine maintenance", "Automobile maintenance checklist: tires, brakes, battery."]:
        vec = enc.encode(text)
        top = sorted(vec.items(), key=lambda x: -x[1])[:12]
        print(f"  {text!r}: {len(vec)} non-zero terms ->", ", ".join(f"{t} {w:.2f}" for t, w in top))
