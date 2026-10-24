"""
neural.py: the three neural scoring architectures from the lecture, with offline fallbacks.

    Bi-encoder     score = cos(E(q), E(d))           one vector per text; documents encoded offline
    Late inter.    score = Σ_i max_j cos(q_i, d_j)   one vector per token; MaxSim (ColBERT)
    Cross-encoder  score = BERT([q; SEP; d])         query and document read together; slow, most accurate

Models, in order of preference:

    Bi-encoder     sentence-transformers/all-MiniLM-L6-v2 (384 d)      → averaged GloVe 50d word vectors
    Token vectors  the same MiniLM, contextual WordPiece token outputs   → one GloVe vector per word
    Cross-encoder  cross-encoder/ms-marco-MiniLM-L-6-v2                 → none (reranking falls back to MaxSim)

Note on late interaction: a real ColBERTv2 checkpoint is trained end to end for MaxSim and projects every token to
128 dimensions. This lab reuses MiniLM's token outputs, which were trained for mean pooling, not for MaxSim, so it
shows how late interaction scores and what it stores, not ColBERTv2's accuracy.

This module does not depend on Streamlit. Try it from the terminal:
    python neural.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

import glove
from engine import STOPWORDS

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class Encoders:
    name: str
    dims: int
    encode: Callable[[list[str]], np.ndarray]                      # unit-length sentence vectors
    tokens: Callable[[str], tuple[list[str], np.ndarray]]          # (token strings, unit-length token vectors)
    real: bool
    token_note: str


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def glove_encoders() -> Encoders:
    g = glove.load()

    def encode(texts: list[str]) -> np.ndarray:
        return np.stack([g.text_vector(t, STOPWORDS) for t in texts]).astype(np.float32)

    def tokens(text: str) -> tuple[list[str], np.ndarray]:
        words = [w for w in re.findall(r"[a-z]+", text.lower()) if w in g.index and w not in STOPWORDS]
        if not words:
            return [], np.zeros((0, g.vectors.shape[1]), dtype=np.float32)
        return words, np.stack([g.vec(w) for w in words])

    return Encoders("Averaged GloVe word vectors (offline fallback)", g.vectors.shape[1], encode, tokens, False,
                    "one static GloVe vector per word (the same word always gets the same vector)")


def load_encoders(prefer_minilm: bool = True) -> tuple[Encoders, str | None]:
    """Return (encoders, reason MiniLM could not be used or None)."""
    if not prefer_minilm:
        return glove_encoders(), None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MINILM, device="cpu")
    except ImportError:
        return glove_encoders(), "sentence-transformers is not installed"
    except Exception as e:                                   # no internet, Hub unavailable, out of memory, …
        return glove_encoders(), f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"

    tok = model.tokenizer
    special = set(tok.all_special_tokens)
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def tokens(text: str) -> tuple[list[str], np.ndarray]:
        emb = model.encode([text], output_value="token_embeddings")[0]
        emb = np.asarray(emb.detach().cpu().numpy() if hasattr(emb, "detach") else emb, dtype=np.float32)
        ids = tok(text, truncation=True, max_length=model.max_seq_length)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)[:len(emb)]
        keep = [i for i, p in enumerate(pieces)
                if p not in special and re.search(r"[a-z0-9]", p.lower()) and p.lower() not in STOPWORDS]
        return [pieces[i] for i in keep], _unit(emb[keep]) if keep else np.zeros((0, emb.shape[1]), np.float32)

    return Encoders("all-MiniLM-L6-v2 (sentence-transformers)", dim_fn(), encode, tokens, True,
                    "contextual MiniLM WordPiece token outputs (the same word gets different vectors in "
                    "different sentences)"), None


@dataclass
class CrossEncoderScorer:
    name: str
    predict: Callable[[list[tuple[str, str]]], np.ndarray]


def load_cross_encoder(prefer: bool = True) -> tuple[CrossEncoderScorer | None, str | None]:
    if not prefer:
        return None, None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(CROSS, device="cpu")
    except ImportError:
        return None, "sentence-transformers is not installed"
    except Exception as e:
        return None, f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
    return CrossEncoderScorer("ms-marco-MiniLM-L-6-v2 cross-encoder",
                              lambda pairs: np.asarray(model.predict(pairs, show_progress_bar=False),
                                                       dtype=np.float32).reshape(-1)), None


def maxsim(Q: np.ndarray, D: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Late interaction score Σ_i max_j cos(q_i, d_j). Returns (score, similarity matrix, best d_j per q_i)."""
    if len(Q) == 0 or len(D) == 0:
        return 0.0, np.zeros((len(Q), len(D))), np.zeros(len(Q), dtype=int)
    S = Q @ D.T
    best = S.argmax(axis=1)
    return float(S.max(axis=1).sum()), S, best


if __name__ == "__main__":
    enc, why = load_encoders()
    print("encoders:", enc.name, enc.dims, "" if why is None else f"(fallback: {why})")
    q = "late interaction token matching"
    d = "ColBERT keeps a contextual embedding for every token and sums the maximum similarity per query token"
    qt, Q = enc.tokens(q)
    dt, D = enc.tokens(d)
    score, S, best = maxsim(Q, D)
    print("MaxSim:", round(score, 4), [(a, dt[j]) for a, j in zip(qt, best)])
    print("bi-encoder cos:", float(enc.encode([q])[0] @ enc.encode([d])[0]))
    ce, why = load_cross_encoder()
    print("cross-encoder:", None if ce is None else ce.name, why or "")
