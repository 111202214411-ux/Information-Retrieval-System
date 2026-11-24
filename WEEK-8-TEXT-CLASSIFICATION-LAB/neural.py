"""
neural.py: sentence embeddings, a logistic-regression head, and SetFit-style few-shot fine-tuning.

Encoders, chosen in the app's sidebar:
  all-MiniLM-L6-v2                         the model in the slide's SetFit code (English, 90 MB)
  paraphrase-multilingual-MiniLM-L12-v2    trained on 50+ languages, Indonesian included (470 MB)
  LSA (offline)                            TF-IDF reduced to 100 dimensions with truncated SVD, fitted on the
                                           training tweets; needs no download and works without PyTorch

Transformer encoders read the raw tweet (URLs removed): they have their own tokenizer and need no stemming.
LSA reads the course's preprocessed text.

SetFit (Tunstall et al., 2022) in two steps:
  1. contrastive fine-tuning: from a few labelled examples, make pairs (same class → target 1, different class →
     target 0) and train the encoder so that cos(E(a), E(b)) moves towards the target (CosineSimilarityLoss)
  2. a classification head (logistic regression) on the fine-tuned embeddings
`setfit_finetune()` implements step 1 with plain PyTorch on a copy of the encoder, so the `setfit` package (and
its `datasets` dependency) is not needed.

This module does not depend on Streamlit.
"""

from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
LSA = "LSA (offline)"


def clean_for_transformer(text: str) -> str:
    return " ".join(re.sub(r"https?://\S+|@\w+", " ", text).split())


@dataclass
class Encoder:
    name: str
    dims: int
    encode: Callable[[list[str]], np.ndarray]      # unit-length rows
    uses_raw: bool                                   # True: encode raw tweets; False: preprocessed text
    model: object = None                             # the SentenceTransformer, when there is one


def load_transformer(key: str) -> tuple[Encoder | None, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[key], device="cpu")
    except ImportError:
        return None, "sentence-transformers is not installed"
    except Exception as e:                            # no internet, Hub unavailable, out of memory, …
        return None, f"the model could not be loaded ({type(e).__name__}: {str(e)[:120]})"
    return transformer_encoder(key, model), None


def transformer_encoder(name: str, model) -> Encoder:
    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode([clean_for_transformer(t) for t in texts], normalize_embeddings=True,
                                       batch_size=64, show_progress_bar=False), dtype=np.float32)

    return Encoder(name, dim_fn(), encode, True, model)


def lsa_encoder(train_texts: list[str], dims: int = 100, seed: int = 0) -> Encoder:
    vec = TfidfVectorizer()
    X = vec.fit_transform(train_texts)
    svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1), random_state=seed).fit(X)

    def encode(texts: list[str]) -> np.ndarray:
        Z = svd.transform(vec.transform(texts))
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return (Z / np.where(n == 0, 1, n)).astype(np.float32)

    return Encoder(LSA, svd.n_components, encode, False)


def logistic_head(seed: int = 0) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def sample_per_class(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of n random examples of every class (all of them if a class is smaller)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        idx.extend(rng.choice(pool, size=min(n, len(pool)), replace=False))
    return np.array(sorted(idx))


# ---------------------------------------------------------------------------
# SetFit step 1: contrastive fine-tuning
# ---------------------------------------------------------------------------
def make_pairs(texts: list[str], labels: list[int], iterations: int, seed: int = 0
               ) -> list[tuple[str, str, float]]:
    """For every example and iteration: one positive pair (same class) and one negative pair (another class)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    pairs = []
    for _ in range(iterations):
        for i, y in enumerate(labels):
            same = [j for j in by_class[y] if j != i]
            other = [j for j in range(len(labels)) if labels[j] != y]
            if same:
                pairs.append((texts[i], texts[rng.choice(same)], 1.0))
            if other:
                pairs.append((texts[i], texts[rng.choice(other)], 0.0))
    rng.shuffle(pairs)
    return pairs


def setfit_finetune(model, texts: list[str], labels: list[int], iterations: int = 10, epochs: int = 1,
                    batch_size: int = 16, lr: float = 2e-5, seed: int = 0,
                    progress: Callable[[float, str], None] | None = None):
    """Return a fine-tuned copy of a SentenceTransformer, trained with CosineSimilarityLoss on generated pairs."""
    import torch
    torch.manual_seed(seed)
    ft = copy.deepcopy(model)
    ft.train()
    pairs = make_pairs([clean_for_transformer(t) for t in texts], labels, iterations, seed)
    prep = getattr(ft, "preprocess", None) or ft.tokenize      # sentence-transformers ≥ 6 renamed tokenize
    opt = torch.optim.AdamW(ft.parameters(), lr=lr)
    steps = epochs * ((len(pairs) + batch_size - 1) // batch_size)
    step, t0 = 0, time.perf_counter()
    for _ in range(epochs):
        for b in range(0, len(pairs), batch_size):
            batch = pairs[b:b + batch_size]
            fa = prep([p[0] for p in batch])
            fb = prep([p[1] for p in batch])
            ea = ft(fa)["sentence_embedding"]
            eb = ft(fb)["sentence_embedding"]
            cos = torch.nn.functional.cosine_similarity(ea, eb)
            loss = torch.nn.functional.mse_loss(cos, torch.tensor([p[2] for p in batch], dtype=cos.dtype))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if progress:
                progress(step / steps, f"step {step}/{steps} · loss {loss.item():.4f} · "
                                       f"{time.perf_counter() - t0:.0f} s")
    ft.eval()
    return ft, len(pairs), steps


@dataclass
class FewShotResult:
    accuracy: float
    f1_macro: float
    f1_weighted: float = float("nan")


def head_scores(train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray, test_y: np.ndarray,
                seed: int = 0) -> tuple[FewShotResult, LogisticRegression]:
    head = logistic_head(seed).fit(train_emb, train_y)
    pred = head.predict(test_emb)
    return FewShotResult(accuracy_score(test_y, pred), f1_score(test_y, pred, average="macro"),
                         f1_score(test_y, pred, average="weighted")), head


if __name__ == "__main__":
    import classifiers as C
    df = C.load()
    s = C.split(df)
    enc = lsa_encoder(list(s.train_x))
    tr, te = enc.encode(list(s.train_x)), enc.encode(list(s.test_x))
    full, _ = head_scores(tr, s.train_y.values, te, s.test_y.values)
    print(f"{enc.name}, {enc.dims} dims, full training set: accuracy {full.accuracy:.4f}, macro F1 {full.f1_macro:.4f}")
    for n in (8, 16, 32, 64):
        accs = []
        for seed in range(5):
            idx = sample_per_class(s.train_y.values, n, seed)
            accs.append(head_scores(tr[idx], s.train_y.values[idx], te, s.test_y.values)[0].accuracy)
        print(f"  {n:>2} per class: accuracy {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(len(make_pairs(["a", "b", "c", "d"], [0, 0, 1, 1], iterations=20)), "pairs for 4 examples × 20 iterations")
