"""
classifiers.py: the lecture notebook's experiment on 903 Indonesian COVID-19 tweets, plus a correct KNN.

  data          data/tweets_sentiment.csv: the tweet, its sentiment (−1 negative, 0 neutral, 1 positive) and the
                course's preprocessed text (columns step01 and ProcessedText of clean_dataset_part01.csv)
  split         train_test_split(test_size=0.3, random_state=42): 632 training and 271 test tweets, as in the notebook
  Naive Bayes   Pipeline(TfidfVectorizer, MultinomialNB) tuned by GridSearchCV(cv=3) over
                ngram_range (1,1)…(1,4) and α ∈ {0.0001, 0.001, 0.1, 1, 10, 100}, exactly as in the notebook
  KNN           Pipeline(TfidfVectorizer, KNeighborsClassifier(metric="cosine")), with k chosen by
                cross-validation on the training set
  notebook KNN  the notebook's version, which LabelEncodes each whole tweet into one integer (alphabetical
                position) and runs KNN on that single number; kept to show why its accuracy is low

This module does not depend on Streamlit. Try it from the terminal:
    python classifiers.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent / "data"
LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
CLASSES = [-1, 0, 1]
NB_GRID = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4)],
           "model__alpha": (0.0001, 0.001, 0.1, 1, 10, 100)}
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31]


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "tweets_sentiment.csv")


@dataclass
class Split:
    train_x: pd.Series
    test_x: pd.Series
    train_y: pd.Series
    test_y: pd.Series
    train_raw: pd.Series
    test_raw: pd.Series


def split(df: pd.DataFrame) -> Split:
    """The notebook's split of the preprocessed text; the raw tweets follow the same index."""
    trx, tex, try_, tey = train_test_split(df["processed"].astype(str), df["sentiment"], test_size=0.3,
                                           random_state=42)
    return Split(trx, tex, try_, tey, df.loc[trx.index, "tweet"], df.loc[tex.index, "tweet"])


@dataclass
class Scores:
    accuracy: float
    precision_w: float
    recall_w: float
    f1_w: float
    f1_macro: float
    confusion: np.ndarray
    per_class: pd.DataFrame


def score(y_true, y_pred) -> Scores:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average="weighted",
                                                    zero_division=0)
    per = pd.DataFrame({"Class": [LABELS[c] for c in CLASSES], "Precision": p, "Recall": r, "F1": f, "Support": s})
    return Scores(accuracy_score(y_true, y_pred), pw, rw, fw,
                  f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
                  confusion_matrix(y_true, y_pred, labels=CLASSES), per)


@dataclass
class Fitted:
    name: str
    model: object
    train: Scores
    test: Scores
    fit_s: float
    predict_ms: float          # per test tweet
    note: str = ""


def _fit(name: str, model, s: Split, note: str = "", fit_args=None) -> Fitted:
    t = time.perf_counter()
    model.fit(s.train_x, s.train_y.values)
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    pred = model.predict(s.test_x)
    pred_ms = (time.perf_counter() - t) * 1000 / len(s.test_x)
    return Fitted(name, model, score(s.train_y, model.predict(s.train_x)), score(s.test_y, pred), fit_s, pred_ms, note)


def naive_bayes_grid(s: Split) -> tuple[Fitted, pd.DataFrame]:
    grid = GridSearchCV(Pipeline([("tfidf", TfidfVectorizer()), ("model", MultinomialNB())]), NB_GRID, cv=3, n_jobs=1)
    f = _fit("Naive Bayes (TF-IDF, grid search)", grid, s)
    cv = pd.DataFrame(grid.cv_results_)
    table = pd.DataFrame({"α": cv["param_model__alpha"].astype(float),
                          "n-grams": cv["param_tfidf__ngram_range"].map(lambda r: f"({r[0]}, {r[1]})"),
                          "CV accuracy": cv["mean_test_score"]})
    f.note = (f"best: α = {grid.best_params_['model__alpha']}, ngram_range = {grid.best_params_['tfidf__ngram_range']}"
              f" (3-fold CV accuracy {grid.best_score_:.3f})")
    return f, table


def naive_bayes(s: Split, alpha: float, ngram: tuple[int, int]) -> Fitted:
    return _fit(f"Naive Bayes (α = {alpha:g}, n-grams {ngram})",
                Pipeline([("tfidf", TfidfVectorizer(ngram_range=ngram)), ("model", MultinomialNB(alpha=alpha))]), s)


def knn_pipeline(k: int, metric: str = "cosine", weights: str = "uniform") -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer()),
                     ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights))])


def knn(s: Split, k: int, metric: str = "cosine", weights: str = "uniform") -> Fitted:
    return _fit(f"KNN (TF-IDF, k = {k}, {metric})", knn_pipeline(k, metric, weights), s)


def knn_k_curve(s: Split, metric: str = "cosine", weights: str = "uniform") -> pd.DataFrame:
    """5-fold CV accuracy on the training set (for choosing k) and test accuracy (for reference) for each k."""
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for k in K_GRID:
        cvs = cross_val_score(knn_pipeline(k, metric, weights), s.train_x, s.train_y.values, cv=cv)
        test = accuracy_score(s.test_y, knn_pipeline(k, metric, weights).fit(s.train_x, s.train_y.values).predict(s.test_x))
        rows.append({"k": k, "CV accuracy (training set)": cvs.mean(), "Test accuracy": test})
    return pd.DataFrame(rows)


class NotebookKNN:
    """The notebook's KNN, reproduced as written: each tweet becomes one integer, its position in the alphabetical
    list of distinct tweets, and the training and test sets are encoded separately."""

    def __init__(self, k: int = 3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, x, y):
        self.model.fit(LabelEncoder().fit_transform(x).reshape(-1, 1), y)
        return self

    def predict(self, x):
        return self.model.predict(LabelEncoder().fit_transform(x).reshape(-1, 1))


def notebook_knn(s: Split, k: int = 3) -> Fitted:
    return _fit("KNN as in the notebook (LabelEncoder)", NotebookKNN(k), s)


def majority(s: Split) -> Fitted:
    from sklearn.dummy import DummyClassifier
    return _fit("Majority class (always positive)", Pipeline([("tfidf", TfidfVectorizer()),
                                                              ("model", DummyClassifier(strategy="most_frequent"))]), s)


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def nb_explain(pipe: Pipeline, text: str) -> tuple[pd.DataFrame, dict[int, float], dict[int, float]]:
    """Per-word contributions x_w · log P(w | c) of MultinomialNB on TF-IDF features.
    Returns (word table, log prior per class, posterior per class)."""
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    nb: MultinomialNB = pipe.named_steps["model"]
    x = vec.transform([text])
    idx = x.nonzero()[1]
    names = vec.get_feature_names_out()
    rows = []
    for j in idx:
        row = {"Feature": names[j], "TF-IDF": x[0, j]}
        for ci, c in enumerate(nb.classes_):
            row[LABELS[int(c)]] = x[0, j] * nb.feature_log_prob_[ci, j]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("TF-IDF", ascending=False) if rows else pd.DataFrame()
    prior = {int(c): float(nb.class_log_prior_[ci]) for ci, c in enumerate(nb.classes_)}
    post = dict(zip((int(c) for c in nb.classes_), pipe.predict_proba([text])[0]))
    return table, prior, post


def knn_neighbors(pipe: Pipeline, text: str, train_x: pd.Series, train_y: pd.Series, raw: pd.Series) -> pd.DataFrame:
    vec = pipe.named_steps["tfidf"]
    est: KNeighborsClassifier = pipe.named_steps["knn"]
    dist, idx = est.kneighbors(vec.transform([text]))
    return pd.DataFrame({"Rank": range(1, len(idx[0]) + 1),
                         "Cosine similarity": 1 - dist[0],
                         "Label": [LABELS[int(train_y.iloc[i])] for i in idx[0]],
                         "Tweet": [raw.iloc[i] for i in idx[0]],
                         "Preprocessed": [train_x.iloc[i] for i in idx[0]]})


if __name__ == "__main__":
    df = load()
    s = split(df)
    print(f"{len(s.train_x)} training, {len(s.test_x)} test tweets")
    nb, _ = naive_bayes_grid(s)
    print(nb.name, nb.note, f"train {nb.train.accuracy:.4f} test {nb.test.accuracy:.4f} F1w {nb.test.f1_w:.4f}")
    print("notebook KNN test", round(notebook_knn(s).test.accuracy, 4))
    curve = knn_k_curve(s)
    print(curve.round(4).to_string(index=False))
    print("majority baseline", round(majority(s).test.accuracy, 4))
