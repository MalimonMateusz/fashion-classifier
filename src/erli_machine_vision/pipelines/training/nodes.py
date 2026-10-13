"""Trening klasyfikatorow na zcache'owanych cechach.

Hiperparametry (C) wybieramy na zbiorze walidacyjnym wg macro-F1, test zostaje
nietkniety do ewaluacji. Klasy sa niezbalansowane (560-1120 zdjec), wiec
uzywamy class_weight="balanced".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from erli_machine_vision.classifiers import (
    LinearProbe,
    SklearnClassifier,
    ZeroShotClassifier,
    l2_normalize,
)

log = logging.getLogger(__name__)


def split_xy(
    features: dict, dataset: pd.DataFrame, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Wybiera wiersze cech nalezace do danego splitu (dopasowanie po product_id)."""
    index = pd.Series(
        np.arange(len(features["product_id"])), index=features["product_id"]
    )
    rows = dataset[dataset["split"] == split]
    X = features["X"][index.loc[rows["product_id"]].to_numpy()]
    return X, rows["label"].to_numpy()


def _select_c(  # noqa: PLR0913
    make_estimator, X_train, y_train, X_val, y_val, grid: list[float], name: str
):
    scores = {}
    for c in grid:
        est = make_estimator(c).fit(X_train, y_train)
        scores[c] = f1_score(y_val, est.predict(X_val), average="macro")
    best_c = max(scores, key=scores.get)
    log.info("%s: val macro-F1 dla C=%s -> najlepsze C=%s", name, scores, best_c)
    return make_estimator(best_c).fit(X_train, y_train), best_c


def train_linear_probe(
    features: dict, dataset: pd.DataFrame, params: dict
) -> LinearProbe:
    X_train, y_train = split_xy(features, dataset, "train")
    X_val, y_val = split_xy(features, dataset, "val")
    est, best_c = _select_c(
        lambda c: LogisticRegression(
            C=c, max_iter=params["max_iter"], class_weight="balanced"
        ),
        l2_normalize(X_train),
        y_train,
        l2_normalize(X_val),
        y_val,
        params["c_grid"],
        "linear probe",
    )
    model = LinearProbe(est, est.classes_)
    model.selected_c = best_c
    return model


def train_handcrafted_svm(
    features: dict, dataset: pd.DataFrame, params: dict
) -> SklearnClassifier:
    X_train, y_train = split_xy(features, dataset, "train")
    X_val, y_val = split_xy(features, dataset, "val")
    pipe, best_c = _select_c(
        lambda c: make_pipeline(
            StandardScaler(),
            PCA(n_components=params["pca_components"], random_state=params["seed"]),
            SVC(
                C=c,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=params["seed"],
            ),
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        params["c_grid"],
        "HOG+HSV SVM",
    )
    model = SklearnClassifier(pipe, pipe.classes_)
    model.selected_c = best_c
    return model


def build_zero_shot(text_embeddings: dict) -> ZeroShotClassifier:
    return ZeroShotClassifier(
        text_embeddings["embeddings"],
        text_embeddings["classes"],
        text_embeddings["logit_scale"],
    )
