"""Klasyfikatory dzialajace na wektorach cech, ze wspolnym interfejsem.

Kazdy ma `classes_` i `predict_proba(X)`. Klasyfikatory liniowe na embeddingach
udostepniaja tez `linear_head()` -> (W, b), co pozwala wyeksportowac caly model
(backbone + L2-normalizacja + warstwa liniowa) do jednego pliku ONNX.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LinearProbe:
    """L2-normalizacja embeddingu + regresja logistyczna (linear probe)."""

    def __init__(self, estimator, classes: list[str]):
        self.estimator = estimator
        self.classes_ = list(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(l2_normalize(X))

    def linear_head(self) -> tuple[np.ndarray, np.ndarray]:
        return self.estimator.coef_, self.estimator.intercept_


class ZeroShotClassifier:
    """CLIP zero-shot: podobienstwo kosinusowe obrazu do opisow klas."""

    def __init__(
        self, text_embeddings: np.ndarray, classes: list[str], logit_scale: float
    ):
        self.text_embeddings = l2_normalize(text_embeddings)
        self.classes_ = list(classes)
        self.logit_scale = logit_scale

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logit_scale * l2_normalize(X) @ self.text_embeddings.T)

    def linear_head(self) -> tuple[np.ndarray, np.ndarray]:
        W = self.logit_scale * self.text_embeddings
        return W, np.zeros(len(self.classes_), dtype=W.dtype)


class SklearnClassifier:
    """Dowolny pipeline sklearn z predict_proba (np. HOG + SVM)."""

    def __init__(self, pipeline: Pipeline, classes: list[str]):
        self.pipeline = pipeline
        self.classes_ = list(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)
