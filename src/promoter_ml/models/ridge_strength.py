"""A dependency-light baseline for promoter strength prediction."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class RidgeStrengthPredictor:
    """Closed-form ridge regression with standardized input features."""

    def __init__(self, alpha: float = 10.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeStrengthPredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected a 2D feature matrix and matching 1D targets")

        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        standardized = (x - self.feature_mean) / self.feature_scale

        self.intercept = float(y.mean())
        centered_targets = y - self.intercept
        gram = standardized.T @ standardized
        regularizer = self.alpha * np.eye(gram.shape[0], dtype=np.float64)
        self.weights = np.linalg.solve(gram + regularizer, standardized.T @ centered_targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None or self.intercept is None:
            raise RuntimeError("The predictor must be fitted before prediction")
        standardized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return standardized @ self.weights + self.intercept

    def save(self, path: str | Path) -> None:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None or self.intercept is None:
            raise RuntimeError("Cannot save an unfitted predictor")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            alpha=self.alpha,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            intercept=self.intercept,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeStrengthPredictor":
        with np.load(path) as state:
            model = cls(alpha=float(state["alpha"]))
            model.feature_mean = state["feature_mean"].copy()
            model.feature_scale = state["feature_scale"].copy()
            model.weights = state["weights"].copy()
            model.intercept = float(state["intercept"])
        return model
