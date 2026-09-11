"""Regression metrics used by the independent strength evaluator."""

from __future__ import annotations

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including correct handling of tied values."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    pearson = float(np.corrcoef(targets, predictions)[0, 1])
    spearman = float(np.corrcoef(_rankdata(targets), _rankdata(predictions))[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman}
