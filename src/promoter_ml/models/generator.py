"""Shared contract for CVAE and autoregressive generator implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ConditionalSequenceGenerator(ABC):
    """Interface that both planned conditional generators must implement."""

    @abstractmethod
    def fit(self, sequences: np.ndarray, conditions: np.ndarray) -> "ConditionalSequenceGenerator":
        """Fit the generator on training sequences and transformed conditions."""

    @abstractmethod
    def generate(self, conditions: np.ndarray, samples_per_condition: int, seed: int) -> np.ndarray:
        """Return generated DNA sequences grouped by requested conditions."""
