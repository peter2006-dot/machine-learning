"""Reproducibility helpers shared by training and evaluation scripts."""

from __future__ import annotations

import os
import random
import numpy as np


def set_random_seed(seed: int) -> np.random.Generator:
    """Seed standard Python and NumPy and return the project RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
