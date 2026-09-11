"""Loading, validating and featurizing promoter sequences."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import numpy as np

DNA_ALPHABET = "ACGT"


def load_promoter_arrays(
    data_directory: str | Path,
    sequence_file: str = "promoter.npy",
    label_file: str = "gene_expression.npy",
    sequence_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the course arrays and enforce the shared data contract."""
    data_dir = Path(data_directory)
    sequences = np.load(data_dir / sequence_file, allow_pickle=False).astype(str)
    strengths = np.load(data_dir / label_file, allow_pickle=False).astype(np.float64)
    sequences = np.char.upper(sequences)

    if sequences.ndim != 1 or strengths.ndim != 1:
        raise ValueError("Sequences and strengths must both be one-dimensional arrays")
    if len(sequences) != len(strengths):
        raise ValueError("Sequence and strength arrays have different lengths")
    if len(sequences) == 0:
        raise ValueError("The dataset is empty")
    if np.any(strengths <= 0) or not np.all(np.isfinite(strengths)):
        raise ValueError("Strength labels must be finite positive values for log10")

    invalid = [seq for seq in sequences if len(seq) != sequence_length or set(seq) - set(DNA_ALPHABET)]
    if invalid:
        raise ValueError(f"Found {len(invalid)} invalid sequences; first example: {invalid[0]!r}")
    return sequences, strengths


def split_indices(
    sample_count: int,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic development split.

    This random split is used only to make the initial pipeline runnable. The
    formal experiments will replace it with a similarity-cluster split.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    train_end = int(sample_count * train_ratio)
    validation_end = train_end + int(sample_count * validation_ratio)
    return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]


class KmerFeaturizer:
    """Convert DNA sequences into normalized k-mer counts and optional GC content."""

    def __init__(self, k_min: int = 1, k_max: int = 3, include_gc: bool = True):
        if k_min < 1 or k_max < k_min:
            raise ValueError("Require 1 <= k_min <= k_max")
        self.k_min = k_min
        self.k_max = k_max
        self.include_gc = include_gc
        self.vocabulary = [
            "".join(chars)
            for k in range(k_min, k_max + 1)
            for chars in product(DNA_ALPHABET, repeat=k)
        ]
        self.index = {token: position for position, token in enumerate(self.vocabulary)}

    @property
    def feature_names(self) -> list[str]:
        names = [f"kmer_{token}" for token in self.vocabulary]
        return names + (["gc_fraction"] if self.include_gc else [])

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        features = np.zeros((len(sequences), len(self.feature_names)), dtype=np.float64)
        for row, sequence in enumerate(sequences):
            for k in range(self.k_min, self.k_max + 1):
                denominator = len(sequence) - k + 1
                for position in range(denominator):
                    token = sequence[position : position + k]
                    features[row, self.index[token]] += 1.0 / denominator
            if self.include_gc:
                features[row, -1] = (sequence.count("G") + sequence.count("C")) / len(sequence)
        return features
