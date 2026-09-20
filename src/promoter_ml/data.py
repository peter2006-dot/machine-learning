"""Loading, validating and featurizing promoter sequences."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import numpy as np

DNA_ALPHABET = "ACGT"


def one_hot_encode_sequences(
    sequences: np.ndarray,
    sequence_length: int,
    alphabet: str = DNA_ALPHABET,
) -> np.ndarray:
    """Return a (N, C, L) one-hot encoding for Conv1d sequence models."""
    values = np.char.upper(np.asarray(sequences).astype(str))
    if values.ndim != 1:
        raise ValueError("Sequences must be a one-dimensional array")
    encoded = np.zeros((len(values), len(alphabet), sequence_length), dtype=np.float32)
    base_index = {base: index for index, base in enumerate(alphabet)}
    for row, sequence in enumerate(values):
        if len(sequence) != sequence_length or set(sequence) - set(alphabet):
            raise ValueError(f"Invalid sequence for one-hot encoding: {sequence!r}")
        for position, base in enumerate(sequence):
            encoded[row, base_index[base], position] = 1.0
    return encoded


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


class PositionAwareKmerFeaturizer:
    """Combine position-specific one-hot, k-mer and composition features."""

    def __init__(
        self,
        sequence_length: int,
        k_min: int = 1,
        k_max: int = 3,
        include_gc: bool = True,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = int(sequence_length)
        self.kmer = KmerFeaturizer(k_min=k_min, k_max=k_max, include_gc=include_gc)

    @property
    def feature_names(self) -> list[str]:
        positional = [
            f"position_{position}_{base}"
            for position in range(self.sequence_length)
            for base in DNA_ALPHABET
        ]
        return positional + self.kmer.feature_names

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        values = np.char.upper(np.asarray(sequences).astype(str))
        invalid = [sequence for sequence in values if len(sequence) != self.sequence_length or set(sequence) - set(DNA_ALPHABET)]
        if invalid:
            raise ValueError(f"Found invalid sequence for position-aware features: {invalid[0]!r}")

        positional = np.zeros((len(values), self.sequence_length * len(DNA_ALPHABET)), dtype=np.float64)
        base_index = {base: index for index, base in enumerate(DNA_ALPHABET)}
        for row, sequence in enumerate(values):
            for position, base in enumerate(sequence):
                positional[row, position * len(DNA_ALPHABET) + base_index[base]] = 1.0
        return np.concatenate((positional, self.kmer.transform(values)), axis=1)


def sequence_identity(first: str, second: str) -> float:
    """Return positional identity for two equal-length DNA sequences."""
    if len(first) != len(second):
        raise ValueError("Sequence identity requires equal-length sequences")
    if not first:
        raise ValueError("Sequence identity is undefined for empty sequences")
    return sum(left == right for left, right in zip(first, second)) / len(first)


def similarity_cluster_split(
    sequences: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
    identity_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split connected similarity clusters without crossing data partitions.

    Two sequences are connected when their positional identity is at least the
    configured threshold. Connected components are assigned as indivisible
    groups, preventing near-duplicate promoters from leaking across splits.
    """
    values = np.char.upper(np.asarray(sequences).astype(str))
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("At least three one-dimensional sequences are required")
    if not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("identity_threshold must be between zero and one")
    test_ratio = 1.0 - float(train_ratio) - float(validation_ratio)
    ratios = np.array([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("All split ratios must be positive and sum to one")

    lengths = {len(sequence) for sequence in values}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("Similarity clustering requires non-empty equal-length sequences")
    encoded = np.asarray([list(sequence) for sequence in values], dtype="U1")

    parent = np.arange(len(values), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(values) - 1):
        identities = np.mean(encoded[left + 1 :] == encoded[left], axis=1)
        for offset in np.flatnonzero(identities >= identity_threshold):
            union(left, left + 1 + int(offset))

    components: dict[int, list[int]] = {}
    for index in range(len(values)):
        components.setdefault(find(index), []).append(index)
    if len(components) < 3:
        raise ValueError(
            "Similarity clustering produced fewer than three clusters; "
            "increase the identity threshold or provide more diverse data"
        )

    rng = np.random.default_rng(seed)
    clusters = list(components.values())
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)
    target_counts = ratios * len(values)
    assigned: list[list[int]] = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    for cluster in clusters:
        deficits = target_counts - counts
        destination = int(np.argmax(deficits))
        assigned[destination].extend(cluster)
        counts[destination] += len(cluster)

    if any(not indices for indices in assigned):
        raise ValueError("Unable to create three non-empty similarity-aware splits")
    return tuple(np.asarray(sorted(indices), dtype=np.int64) for indices in assigned)
