from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.data import PositionAwareKmerFeaturizer, one_hot_encode_sequences, similarity_cluster_split


class PositionAwareFeaturesTest(unittest.TestCase):
    def test_feature_shape_and_position_signal(self) -> None:
        sequences = np.asarray(["A" * 50, "C" + "A" * 49])
        featurizer = PositionAwareKmerFeaturizer(50, k_min=1, k_max=3, include_gc=True)
        features = featurizer.transform(sequences)
        self.assertEqual(features.shape, (2, 285))
        self.assertFalse(np.array_equal(features[0], features[1]))


class OneHotEncodingTest(unittest.TestCase):
    def test_shape_and_base_channel(self) -> None:
        encoded = one_hot_encode_sequences(np.asarray(["ACGT" + "A" * 46]), 50)
        self.assertEqual(encoded.shape, (1, 4, 50))
        self.assertEqual(encoded[0, 0, 0], 1.0)
        self.assertEqual(encoded[0, 1, 1], 1.0)
        self.assertEqual(encoded[0, 2, 2], 1.0)
        self.assertEqual(encoded[0, 3, 3], 1.0)


class SimilaritySplitTest(unittest.TestCase):
    def test_similar_sequences_stay_in_one_partition(self) -> None:
        sequences = np.asarray(
            [
                "A" * 50,
                "A" * 49 + "C",
                "C" * 50,
                "C" * 49 + "G",
                "G" * 50,
                "G" * 49 + "T",
                "T" * 50,
                "T" * 49 + "A",
            ]
        )
        splits = similarity_cluster_split(sequences, 0.5, 0.25, seed=7, identity_threshold=0.9)
        ownership = {}
        for split_id, indices in enumerate(splits):
            for index in indices:
                ownership[int(index)] = split_id
        for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertEqual(ownership[left], ownership[right])
        self.assertEqual(set().union(*(set(values.tolist()) for values in splits)), set(range(len(sequences))))


if __name__ == "__main__":
    unittest.main()
