#!/usr/bin/env python3
"""Run a dependency-light end-to-end check using synthetic DNA data."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.data import KmerFeaturizer
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import RidgeStrengthPredictor


def main() -> None:
    rng = np.random.default_rng(7)
    alphabet = np.array(list("ACGT"))
    sequences = np.array(["".join(rng.choice(alphabet, size=50)) for _ in range(200)])
    targets = np.array([(seq.count("G") + seq.count("C")) / 50 for seq in sequences])
    featurizer = KmerFeaturizer(k_min=1, k_max=2, include_gc=True)
    features = featurizer.transform(sequences)
    model = RidgeStrengthPredictor(alpha=1.0).fit(features[:150], targets[:150])
    predictions = model.predict(features[150:])
    metrics = regression_metrics(targets[150:], predictions)

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.npz"
        model.save(model_path)
        restored = RidgeStrengthPredictor.load(model_path)
        if not np.allclose(predictions, restored.predict(features[150:])):
            raise AssertionError("Saved and loaded predictions do not match")
        logger = configure_logging("smoke_test", Path(directory) / "logs")
        logger.info("Smoke test metrics: %s", metrics)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if not np.all(np.isfinite(predictions)) or metrics["pearson"] < 0.95:
        raise AssertionError(f"Unexpected smoke test result: {metrics}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
