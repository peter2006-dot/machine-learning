#!/usr/bin/env python3
"""Train and evaluate the initial k-mer ridge strength baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import KmerFeaturizer, load_promoter_arrays, split_indices
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import RidgeStrengthPredictor
from promoter_ml.reproducibility import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/ridge_strength_dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_config = config["logging"]
    logger = configure_logging(
        "train_strength_baseline",
        REPOSITORY_ROOT / log_config["directory"],
        filename=log_config["filename"],
        level=log_config["level"],
    )

    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    data_config = config["data"]
    logger.info("Loading data from %s", args.data_dir)
    sequences, strengths = load_promoter_arrays(
        args.data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    targets = np.log10(strengths)
    train_idx, validation_idx, test_idx = split_indices(
        len(sequences),
        float(data_config["train_ratio"]),
        float(data_config["validation_ratio"]),
        seed,
    )
    logger.info(
        "Development split created: train=%d validation=%d test=%d",
        len(train_idx), len(validation_idx), len(test_idx),
    )

    feature_config = config["features"]
    featurizer = KmerFeaturizer(
        k_min=int(feature_config["k_min"]),
        k_max=int(feature_config["k_max"]),
        include_gc=bool(feature_config["include_gc"]),
    )
    logger.info("Building %d k-mer and composition features", len(featurizer.feature_names))
    features = featurizer.transform(sequences)

    model = RidgeStrengthPredictor(alpha=float(config["model"]["ridge_alpha"]))
    model.fit(features[train_idx], targets[train_idx])
    validation_metrics = regression_metrics(targets[validation_idx], model.predict(features[validation_idx]))
    test_metrics = regression_metrics(targets[test_idx], model.predict(features[test_idx]))

    result = {
        "status": "development_baseline_not_final_evaluation",
        "seed": seed,
        "split_strategy": data_config["split_strategy"],
        "sample_counts": {
            "all": len(sequences),
            "train": len(train_idx),
            "validation": len(validation_idx),
            "test": len(test_idx),
        },
        "feature_count": len(featurizer.feature_names),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    model.save(output_dir / "model.npz")
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Validation metrics: %s", validation_metrics)
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Saved model and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
