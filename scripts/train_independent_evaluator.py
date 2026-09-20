#!/usr/bin/env python3
"""Train the independent CNN promoter strength evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import (
    PositionAwareKmerFeaturizer,
    load_promoter_arrays,
    similarity_cluster_split,
)
from promoter_ml.logging_utils import configure_logging
from promoter_ml.metrics import regression_metrics
from promoter_ml.models import CNNStrengthPredictor, RidgeStrengthPredictor
from promoter_ml.reproducibility import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/validator.toml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/independent_evaluator")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def predictor_from_config(config: dict) -> CNNStrengthPredictor:
    model_config = config["model"]
    data_config = config["data"]
    return CNNStrengthPredictor(
        sequence_length=int(data_config["sequence_length"]),
        alphabet=str(data_config["alphabet"]),
        stem_channels=int(model_config.get("stem_channels", 32)),
        kernel_sizes=tuple(int(value) for value in model_config.get("kernel_sizes", [3, 5, 7])),
        merge_channels=int(model_config.get("merge_channels", 64)),
        hidden_size=int(model_config.get("hidden_size", 64)),
        position_hidden_size=int(model_config.get("position_hidden_size", 64)),
        dropout=float(model_config.get("dropout", 0.35)),
        learning_rate=float(model_config.get("learning_rate", 5e-4)),
        weight_decay=float(model_config.get("weight_decay", 1e-3)),
        batch_size=int(model_config.get("batch_size", 128)),
        max_epochs=int(model_config.get("max_epochs", 100)),
        patience=int(model_config.get("patience", 20)),
        min_delta=float(model_config.get("min_delta", 0.002)),
        loss_name=str(model_config.get("loss_name", "huber")),
        selection_metric=str(model_config.get("selection_metric", "pearson")),
        calibration=str(model_config.get("calibration", "median_bias")),
        correlation_weight=float(model_config.get("correlation_weight", 0.5)),
        mutation_rate=float(model_config.get("mutation_rate", 0.25)),
    )


def main() -> None:
    args = parse_args()
    config_path = REPOSITORY_ROOT / args.config
    config = load_config(config_path)
    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(
        "train_independent_evaluator",
        REPOSITORY_ROOT / config["logging"]["directory"],
        filename=config["logging"]["filename"],
        level=config["logging"]["level"],
    )
    seed = int(config["project"]["seed"])
    set_random_seed(seed)
    data_config = config["data"]
    sequences, strengths = load_promoter_arrays(
        args.data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    targets = np.log10(strengths)

    train_idx, validation_idx, test_idx = similarity_cluster_split(
        sequences,
        train_ratio=float(data_config["train_ratio"]),
        validation_ratio=float(data_config["validation_ratio"]),
        seed=seed,
        identity_threshold=float(config["split"]["identity_threshold"]),
    )
    logger.info(
        "Similarity-aware split created: train=%d validation=%d test=%d",
        len(train_idx),
        len(validation_idx),
        len(test_idx),
    )

    model = predictor_from_config(config)
    logger.info("Training independent CNN evaluator on device %s", model.device)
    model.fit(
        sequences[train_idx],
        targets[train_idx],
        validation_sequences=sequences[validation_idx],
        validation_targets=targets[validation_idx],
        seed=seed,
    )
    validation_metrics = regression_metrics(targets[validation_idx], model.predict(sequences[validation_idx]))
    test_metrics = regression_metrics(targets[test_idx], model.predict(sequences[test_idx]))
    logger.info("CNN validation metrics: %s", validation_metrics)
    logger.info("CNN test metrics: %s", test_metrics)

    feature_config = config["features"]
    featurizer = PositionAwareKmerFeaturizer(
        sequence_length=int(data_config["sequence_length"]),
        k_min=int(feature_config["k_min"]),
        k_max=int(feature_config["k_max"]),
        include_gc=bool(feature_config["include_gc"]),
    )
    ridge_features = featurizer.transform(sequences)
    ridge = RidgeStrengthPredictor(alpha=float(config["model"].get("ridge_alpha", 10.0)))
    ridge.fit(ridge_features[train_idx], targets[train_idx])
    ridge_validation = regression_metrics(targets[validation_idx], ridge.predict(ridge_features[validation_idx]))
    ridge_test = regression_metrics(targets[test_idx], ridge.predict(ridge_features[test_idx]))
    logger.info("Ridge comparison validation metrics: %s", ridge_validation)
    logger.info("Ridge comparison test metrics: %s", ridge_test)

    model.evaluation_metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }
    model.save(output_dir / "model.pt")
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train=train_idx,
        validation=validation_idx,
        test=test_idx,
    )
    metadata = {
        "status": "independent_strength_evaluator",
        "config": str(config_path.relative_to(REPOSITORY_ROOT)),
        "seed": seed,
        "split_strategy": data_config["split_strategy"],
        "identity_threshold": float(config["split"]["identity_threshold"]),
        "sample_counts": {
            "all": int(len(sequences)),
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "test": int(len(test_idx)),
        },
        "model_file": "model.pt",
        "model_spec": model.spec,
        "best_epoch": model.best_epoch,
        "data_files": {
            "sequences": {
                "name": data_config["sequence_file"],
                "sha256": sha256_file(Path(args.data_dir) / data_config["sequence_file"]),
            },
            "labels": {
                "name": data_config["label_file"],
                "sha256": sha256_file(Path(args.data_dir) / data_config["label_file"]),
            },
        },
        "cnn": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "position_kmer_ridge": {
            "validation": ridge_validation,
            "test": ridge_test,
            "feature_count": int(ridge_features.shape[1]),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved evaluator artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
