#!/usr/bin/env python3
"""Evaluate generated promoters with strength, motif, novelty and diversity metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.config import load_config
from promoter_ml.data import load_promoter_arrays
from promoter_ml.evaluation import evaluate_acceptance, evaluate_generated_promoters
from promoter_ml.logging_utils import configure_logging
from promoter_ml.models import CNNStrengthPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, help="Generated .npy, FASTA, CSV, TSV or text file")
    parser.add_argument("--reference-data-dir", required=True, help="Directory containing course reference arrays")
    parser.add_argument("--model", required=True, help="Trained independent evaluator model.pt")
    parser.add_argument("--targets", help="Optional .npy or text file with one target strength per sequence")
    parser.add_argument(
        "--targets-log10",
        action="store_true",
        help="Interpret --targets as already log10 transformed; raw positive strengths are expected otherwise",
    )
    parser.add_argument("--config", default="configs/validator.toml")
    parser.add_argument("--output-dir", default="outputs/generated_validation")
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Exit with status 2 when configured acceptance rules reject the generated set",
    )
    return parser.parse_args()


def load_generated(path: Path) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Load generated sequences and optional embedded targets.

    The returned boolean indicates whether embedded targets are already log10.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(str), None, False
    if suffix in {".fa", ".fasta", ".fna"}:
        sequences = []
        current: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
        return np.asarray(sequences, dtype=str), None, False
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames or "sequence" not in reader.fieldnames:
                raise ValueError("Generated table must contain a 'sequence' column")
            records = list(reader)
        sequences = np.asarray([record["sequence"] for record in records], dtype=str)
        if "target_log10_strength" in reader.fieldnames:
            targets = np.asarray([float(record["target_log10_strength"]) for record in records])
            return sequences, targets, True
        if "target_strength" in reader.fieldnames:
            targets = np.asarray([float(record["target_strength"]) for record in records])
            return sequences, targets, False
        return sequences, None, False
    sequences = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray(sequences, dtype=str), None, False


def load_targets(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64)
    return np.asarray(
        [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=np.float64,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    def flattened(items: dict[str, Any], prefix: str = ""):
        for name, value in items.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            if isinstance(value, dict):
                yield from flattened(value, key)
            else:
                yield key, value

    lines = ["# Generated promoter validation report", ""]
    for section, metrics in summary.items():
        lines.extend((f"## {section.replace('_', ' ').title()}", ""))
        if isinstance(metrics, dict):
            lines.extend(("| Metric | Value |", "|---|---:|"))
            for name, value in flattened(metrics):
                rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
                lines.append(f"| {name} | {rendered} |")
        else:
            lines.append(str(metrics))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    output_dir = REPOSITORY_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        "evaluate_generated_promoters",
        REPOSITORY_ROOT / config["logging"]["directory"],
        filename=config["logging"]["filename"],
        level=config["logging"]["level"],
    )

    generated, embedded_targets, embedded_log10 = load_generated(Path(args.generated))
    data_config = config["data"]
    references, _ = load_promoter_arrays(
        args.reference_data_dir,
        sequence_file=data_config["sequence_file"],
        label_file=data_config["label_file"],
        sequence_length=int(data_config["sequence_length"]),
    )
    logger.info("Loaded %d generated and %d reference promoters", len(generated), len(references))

    target_values = load_targets(Path(args.targets)) if args.targets else embedded_targets
    targets_are_log10 = args.targets_log10 if args.targets else embedded_log10
    target_log10 = None
    if target_values is not None:
        if len(target_values) != len(generated):
            raise ValueError("Target strength count does not match generated sequence count")
        if targets_are_log10:
            if not np.all(np.isfinite(target_values)):
                raise ValueError("Log10 target strengths must be finite")
            target_log10 = target_values
        else:
            if np.any(target_values <= 0.0) or not np.all(np.isfinite(target_values)):
                raise ValueError("Raw target strengths must be finite and positive before log10")
            target_log10 = np.log10(target_values)

    sequence_length = int(data_config["sequence_length"])
    valid_mask = np.asarray(
        [len(value) == sequence_length and not (set(value.upper()) - set(data_config["alphabet"])) for value in generated]
    )
    predictions = np.full(len(generated), np.nan, dtype=np.float64)
    model = CNNStrengthPredictor.load(args.model)
    if model.sequence_length != sequence_length or model.alphabet != data_config["alphabet"]:
        raise ValueError("Evaluator checkpoint does not match the validator sequence contract")
    if valid_mask.any():
        predictions[valid_mask] = model.predict(generated[valid_mask])
    logger.info(
        "Loaded CNN evaluator: length=%s alphabet=%s kernels=%s feature_count=%s",
        model.sequence_length,
        model.alphabet,
        model.kernel_sizes,
        model.spec["feature_count"],
    )

    motif_config = config["motifs"]
    validator_config = config["validator"]
    summary, rows = evaluate_generated_promoters(
        generated,
        sequence_length=sequence_length,
        references=references,
        predictions=predictions,
        target_log10_strengths=target_log10,
        alphabet=data_config["alphabet"],
        minus35=motif_config["minus35"],
        minus10=motif_config["minus10"],
        relative_score_threshold=float(motif_config["relative_score_threshold"]),
        spacer_min=int(motif_config["spacer_min"]),
        spacer_max=int(motif_config["spacer_max"]),
        novelty_identity_threshold=float(validator_config["novelty_identity_threshold"]),
        target_tolerances=tuple(float(value) for value in validator_config["target_tolerances_log10"]),
        max_pairwise_sequences=int(validator_config["max_pairwise_sequences"]),
        kmer_comparison_max=int(validator_config["kmer_comparison_max"]),
        seed=int(config["project"]["seed"]),
    )
    summary["evaluator_quality"] = model.evaluation_metrics or None
    acceptance_config = config.get("acceptance")
    if acceptance_config and bool(acceptance_config.get("enabled", True)):
        summary["acceptance"] = evaluate_acceptance(summary, acceptance_config)
    summary["run"] = {
        "config": str(args.config),
        "generated_file": str(Path(args.generated).resolve()),
        "generated_sha256": sha256_file(Path(args.generated)),
        "reference_data_dir": str(Path(args.reference_data_dir).resolve()),
        "model_file": str(Path(args.model).resolve()),
        "model_sha256": sha256_file(Path(args.model)),
        "targets_file": str(Path(args.targets).resolve()) if args.targets else None,
        "targets_sha256": sha256_file(Path(args.targets)) if args.targets else None,
        "model_spec": model.spec,
    }

    clean_summary = json_ready(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(clean_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    clean_rows = [json_ready(row) for row in rows]
    with (output_dir / "per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)
    write_markdown_report(output_dir / "report.md", clean_summary)
    logger.info("Validation complete; outputs saved to %s", output_dir)
    if args.fail_on_rejection and summary.get("acceptance", {}).get("status") != "pass":
        logger.error("Generated promoters rejected by configured acceptance rules")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
