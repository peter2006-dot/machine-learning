#!/usr/bin/env python3
"""Run the independent evaluator and generated-sequence validator end to end."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(17)
    alphabet = np.asarray(list("ACGT"))
    references = np.asarray(["".join(rng.choice(alphabet, size=50)) for _ in range(180)])
    log10_strengths = np.asarray(
        [
            0.8
            + 1.5 * (sequence.count("G") + sequence.count("C")) / 50
            + 0.2 * (sequence[10] == "A")
            for sequence in references
        ]
    )
    strengths = np.power(10.0, log10_strengths)
    generated = references[:12].copy()
    generated[0] = "AAAAA" + "TTGACA" + "A" * 15 + "TATAAT" + "A" * 18
    targets = np.power(10.0, np.full(len(generated), 1.5))

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        data_dir = temporary / "data"
        model_dir = temporary / "model"
        validation_dir = temporary / "validation"
        strict_validation_dir = temporary / "strict_validation"
        data_dir.mkdir()
        np.save(data_dir / "promoter.npy", references)
        np.save(data_dir / "gene_expression.npy", strengths)
        generated_path = temporary / "generated.npy"
        target_path = temporary / "targets.npy"
        np.save(generated_path, generated)
        np.save(target_path, targets)

        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train_independent_evaluator.py"),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(model_dir),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        evaluation_command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_generated_promoters.py"),
            "--generated",
            str(generated_path),
            "--reference-data-dir",
            str(data_dir),
            "--model",
            str(model_dir / "model.pt"),
            "--targets",
            str(target_path),
        ]
        subprocess.run(
            [*evaluation_command, "--output-dir", str(validation_dir)],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        summary = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
        required_sections = {
            "validity",
            "motifs",
            "diversity",
            "novelty",
            "distribution",
            "target_control",
            "evaluator_quality",
            "acceptance",
        }
        if not required_sections.issubset(summary):
            raise AssertionError(f"Missing validation sections: {sorted(required_sections - set(summary))}")
        if summary["counts"]["valid"] != len(generated):
            raise AssertionError("Validator did not accept all synthetic promoters")
        if not summary["acceptance"]["rules"]:
            raise AssertionError("Validator did not evaluate configured acceptance rules")
        if not (validation_dir / "per_sequence.csv").exists() or not (validation_dir / "report.md").exists():
            raise AssertionError("Validator did not create all expected output files")
        strict_result = subprocess.run(
            [
                *evaluation_command,
                "--output-dir",
                str(strict_validation_dir),
                "--fail-on-rejection",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if strict_result.returncode != 2:
            raise AssertionError(f"Strict rejection returned {strict_result.returncode}, expected 2")

    print("Validator smoke test passed")


if __name__ == "__main__":
    main()
