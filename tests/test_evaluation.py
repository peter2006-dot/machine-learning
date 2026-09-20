from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from promoter_ml.evaluation import evaluate_acceptance, evaluate_generated_promoters, scan_promoter_motifs
from promoter_ml.models import CNNStrengthPredictor


MOTIF_PROMOTER = "AAAAA" + "TTGACA" + "A" * 15 + "TATAAT" + "A" * 18
NARROW_SPACER = "AAAAA" + "TTGACA" + "A" * 14 + "TATAAT" + "A" * 19
RANDOM_PROMOTER = "ACGT" * 12 + "AC"


class MotifScannerTest(unittest.TestCase):
    def test_finds_ordered_core_motifs_and_spacing(self) -> None:
        result = scan_promoter_motifs(MOTIF_PROMOTER)
        self.assertEqual(result["minus35_start"], 5)
        self.assertEqual(result["minus10_start"], 26)
        self.assertEqual(result["spacer_length"], 15)
        self.assertGreaterEqual(result["minus35_score"], 0.80)
        self.assertGreaterEqual(result["minus10_score"], 0.80)
        self.assertTrue(result["motif_pair_valid"])

    def test_rejects_canonical_pair_with_invalid_spacing(self) -> None:
        result = scan_promoter_motifs(NARROW_SPACER)
        self.assertEqual(result["spacer_length"], 14)
        self.assertTrue(result["motif_pair_present"])
        self.assertFalse(result["spacing_valid"])
        self.assertFalse(result["motif_pair_valid"])

    def test_random_sequence_is_below_pwm_threshold(self) -> None:
        result = scan_promoter_motifs(RANDOM_PROMOTER)
        self.assertFalse(result["motif_pair_valid"])
        self.assertLess(result["minus35_score"], 0.80)

class GeneratedEvaluationTest(unittest.TestCase):
    def test_combines_validity_strength_novelty_and_diversity(self) -> None:
        novel_promoter = MOTIF_PROMOTER[:-1] + "C"
        sequences = np.asarray([MOTIF_PROMOTER, novel_promoter, novel_promoter, "ACGT"])
        references = np.asarray([MOTIF_PROMOTER, "C" * 50])
        predictions = np.asarray([1.0, 1.3, 1.3, np.nan])
        targets = np.asarray([1.1, 1.0, 1.0, 1.0])
        summary, rows = evaluate_generated_promoters(
            sequences,
            sequence_length=50,
            references=references,
            predictions=predictions,
            target_log10_strengths=targets,
            seed=3,
        )
        self.assertEqual(summary["counts"]["valid"], 3)
        self.assertEqual(summary["counts"]["unique_valid"], 2)
        self.assertAlmostEqual(summary["validity"]["valid_rate"], 0.75)
        self.assertTrue(rows[0]["exact_reference_match"])
        self.assertFalse(rows[1]["exact_reference_match"])
        self.assertTrue(rows[1]["duplicate"])
        self.assertIsNone(rows[3]["predicted_log10_strength"])
        self.assertIsNone(rows[3]["target_hit_within_0.25"])
        self.assertEqual(summary["target_control"]["evaluable_count"], 3)
        self.assertAlmostEqual(summary["target_control"]["mae_log10"], 0.7 / 3.0)
        self.assertEqual(summary["motifs"]["scanner"], "sigma70_pwm_relative_score")

    def test_acceptance_rules_report_pass_fail_and_skipped_metrics(self) -> None:
        summary = {
            "validity": {"valid_rate": 1.0, "unique_rate_among_valid": 0.8},
            "target_control": {"mae_log10": None},
        }
        result = evaluate_acceptance(
            summary,
            {
                "min_valid_rate": 1.0,
                "min_unique_rate_among_valid": 0.95,
                "max_target_mae_log10": 0.5,
            },
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["passed_rule_count"], 1)
        self.assertEqual(result["failed_rule_count"], 1)
        self.assertEqual(result["skipped_rule_count"], 1)

    def test_acceptance_can_require_selected_metrics(self) -> None:
        result = evaluate_acceptance(
            {"validity": {"valid_rate": 1.0}},
            {
                "required_rules": ["min_evaluator_test_pearson"],
                "min_valid_rate": 1.0,
                "min_evaluator_test_pearson": 0.2,
            },
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["rules"]["min_evaluator_test_pearson"]["status"], "fail")


class CNNStrengthPredictorTest(unittest.TestCase):
    def test_checkpoint_stores_architecture_and_reloads(self) -> None:
        rng = np.random.default_rng(4)
        alphabet = np.asarray(list("ACGT"))
        sequences = np.asarray(["".join(rng.choice(alphabet, size=50)) for _ in range(80)])
        targets = np.asarray(
            [
                0.4 + 1.8 * (sequence.count("G") + sequence.count("C")) / 50
                for sequence in sequences
            ]
        )
        model = CNNStrengthPredictor(
            sequence_length=50,
            stem_channels=8,
            kernel_sizes=(3, 5),
            merge_channels=16,
            hidden_size=16,
            position_hidden_size=8,
            dropout=0.0,
            batch_size=16,
            max_epochs=8,
            patience=8,
            min_delta=0.0,
            loss_name="mse",
            selection_metric="rmse",
            calibration="median_bias",
            device="cpu",
        )
        model.fit(sequences[:56], targets[:56], sequences[56:68], targets[56:68], seed=4)
        predicted = model.predict(sequences[68:])
        model.evaluation_metrics = {"test": {"mae": 0.25, "pearson": 0.5}}
        self.assertEqual(predicted.shape, (12,))
        self.assertEqual(model.spec["feature_layout"], "one_hot_channels_by_length")
        self.assertEqual(model.spec["feature_count"], 200)
        self.assertEqual(model.spec["input_encoding"], "one_hot_ncl")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = CNNStrengthPredictor.load(path, device="cpu")
        np.testing.assert_allclose(loaded.predict(sequences[68:]), predicted, rtol=1e-5, atol=1e-5)
        self.assertEqual(loaded.spec["stem_channels"], 8)
        self.assertEqual(loaded.spec["kernel_sizes"], [3, 5])
        self.assertEqual(loaded.spec["position_hidden_size"], 8)
        self.assertEqual(loaded.spec["loss_name"], "mse")
        self.assertEqual(loaded.spec["selection_metric"], "rmse")
        self.assertEqual(loaded.spec["calibration"], "median_bias")
        self.assertEqual(loaded.evaluation_metrics["test"]["mae"], 0.25)

    def test_rejects_legacy_ridge_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, alpha=1.0)
            with self.assertRaises(ValueError):
                CNNStrengthPredictor.load(path)

    def test_loads_checkpoint_from_before_position_branch(self) -> None:
        model = CNNStrengthPredictor(
            sequence_length=50,
            stem_channels=8,
            kernel_sizes=(3, 5),
            merge_channels=16,
            hidden_size=16,
            position_hidden_size=0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_model.pt"
            model.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            for name in (
                "position_hidden_size",
                "position_feature_source",
                "loss_name",
                "selection_metric",
                "calibration",
            ):
                checkpoint["spec"].pop(name, None)
            torch.save(checkpoint, path)
            loaded = CNNStrengthPredictor.load(path, device="cpu")
        self.assertEqual(loaded.position_hidden_size, 0)
        self.assertEqual(loaded.loss_name, "huber")
        self.assertEqual(loaded.selection_metric, "pearson")
        self.assertEqual(loaded.calibration, "none")


if __name__ == "__main__":
    unittest.main()
