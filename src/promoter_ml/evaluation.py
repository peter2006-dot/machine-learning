"""Model-independent evaluation of generated promoter sequences."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import DNA_ALPHABET, KmerFeaturizer
from .metrics import regression_metrics


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else None


def _finite_std(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.std()) if len(finite) else None


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _jensen_shannon_divergence(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left = left / left.sum() if left.sum() else np.full_like(left, 1.0 / len(left))
    right = right / right.sum() if right.sum() else np.full_like(right, 1.0 / len(right))
    midpoint = 0.5 * (left + right)

    def kl_divergence(values: np.ndarray, reference: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def sequence_is_valid(sequence: str, sequence_length: int, alphabet: str = DNA_ALPHABET) -> bool:
    """Return whether a sequence follows the fixed promoter data contract."""
    return len(sequence) == sequence_length and not (set(sequence) - set(alphabet))


_BASE_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}

# Approximate E. coli σ70 frequency matrices, rows A/C/G/T, columns 5'→3'.
# Values follow common compilations of -35 TTGACA and -10 TATAAT sites.
SIGMA70_MINUS35_FREQUENCIES = np.array(
    [
        [0.03, 0.06, 0.15, 0.58, 0.18, 0.52],
        [0.10, 0.06, 0.12, 0.14, 0.54, 0.21],
        [0.15, 0.12, 0.62, 0.16, 0.16, 0.13],
        [0.72, 0.76, 0.11, 0.12, 0.12, 0.14],
    ],
    dtype=np.float64,
)
SIGMA70_MINUS10_FREQUENCIES = np.array(
    [
        [0.04, 0.82, 0.07, 0.54, 0.53, 0.06],
        [0.07, 0.06, 0.06, 0.18, 0.21, 0.07],
        [0.09, 0.06, 0.12, 0.17, 0.12, 0.09],
        [0.80, 0.06, 0.75, 0.11, 0.14, 0.78],
    ],
    dtype=np.float64,
)


def pwm_log_odds(
    frequencies: np.ndarray,
    background: float = 0.25,
    pseudocount: float = 0.01,
) -> np.ndarray:
    """Convert base frequencies into log2 odds versus a uniform background."""
    values = np.asarray(frequencies, dtype=np.float64) + pseudocount
    values = values / values.sum(axis=0, keepdims=True)
    return np.log2(values / background)


def pwm_from_consensus(motif: str, match_prob: float = 0.70) -> np.ndarray:
    """Build a peaked PWM when a literature matrix is not available."""
    motif = motif.upper()
    if set(motif) - set(DNA_ALPHABET):
        raise ValueError(f"Motif contains non-DNA characters: {motif!r}")
    mismatch = (1.0 - match_prob) / (len(DNA_ALPHABET) - 1)
    frequencies = np.full((len(DNA_ALPHABET), len(motif)), mismatch, dtype=np.float64)
    for position, base in enumerate(motif):
        frequencies[_BASE_INDEX[base], position] = match_prob
    return pwm_log_odds(frequencies)


def motif_pwm(motif: str) -> np.ndarray:
    """Return the log-odds PWM used for a canonical or custom motif."""
    motif = motif.upper()
    if motif == "TTGACA":
        return pwm_log_odds(SIGMA70_MINUS35_FREQUENCIES)
    if motif == "TATAAT":
        return pwm_log_odds(SIGMA70_MINUS10_FREQUENCIES)
    return pwm_from_consensus(motif)


def _relative_pwm_score(score: float, pwm: np.ndarray) -> float:
    maximum = float(pwm.max(axis=0).sum())
    minimum = float(pwm.min(axis=0).sum())
    if abs(maximum - minimum) < 1e-12:
        return 0.0
    return float((score - minimum) / (maximum - minimum))


def _pwm_window_hits(sequence: str, motif: str, pwm: np.ndarray) -> list[tuple[int, int, float, float]]:
    width = pwm.shape[1]
    if len(sequence) < width:
        return []
    encoded = np.array([_BASE_INDEX.get(base, -1) for base in sequence], dtype=np.int16)
    if np.any(encoded < 0):
        return []
    windows = np.lib.stride_tricks.sliding_window_view(encoded, width)
    position_index = np.arange(width)
    scores = pwm[windows, position_index].sum(axis=1)
    consensus = np.array([_BASE_INDEX[base] for base in motif], dtype=np.int16)
    mismatches = np.sum(windows != consensus, axis=1)
    hits = []
    for start, mismatch_count, score in zip(range(len(scores)), mismatches, scores):
        value = float(score)
        hits.append((start, int(mismatch_count), value, _relative_pwm_score(value, pwm)))
    return hits


def _empty_motif_result() -> dict[str, Any]:
    return {
        "minus35_start": None,
        "minus35_mismatches": None,
        "minus35_score": None,
        "minus35_pwm_score": None,
        "minus10_start": None,
        "minus10_mismatches": None,
        "minus10_score": None,
        "minus10_pwm_score": None,
        "spacer_length": None,
        "minus35_present": False,
        "minus10_present": False,
        "motif_pair_present": False,
        "spacing_valid": False,
        "motif_pair_valid": False,
        "relative_score_threshold": None,
    }


def scan_promoter_motifs(
    sequence: str,
    minus35: str = "TTGACA",
    minus10: str = "TATAAT",
    relative_score_threshold: float = 0.60,
    spacer_min: int = 15,
    spacer_max: int = 19,
) -> dict[str, Any]:
    """Find the highest-scoring ordered -35/-10 PWM pair in one promoter.

    Presence uses a relative PWM score threshold, not Hamming distance. The
    default 0.60 is slightly more permissive than two substitutions in the
    canonical σ70 matrices, which keeps known box variants while rejecting
    random hexamers.
    """
    minus35 = minus35.upper()
    minus10 = minus10.upper()
    minus35_pwm = motif_pwm(minus35)
    minus10_pwm = motif_pwm(minus10)
    candidates = []
    midpoint = (spacer_min + spacer_max) / 2.0
    for hit35 in _pwm_window_hits(sequence, minus35, minus35_pwm):
        for hit10 in _pwm_window_hits(sequence, minus10, minus10_pwm):
            spacer = hit10[0] - (hit35[0] + len(minus35))
            if spacer < 0:
                continue
            in_range = spacer_min <= spacer <= spacer_max
            candidates.append(
                (
                    hit35[3] + hit10[3],
                    int(in_range),
                    -abs(spacer - midpoint),
                    hit35,
                    hit10,
                    spacer,
                )
            )
    if not candidates:
        result = _empty_motif_result()
        result["relative_score_threshold"] = float(relative_score_threshold)
        return result

    _, _, _, hit35, hit10, spacer = max(candidates, key=lambda item: item[:3])
    minus35_present = hit35[3] >= relative_score_threshold
    minus10_present = hit10[3] >= relative_score_threshold
    pair_present = minus35_present and minus10_present
    spacing_valid = spacer_min <= spacer <= spacer_max
    return {
        "minus35_start": int(hit35[0]),
        "minus35_mismatches": int(hit35[1]),
        "minus35_score": float(hit35[3]),
        "minus35_pwm_score": float(hit35[2]),
        "minus10_start": int(hit10[0]),
        "minus10_mismatches": int(hit10[1]),
        "minus10_score": float(hit10[3]),
        "minus10_pwm_score": float(hit10[2]),
        "spacer_length": int(spacer),
        "minus35_present": bool(minus35_present),
        "minus10_present": bool(minus10_present),
        "motif_pair_present": bool(pair_present),
        "spacing_valid": bool(spacing_valid),
        "motif_pair_valid": bool(pair_present and spacing_valid),
        "relative_score_threshold": float(relative_score_threshold),
    }


def nearest_reference_identities(sequences: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return maximum positional identity to the reference set per sequence."""
    generated = np.asarray([list(value) for value in sequences], dtype="U1")
    reference = np.asarray([list(value) for value in references], dtype="U1")
    if generated.ndim != 2 or reference.ndim != 2 or generated.shape[1] != reference.shape[1]:
        raise ValueError("Generated and reference sequences must have the same fixed length")
    identities = np.empty(len(generated), dtype=np.float64)
    for index, sequence in enumerate(generated):
        identities[index] = float(np.max(np.mean(reference == sequence, axis=1)))
    return identities


def pairwise_diversity(
    sequences: np.ndarray,
    max_sequences: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Calculate mean pairwise and nearest-neighbor normalized Hamming distance."""
    values = np.asarray(sequences).astype(str)
    if len(values) < 2:
        return {
            "evaluated_sequence_count": int(len(values)),
            "mean_pairwise_hamming_distance": None,
            "mean_nearest_neighbor_hamming_distance": None,
        }
    if len(values) > max_sequences:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), size=max_sequences, replace=False)]
    encoded = np.asarray([list(value) for value in values], dtype="U1")
    pair_sum = 0.0
    pair_count = 0
    nearest = np.ones(len(values), dtype=np.float64)
    for left in range(len(values) - 1):
        distances = np.mean(encoded[left + 1 :] != encoded[left], axis=1)
        pair_sum += float(distances.sum())
        pair_count += len(distances)
        nearest[left] = min(nearest[left], float(distances.min()))
        nearest[left + 1 :] = np.minimum(nearest[left + 1 :], distances)
    return {
        "evaluated_sequence_count": int(len(values)),
        "mean_pairwise_hamming_distance": pair_sum / pair_count,
        "mean_nearest_neighbor_hamming_distance": float(nearest.mean()),
    }


def _distribution_metrics(
    generated: np.ndarray,
    references: np.ndarray,
    kmer_max: int,
) -> dict[str, float | None]:
    featurizer = KmerFeaturizer(k_min=1, k_max=kmer_max, include_gc=False)
    generated_profile = featurizer.transform(generated).mean(axis=0)
    reference_profile = featurizer.transform(references).mean(axis=0)
    generated_gc = np.asarray([(value.count("G") + value.count("C")) / len(value) for value in generated])
    reference_gc = np.asarray([(value.count("G") + value.count("C")) / len(value) for value in references])
    bins = np.linspace(0.0, 1.0, 21)
    generated_histogram, _ = np.histogram(generated_gc, bins=bins)
    reference_histogram, _ = np.histogram(reference_gc, bins=bins)
    return {
        "kmer_profile_pearson": _safe_correlation(generated_profile, reference_profile),
        "kmer_profile_js_divergence": _jensen_shannon_divergence(generated_profile, reference_profile),
        "gc_histogram_js_divergence": _jensen_shannon_divergence(generated_histogram, reference_histogram),
        "reference_gc_mean": float(reference_gc.mean()),
        "generated_gc_mean": float(generated_gc.mean()),
    }


def evaluate_generated_promoters(
    sequences: np.ndarray,
    sequence_length: int,
    references: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    target_log10_strengths: np.ndarray | None = None,
    alphabet: str = DNA_ALPHABET,
    minus35: str = "TTGACA",
    minus10: str = "TATAAT",
    relative_score_threshold: float = 0.60,
    spacer_min: int = 15,
    spacer_max: int = 19,
    novelty_identity_threshold: float = 0.9,
    target_tolerances: tuple[float, ...] = (0.25, 0.5),
    max_pairwise_sequences: int = 2000,
    kmer_comparison_max: int = 3,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate generated promoters and return aggregate and per-sequence results."""
    values = np.char.upper(np.asarray(sequences).astype(str))
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Generated sequences must be a non-empty one-dimensional array")
    for optional, name in ((predictions, "predictions"), (target_log10_strengths, "targets")):
        if optional is not None and len(optional) != len(values):
            raise ValueError(f"{name} must have the same length as generated sequences")

    counts = Counter(values.tolist())
    valid_mask = np.asarray([sequence_is_valid(value, sequence_length, alphabet) for value in values])
    valid_values = values[valid_mask]
    rows: list[dict[str, Any]] = []

    nearest_by_index: dict[int, float] = {}
    reference_values: np.ndarray | None = None
    if references is not None:
        candidate_references = np.char.upper(np.asarray(references).astype(str))
        reference_mask = np.asarray(
            [sequence_is_valid(value, sequence_length, alphabet) for value in candidate_references]
        )
        reference_values = candidate_references[reference_mask]
        if len(reference_values) == 0:
            raise ValueError("Reference set contains no valid sequences")
        if len(valid_values):
            identities = nearest_reference_identities(valid_values, reference_values)
            for original_index, identity in zip(np.flatnonzero(valid_mask), identities):
                nearest_by_index[int(original_index)] = float(identity)

    prediction_values = None if predictions is None else np.asarray(predictions, dtype=np.float64)
    target_values = None if target_log10_strengths is None else np.asarray(target_log10_strengths, dtype=np.float64)
    for index, sequence in enumerate(values):
        valid = bool(valid_mask[index])
        row: dict[str, Any] = {
            "index": index,
            "sequence": sequence,
            "valid": valid,
            "duplicate": counts[sequence] > 1,
            "gc_fraction": (
                (sequence.count("G") + sequence.count("C")) / len(sequence) if valid else None
            ),
        }
        row.update(
            scan_promoter_motifs(
                sequence,
                minus35=minus35,
                minus10=minus10,
                relative_score_threshold=relative_score_threshold,
                spacer_min=spacer_min,
                spacer_max=spacer_max,
            )
            if valid
            else scan_promoter_motifs("")
        )
        nearest_identity = nearest_by_index.get(index)
        row["nearest_reference_identity"] = nearest_identity
        row["exact_reference_match"] = nearest_identity == 1.0 if nearest_identity is not None else None
        row["novel_at_identity_threshold"] = (
            nearest_identity < novelty_identity_threshold if nearest_identity is not None else None
        )

        prediction_candidate = float(prediction_values[index]) if prediction_values is not None else None
        target_candidate = float(target_values[index]) if target_values is not None else None
        prediction = (
            prediction_candidate
            if prediction_candidate is not None and np.isfinite(prediction_candidate)
            else None
        )
        target = target_candidate if target_candidate is not None and np.isfinite(target_candidate) else None
        row["predicted_log10_strength"] = prediction
        row["target_log10_strength"] = target
        error = abs(prediction - target) if prediction is not None and target is not None else None
        row["absolute_log10_error"] = error
        for tolerance in target_tolerances:
            row[f"target_hit_within_{tolerance:.2f}"] = error <= tolerance if error is not None else None
        rows.append(row)

    valid_count = int(valid_mask.sum())
    unique_count = len(set(valid_values.tolist()))
    summary: dict[str, Any] = {
        "counts": {
            "all": int(len(values)),
            "valid": valid_count,
            "invalid": int(len(values) - valid_count),
            "unique_valid": int(unique_count),
        },
        "validity": {
            "valid_rate": valid_count / len(values),
            "unique_rate_among_valid": unique_count / valid_count if valid_count else None,
        },
    }
    if valid_count:
        valid_rows = [row for row in rows if row["valid"]]
        detected_pairs = [row for row in valid_rows if row["motif_pair_present"]]
        summary["motifs"] = {
            "scanner": "sigma70_pwm_relative_score",
            "relative_score_threshold": float(relative_score_threshold),
            "minus35_presence_rate": float(np.mean([row["minus35_present"] for row in valid_rows])),
            "minus10_presence_rate": float(np.mean([row["minus10_present"] for row in valid_rows])),
            "motif_pair_presence_rate": float(np.mean([row["motif_pair_present"] for row in valid_rows])),
            "valid_spacing_rate_among_detected_pairs": (
                float(np.mean([row["spacing_valid"] for row in detected_pairs])) if detected_pairs else None
            ),
            "motif_pair_valid_rate": float(np.mean([row["motif_pair_valid"] for row in valid_rows])),
            "mean_minus35_relative_score": float(np.mean([row["minus35_score"] for row in valid_rows])),
            "mean_minus10_relative_score": float(np.mean([row["minus10_score"] for row in valid_rows])),
        }
        gc_values = np.asarray([row["gc_fraction"] for row in rows if row["valid"]], dtype=np.float64)
        summary["composition"] = {
            "gc_mean": float(gc_values.mean()),
            "gc_std": float(gc_values.std()),
        }
        summary["diversity"] = pairwise_diversity(
            valid_values,
            max_sequences=max_pairwise_sequences,
            seed=seed,
        )
    else:
        summary["motifs"] = {}
        summary["composition"] = {}
        summary["diversity"] = pairwise_diversity(valid_values)

    if reference_values is not None and valid_count:
        nearest_values = np.asarray([nearest_by_index[index] for index in np.flatnonzero(valid_mask)])
        summary["novelty"] = {
            "exact_novelty_rate": float(np.mean(nearest_values < 1.0)),
            "novelty_identity_threshold": float(novelty_identity_threshold),
            "novel_at_threshold_rate": float(np.mean(nearest_values < novelty_identity_threshold)),
            "mean_nearest_reference_identity": float(nearest_values.mean()),
            "max_nearest_reference_identity": float(nearest_values.max()),
        }
        summary["distribution"] = _distribution_metrics(
            valid_values,
            reference_values,
            kmer_max=kmer_comparison_max,
        )

    if prediction_values is not None:
        valid_predictions = prediction_values[valid_mask]
        summary["predicted_strength"] = {
            "mean_log10_strength": _finite_mean(valid_predictions),
            "std_log10_strength": _finite_std(valid_predictions),
        }
    if prediction_values is not None and target_values is not None:
        evaluable = valid_mask & np.isfinite(prediction_values) & np.isfinite(target_values)
        errors = np.abs(prediction_values[evaluable] - target_values[evaluable])
        regression = (
            regression_metrics(target_values[evaluable], prediction_values[evaluable])
            if evaluable.any()
            else None
        )
        strength_metrics: dict[str, Any] = {
            "evaluable_count": int(evaluable.sum()),
            "mae_log10": regression["mae"] if regression else None,
            "rmse_log10": regression["rmse"] if regression else None,
            "pearson": regression["pearson"] if regression else None,
            "spearman": regression["spearman"] if regression else None,
            "mean_signed_error_log10": (
                float(np.mean(prediction_values[evaluable] - target_values[evaluable])) if len(errors) else None
            ),
        }
        for tolerance in target_tolerances:
            strength_metrics[f"hit_rate_within_{tolerance:.2f}_log10"] = (
                float(np.mean(errors <= tolerance)) if len(errors) else None
            )
        summary["target_control"] = strength_metrics
    return summary, rows


_ACCEPTANCE_RULES: dict[str, tuple[tuple[str, ...], str]] = {
    "min_valid_rate": (("validity", "valid_rate"), ">="),
    "min_unique_rate_among_valid": (("validity", "unique_rate_among_valid"), ">="),
    "min_motif_pair_valid_rate": (("motifs", "motif_pair_valid_rate"), ">="),
    "min_novel_at_threshold_rate": (("novelty", "novel_at_threshold_rate"), ">="),
    "min_mean_pairwise_hamming_distance": (("diversity", "mean_pairwise_hamming_distance"), ">="),
    "max_kmer_profile_js_divergence": (("distribution", "kmer_profile_js_divergence"), "<="),
    "max_gc_histogram_js_divergence": (("distribution", "gc_histogram_js_divergence"), "<="),
    "max_target_mae_log10": (("target_control", "mae_log10"), "<="),
    "min_target_hit_rate_within_0_50_log10": (
        ("target_control", "hit_rate_within_0.50_log10"),
        ">=",
    ),
    "max_evaluator_test_mae_log10": (("evaluator_quality", "test", "mae"), "<="),
    "min_evaluator_test_pearson": (("evaluator_quality", "test", "pearson"), ">="),
}


def evaluate_acceptance(summary: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Evaluate configured quality gates against a validation summary."""
    control_keys = {"enabled", "require_all_metrics", "required_rules"}
    unknown = set(rules) - control_keys - set(_ACCEPTANCE_RULES)
    if unknown:
        raise ValueError(f"Unknown acceptance rules: {sorted(unknown)}")
    require_all = bool(rules.get("require_all_metrics", False))
    required_rules = {str(name) for name in rules.get("required_rules", [])}
    unknown_required = required_rules - set(_ACCEPTANCE_RULES)
    if unknown_required:
        raise ValueError(f"Unknown required acceptance rules: {sorted(unknown_required)}")
    results: dict[str, dict[str, Any]] = {}
    for name, (path, operator) in _ACCEPTANCE_RULES.items():
        if name not in rules:
            continue
        threshold = float(rules[name])
        value: Any = summary
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
            if value is None:
                break
        if value is None or not np.isfinite(float(value)):
            status = "fail" if require_all or name in required_rules else "skipped"
            actual = None
        else:
            actual = float(value)
            passed = actual >= threshold if operator == ">=" else actual <= threshold
            status = "pass" if passed else "fail"
        results[name] = {
            "metric": ".".join(path),
            "operator": operator,
            "threshold": threshold,
            "actual": actual,
            "status": status,
        }

    failed = sum(result["status"] == "fail" for result in results.values())
    passed = sum(result["status"] == "pass" for result in results.values())
    skipped = sum(result["status"] == "skipped" for result in results.values())
    status = "fail" if failed else ("pass" if passed else "not_evaluated")
    return {
        "status": status,
        "passed_rule_count": int(passed),
        "failed_rule_count": int(failed),
        "skipped_rule_count": int(skipped),
        "require_all_metrics": require_all,
        "required_rules": sorted(required_rules),
        "rules": results,
    }
