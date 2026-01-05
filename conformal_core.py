"""Deterministic conformal prediction helpers for multiclass classifiers."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np


def _as_alpha_fraction(alpha: float | Fraction) -> Fraction:
    value = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
    if not 0 < value < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    return value


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_observations, n_classes) with at least two classes.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and nonnegative.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-8):
        raise ValueError("each probability row must sum to one.")
    return values


def _validate_labels(labels: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("labels must be a one-dimensional array matching the number of probability rows.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must contain zero-based integer class indices.")
    if (values < 0).any() or (values >= n_classes).any():
        raise ValueError("labels are outside the class range.")
    return values.astype(int, copy=False)


def adaptive_scores(
    probabilities: np.ndarray,
    *,
    regularization: float = 0.0,
    k_reg: int = 0,
) -> np.ndarray:
    """Return deterministic APS/RAPS-style nonconformity scores for every class.

    A score is the cumulative probability through a candidate class in descending
    probability order, plus ``regularization * max(rank - k_reg, 0)``. Exact
    probability ties receive the score at the end of their tied block, so a set
    never separates equal-probability classes because of class-index ordering.
    """
    values = _validate_probabilities(probabilities)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not isinstance(k_reg, (int, np.integer)) or k_reg < 0:
        raise ValueError("k_reg must be a nonnegative integer.")

    order = np.argsort(-values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(sorted_values, axis=1)
    sorted_scores = np.empty_like(sorted_values)

    for row in range(len(values)):
        start = 0
        while start < values.shape[1]:
            end = start
            while end + 1 < values.shape[1] and sorted_values[row, end + 1] == sorted_values[row, start]:
                end += 1
            rank = end + 1
            sorted_scores[row, start : end + 1] = cumulative[row, end] + regularization * max(rank - k_reg, 0)
            start = end + 1

    scores = np.empty_like(values)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def true_label_scores(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float = 0.0,
    k_reg: int = 0,
) -> np.ndarray:
    """Return APS/RAPS-style scores for the observed class labels."""
    values = _validate_probabilities(probabilities)
    y = _validate_labels(labels, len(values), values.shape[1])
    return adaptive_scores(values, regularization=regularization, k_reg=k_reg)[np.arange(len(values)), y]


def pooled_threshold(calibration_scores: np.ndarray, alpha: float | Fraction) -> float:
    """Return the finite-sample split-conformal threshold without interpolation.

    It is the ``ceil((n + 1) * (1 - alpha))`` order statistic, with an extra
    point mass at infinity. The usual marginal coverage statement requires an
    exchangeable calibration point and future test point.
    """
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration_scores must be a nonempty, finite one-dimensional array.")
    target = (len(scores) + 1) * (1 - _as_alpha_fraction(alpha))
    rank = (target.numerator + target.denominator - 1) // target.denominator
    if rank > len(scores):
        return float("inf")
    return float(np.sort(scores, kind="stable")[rank - 1])


def hierarchical_threshold(
    calibration_scores: np.ndarray,
    groups: np.ndarray,
    alpha: float | Fraction,
) -> float:
    """Return a group-balanced HCP threshold for a first point from a new group.

    Each of ``G`` calibration groups contributes total mass ``1 / (G + 1)``,
    divided equally across its observations; the remaining ``1 / (G + 1)`` is
    placed at infinity for the unseen group. This is a non-interpolated weighted
    empirical quantile. Its formal coverage guarantee needs hierarchical
    exchangeability across and within groups.
    """
    scores = np.asarray(calibration_scores, dtype=float)
    group_values = np.asarray(groups)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration_scores must be a nonempty, finite one-dimensional array.")
    if group_values.ndim != 1 or len(group_values) != len(scores):
        raise ValueError("groups must be one-dimensional and match calibration_scores.")

    _, inverse, counts = np.unique(group_values, return_inverse=True, return_counts=True)
    n_groups = len(counts)
    target = (n_groups + 1) * (1 - _as_alpha_fraction(alpha))
    if target > n_groups:
        return float("inf")

    weights = 1.0 / counts[inverse]
    order = np.argsort(scores, kind="stable")
    target_weight = float(target)
    cumulative = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        cumulative += float(weights[order[start:end]].sum())
        if cumulative + 1e-12 >= target_weight:
            return float(scores[order[start]])
        start = end
    raise AssertionError("the finite hierarchical quantile was not reached.")


def prediction_sets(
    probabilities: np.ndarray,
    threshold: float,
    *,
    regularization: float = 0.0,
    k_reg: int = 0,
) -> np.ndarray:
    """Construct a nonempty boolean prediction-set matrix from a conformal threshold."""
    if np.isnan(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative or positive infinity.")
    values = _validate_probabilities(probabilities)
    scores = adaptive_scores(values, regularization=regularization, k_reg=k_reg)
    sets = scores <= threshold
    sets |= values == values.max(axis=1, keepdims=True)
    return sets


def set_metrics(prediction_set: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    """Summarize coverage, ambiguity, and the singleton auto-accept policy."""
    sets = np.asarray(prediction_set, dtype=bool)
    if sets.ndim != 2 or sets.shape[0] == 0:
        raise ValueError("prediction_set must be a nonempty two-dimensional boolean array.")
    y = _validate_labels(labels, len(sets), sets.shape[1])
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(sets)), y]
    singleton = sizes == 1
    singleton_errors = singleton & ~covered
    n_singleton = int(singleton.sum())
    return {
        "n": int(len(sets)),
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "review_rate": float((~singleton).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "n_singleton": n_singleton,
        "n_singleton_errors": int(singleton_errors.sum()),
        "singleton_error_rate": float(singleton_errors.sum() / n_singleton) if n_singleton else None,
    }


def group_set_metrics(prediction_set: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> list[dict[str, Any]]:
    """Return ``set_metrics`` separately for each observed group."""
    sets = np.asarray(prediction_set, dtype=bool)
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != len(sets):
        raise ValueError("groups must be one-dimensional and match prediction_set rows.")
    rows = []
    for group in np.unique(group_values):
        mask = group_values == group
        rows.append({"group": group.item() if isinstance(group, np.generic) else group} | set_metrics(sets[mask], np.asarray(labels)[mask]))
    return rows


def _self_test() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
    scores = adaptive_scores(probabilities)
    assert np.allclose(scores[0], [0.7, 0.9, 1.0])
    assert np.allclose(scores[1], [0.4, 1.0, 1.0])
    assert np.allclose(true_label_scores(probabilities, np.array([0, 2])), [0.7, 1.0])
    assert pooled_threshold(np.array([0.1, 0.2, 0.3, 0.4]), 0.2) == 0.4
    assert np.isinf(pooled_threshold(np.array([0.1, 0.2, 0.3, 0.4]), 0.1))
    assert hierarchical_threshold(np.array([0.1, 0.2, 0.9, 0.95]), np.array([1, 1, 2, 2]), 0.4) == 0.95
    assert np.isinf(hierarchical_threshold(np.array([0.1, 0.2, 0.9, 0.95]), np.array([1, 1, 2, 2]), 0.2))
    sets = prediction_sets(probabilities, 0.7)
    metrics = set_metrics(sets, np.array([0, 1]))
    assert metrics["n"] == 2 and metrics["n_singleton"] == 2 and metrics["n_singleton_errors"] == 1
    assert prediction_sets(probabilities, 0.1).sum(axis=1).min() >= 1


if __name__ == "__main__":
    _self_test()
