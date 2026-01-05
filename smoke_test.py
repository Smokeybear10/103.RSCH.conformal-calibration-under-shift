"""Exercise the experiment on synthetic participants without downloading data.

This checks that fitting, calibration, and evaluation work together. Its
metrics are synthetic diagnostics, not evidence for the published findings.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

import conformal_core as conformal
from run_experiment import ALPHAS, METHODS, make_splits, select_models, validate_probabilities


def main() -> None:
    rng = np.random.default_rng(42)
    subjects = np.repeat(np.arange(1, 31), 24)
    labels = np.tile(np.repeat(np.arange(6), 4), 30)
    centers = 2.0 * np.eye(6)
    person_offsets = rng.normal(0, 0.25, size=(30, 6))
    features = centers[labels] + person_offsets[subjects - 1] + rng.normal(0, 0.8, size=(720, 6))
    split = make_splits(subjects)[0]
    masks = {role: np.isin(subjects, ids) for role, ids in split["subjects"].items()}

    print("Synthetic smoke check: 30 participants; 12 fit, 12 calibration, 6 test.")
    print("These metrics do not reproduce the UCI HAR or WISDM findings.")
    print(f"{'model':<10} {'method':<18} {'alpha':>5} {'coverage':>10} {'review':>10}")
    conditions = 0
    with threadpool_limits(limits=2):
        models, _, _ = select_models(features[masks["fit"]], labels[masks["fit"]], subjects[masks["fit"]], 42)
        for family, model in models.items():
            fitted = clone(model).fit(features[masks["fit"]], labels[masks["fit"]])
            calibration = fitted.predict_proba(features[masks["calibration"]])
            test = fitted.predict_proba(features[masks["test"]])
            validate_probabilities(calibration)
            validate_probabilities(test)
            for method, settings in METHODS.items():
                scoring = {key: settings[key] for key in ("regularization", "k_reg")}
                scores = conformal.true_label_scores(calibration, labels[masks["calibration"]], **scoring)
                for alpha in ALPHAS:
                    if settings["calibration"] == "window":
                        threshold = conformal.pooled_threshold(scores, alpha)
                    else:
                        threshold = conformal.hierarchical_threshold(scores, subjects[masks["calibration"]], alpha)
                    sets = conformal.prediction_sets(test, threshold, **scoring)
                    if not np.isfinite(threshold) or sets.shape != (144, 6) or not sets.any(axis=1).all():
                        raise AssertionError("Unexpected threshold or prediction sets in synthetic pipeline.")
                    metrics = conformal.set_metrics(sets, labels[masks["test"]])
                    per_person = conformal.group_set_metrics(sets, labels[masks["test"]], subjects[masks["test"]])
                    if len(per_person) != 6 or sum(row["n"] for row in per_person) != metrics["n"]:
                        raise AssertionError("Participant metrics do not account for the test rows.")
                    print(f"{family:<10} {method:<18} {alpha:5.2f} {metrics['coverage']:10.3f} {metrics['review_rate']:10.3f}")
                    conditions += 1
    if conditions != 16:
        raise AssertionError("Expected both models, four methods, and two alpha values.")
    print("Passed all 16 model/method/alpha conditions. No files were written.")


if __name__ == "__main__":
    main()
