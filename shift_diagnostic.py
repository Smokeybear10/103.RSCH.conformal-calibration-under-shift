from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_SOURCE = HERE / "data" / "UCI HAR Dataset"
METHOD_LABELS = {
    "window_aps": "Window APS",
    "window_raps": "Window RAPS",
    "participant_aps": "Participant APS",
    "participant_raps": "Participant RAPS",
}
METHOD_COLORS = {
    "window_aps": "#5b728c",
    "window_raps": "#1f77b4",
    "participant_aps": "#a76d35",
    "participant_raps": "#c44e52",
}


def load_features(source_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    required = source_dir / "train" / "X_train.txt"
    if not required.exists():
        expected_default = HERE / "data" / "UCI HAR Dataset"
        if source_dir != expected_default:
            raise FileNotFoundError(
                f"Missing UCI HAR features at {source_dir}. Run run_experiment.py first or pass its data directory."
            )
        # Keep this diagnostic independently reproducible without trusting an
        # unverified local copy: reuse the main runner's verified downloader.
        from run_experiment import load_data

        load_data(source_dir.parent)
    X = np.r_[
        np.loadtxt(source_dir / "train" / "X_train.txt"),
        np.loadtxt(source_dir / "test" / "X_test.txt"),
    ]
    subjects = np.r_[
        np.loadtxt(source_dir / "train" / "subject_train.txt", dtype=int),
        np.loadtxt(source_dir / "test" / "subject_test.txt", dtype=int),
    ]
    if X.shape != (10299, 561) or subjects.shape != (10299,):
        raise ValueError("Unexpected UCI HAR feature or participant shape.")
    return X, subjects


def source_only_shift_rows(X: np.ndarray, subjects: np.ndarray, splits: list[dict]) -> pd.DataFrame:
    rows = []
    for split in splits:
        reference_people = np.asarray(split["subjects"]["fit"] + split["subjects"]["calibration"])
        reference = X[np.isin(subjects, reference_people)]
        reference_mean = reference.mean(axis=0)
        reference_std = reference.std(axis=0)
        reference_std[reference_std < 1e-9] = 1.0
        for subject in split["subjects"]["test"]:
            target_mean = X[subjects == subject].mean(axis=0)
            shift_score = float(np.sqrt(np.mean(((target_mean - reference_mean) / reference_std) ** 2)))
            rows.append(
                {
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "seed": split["seed"],
                    "subject": int(subject),
                    "feature_shift": shift_score,
                    "reference_people": len(reference_people),
                    "uses_target_labels": False,
                }
            )
    shifts = pd.DataFrame(rows)
    if len(shifts) != 90 or shifts.duplicated(["repeat", "subject"]).any():
        raise AssertionError("A participant should appear in exactly one test fold per repeat.")
    return shifts


def aggregate_diagnostic(shifts: pd.DataFrame, subject_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = subject_metrics[np.isclose(subject_metrics["alpha"], 0.10)].copy()
    merged = metrics.merge(shifts, on=["repeat", "fold", "seed", "subject"], how="inner", validate="many_to_one")
    if len(merged) != len(metrics):
        raise AssertionError("Every 90% coverage measurement needs a matching source-only shift score.")

    rows = []
    for (family, method), frame in merged.groupby(["family", "method"], sort=True):
        people = frame.groupby("subject", as_index=False).agg(
            feature_shift=("feature_shift", "mean"),
            coverage=("coverage", "mean"),
            review_rate=("review_rate", "mean"),
            repeats=("repeat", "nunique"),
        )
        if len(people) != 30 or not people["repeats"].eq(3).all():
            raise AssertionError("Every method must evaluate all participants across three repeats.")
        rho = float(people[["feature_shift", "coverage"]].corr(method="spearman").iloc[0, 1])
        people = people.assign(
            family=family,
            method=method,
            method_label=METHOD_LABELS[method],
            alpha=0.10,
            spearman_rho=rho,
            diagnostic="Post-hoc descriptive association; feature shift uses fit+calibration X only, never target labels.",
        )
        rows.append(people)
    return pd.concat(rows, ignore_index=True).sort_values(["family", "method", "subject"], kind="stable")


def make_figure(diagnostic: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for axis, family in zip(axes, ("Logistic", "Forest")):
        family_rows = diagnostic[diagnostic["family"] == family]
        for method, frame in family_rows.groupby("method", sort=False):
            axis.scatter(
                frame["feature_shift"],
                100 * frame["coverage"],
                s=34,
                alpha=0.76,
                color=METHOD_COLORS[method],
                label=f"{METHOD_LABELS[method]} ($\\rho$={frame['spearman_rho'].iat[0]:.2f})",
            )
        axis.axhline(90, color="#991b1b", linestyle="--", linewidth=1)
        axis.grid(axis="y", alpha=0.2)
        axis.set(title=family, xlabel="Source-only standardized feature-mean shift")
    axes[0].set_ylabel("Mean held-out-person coverage across repeats (%)")
    axes[1].legend(loc="lower left", fontsize=8)
    figure.suptitle("People farther from the reference feature distribution tend to lose coverage", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Shift uses only fit + calibration features. Correlations are post-hoc descriptive, not causal or inferential claims.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run(source_dir: Path, results_dir: Path) -> pd.DataFrame:
    X, subjects = load_features(source_dir)
    protocol = json.loads((results_dir / "protocol.json").read_text())
    shifts = source_only_shift_rows(X, subjects, protocol["splits"])
    subject_metrics = pd.read_csv(results_dir / "subject_metrics.csv")
    diagnostic = aggregate_diagnostic(shifts, subject_metrics)
    diagnostic.to_csv(results_dir / "shift_diagnostic.csv", index=False)
    make_figure(diagnostic, results_dir / "shift_vs_coverage.png")
    summary = (
        diagnostic.groupby(["family", "method", "method_label"], as_index=False)["spearman_rho"]
        .first()
        .sort_values(["family", "method"], kind="stable")
    )
    print(summary.to_string(index=False), flush=True)
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute a post-hoc, source-only feature-shift diagnostic for held-out UCI HAR participants.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    arguments = parser.parse_args()
    run(arguments.source_dir, arguments.results_dir)
