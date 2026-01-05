"""Show when pooled conformal calibration targets the wrong population.

This is a controlled score-level simulation, not a model of UCI HAR. Each
person is easy or hard, and conformal scores are conditionally i.i.d. within a
person. The equal-size regime gives every calibration person 200 windows. The
size-biased regime gives easy people 400 windows and hard people 40 windows.
The target population chooses people uniformly, so pooled windows target a
different population in the second regime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conformal_core as conformal


HERE = Path(__file__).resolve().parent
ALPHA = 0.10
SEED = 20261105
REPETITIONS = 1000
N_CALIBRATION_PEOPLE = 20
N_EASY_CALIBRATION_PEOPLE = 15
N_TEST_PEOPLE = 80
N_EASY_TEST_PEOPLE = 60
N_TEST_WINDOWS = 200
REGIMES = {
    "equal_windows": {"easy_windows": 200, "hard_windows": 200},
    "size_biased_windows": {"easy_windows": 400, "hard_windows": 40},
}
METHODS = ("pooled_window", "participant_balanced")
COLORS = {"pooled_window": "#2563a6", "participant_balanced": "#bf6330"}
LABELS = {"pooled_window": "Pooled windows", "participant_balanced": "Participant balanced"}


def draw_scores(rng: np.random.Generator, hard: bool, size: int) -> np.ndarray:
    """Low scores represent easy examples; high scores represent hard ones."""
    return rng.beta(4.0, 4.0, size=size) if hard else rng.beta(1.5, 25.0, size=size)


def make_calibration_scores(rng: np.random.Generator, regime: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    scores, groups = [], []
    for person in range(N_CALIBRATION_PEOPLE):
        hard = person >= N_EASY_CALIBRATION_PEOPLE
        size = regime["hard_windows"] if hard else regime["easy_windows"]
        scores.append(draw_scores(rng, hard, size))
        groups.append(np.full(size, person, dtype=int))
    return np.concatenate(scores), np.concatenate(groups)


def evaluate_threshold(rng: np.random.Generator, threshold: float) -> dict[str, float]:
    person_coverages = []
    easy_coverages, hard_coverages = [], []
    for person in range(N_TEST_PEOPLE):
        hard = person >= N_EASY_TEST_PEOPLE
        coverage = float((draw_scores(rng, hard, N_TEST_WINDOWS) <= threshold).mean())
        person_coverages.append(coverage)
        (hard_coverages if hard else easy_coverages).append(coverage)
    coverage = np.asarray(person_coverages)
    return {
        "equal_person_coverage": float(coverage.mean()),
        "easy_person_coverage": float(np.mean(easy_coverages)),
        "hard_person_coverage": float(np.mean(hard_coverages)),
        "p10_person_coverage": float(np.quantile(coverage, 0.10)),
        "fraction_people_below_target": float((coverage < 1 - ALPHA).mean()),
    }


def run_simulation(repetitions: int = REPETITIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    rng = np.random.default_rng(SEED)
    records = []
    for regime_name, regime in REGIMES.items():
        for repetition in range(repetitions):
            calibration_scores, groups = make_calibration_scores(rng, regime)
            thresholds = {
                "pooled_window": conformal.pooled_threshold(calibration_scores, ALPHA),
                "participant_balanced": conformal.hierarchical_threshold(calibration_scores, groups, ALPHA),
            }
            for method, threshold in thresholds.items():
                metrics = evaluate_threshold(rng, threshold)
                records.append(
                    {
                        "regime": regime_name,
                        "method": method,
                        "repetition": repetition,
                        "threshold": threshold,
                        **metrics,
                    }
                )
    draws = pd.DataFrame(records)
    rows = []
    for (regime, method), frame in draws.groupby(["regime", "method"], sort=False):
        settings = REGIMES[regime]
        rows.append(
            {
                "regime": regime,
                "method": method,
                "repetitions": repetitions,
                "alpha": ALPHA,
                "calibration_people": N_CALIBRATION_PEOPLE,
                "easy_calibration_people": N_EASY_CALIBRATION_PEOPLE,
                "hard_calibration_people": N_CALIBRATION_PEOPLE - N_EASY_CALIBRATION_PEOPLE,
                "easy_windows_per_calibration_person": settings["easy_windows"],
                "hard_windows_per_calibration_person": settings["hard_windows"],
                "test_people": N_TEST_PEOPLE,
                "test_windows_per_person": N_TEST_WINDOWS,
                "mean_threshold": float(frame["threshold"].mean()),
                "mean_equal_person_coverage": float(frame["equal_person_coverage"].mean()),
                "mean_easy_person_coverage": float(frame["easy_person_coverage"].mean()),
                "mean_hard_person_coverage": float(frame["hard_person_coverage"].mean()),
                "mean_p10_person_coverage": float(frame["p10_person_coverage"].mean()),
                "mean_fraction_people_below_target": float(frame["fraction_people_below_target"].mean()),
                "hard_coverage_q05": float(frame["hard_person_coverage"].quantile(0.05)),
                "hard_coverage_q95": float(frame["hard_person_coverage"].quantile(0.95)),
            }
        )
    return draws, pd.DataFrame(rows)


def make_figure(draws: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axis, (regime, frame) in zip(axes, draws.groupby("regime", sort=False)):
        positions = np.arange(len(METHODS))
        width = 0.32
        for offset, column, label in ((-width / 2, "easy_person_coverage", "Easy new people"), (width / 2, "hard_person_coverage", "Hard new people")):
            values, lower, upper = [], [], []
            for method in METHODS:
                series = frame.loc[frame["method"] == method, column]
                values.append(series.mean())
                lower.append(series.mean() - series.quantile(0.05))
                upper.append(series.quantile(0.95) - series.mean())
            axis.bar(
                positions + offset,
                values,
                width,
                yerr=np.vstack([lower, upper]),
                capsize=3,
                color="#5ba870" if column.startswith("easy") else "#b85c5c",
                alpha=0.9,
                label=label,
            )
        axis.axhline(1 - ALPHA, color="#991b1b", linestyle="--", linewidth=1)
        settings = REGIMES[regime]
        subtitle = "Equal 200-window calibration people" if regime == "equal_windows" else "Easy: 400 windows; hard: 40 windows"
        axis.set(
            title=subtitle,
            xticks=positions,
            xticklabels=[LABELS[method] for method in METHODS],
            xlabel="Calibration rule",
        )
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Coverage for a uniformly chosen new person")
    axes[0].set_ylim(0, 1.03)
    axes[1].legend(loc="lower right")
    figure.suptitle("Pooled calibration targets a size-weighted population", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Bars are mean coverage over 1,000 simulated calibration samples; whiskers are 5th–95th percentiles. Dashed line: 90% target.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main(output_dir: Path, repetitions: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    draws, summary = run_simulation(repetitions)
    summary.to_csv(output_dir / "simulation_summary.csv", index=False)
    make_figure(draws, output_dir / "simulation_mechanism.png")
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate pooled and participant-balanced conformal calibration under hierarchical size bias.")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    arguments = parser.parse_args()
    main(arguments.output, arguments.repetitions)
