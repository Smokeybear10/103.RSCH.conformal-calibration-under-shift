"""External unseen-participant replication on WISDM phone accelerometer windows.

This intentionally mirrors the pooled-versus-participant-balanced APS/RAPS
comparison in ``run_experiment.py`` while using a separate acquisition study.
It consumes only WISDM's supplied transformed ARFF feature windows; raw sensor
streams are never extracted or written to the results directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import conformal_core as conformal


HERE = Path(__file__).resolve().parent
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset.zip"
)
SOURCE_BYTES = 310_292_805
SOURCE_SHA256 = "6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469"
INNER_ARCHIVE = "wisdm-dataset.zip"
INNER_BYTES = 309_713_877
INNER_SHA256 = "1cf32e8d3a0ecc101bd299fba22545273dcb7de1c9565e0b7844abdfbc6b1695"
ARFF_PATTERN = re.compile(r"wisdm-dataset/arff_files/phone/accel/data_(\d+)_accel_phone\.arff$")
REPEAT_SEEDS = (42, 71, 103)
ALPHAS = (0.10, 0.20)
METHODS = {
    "window_aps": {"calibration": "window", "regularization": 0.0, "k_reg": 0},
    "window_raps": {"calibration": "window", "regularization": 0.01, "k_reg": 1},
    "participant_aps": {"calibration": "participant", "regularization": 0.0, "k_reg": 0},
    "participant_raps": {"calibration": "participant", "regularization": 0.01, "k_reg": 1},
}
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


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str, description: str) -> None:
    if not path.exists():
        raise ValueError(f"Missing {description}: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{description} has an unexpected byte count.")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{description} differs from the verified official archive hash.")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{description} is not a valid ZIP archive.")


def fetch_data(data_dir: Path) -> tuple[Path, Path]:
    """Download and cryptographically verify the official outer and inner archives."""
    data_dir.mkdir(parents=True, exist_ok=True)
    outer_path = data_dir / "source.zip"
    if not outer_path.exists():
        temporary = outer_path.with_suffix(".zip.part")
        request = Request(DATA_URL, headers={"User-Agent": "person-shift-conformal-replication/1.0"})
        with urlopen(request, timeout=90) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        temporary.replace(outer_path)
    verify_file(outer_path, SOURCE_BYTES, SOURCE_SHA256, "WISDM outer archive")

    inner_path = data_dir / INNER_ARCHIVE
    if not inner_path.exists() or sha256_file(inner_path) != INNER_SHA256:
        temporary = inner_path.with_suffix(".zip.part")
        with zipfile.ZipFile(outer_path) as outer:
            if INNER_ARCHIVE not in outer.namelist():
                raise ValueError(f"Verified WISDM outer archive does not contain {INNER_ARCHIVE}.")
            with outer.open(INNER_ARCHIVE) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        temporary.replace(inner_path)
    verify_file(inner_path, INNER_BYTES, INNER_SHA256, "WISDM nested dataset archive")
    return outer_path, inner_path


def decode_nominal(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def clean_field_name(name: str) -> str:
    return name.strip().strip('"').strip("'")


def parse_arff(payload: bytes, filename_subject: str) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Parse and validate one supplied transformed ARFF file without extracting it."""
    try:
        table, _ = arff.loadarff(StringIO(payload.decode("utf-8")))
    except Exception as error:  # scipy exposes several parser exception types.
        raise ValueError(f"Could not parse {filename_subject}'s transformed ARFF file.") from error

    raw_names = list(table.dtype.names or ())
    names = [clean_field_name(name) for name in raw_names]
    if len(raw_names) < 4 or names[0] != "ACTIVITY" or names[-1].lower() != "class":
        raise ValueError(
            f"Unexpected WISDM ARFF schema for subject {filename_subject}: expected ACTIVITY first and class last."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"WISDM ARFF for subject {filename_subject} contains duplicate field names.")
    feature_names = names[1:-1]
    if not feature_names:
        raise ValueError(f"WISDM ARFF for subject {filename_subject} has no numeric feature columns.")
    try:
        features = np.column_stack([np.asarray(table[name], dtype=float) for name in raw_names[1:-1]])
    except (TypeError, ValueError) as error:
        raise ValueError(f"WISDM ARFF for subject {filename_subject} has a nonnumeric feature column.") from error
    if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
        raise ValueError(f"WISDM ARFF for subject {filename_subject} contains empty or non-finite feature rows.")

    activities = np.asarray([decode_nominal(value) for value in table[raw_names[0]]], dtype=str)
    declared_subjects = {decode_nominal(value) for value in table[raw_names[-1]]}
    if len(activities) != len(features) or not np.all(activities != ""):
        raise ValueError(f"WISDM ARFF for subject {filename_subject} has malformed activity labels.")
    if len(declared_subjects) != 1:
        raise ValueError(f"WISDM ARFF for subject {filename_subject} has more than one declared person id.")
    return features, activities, feature_names, next(iter(declared_subjects))


def load_data(data_dir: Path) -> dict:
    """Load unique phone-accelerometer windows and defensively validate the schema."""
    outer_path, inner_path = fetch_data(data_dir)
    with zipfile.ZipFile(inner_path) as archive:
        entries = []
        for name in archive.namelist():
            match = ARFF_PATTERN.fullmatch(name)
            if match:
                payload = archive.read(name)
                entries.append(
                    {
                        "path": name,
                        "filename_subject": match.group(1),
                        "payload": payload,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )

    if len(entries) < 10:
        raise ValueError("Expected at least ten transformed phone-accelerometer ARFF files in WISDM.")

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_hash[entry["sha256"]].append(entry)

    retained: list[dict] = []
    duplicate_aliases: list[dict] = []
    for digest, group in sorted(by_hash.items()):
        representative = group[0]
        _, _, _, declared_subject = parse_arff(representative["payload"], representative["filename_subject"])
        names = sorted(entry["filename_subject"] for entry in group)
        if len(group) == 1:
            if declared_subject != names[0]:
                raise ValueError(
                    "A nonduplicated transformed ARFF file has a filename/person-id disagreement: "
                    f"filename={names[0]}, declared={declared_subject}."
                )
            retained.append(representative)
            continue
        if declared_subject not in names:
            raise ValueError(
                "Byte-identical transformed ARFF files do not contain a canonical declared person id: "
                f"filenames={names}, declared={declared_subject}."
            )
        canonical = next(entry for entry in group if entry["filename_subject"] == declared_subject)
        retained.append(canonical)
        duplicate_aliases.append(
            {
                "canonical_subject": declared_subject,
                "dropped_filename_subjects": [name for name in names if name != declared_subject],
                "sha256": digest,
            }
        )

    retained.sort(key=lambda entry: int(entry["filename_subject"]))
    matrices: list[np.ndarray] = []
    activities: list[np.ndarray] = []
    people: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    feature_names: list[str] | None = None
    for entry in retained:
        X_person, activity_person, columns, declared_subject = parse_arff(
            entry["payload"], entry["filename_subject"]
        )
        if declared_subject != entry["filename_subject"]:
            raise AssertionError("Only canonical filename/person-id matches should remain after duplicate handling.")
        if feature_names is None:
            feature_names = columns
        elif columns != feature_names:
            raise ValueError("Transformed phone-accelerometer ARFF files do not share one feature schema.")
        matrices.append(X_person)
        activities.append(activity_person)
        people.append(np.repeat(declared_subject, len(X_person)))
        hashes[declared_subject] = entry["sha256"]

    if feature_names is None:
        raise AssertionError("No WISDM ARFF schema was loaded.")
    X = np.concatenate(matrices)
    activity_names = np.concatenate(activities)
    subjects = np.concatenate(people)
    labels = sorted(np.unique(activity_names).tolist())
    if len(labels) < 2:
        raise ValueError("WISDM transformed windows do not contain at least two activity classes.")
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([label_index[label] for label in activity_names], dtype=int)

    per_subject_labels = {
        subject: sorted(np.unique(activity_names[subjects == subject]).tolist()) for subject in np.unique(subjects)
    }
    activity_subject_counts = {
        label: int(sum(label in values for values in per_subject_labels.values())) for label in labels
    }
    missing_by_subject = {
        subject: [label for label in labels if label not in values]
        for subject, values in per_subject_labels.items()
        if len(values) != len(labels)
    }
    if any(count == 0 for count in activity_subject_counts.values()):
        missing = [label for label, count in activity_subject_counts.items() if count == 0]
        raise ValueError(f"WISDM activity labels have no observed participant support: {missing}.")

    return {
        "X": X,
        "y": y,
        "subjects": subjects,
        "labels": labels,
        "feature_names": feature_names,
        "source_sha256": sha256_file(outer_path),
        "nested_source_sha256": sha256_file(inner_path),
        "input_hashes": hashes,
        "n_source_arff_files": len(entries),
        "duplicate_aliases": duplicate_aliases,
        "activity_subject_counts": activity_subject_counts,
        "missing_activities_by_subject": missing_by_subject,
    }


def validate_probabilities(probabilities: np.ndarray, n_classes: int) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != n_classes:
        raise ValueError(f"Expected a {n_classes}-class probability matrix.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model produced invalid probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Model probabilities do not sum to one.")


def model_probabilities(fitted: object, X: np.ndarray, n_classes: int) -> np.ndarray:
    classes = np.asarray(getattr(fitted, "classes_", ()))
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(
            "A fitted model did not see every activity class in its fit partition; "
            "the selected participant split is invalid."
        )
    probabilities = fitted.predict_proba(X)
    validate_probabilities(probabilities, n_classes)
    return probabilities


def candidate_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        f"Logistic C={value:g}": make_pipeline(
            StandardScaler(), LogisticRegression(C=value, max_iter=3000, random_state=seed)
        )
        for value in (0.1, 1.0, 10.0)
    }
    models["Forest"] = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=2,
    )
    return models


def select_models(
    X: np.ndarray, y: np.ndarray, subjects: np.ndarray, seed: int, n_classes: int
) -> tuple[dict[str, object], pd.DataFrame, str]:
    candidates = candidate_models(seed)
    rows = []
    if len(np.unique(subjects)) < 3:
        raise ValueError("At least three fit participants are required for inner GroupKFold model selection.")
    for name, candidate in candidates.items():
        if name == "Forest":
            continue
        for inner_fold, (fit, validation) in enumerate(GroupKFold(n_splits=3).split(X, y, subjects), start=1):
            if set(subjects[fit]) & set(subjects[validation]):
                raise AssertionError("A participant leaked across an inner split.")
            if set(y[fit]) != set(range(n_classes)):
                raise ValueError("An inner fit split lost an activity class; cannot compare class probabilities fairly.")
            fitted = clone(candidate).fit(X[fit], y[fit])
            probabilities = model_probabilities(fitted, X[validation], n_classes)
            rows.append(
                {
                    "candidate": name,
                    "inner_fold": inner_fold,
                    "n": len(validation),
                    "nll": log_loss(y[validation], probabilities, labels=np.arange(n_classes)),
                    "accuracy": float((probabilities.argmax(axis=1) == y[validation]).mean()),
                }
            )
    scores = pd.DataFrame(rows)
    logistic_name = (
        scores[scores["candidate"].str.startswith("Logistic")]
        .groupby("candidate", sort=False)["nll"]
        .mean()
        .sort_values(kind="stable")
        .index[0]
    )
    return {"Logistic": candidates[logistic_name], "Forest": candidates["Forest"]}, scores, logistic_name


def role_has_all_activities(
    subjects: np.ndarray, y: np.ndarray, roles: dict[str, np.ndarray], n_classes: int
) -> tuple[bool, dict[str, list[int]]]:
    missing: dict[str, list[int]] = {}
    for role, ids in roles.items():
        labels = set(y[np.isin(subjects, ids)])
        absent = sorted(set(range(n_classes)) - labels)
        if absent:
            missing[role] = absent
    return not missing, missing


def make_splits(subjects: np.ndarray, y: np.ndarray, labels: list[str]) -> list[dict]:
    ids = np.unique(subjects)
    n_classes = len(labels)
    if len(ids) < 12:
        raise ValueError("At least twelve unique WISDM participants are required for four-fold unseen-person evaluation.")
    splits = []
    for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
        outer_folds = np.array_split(np.random.default_rng(seed).permutation(ids), 4)
        for fold, test in enumerate(outer_folds, start=1):
            remaining = np.setdiff1d(ids, test)
            chosen_roles: dict[str, np.ndarray] | None = None
            missing_detail: dict[str, list[int]] = {}
            for attempt in range(100):
                permuted = np.random.default_rng(seed + 1000 * fold + attempt).permutation(remaining)
                fit_count = len(permuted) // 2
                candidate = {"fit": permuted[:fit_count], "calibration": permuted[fit_count:], "test": test}
                valid, missing_detail = role_has_all_activities(subjects, y, candidate, n_classes)
                if valid:
                    chosen_roles = candidate
                    break
            if chosen_roles is None:
                readable = {role: [labels[index] for index in absent] for role, absent in missing_detail.items()}
                raise ValueError(
                    "WISDM's incomplete per-person activity coverage prevented an all-class participant split "
                    f"for repeat={repeat}, fold={fold}; last missing labels={readable}."
                )
            all_ids = np.concatenate(list(chosen_roles.values()))
            if len(all_ids) != len(np.unique(all_ids)) or set(all_ids) != set(ids):
                raise AssertionError("Outer split does not partition retained WISDM participants.")
            splits.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "subjects": {role: values.tolist() for role, values in chosen_roles.items()},
                }
            )
    return splits


def helper_tests() -> None:
    conformal._self_test()
    scores = np.linspace(0.1, 0.9, 12)
    groups = np.arange(12)
    if not np.isfinite(conformal.hierarchical_threshold(scores, groups, 0.10)):
        raise AssertionError("Twelve calibration participants should make a 90% HCP threshold finite.")
    if not np.isinf(conformal.hierarchical_threshold(scores[:8], groups[:8], 0.10)):
        raise AssertionError("Eight calibration participants cannot make a 90% HCP threshold finite.")
    duplicated_scores = np.r_[scores[:4], np.repeat(scores[4:8], 3), scores[8:]]
    duplicated_groups = np.r_[groups[:4], np.repeat(groups[4:8], 3), groups[8:]]
    if conformal.hierarchical_threshold(scores, groups, 0.20) != conformal.hierarchical_threshold(
        duplicated_scores, duplicated_groups, 0.20
    ):
        raise AssertionError("Hierarchical calibration changed when a person's score rows were duplicated.")


def aggregate_repeats(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["repeat", "family", "method", "alpha"]
    for keys, frame in subject_metrics.groupby(columns, sort=True):
        total_n = int(frame["n"].sum())
        total_singleton = int(frame["n_singleton"].sum())
        row = dict(zip(columns, keys))
        row.update(
            {
                "n_subjects": int(frame["subject"].nunique()),
                "n_windows": total_n,
                "coverage_equal_subject": float(frame["coverage"].mean()),
                "coverage_pooled": float(np.average(frame["coverage"], weights=frame["n"])),
                "coverage_p10_subject": float(frame["coverage"].quantile(0.10)),
                "coverage_min_subject": float(frame["coverage"].min()),
                "people_below_target": int((frame["coverage"] < 1 - keys[-1]).sum()),
                "fraction_people_below_target": float((frame["coverage"] < 1 - keys[-1]).mean()),
                "mean_set_size_equal_subject": float(frame["mean_set_size"].mean()),
                "mean_set_size_pooled": float(np.average(frame["mean_set_size"], weights=frame["n"])),
                "singleton_rate_equal_subject": float(frame["singleton_rate"].mean()),
                "review_rate_equal_subject": float(frame["review_rate"].mean()),
                "singleton_error_rate": (
                    int(frame["n_singleton_errors"].sum()) / total_singleton if total_singleton else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(output_dir: Path, repeat_summary: pd.DataFrame, subject_metrics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {"figure.dpi": 120, "savefig.dpi": 180, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    primary = repeat_summary[repeat_summary["alpha"] == 0.10]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for axis, family in zip(axes, ("Logistic", "Forest")):
        for method, frame in primary[primary["family"] == family].groupby("method", sort=False):
            axis.scatter(
                100 * frame["review_rate_equal_subject"],
                100 * frame["coverage_p10_subject"],
                color=METHOD_COLORS[method],
                s=72,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            for _, row in frame.iterrows():
                axis.annotate(
                    str(int(row["repeat"])),
                    (100 * row["review_rate_equal_subject"], 100 * row["coverage_p10_subject"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.axhline(90, color="#991b1b", linestyle="--", linewidth=1)
        axis.set(title=family, xlabel="Review rate across held-out people (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("10th-percentile held-out-person coverage (%)\n(diagnostic, not a per-person guarantee)")
    axes[1].legend(loc="lower left", fontsize=8)
    figure.suptitle("A pooled coverage claim can hide low coverage for a new person", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Each dot is one full four-fold subject partition. The dashed line is a 90% diagnostic benchmark, not a per-person guarantee.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output_dir / "coverage_vs_review.png", bbox_inches="tight")
    plt.close(figure)

    primary_subjects = subject_metrics[subject_metrics["alpha"] == 0.10]
    figure, axes = plt.subplots(2, 1, figsize=(16.5, 5.8), sharex=True, sharey=True, layout="constrained")
    for axis, family in zip(axes, ("Logistic", "Forest")):
        frame = (
            primary_subjects[primary_subjects["family"] == family]
            .groupby(["method", "subject"], as_index=False)["coverage"]
            .mean()
        )
        methods = list(METHODS)
        subjects = sorted(frame["subject"].unique(), key=int)
        matrix = frame.pivot(index="method", columns="subject", values="coverage").reindex(index=methods, columns=subjects)
        image = axis.imshow(matrix.to_numpy(), vmin=0.65, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axis.set_yticks(np.arange(len(methods)), [METHOD_LABELS[method] for method in methods])
        axis.set(title=f"{family}: coverage by held-out participant", ylabel="Calibration rule")
        for row, method in enumerate(methods):
            for column, subject in enumerate(subjects):
                axis.text(column, row, f"{100 * matrix.loc[method, subject]:.0f}", ha="center", va="center", fontsize=5.6)
    axes[-1].set_xticks(np.arange(len(subjects)), subjects, fontsize=6)
    axes[-1].set_xlabel("Retained transformed-window participant")
    figure.colorbar(image, ax=axes, label="Coverage", shrink=0.86, pad=0.02)
    figure.savefig(output_dir / "subject_coverage.png", bbox_inches="tight")
    plt.close(figure)


def write_report(
    output_dir: Path,
    repeat_summary: pd.DataFrame,
    started: str,
    elapsed_seconds: float,
    data: dict,
) -> None:
    primary = repeat_summary[repeat_summary["alpha"] == 0.10].copy()
    raps_findings = {}
    for family in ("Logistic", "Forest"):
        values = primary[(primary["family"] == family) & primary["method"].isin(["window_raps", "participant_raps"])]
        raps_findings[family] = values.groupby("method")[
            ["coverage_p10_subject", "review_rate_equal_subject", "fraction_people_below_target"]
        ].mean()
    logistic_window, logistic_participant = raps_findings["Logistic"].loc[
        ["window_raps", "participant_raps"]
    ].to_dict("records")
    forest_window, forest_participant = raps_findings["Forest"].loc[
        ["window_raps", "participant_raps"]
    ].to_dict("records")
    observed_tradeoff = (
        f"For Logistic RAPS, participant balancing moved average 10th-percentile participant coverage from "
        f"{100 * logistic_window['coverage_p10_subject']:.1f}% to {100 * logistic_participant['coverage_p10_subject']:.1f}% "
        f"and review from {100 * logistic_window['review_rate_equal_subject']:.1f}% to "
        f"{100 * logistic_participant['review_rate_equal_subject']:.1f}%. For Forest RAPS, 10th-percentile "
        f"coverage moved from {100 * forest_window['coverage_p10_subject']:.1f}% to "
        f"{100 * forest_participant['coverage_p10_subject']:.1f}%, with review moving from "
        f"{100 * forest_window['review_rate_equal_subject']:.1f}% to "
        f"{100 * forest_participant['review_rate_equal_subject']:.1f}%."
    )
    display = primary.groupby(["family", "method"], as_index=False).agg(
        coverage_equal_subject=("coverage_equal_subject", "mean"),
        coverage_p10_subject=("coverage_p10_subject", "mean"),
        fraction_people_below_target=("fraction_people_below_target", "mean"),
        mean_set_size_equal_subject=("mean_set_size_equal_subject", "mean"),
        review_rate_equal_subject=("review_rate_equal_subject", "mean"),
        singleton_error_rate=("singleton_error_rate", "mean"),
    )
    review_low = 100 * float(display["review_rate_equal_subject"].min())
    review_high = 100 * float(display["review_rate_equal_subject"].max())
    set_low = float(display["mean_set_size_equal_subject"].min())
    set_high = float(display["mean_set_size_equal_subject"].max())
    display["method"] = display["method"].map(METHOD_LABELS)
    for column in (
        "coverage_equal_subject",
        "coverage_p10_subject",
        "fraction_people_below_target",
        "review_rate_equal_subject",
        "singleton_error_rate",
    ):
        display[column] = 100 * display[column]
    display = display.rename(
        columns={
            "family": "Model",
            "method": "Method",
            "coverage_equal_subject": "Mean person coverage (%)",
            "coverage_p10_subject": "10th-person coverage (%)",
            "fraction_people_below_target": "People below 90% (%)",
            "mean_set_size_equal_subject": "Mean set size",
            "review_rate_equal_subject": "Review rate (%)",
            "singleton_error_rate": "Singleton error (%)",
        }
    ).round(2)
    duplicate_note = (
        "The official transformed phone-accelerometer folder has "
        f"{data['n_source_arff_files']} files, but {sum(len(item['dropped_filename_subjects']) for item in data['duplicate_aliases'])} "
        "filename aliases were byte-identical copies of one file. They were removed before splitting, leaving "
        f"{len(np.unique(data['subjects']))} deduplicated file-level subjects."
    )
    operational_note = (
        f"With 18 activities, every condition here reviews {review_low:.1f}%–{review_high:.1f}% of cases "
        f"and emits mean sets of {set_low:.1f}–{set_high:.1f} labels. This is an external stress test of the "
        "coverage tradeoff, not a usable automatic-decision policy."
    )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Who gets the guarantee? WISDM replication</title>
  <style>
    body {{ max-width: 1040px; margin: 48px auto; padding: 0 24px 48px; color: #172033; font: 16px/1.55 -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    h1 {{ font-size: 2.45rem; letter-spacing: -0.045em; margin-bottom: 0; }}
    h2 {{ margin-top: 2.4rem; }}
    .dek {{ color: #526174; font-size: 1.15rem; max-width: 800px; }}
    .note {{ background: #f3f6fa; border-left: 3px solid #4278a5; padding: 12px 16px; }}
    img {{ width: 100%; border: 1px solid #e0e6ee; margin: 12px 0 4px; }}
    .summary {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
    .summary th, .summary td {{ border-bottom: 1px solid #dbe3ec; padding: 8px; text-align: right; }}
    .summary th:nth-child(-n+2), .summary td:nth-child(-n+2) {{ text-align: left; }}
    .summary th {{ background: #f3f6fa; }}
    a {{ color: #1d5b89; }}
  </style>
</head>
<body>
  <h1>Who gets the guarantee?</h1>
  <p class=\"dek\">External WISDM replication of conformal prediction under unseen-person shift.</p>
  <p>This is a separate activity-recognition study with supplied 10-second transformed phone-accelerometer windows. It tests the same calibration comparison on a different device, feature pipeline, participant set, and 18-activity taxonomy.</p>
  <div class=\"note\"><strong>Decision rule:</strong> a singleton label is automatically accepted. A set with two or more activities is escalated for review.</div>
  <h2>Data integrity and design</h2>
  <p>{duplicate_note} The retained data have {len(data['labels'])} activities and {len(data['feature_names'])} engineered features. Every role in every split is checked to contain every activity; if incomplete activity coverage made that impossible, the run would fail rather than silently drop a class.</p>
  <p>Across three repeated four-fold participant partitions, each fold fits models on roughly half of the retained people, calibrates on the other half, and evaluates entirely unseen people. Window APS/RAPS gives every calibration window equal mass. Participant APS/RAPS gives every calibration participant equal mass and reserves finite-sample mass for the unseen participant. To avoid empty outputs, each set always includes the most likely label if a threshold would otherwise exclude it; this can only enlarge a conformal set.</p>
  <p>Prediction sets use established adaptive conformal scores rather than a proposed new algorithm. The participant-balanced rule requires hierarchical exchangeability for its formal guarantee; these repeated held-out-person results are an empirical stress test, not a per-person deployment guarantee.</p>
  <h2>90% nominal-coverage summary</h2>
  {display.to_html(index=False, border=0, classes='summary', na_rep='—')}
  <h2>Coverage versus review burden</h2>
  <img src=\"coverage_vs_review.png\" alt=\"Review rate versus tenth-percentile person coverage for four conformal methods and two models\">
  <h2>Who is left behind?</h2>
  <img src=\"subject_coverage.png\" alt=\"Participant-level coverage heatmaps for logistic regression and random forest\">
  <h2>Interpretation</h2>
  <p>{observed_tradeoff}</p>
  <p class="note"><strong>Practical limit:</strong> {operational_note}</p>
  <p>Methods: <a href=\"https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf\">Adaptive Prediction Sets</a>; <a href=\"https://arxiv.org/abs/2306.06342\">distribution-free inference with hierarchical data</a>. WISDM source: <a href=\"https://archive.ics.uci.edu/dataset/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset\">UCI dataset 507</a>.</p>
  <p style=\"color:#667085;font-size:.85rem\">Executed {started}; {elapsed_seconds:.1f}s. The results directory contains only derived metrics, predictions, figures, and hashes—not raw WISDM sensor data.</p>
</body>
</html>"""
    (output_dir / "report.html").write_text(html)


def run_experiment(data_dir: Path, output_dir: Path) -> dict:
    helper_tests()
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    labels = data["labels"]
    n_classes = len(labels)
    splits = make_splits(subjects, y, labels)
    protocol = {
        "question": "Does nominal conformal coverage protect a completely unseen participant on an external activity-recognition study?",
        "dataset": "UCI WISDM Smartphone and Smartwatch Activity and Biometrics Dataset; supplied transformed phone-accelerometer windows only.",
        "dataset_source": DATA_URL,
        "schema": {
            "n_features": len(data["feature_names"]),
            "n_activities": n_classes,
            "activity_labels": labels,
            "source_transformed_arff_files": data["n_source_arff_files"],
            "retained_independent_file_participants": int(len(np.unique(subjects))),
            "activity_subject_counts": data["activity_subject_counts"],
            "missing_activities_by_subject": data["missing_activities_by_subject"],
            "duplicate_file_aliases_removed": data["duplicate_aliases"],
        },
        "integrity": {
            "outer_archive_sha256": data["source_sha256"],
            "nested_archive_sha256": data["nested_source_sha256"],
            "transformed_arff_sha256_by_retained_subject": data["input_hashes"],
            "raw_data_policy": "Raw sensor streams are never extracted or written to wisdm_results; only supplied transformed ARFF features are parsed in memory.",
        },
        "outer_protocol": "Three seeds × four participant folds. With 44 retained people, each fold has 16 model-fit people, 17 conformal-calibration people, and 11 unseen test people. Every role must contain every activity class.",
        "models": "Logistic C selected by three-fold GroupKFold NLL among C=[0.1, 1, 10]. Forest fixed at 250 trees, min_samples_leaf=2, max_features=sqrt.",
        "methods": {
            "window_aps": "Standard split-conformal adaptive score; every calibration window has equal mass.",
            "window_raps": "Window APS plus fixed rank penalty lambda=0.01 after top rank.",
            "participant_aps": "Hierarchical participant-balanced APS. Every calibration participant contributes equal mass; a pseudo-score at infinity reserves mass for a new person.",
            "participant_raps": "Participant-balanced RAPS with the same prespecified lambda=0.01 and k_reg=1.",
        },
        "primary_alpha": 0.10,
        "alphas": list(ALPHAS),
        "review_policy": "A singleton prediction set is auto-accepted. Any larger set is reviewed.",
        "nonempty_set_convention": "If a deterministic APS/RAPS threshold would return an empty set, include every tied top-probability label. This can only enlarge a set and increase coverage.",
        "limitations": [
            "The WISDM transformed phone-accelerometer folder has incomplete activity coverage for a small number of file-level participants; split construction validates all activity classes rather than dropping them.",
            "The official transformed folder includes byte-identical filename aliases, which are removed before participant splitting to avoid train/test duplication.",
            "Participant-balanced calibration targets a random person/window under hierarchical exchangeability, not a per-person guarantee.",
            "These are supplied engineered windows, not raw time-series models; no human review accuracy, cost, or downstream action is observed.",
        ],
        "splits": splits,
    }
    write_json(output_dir / "protocol.json", protocol)

    fold_rows: list[dict] = []
    subject_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []

    with threadpool_limits(limits=2):
        for split in splits:
            repeat, fold, seed = split["repeat"], split["fold"], split["seed"]
            print(f"repeat {repeat}/{len(REPEAT_SEEDS)}, fold {fold}/4", flush=True)
            masks = {role: np.isin(subjects, ids) for role, ids in split["subjects"].items()}
            for role, mask in masks.items():
                observed = set(y[mask])
                if observed != set(range(n_classes)):
                    missing = [labels[index] for index in sorted(set(range(n_classes)) - observed)]
                    raise AssertionError(f"{role} partition lost activity classes: {missing}.")

            candidates, cv_scores, selected_logistic = select_models(
                X[masks["fit"]], y[masks["fit"]], subjects[masks["fit"]], seed, n_classes
            )
            cv_frames.append(cv_scores.assign(repeat=repeat, fold=fold))
            calibration_subjects = subjects[masks["calibration"]]
            test_subjects = subjects[masks["test"]]
            test_rows = np.flatnonzero(masks["test"])

            for family, candidate in candidates.items():
                fitted = clone(candidate).fit(X[masks["fit"]], y[masks["fit"]])
                calibration_probabilities = model_probabilities(fitted, X[masks["calibration"]], n_classes)
                test_probabilities = model_probabilities(fitted, X[masks["test"]], n_classes)

                for method, settings in METHODS.items():
                    calibration_scores = conformal.true_label_scores(
                        calibration_probabilities,
                        y[masks["calibration"]],
                        regularization=settings["regularization"],
                        k_reg=settings["k_reg"],
                    )
                    test_scores = conformal.adaptive_scores(
                        test_probabilities,
                        regularization=settings["regularization"],
                        k_reg=settings["k_reg"],
                    )
                    top_class = test_probabilities == test_probabilities.max(axis=1, keepdims=True)
                    for alpha in ALPHAS:
                        if settings["calibration"] == "window":
                            threshold = conformal.pooled_threshold(calibration_scores, alpha)
                        else:
                            threshold = conformal.hierarchical_threshold(calibration_scores, calibration_subjects, alpha)
                        prediction_set = conformal.prediction_sets(
                            test_probabilities,
                            threshold,
                            regularization=settings["regularization"],
                            k_reg=settings["k_reg"],
                        )
                        covered = prediction_set[np.arange(len(test_subjects)), y[masks["test"]]]
                        expected_covered = (
                            test_scores[np.arange(len(test_subjects)), y[masks["test"]]] <= threshold
                        ) | top_class[np.arange(len(test_subjects)), y[masks["test"]]]
                        if not np.array_equal(covered, expected_covered):
                            raise AssertionError("Prediction-set coverage no longer matches its score rule.")
                        metrics = conformal.set_metrics(prediction_set, y[masks["test"]])
                        identifiers = {
                            "repeat": repeat,
                            "fold": fold,
                            "seed": seed,
                            "family": family,
                            "method": method,
                            "alpha": alpha,
                            "threshold": threshold,
                            "selected_logistic": selected_logistic,
                            "n_calibration_people": int(len(np.unique(calibration_subjects))),
                            "n_classes": n_classes,
                        }
                        fold_rows.append(identifiers | metrics)
                        for row in conformal.group_set_metrics(prediction_set, y[masks["test"]], test_subjects):
                            subject_rows.append(identifiers | {"subject": str(row.pop("group"))} | row)

                        sizes = prediction_set.sum(axis=1)
                        prediction_frames.append(
                            pd.DataFrame(
                                {
                                    "row_id": test_rows,
                                    "subject": test_subjects,
                                    "y": y[masks["test"]],
                                    "predicted": test_probabilities.argmax(axis=1),
                                    "covered": covered,
                                    "set_size": sizes,
                                    "singleton": sizes == 1,
                                    "review": sizes > 1,
                                }
                            ).assign(**identifiers)
                        )

    fold_metrics = pd.DataFrame(fold_rows)
    subject_metrics = pd.DataFrame(subject_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    cv_scores = pd.concat(cv_frames, ignore_index=True)
    expected_predictions = len(X) * len(REPEAT_SEEDS) * len(METHODS) * len(ALPHAS) * 2
    if len(predictions) != expected_predictions:
        raise AssertionError("Unexpected prediction count.")
    if predictions.duplicated(["repeat", "family", "method", "alpha", "row_id"]).any():
        raise AssertionError("A test window appeared twice in the same experimental condition.")
    if subject_metrics.groupby(["repeat", "subject"])["fold"].nunique().max() != 1:
        raise AssertionError("A participant was held out more than once in a repeat.")

    repeat_summary = aggregate_repeats(subject_metrics)
    tables = {
        "fold_metrics.csv": fold_metrics,
        "subject_metrics.csv": subject_metrics,
        "repeat_summary.csv": repeat_summary,
        "cv_scores.csv": cv_scores,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    make_figures(output_dir, repeat_summary, subject_metrics)
    elapsed_seconds = time.monotonic() - started_clock
    write_report(output_dir, repeat_summary, started, elapsed_seconds, data)
    summary = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "source_sha256": data["source_sha256"],
        "nested_source_sha256": data["nested_source_sha256"],
        "input_hashes": data["input_hashes"],
        "n_retained_participants": int(len(np.unique(subjects))),
        "n_classes": n_classes,
        "rows": {name: len(table) for name, table in tables.items()} | {"predictions": len(predictions)},
        "primary_summary": json.loads(repeat_summary[repeat_summary["alpha"] == 0.10].to_json(orient="records")),
    }
    write_json(output_dir / "summary.json", summary)
    print(repeat_summary[repeat_summary["alpha"] == 0.10].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Externally replicate pooled and participant-balanced conformal prediction on WISDM."
    )
    parser.add_argument("--data-dir", type=Path, default=HERE / "wisdm_data")
    parser.add_argument("--output", type=Path, default=HERE / "wisdm_results")
    arguments = parser.parse_args()
    run_experiment(arguments.data_dir, arguments.output)
