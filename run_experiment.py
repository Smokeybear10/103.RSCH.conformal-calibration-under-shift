from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
SOURCE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
SOURCE_BYTES = 61005872
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
FAMILY_COLORS = {"Logistic": "#2563a6", "Forest": "#bf6330"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "source.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        with urlopen(DATA_URL, timeout=60) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        temporary.replace(archive)
    if archive.stat().st_size != SOURCE_BYTES:
        raise ValueError("The UCI archive has an unexpected byte count.")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("The UCI archive differs from the verified source hash.")
    return archive


def load_data(data_dir: Path) -> dict:
    archive = fetch_data(data_dir)
    expected = (
        "features.txt",
        "activity_labels.txt",
        "train/X_train.txt",
        "train/y_train.txt",
        "train/subject_train.txt",
        "test/X_test.txt",
        "test/y_test.txt",
        "test/subject_test.txt",
    )
    root = data_dir / "UCI HAR Dataset"
    input_hashes = {}
    with zipfile.ZipFile(archive) as outer:
        nested_name = next(name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) as nested:
            for relative in expected:
                payload = nested.read(f"UCI HAR Dataset/{relative}")
                destination = root / relative
                digest = hashlib.sha256(payload).hexdigest()
                if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Cached source file differs from the verified archive: {destination}")
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                input_hashes[relative] = digest

    data = {}
    for split, expected_rows in (("train", 7352), ("test", 2947)):
        data[f"X_{split}"] = np.loadtxt(root / split / f"X_{split}.txt")
        data[f"y_{split}"] = np.loadtxt(root / split / f"y_{split}.txt", dtype=int) - 1
        data[f"subject_{split}"] = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        if data[f"X_{split}"].shape != (expected_rows, 561):
            raise ValueError(f"Unexpected UCI feature shape for {split}.")
        if set(data[f"y_{split}"]) != set(range(6)):
            raise ValueError(f"Unexpected activity labels in {split}.")

    data["labels"] = [line.split(maxsplit=1)[1] for line in (root / "activity_labels.txt").read_text().splitlines()]
    data["input_hashes"] = input_hashes
    return data


def validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != 6:
        raise ValueError("Expected a six-class probability matrix.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model produced invalid probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Model probabilities do not sum to one.")


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


def select_models(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[dict[str, object], pd.DataFrame, str]:
    candidates = candidate_models(seed)
    rows = []
    for name, candidate in candidates.items():
        if name == "Forest":
            continue
        for inner_fold, (fit, validation) in enumerate(GroupKFold(n_splits=3).split(X, y, subjects), start=1):
            if set(subjects[fit]) & set(subjects[validation]):
                raise AssertionError("A participant leaked across an inner split.")
            fitted = clone(candidate).fit(X[fit], y[fit])
            probabilities = fitted.predict_proba(X[validation])
            validate_probabilities(probabilities)
            rows.append(
                {
                    "candidate": name,
                    "inner_fold": inner_fold,
                    "n": len(validation),
                    "nll": log_loss(y[validation], probabilities, labels=np.arange(6)),
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


def make_splits(subjects: np.ndarray) -> list[dict]:
    ids = np.unique(subjects)
    if len(ids) != 30:
        raise ValueError("The study expects exactly 30 UCI participants.")
    splits = []
    for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
        outer_folds = np.array_split(np.random.default_rng(seed).permutation(ids), 5)
        for fold, test in enumerate(outer_folds, start=1):
            remaining = np.setdiff1d(ids, test)
            remaining = np.random.default_rng(seed + 1000 * fold).permutation(remaining)
            roles = {"fit": remaining[:12], "calibration": remaining[12:], "test": test}
            all_ids = np.concatenate(list(roles.values()))
            if len(all_ids) != len(np.unique(all_ids)) or set(all_ids) != set(ids):
                raise AssertionError("Outer split does not partition participants.")
            splits.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "fold": fold,
                    "subjects": {role: values.tolist() for role, values in roles.items()},
                }
            )
    return splits


def helper_tests() -> None:
    conformal._self_test()
    scores = np.linspace(0.1, 0.9, 12)
    groups = np.repeat(np.arange(12), 1)
    if not np.isfinite(conformal.hierarchical_threshold(scores, groups, 0.10)):
        raise AssertionError("Twelve calibration participants should make a 90% HCP threshold finite.")
    if not np.isinf(conformal.hierarchical_threshold(scores[:8], groups[:8], 0.10)):
        raise AssertionError("Eight calibration participants cannot make a 90% HCP threshold finite.")
    duplicated_scores = np.r_[scores[:4], np.repeat(scores[4:8], 3), scores[8:]]
    duplicated_groups = np.r_[groups[:4], np.repeat(groups[4:8], 3), groups[8:]]
    original = conformal.hierarchical_threshold(scores, groups, 0.20)
    duplicate = conformal.hierarchical_threshold(duplicated_scores, duplicated_groups, 0.20)
    if original != duplicate:
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
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
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
                axis.annotate(str(int(row["repeat"])), (100 * row["review_rate_equal_subject"], 100 * row["coverage_p10_subject"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.axhline(90, color="#991b1b", linestyle="--", linewidth=1)
        axis.set(title=family, xlabel="Review rate across held-out people (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("10th-percentile held-out-person coverage (%)\n(diagnostic, not a per-person guarantee)")
    axes[1].legend(loc="lower left", fontsize=8)
    figure.suptitle("A pooled coverage claim can hide low coverage for a new person", fontsize=14)
    figure.text(0.5, 0.01, "Each dot is one full five-fold subject partition. The dashed line is a 90% diagnostic benchmark, not a per-person guarantee.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    figure.savefig(output_dir / "coverage_vs_review.png", bbox_inches="tight")
    plt.close(figure)

    primary_subjects = subject_metrics[subject_metrics["alpha"] == 0.10]
    figure, axes = plt.subplots(2, 1, figsize=(14.5, 5.8), sharex=True, sharey=True, layout="constrained")
    for axis, family in zip(axes, ("Logistic", "Forest")):
        frame = (
            primary_subjects[primary_subjects["family"] == family]
            .groupby(["method", "subject"], as_index=False)["coverage"]
            .mean()
        )
        methods = list(METHODS)
        subjects = sorted(frame["subject"].unique())
        matrix = frame.pivot(index="method", columns="subject", values="coverage").reindex(index=methods, columns=subjects)
        image = axis.imshow(matrix.to_numpy(), vmin=0.65, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axis.set_yticks(np.arange(len(methods)), [METHOD_LABELS[method] for method in methods])
        axis.set(title=f"{family}: coverage by held-out participant", ylabel="Calibration rule")
        for row, method in enumerate(methods):
            for column, subject in enumerate(subjects):
                value = matrix.loc[method, subject]
                axis.text(column, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=7)
    axes[-1].set_xticks(np.arange(len(subjects)), subjects)
    axes[-1].set_xlabel("Held-out participant")
    figure.colorbar(image, ax=axes, label="Coverage", shrink=0.86, pad=0.02)
    figure.savefig(output_dir / "subject_coverage.png", bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, repeat_summary: pd.DataFrame, started: str, elapsed_seconds: float) -> None:
    primary = repeat_summary[repeat_summary["alpha"] == 0.10].copy()
    raps_findings = {}
    for family in ("Logistic", "Forest"):
        values = primary[(primary["family"] == family) & primary["method"].isin(["window_raps", "participant_raps"])]
        raps_findings[family] = values.groupby("method")[
            ["coverage_p10_subject", "review_rate_equal_subject", "fraction_people_below_target"]
        ].mean()
    logistic_window, logistic_participant = raps_findings["Logistic"].loc[["window_raps", "participant_raps"]].to_dict("records")
    forest_window, forest_participant = raps_findings["Forest"].loc[["window_raps", "participant_raps"]].to_dict("records")
    observed_tradeoff = (
        f"For Logistic RAPS, participant balancing raised average 10th-percentile participant coverage from "
        f"{100 * logistic_window['coverage_p10_subject']:.1f}% to {100 * logistic_participant['coverage_p10_subject']:.1f}% "
        f"and reduced the mean share of people below 90% from {100 * logistic_window['fraction_people_below_target']:.1f}% "
        f"to {100 * logistic_participant['fraction_people_below_target']:.1f}%, but increased review from "
        f"{100 * logistic_window['review_rate_equal_subject']:.1f}% to {100 * logistic_participant['review_rate_equal_subject']:.1f}%. "
        f"For Forest RAPS, the corresponding 10th-percentile coverage moved from "
        f"{100 * forest_window['coverage_p10_subject']:.1f}% to {100 * forest_participant['coverage_p10_subject']:.1f}%, "
        f"with review moving from {100 * forest_window['review_rate_equal_subject']:.1f}% to "
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
    display["method"] = display["method"].map(METHOD_LABELS)
    for column in ("coverage_equal_subject", "coverage_p10_subject", "fraction_people_below_target", "review_rate_equal_subject", "singleton_error_rate"):
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
    table = display.to_html(index=False, border=0, classes="summary", na_rep="—")
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Who gets the guarantee?</title>
  <style>
    body {{ max-width: 1040px; margin: 48px auto; padding: 0 24px 48px; color: #172033; font: 16px/1.55 -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }}
    h1 {{ font-size: 2.45rem; letter-spacing: -0.045em; margin-bottom: 0; }}
    h2 {{ margin-top: 2.4rem; }}
    .dek {{ color: #526174; font-size: 1.15rem; max-width: 760px; }}
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
  <p class=\"dek\">Conformal prediction under unseen-person shift in smartphone activity recognition.</p>
  <p>This experiment asks whether a nominal coverage guarantee remains useful when the thing that changes at deployment is an entire person, not a random sensor window.</p>
  <div class=\"note\"><strong>Decision rule:</strong> a singleton label is automatically accepted. A set with two or more activities is escalated for review.</div>
  <h2>Design</h2>
  <p>Across three repeated five-fold participant partitions, each fold fits models on 12 people, calibrates on 12 different people, and evaluates six unseen people. Window APS/RAPS gives every calibration window equal mass. Participant APS/RAPS gives every calibration participant equal mass and reserves finite-sample mass for the unseen participant. To avoid empty outputs, each set always includes the most likely label if a threshold would otherwise exclude it; this can only enlarge a conformal set.</p>
  <p>Prediction sets use established adaptive conformal scores rather than a proposed new algorithm. The hierarchical rule requires hierarchical exchangeability for its formal guarantee; UCI HAR has overlapping windows and only 30 participants, so these results are an empirical stress test, not a deployment claim.</p>
  <h2>90% nominal-coverage summary</h2>
  {table}
  <h2>Coverage versus review burden</h2>
  <img src=\"coverage_vs_review.png\" alt=\"Review rate versus tenth-percentile person coverage diagnostic for four conformal methods and two models\">
  <h2>Who is left behind?</h2>
  <img src=\"subject_coverage.png\" alt=\"Participant-level coverage heatmaps for logistic regression and random forest\">
  <h2>Interpretation</h2>
  <p>{observed_tradeoff}</p>
  <p>Population-average coverage is not enough if a system acts on individual people. The useful comparison is whether participant-balanced calibration improves low-coverage people without making too many predictions ambiguous. These repeated splits show that tradeoff, not a new universal guarantee.</p>
  <p>Methods: <a href=\"https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf\">Adaptive Prediction Sets</a>; <a href=\"https://arxiv.org/abs/2306.06342\">distribution-free inference with hierarchical data</a>. UCI HAR source: <a href=\"https://doi.org/10.24432/C54S4K\">Reyes-Ortiz et al.</a>.</p>
  <p style=\"color:#667085;font-size:.85rem\">Executed {started}; {elapsed_seconds:.1f}s. See CSV tables and <code>run_experiment.py</code> for every calculation.</p>
</body>
</html>"""
    (output_dir / "report.html").write_text(html)


def run_experiment(data_dir: Path, output_dir: Path) -> dict:
    helper_tests()
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    X = np.concatenate([data["X_train"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])
    subjects = np.concatenate([data["subject_train"], data["subject_test"]])
    splits = make_splits(subjects)
    protocol = {
        "question": "Does a nominal conformal coverage guarantee protect a completely unseen participant?",
        "dataset": "UCI Human Activity Recognition Using Smartphones; six activities, 30 participants, 10,299 engineered sensor windows.",
        "outer_protocol": "Three seeds × five participant folds. Each fold: 12 fit people, 12 conformal-calibration people, six unseen test people.",
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
            "The data contain only 30 people from one acquisition study.",
            "UCI HAR windows overlap by 50%, so independent-window assumptions do not literally hold.",
            "Participant-balanced calibration targets a random person/window under hierarchical exchangeability, not a per-person guarantee.",
            "No human review accuracy, cost, or downstream action is observed.",
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
            print(f"repeat {repeat}/3, fold {fold}/5", flush=True)
            masks = {role: np.isin(subjects, ids) for role, ids in split["subjects"].items()}
            for role, mask in masks.items():
                if set(y[mask]) != set(range(6)):
                    raise AssertionError(f"{role} partition lost an activity class.")

            candidates, cv_scores, selected_logistic = select_models(X[masks["fit"]], y[masks["fit"]], subjects[masks["fit"]], seed)
            cv_frames.append(cv_scores.assign(repeat=repeat, fold=fold))
            calibration_subjects = subjects[masks["calibration"]]
            test_subjects = subjects[masks["test"]]
            test_rows = np.flatnonzero(masks["test"])

            for family, candidate in candidates.items():
                fitted = clone(candidate).fit(X[masks["fit"]], y[masks["fit"]])
                calibration_probabilities = fitted.predict_proba(X[masks["calibration"]])
                test_probabilities = fitted.predict_proba(X[masks["test"]])
                validate_probabilities(calibration_probabilities)
                validate_probabilities(test_probabilities)

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
                        expected_covered = (test_scores[np.arange(len(test_subjects)), y[masks["test"]]] <= threshold) | top_class[np.arange(len(test_subjects)), y[masks["test"]]]
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
                        }
                        fold_rows.append(identifiers | metrics)
                        for row in conformal.group_set_metrics(prediction_set, y[masks["test"]], test_subjects):
                            subject_rows.append(identifiers | {"subject": int(row.pop("group"))} | row)

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
    write_report(output_dir, repeat_summary, started, elapsed_seconds)
    summary = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "source_sha256": SOURCE_SHA256,
        "input_hashes": data["input_hashes"],
        "rows": {name: len(table) for name, table in tables.items()} | {"predictions": len(predictions)},
        "primary_summary": json.loads(repeat_summary[repeat_summary["alpha"] == 0.10].to_json(orient="records")),
    }
    write_json(output_dir / "summary.json", summary)
    print(repeat_summary[repeat_summary["alpha"] == 0.10].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pooled and participant-balanced conformal prediction under unseen-person shift.")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    arguments = parser.parse_args()
    run_experiment(arguments.data_dir, arguments.output)
