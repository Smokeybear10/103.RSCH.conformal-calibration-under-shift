# Conformal Calibration | Unseen Participants

Compares two ways of calibrating conformal prediction for people held out from training and calibration.

## Quick Start

Use Python 3.11 or 3.13. The [notebook](unseen_person_conformal.ipynb) also has saved outputs you can read without installing anything.

```bash
git clone https://github.com/Smokeybear10/conformal-calibration-under-shift.git
cd conformal-calibration-under-shift
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab unseen_person_conformal.ipynb
```

## What It Does

Fits logistic regression and random forest on UCI HAR[^1] and WISDM[^2], then builds APS/RAPS prediction sets[^3] at a 90% nominal coverage target. Window calibration weights each window equally. Participant-balanced calibration gives each person equal total weight. A one-label set is accepted; larger sets go to review.

The notebook includes both datasets, a feature-shift diagnostic, and a simulation of unequal window counts per person.

## Results

On UCI HAR, participant-balanced RAPS improved lower-tail coverage at the cost of more review.

| Model | Calibration | 10th-percentile coverage | Review rate |
| --- | --- | ---: | ---: |
| Logistic | Window | 85.8% | 6.7% |
| Logistic | Participant-balanced | 92.6% | 44.2% |
| Forest | Window | 91.6% | 45.0% |
| Forest | Participant-balanced | 95.3% | 68.1% |

Coverage is the 10th percentile of participant coverage rates. Both columns are averaged over three repeated participant partitions; review rates weight participants equally.

These are the original saved results. A fresh-environment rerun preserved the tradeoff but changed numerical values, including two rounded table entries. The cause remains unresolved; the [reproduction record](REPRODUCIBILITY.md) documents the differences.

![Coverage versus review burden on UCI HAR](results/coverage_vs_review.png)

WISDM uses 44 file-level subjects, 18 activities, and 20,650 windows after removing six byte-identical subject-file aliases. Sets were wide and nearly every prediction went to review, making the singleton rule impractical there.

## Reproduce the Results

Run from the repository root. New outputs go to `work/`:

```bash
python run_experiment.py --output work/uci
python shift_diagnostic.py --results-dir work/uci
python hierarchical_simulation.py --output work/uci
python wisdm_replication.py --output work/wisdm
```

The first run downloads about 61 MB for UCI HAR and 310 MB for WISDM, verifies archive hashes, and caches the data locally. Compare each new `repeat_summary.csv` with the saved [UCI HAR](results/repeat_summary.csv) and [WISDM](wisdm_results/repeat_summary.csv) tables. Full splits are in the [UCI HAR](results/protocol.json) and [WISDM](wisdm_results/protocol.json) protocols.

Checks run without dataset downloads; they do not reproduce the empirical results:

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

## Limits

Participants stay separate across fitting, calibration, and testing. Model selection uses fit participants only. Participant-balanced coverage relies on hierarchical exchangeability[^4] and does not guarantee 90% coverage for every person. UCI HAR windows overlap, both datasets have within-person dependence, and human review accuracy and cost were not measured. The feature-shift diagnostic is correlational; the size-imbalance simulation is illustrative.

[^1]: [UCI Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K)
[^2]: [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://doi.org/10.24432/C5HK59)
[^3]: [Adaptive Prediction Sets](https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)
[^4]: [Distribution-Free Inference With Hierarchical Data](https://arxiv.org/abs/2306.06342)

Code and documentation: [MIT License](LICENSE). Datasets retain their own terms.

---

Built by Thomas Ou
