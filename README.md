# Conformal Calibration Under New-Participant Shift

I compared two ways of calibrating conformal prediction when the test person was never seen during training or calibration.

## Start Here

Open [unseen_person_conformal.ipynb](unseen_person_conformal.ipynb). It has the UCI HAR results, a feature-shift check, a small size-imbalance simulation, and the external WISDM run.

## What It Compares

- **Window calibration:** every calibration window gets equal weight.
- **Participant-balanced calibration:** every calibration participant gets equal total weight.
- **Decision rule:** a one-label set is accepted; a larger set is reviewed.

The models are logistic regression and random forest. Prediction sets use APS/RAPS scores with a 90% nominal coverage target.

## UCI HAR Result

Participant-balanced calibration improved the lower tail of held-out-person coverage, but it also raised the review rate.

| Model | RAPS rule | 10th-person coverage | Review rate |
| --- | --- | ---: | ---: |
| Logistic | Window | 85.8% | 6.7% |
| Logistic | Participant-balanced | 92.6% | 44.2% |
| Forest | Window | 91.6% | 45.0% |
| Forest | Participant-balanced | 95.3% | 68.1% |

The feature-shift plot is only a correlation check. The toy simulation shows what can happen when a few people contribute most of the windows.

## WISDM Check

The same setup is run on WISDM's transformed phone-accelerometer windows. The official folder contained six files that were byte-for-byte copies of subject 1610's file, so the code drops those aliases before splitting. The run uses 44 file-level subjects, 18 activities, and 20,650 windows.

I included WISDM as a second check. The review policy itself is not very practical there because the 18-class sets are wide.

## Run It

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

To regenerate the saved outputs:

```bash
python run_experiment.py
python shift_diagnostic.py
python hierarchical_simulation.py
python wisdm_replication.py
```

The WISDM download is roughly 296 MB and its full run takes a few minutes. The repository includes derived tables and figures, not raw source archives or prediction dumps.

## Data and Methods

- [UCI Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K)
- [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset](https://doi.org/10.24432/C5HK59)
- [Adaptive Prediction Sets](https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)
- [Distribution-Free Inference With Hierarchical Data](https://arxiv.org/abs/2306.06342)
