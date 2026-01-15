# Conformal Calibration Under New-Participant Shift

A notebook and set of experiments comparing pooled window-level conformal calibration with participant-balanced calibration when evaluation is on people the model has not seen during fitting or calibration.

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

The feature-shift plot is only a correlation check. The simulation isolates why pooling windows can overweight people who contribute more data.

## WISDM Check

The same setup is run on WISDM's transformed phone-accelerometer windows. The official folder contained six files that were byte-for-byte copies of subject 1610's file, so the code drops those aliases before splitting. The run uses 44 file-level subjects, 18 activities, and 20,650 windows.

WISDM is much harder operationally: the sets are wide and almost every example is reviewed. It is useful as a second dataset check, not as a proposed automatic-decision policy.

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
