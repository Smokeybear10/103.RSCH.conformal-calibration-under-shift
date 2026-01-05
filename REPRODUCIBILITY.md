# Reproduction check: 2026-11-05

An automated rerun in a fresh Python environment preserved the main findings, but some reported values changed. This check was performed with AI assistance; it is not an independent human replication.

The rerun used the experiment scripts from commit [`d8a1899e8269f51719303411c202240debadb391`](https://github.com/Smokeybear10/conformal-calibration-under-shift/commit/d8a1899e8269f51719303411c202240debadb391), Python 3.13.5 on macOS arm64, and the pinned `requirements.txt`. Both full experiments ran into separate output directories. The original reference files in `results/` and `wisdm_results/` were preserved.

For both UCI HAR and WISDM, the rerun matched the complete saved protocol, including participant splits; source-archive and input-file hashes; output row counts; and selected logistic hyperparameters. Summary comparisons aligned rows by repeat, model, method, and alpha. CV comparisons also aligned candidate and inner-fold identifiers.

The UCI HAR headline comparison below averages the three repeats at alpha = 0.10. Coverage is the 10th percentile of participant coverage; review rate gives participants equal weight.

| Model / RAPS calibration | Original coverage | Rerun coverage | Original review | Rerun review |
| --- | ---: | ---: | ---: | ---: |
| Logistic / window | 85.8% | 85.8% | 6.7% | 6.8% |
| Logistic / participant-balanced | 92.6% | 92.8% | 44.2% | 44.2% |
| Forest / window | 91.6% | 91.6% | 45.0% | 45.0% |
| Forest / participant-balanced | 95.3% | 95.3% | 68.1% | 68.1% |

Across all repeat/model/method/alpha conditions, the maximum absolute changes were:

| Metric | UCI HAR | WISDM |
| --- | ---: | ---: |
| 10th-percentile participant coverage | 0.528486 percentage points | 0.215163 percentage points |
| Equal-participant review rate | 0.499959 percentage points | 0.014029 percentage points |

Most variation occurred in logistic models. Forest 10th-percentile coverage was unchanged in both datasets, although some other forest metrics varied. WISDM still produced very wide sets and nearly universal review under the singleton acceptance rule.

The differences have not been traced to a specific cause. Matching data and splits does not establish identical numerical execution. The original saved summaries did not capture a complete environment or executed code revision, so this check cannot isolate environment sensitivity from other execution differences. It does not establish a per-participant coverage guarantee.

The [rerun environment](reproduction/2026-11-05-environment.json) and [measured comparisons](reproduction/2026-11-05-comparison.json) record the available evidence. The README table continues to report the original reference run.
