# Contributing

This study compares window-weighted and participant-balanced conformal calibration on people excluded from fitting and calibration. Contributions that make the experiment easier to reproduce or clarify the coverage/review tradeoff are useful.

## Start locally

Follow the [README setup](README.md#quick-start), then run:

```bash
python -m unittest discover -s tests -v
python smoke_test.py
```

Both commands run without dataset downloads. The smoke check fits models to synthetic participants; passing it does not reproduce the published results. The test suite also checks that the committed participant-level tables agree with the committed repeat summaries.

For a full replication, use the [reproduction commands](README.md#reproduce-the-results). Write new outputs to a separate directory so you can compare them with the committed tables.

## Ways to help

- **Independent reproduction:** run the study in a clean environment and report the commit, Python version, operating system, commands, and differences from the saved tables. A discrepancy report is useful even without a code change.
- **Statistical checks:** contribute a small, independently worked example for a threshold, tie, or participant-weighting edge case. Explain what should happen and why.
- **Research extensions:** discuss the question and evaluation protocol in an issue before adding a dataset, model, or calibration rule. Include the expected runtime and data source.

See the [open issues](https://github.com/Smokeybear10/conformal-calibration-under-shift/issues) for specific tasks. Check for an existing issue or pull request before starting the same work.

## Preserve the experimental boundaries

Fitting, model selection, calibration, and testing have distinct roles. Keep each participant entirely within one outer role. Fit preprocessing and select hyperparameters using fit participants only. Specify tuning and comparison rules before inspecting new test results. Do not select a method or threshold using the held-out test outcomes.

Report coverage together with review rate and set size. Participant-balanced calibration is not a guarantee for every participant. The data have within-person dependence, and the review policy has no measured human-review accuracy or cost.

If a code change affects a scientific result, show the old and new protocol and explain which outputs need regeneration. Keep methodological changes separate from formatting or documentation edits. Do not commit raw data archives, prediction dumps, local environments, or access credentials.

## Pull requests and reviews

Keep each pull request focused on one problem. Describe the observed behavior, the change, and the checks you ran. Include a failing example for a bug fix when possible. Tests should exercise an independently justified property or expected value.

For a review, say which commit and commands you tested. Distinguish an observed failure from a question or a possible concern. Reproduction reports, documentation corrections, and careful reviews are contributions too.

If AI tools helped, describe their role where it affects how the work should be reviewed. Verify the code and claims yourself; do not describe an automated run as an independent human replication.
