"""Check reporting arithmetic and consistency of committed derived tables.

These checks do not rerun model fitting or validate the source datasets.
"""

from pathlib import Path
import unittest

import pandas as pd

import run_experiment as uci
import wisdm_replication as wisdm


ROOT = Path(__file__).resolve().parents[1]
CONDITION = ["repeat", "family", "method", "alpha"]


class ReportingTests(unittest.TestCase):
    def test_equal_person_and_pooled_coverage_use_different_weights(self):
        common = {"repeat": 1, "family": "Logistic", "method": "window_aps", "alpha": 0.1}
        people = pd.DataFrame([
            common | {
                "subject": 1, "n": 1, "coverage": 0.0, "mean_set_size": 1.0,
                "singleton_rate": 1.0, "review_rate": 0.0,
                "n_singleton": 1, "n_singleton_errors": 1,
            },
            common | {
                "subject": 2, "n": 9, "coverage": 1.0, "mean_set_size": 2.0,
                "singleton_rate": 0.0, "review_rate": 1.0,
                "n_singleton": 0, "n_singleton_errors": 0,
            },
        ])
        for experiment in (uci, wisdm):
            with self.subTest(experiment=experiment.__name__):
                row = experiment.aggregate_repeats(people).iloc[0]
                self.assertEqual(row["n_subjects"], 2)
                self.assertEqual(row["n_windows"], 10)
                self.assertEqual(row["coverage_equal_subject"], 0.5)
                self.assertEqual(row["coverage_pooled"], 0.9)
                self.assertAlmostEqual(row["coverage_p10_subject"], 0.1)
                self.assertEqual(row["people_below_target"], 1)
                self.assertEqual(row["mean_set_size_equal_subject"], 1.5)
                self.assertEqual(row["mean_set_size_pooled"], 1.9)
                self.assertEqual(row["singleton_error_rate"], 1.0)

    def test_saved_summaries_can_be_rebuilt_from_saved_participant_metrics(self):
        for experiment, directory, people_count in ((uci, "results", 30), (wisdm, "wisdm_results", 44)):
            with self.subTest(experiment=experiment.__name__):
                people = pd.read_csv(ROOT / directory / "subject_metrics.csv")
                saved = pd.read_csv(ROOT / directory / "repeat_summary.csv")
                self.assertFalse(people.duplicated(CONDITION + ["subject"]).any())
                self.assertTrue((people.groupby(CONDITION)["subject"].nunique() == people_count).all())
                self.assertTrue((people.groupby(["repeat", "subject"])["fold"].nunique() == 1).all())
                self.assertEqual(len(saved), 48)  # 3 repeats x 2 models x 4 methods x 2 alpha values.
                rebuilt = experiment.aggregate_repeats(people)
                pd.testing.assert_frame_equal(
                    rebuilt.set_index(CONDITION).sort_index(),
                    saved.set_index(CONDITION).sort_index(),
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
