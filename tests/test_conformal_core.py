"""Offline numerical checks, not evidence of population coverage guarantees."""

from fractions import Fraction
import unittest

import numpy as np

import conformal_core as conformal


class ThresholdTests(unittest.TestCase):
    def test_pooled_quantile_uses_an_observed_order_statistic(self):
        # ceil((4 + 1) * (1 - 1/2)) = 3: no interpolation between scores.
        scores = np.array([0.9, 0.1, 0.6, 0.2])
        self.assertEqual(conformal.pooled_threshold(scores, Fraction(1, 2)), 0.6)

    def test_pooled_finite_sample_boundary_includes_infinity_mass(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        for alpha in (0.2, Fraction(1, 5)):
            with self.subTest(alpha=alpha):
                self.assertEqual(conformal.pooled_threshold(scores, alpha), 0.4)
        self.assertTrue(np.isinf(conformal.pooled_threshold(scores, Fraction(199, 1000))))

    def test_group_balancing_does_not_give_large_participants_more_mass(self):
        # Three low scores belong to one person, one high score to another.
        # Their total masses are equal; 50% coverage needs the second person
        # once the additional unseen-person mass is included.
        scores = np.array([0.1, 0.2, 0.4, 0.8])
        groups = np.array(["a", "a", "a", "b"])
        self.assertEqual(conformal.pooled_threshold(scores, 0.5), 0.4)
        self.assertEqual(conformal.hierarchical_threshold(scores, groups, 0.5), 0.8)

    def test_hierarchical_boundary_depends_on_people_not_window_count(self):
        scores = np.arange(1, 10, dtype=float) / 10
        self.assertEqual(conformal.hierarchical_threshold(scores, np.arange(9), 0.1), 0.9)
        eight_people = np.repeat(np.arange(8), 20)
        many_windows = np.tile(np.linspace(0.1, 0.8, 20), 8)
        self.assertTrue(np.isinf(conformal.hierarchical_threshold(many_windows, eight_people, 0.1)))

    def test_repeating_all_windows_of_one_person_preserves_group_quantile(self):
        scores = np.array([0.2, 0.8, 0.5, 0.9, 0.7])
        groups = np.array(["a", "a", "b", "b", "c"])
        # In units of participant mass: 0.2 -> .5, 0.5 -> 1,
        # 0.7 -> 2, 0.8 -> 2.5. The required mass is 4 * .6 = 2.4.
        self.assertEqual(conformal.hierarchical_threshold(scores, groups, 0.4), 0.8)
        repeats = np.array([7, 7, 1, 1, 1])
        self.assertEqual(
            conformal.hierarchical_threshold(np.repeat(scores, repeats), np.repeat(groups, repeats), 0.4),
            0.8,
        )

    def test_hierarchical_quantile_is_invariant_to_row_order_and_group_names(self):
        scores = np.array([0.2, 0.8, 0.5, 0.9, 0.7])
        groups = np.array([100, 100, -5, -5, 88])
        order = np.array([4, 1, 3, 0, 2])
        self.assertEqual(conformal.hierarchical_threshold(scores[order], groups[order], 0.4), 0.8)

    def test_invalid_calibration_inputs_are_rejected(self):
        for scores in (np.array([]), np.array([np.nan]), np.array([np.inf]), np.ones((2, 2))):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    conformal.pooled_threshold(scores, 0.2)
                with self.assertRaises(ValueError):
                    conformal.hierarchical_threshold(scores, np.array([]), 0.2)
        for alpha in (0, 1, -0.1, 1.1):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    conformal.pooled_threshold(np.array([0.2]), alpha)
                with self.assertRaises(ValueError):
                    conformal.hierarchical_threshold(np.array([0.2]), np.array([1]), alpha)
        with self.assertRaises(ValueError):
            conformal.hierarchical_threshold(np.array([0.2, 0.4]), np.array([1]), 0.5)


class PredictionSetTests(unittest.TestCase):
    def test_tied_scores_use_the_end_of_the_tied_rank_block(self):
        probabilities = np.array([[0.375, 0.25, 0.375], [0.5, 0.25, 0.25]])
        np.testing.assert_array_equal(
            conformal.adaptive_scores(probabilities),
            [[0.75, 1.0, 0.75], [0.5, 1.0, 1.0]],
        )
        np.testing.assert_array_equal(
            conformal.adaptive_scores(probabilities, regularization=0.125, k_reg=1),
            [[0.875, 1.25, 0.875], [0.5, 1.25, 1.25]],
        )

    def test_true_label_scores_keep_original_class_indices(self):
        probabilities = np.array([[0.25, 0.5, 0.25], [0.375, 0.25, 0.375]])
        np.testing.assert_array_equal(
            conformal.true_label_scores(probabilities, np.array([1, 1])),
            [0.5, 1.0],
        )

    def test_nonempty_convention_retains_every_tied_top_class(self):
        probabilities = np.array([[0.375, 0.25, 0.375], [0.5, 0.25, 0.25]])
        np.testing.assert_array_equal(
            conformal.prediction_sets(probabilities, 0),
            [[True, False, True], [True, False, False]],
        )

    def test_threshold_equality_includes_the_entire_tie(self):
        probabilities = np.array([[0.5, 0.25, 0.25]])
        np.testing.assert_array_equal(conformal.prediction_sets(probabilities, 0.875), [[True, False, False]])
        np.testing.assert_array_equal(conformal.prediction_sets(probabilities, 1.0), [[True, True, True]])

    def test_reordering_class_columns_does_not_change_set_membership(self):
        probabilities = np.array([[0.375, 0.25, 0.375], [0.5, 0.25, 0.25]])
        permutation = np.array([2, 0, 1])
        expected = np.array([[True, False, True], [True, False, False]])
        np.testing.assert_array_equal(
            conformal.prediction_sets(probabilities[:, permutation], 0.75),
            expected[:, permutation],
        )

    def test_increasing_threshold_never_removes_a_class(self):
        probabilities = np.array([[0.125, 0.25, 0.625], [0.375, 0.25, 0.375]])
        previous = conformal.prediction_sets(probabilities, 0)
        for threshold in (0.5, 0.75, 0.875, 1.0, np.inf):
            current = conformal.prediction_sets(probabilities, threshold)
            self.assertTrue(np.all(~previous | current))
            previous = current
        self.assertTrue(previous.all())

    def test_invalid_probabilities_and_thresholds_are_rejected(self):
        for probabilities in (
            np.array([[0.2, 0.2]]),
            np.array([[-0.1, 1.1]]),
            np.array([[np.nan, 1.0]]),
            np.array([[1.0]]),
            np.empty((0, 2)),
        ):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    conformal.prediction_sets(probabilities, 1.0)
        for threshold in (-0.1, -np.inf, np.nan):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    conformal.prediction_sets(np.array([[0.5, 0.5]]), threshold)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.sets = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool)
        self.labels = np.array([0, 0, 1, 0])

    def test_coverage_and_singleton_error_have_different_denominators(self):
        metrics = conformal.set_metrics(self.sets, self.labels)
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["mean_set_size"], 1.5)
        self.assertEqual(metrics["singleton_rate"], 0.5)
        self.assertEqual(metrics["review_rate"], 0.5)
        self.assertEqual(metrics["empty_rate"], 0.0)
        self.assertEqual(metrics["n_singleton"], 2)
        self.assertEqual(metrics["n_singleton_errors"], 1)
        self.assertEqual(metrics["singleton_error_rate"], 0.5)

    def test_no_singletons_reports_undefined_error_rate(self):
        metrics = conformal.set_metrics(self.sets[2:], self.labels[2:])
        self.assertEqual(metrics["n_singleton"], 0)
        self.assertEqual(metrics["review_rate"], 1.0)
        self.assertIsNone(metrics["singleton_error_rate"])

    def test_group_metrics_use_only_each_participants_rows(self):
        rows = conformal.group_set_metrics(self.sets, self.labels, np.array(["a", "a", "a", "b"]))
        by_group = {row["group"]: row for row in rows}
        self.assertEqual(by_group["a"]["n"], 3)
        self.assertAlmostEqual(by_group["a"]["coverage"], 2 / 3)
        self.assertEqual(by_group["a"]["singleton_error_rate"], 0.5)
        self.assertEqual(by_group["b"]["n"], 1)
        self.assertEqual(by_group["b"]["coverage"], 0.0)
        self.assertIsNone(by_group["b"]["singleton_error_rate"])


if __name__ == "__main__":
    unittest.main()
