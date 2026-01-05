"""Exercise production split builders with synthetic participants, without downloads."""

from collections import Counter, defaultdict
import unittest

import numpy as np

import run_experiment as uci
import wisdm_replication as wisdm


class ParticipantSplitTests(unittest.TestCase):
    def assert_partitions(self, splits, subjects, role_sizes, folds_per_repeat):
        ids = set(np.unique(subjects))
        test_counts = defaultdict(Counter)
        fold_counts = Counter()
        for split in splits:
            with self.subTest(repeat=split["repeat"], fold=split["fold"]):
                roles = split["subjects"]
                self.assertEqual(set(roles), {"fit", "calibration", "test"})
                assigned = [subject for role in roles.values() for subject in role]
                self.assertEqual(len(assigned), len(ids))
                self.assertEqual(set(assigned), ids)
                self.assertEqual(len(assigned), len(set(assigned)))
                for role, size in role_sizes.items():
                    self.assertEqual(len(roles[role]), size)
                # Every window, including repeated windows from one person,
                # belongs to exactly one role in an outer fold.
                masks = [np.isin(subjects, role) for role in roles.values()]
                np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(subjects), dtype=int))
                test_counts[split["repeat"]].update(roles["test"])
                fold_counts[split["repeat"]] += 1
        self.assertEqual(len(test_counts), 3)
        for repeat, counts in test_counts.items():
            with self.subTest(repeat=repeat):
                self.assertEqual(fold_counts[repeat], folds_per_repeat)
                self.assertEqual(counts, Counter({subject: 1 for subject in ids}))

    def test_uci_participants_are_disjoint_and_tested_once_per_repeat(self):
        subjects = np.repeat(np.arange(101, 131), np.arange(1, 31))
        splits = uci.make_splits(subjects)
        self.assertEqual(len(splits), 15)
        self.assert_partitions(splits, subjects, {"fit": 12, "calibration": 12, "test": 6}, 5)

    def test_uci_assignments_do_not_depend_on_window_order_or_count(self):
        ids = np.arange(101, 131)
        expanded = np.repeat(ids, np.arange(1, 31))[::-1]
        self.assertEqual(uci.make_splits(ids), uci.make_splits(expanded))
        self.assertEqual(uci.make_splits(ids), uci.make_splits(ids))

    def test_uci_rejects_wrong_participant_count(self):
        for count in (0, 29, 31):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "exactly 30"):
                    uci.make_splits(np.arange(count))

    def test_wisdm_partitions_preserve_all_people_and_activity_classes(self):
        ids = np.array([f"subject-{number}" for number in range(1600, 1644)])
        subjects = np.repeat(ids, 3)
        y = np.tile(np.arange(3), len(ids))
        labels = ["walking", "standing", "jogging"]
        splits = wisdm.make_splits(subjects, y, labels)
        self.assertEqual(len(splits), 12)
        self.assert_partitions(splits, subjects, {"fit": 16, "calibration": 17, "test": 11}, 4)
        for split in splits:
            for role, participants in split["subjects"].items():
                with self.subTest(repeat=split["repeat"], fold=split["fold"], role=role):
                    self.assertEqual(set(y[np.isin(subjects, participants)]), {0, 1, 2})

    def test_wisdm_assignments_are_reproducible_after_window_reordering(self):
        subjects = np.repeat(np.arange(44), 3)
        y = np.tile(np.arange(3), 44)
        labels = ["walking", "standing", "jogging"]
        expected = wisdm.make_splits(subjects, y, labels)
        order = np.random.default_rng(2026).permutation(len(subjects))
        self.assertEqual(expected, wisdm.make_splits(subjects[order], y[order], labels))
        self.assertEqual(expected, wisdm.make_splits(np.repeat(subjects, 2), np.repeat(y, 2), labels))

    def test_wisdm_rejects_insufficient_people(self):
        with self.assertRaisesRegex(ValueError, "At least twelve"):
            wisdm.make_splits(np.arange(11), np.zeros(11, dtype=int), ["walking"])

    def test_wisdm_fails_explicitly_when_an_activity_cannot_appear_in_all_roles(self):
        # No participant has the second declared activity. A valid split is
        # impossible, so the study must not silently drop that class.
        with self.assertRaisesRegex(ValueError, "all-class participant split"):
            wisdm.make_splits(np.arange(12), np.zeros(12, dtype=int), ["walking", "standing"])

    def test_missing_activity_diagnostics_identify_the_affected_roles(self):
        subjects = np.array([1, 1, 2, 3, 3])
        y = np.array([0, 1, 0, 1, 2])
        roles = {"fit": np.array([1]), "calibration": np.array([2]), "test": np.array([3])}
        valid, missing = wisdm.role_has_all_activities(subjects, y, roles, 3)
        self.assertFalse(valid)
        self.assertEqual(missing, {"fit": [2], "calibration": [1, 2], "test": [0]})


if __name__ == "__main__":
    unittest.main()
