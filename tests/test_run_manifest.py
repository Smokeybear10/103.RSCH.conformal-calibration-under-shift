"""Check provenance accuracy, privacy, and optional-metadata failure handling."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import run_experiment
import run_manifest
import wisdm_replication


def git(repository, *arguments):
    return subprocess.run(
        ["git", "-c", "user.name=Provenance Test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=repository, check=True, capture_output=True, text=True,
    ).stdout.strip()


def initialize_repository(repository):
    git(repository, "init", "-q")
    (repository / ".gitignore").write_text("ignored/\n")
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-qm", "Fixture")
    return git(repository, "rev-parse", "HEAD")


class RunManifestTests(unittest.TestCase):
    def test_git_state_tracks_changes_but_not_ignored_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            commit = initialize_repository(repository)
            self.assertEqual(run_manifest.capture_git_state(repository), {"commit": commit, "dirty": False})
            (repository / "ignored").mkdir()
            (repository / "ignored" / "result.txt").write_text("generated")
            self.assertFalse(run_manifest.capture_git_state(repository)["dirty"])
            untracked = repository / "new_code.py"
            untracked.write_text("# new code\n")
            self.assertTrue(run_manifest.capture_git_state(repository)["dirty"])
            untracked.unlink()
            (repository / ".gitignore").write_text("ignored/\nmore/\n")
            self.assertTrue(run_manifest.capture_git_state(repository)["dirty"])

    def test_source_archive_does_not_claim_a_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.assertEqual(run_manifest.capture_git_state(repository), {"commit": None, "dirty": None})
            initialize_repository(repository)
            archive = repository / "source_archive"
            archive.mkdir()
            self.assertEqual(run_manifest.capture_git_state(archive), {"commit": None, "dirty": None})

    def test_missing_git_or_timed_out_git_is_optional(self):
        for error in (FileNotFoundError("private path"), subprocess.TimeoutExpired("git", 5)):
            with self.subTest(error=type(error).__name__), patch("run_manifest.subprocess.run", side_effect=error):
                self.assertEqual(run_manifest.capture_git_state(Path(".")), {"commit": None, "dirty": None})

    def test_json_contains_allowlisted_metadata_without_machine_identifiers(self):
        private_value = "private-machine-or-secret"
        pool = {
            "user_api": "blas", "internal_api": "openblas", "version": "0.3.27",
            "num_threads": 2, "threading_layer": "pthreads", "architecture": "ARMV8",
            "filepath": f"/Users/{private_value}/libblas.dylib", "prefix": private_value,
        }
        with patch("run_manifest.threadpool_info", return_value=[pool]), patch.dict(os.environ, {"SECRET": private_value}):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.assertTrue(run_manifest.write_run_manifest(output, {"commit": "a" * 40, "dirty": False}, "2026-11-19T00:00:00+00:00", 2))
                raw = (output / "run_manifest.json").read_text()
        manifest = json.loads(raw)
        self.assertNotIn(private_value, raw)
        self.assertNotIn("filepath", raw)
        self.assertEqual(set(manifest["packages"]), {"numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "threadpoolctl"})
        self.assertEqual(manifest["threadpools"]["libraries"][0], {
            "user_api": "blas", "internal_api": "openblas", "version": "0.3.27",
            "num_threads": 2, "threading_layer": "pthreads", "architecture": "ARMV8",
        })
        json.dumps(manifest, allow_nan=False)

    def test_unavailable_metadata_does_not_hide_other_fields(self):
        def version(name):
            if name == "numpy":
                raise run_manifest.metadata.PackageNotFoundError(name)
            return "1.0"

        with patch("run_manifest.metadata.version", side_effect=version), patch("run_manifest.platform.release", side_effect=OSError("private path")), patch("run_manifest.threadpool_info", side_effect=RuntimeError("private path")):
            manifest = run_manifest.collect_run_manifest({"commit": None, "dirty": None}, "start", 2)
        self.assertIsNone(manifest["packages"]["numpy"])
        self.assertEqual(manifest["packages"]["scipy"], "1.0")
        self.assertIsNone(manifest["platform"]["release"])
        self.assertIsNotNone(manifest["python"]["version"])
        self.assertIsNone(manifest["threadpools"]["libraries"])
        self.assertNotIn("private path", json.dumps(manifest, allow_nan=False))

    def test_active_thread_limit_is_observed(self):
        with threadpool_limits(limits=1):
            manifest = run_manifest.collect_run_manifest({"commit": None, "dirty": None}, "start", 1)
        blas = [pool for pool in manifest["threadpools"]["libraries"] if pool["user_api"] == "blas"]
        if not blas:
            self.skipTest("No BLAS library supported by threadpoolctl is loaded")
        self.assertTrue(all(pool["num_threads"] == 1 for pool in blas))

    def test_manifest_failures_warn_and_remove_stale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "run_manifest.json"
            for target, error in (("run_manifest.collect_run_manifest", RuntimeError("private path")), ("run_manifest.Path.write_text", OSError("private path"))):
                with self.subTest(target=target):
                    path.write_text('{"previous_run": true}')
                    warnings = io.StringIO()
                    with patch(target, side_effect=error), redirect_stderr(warnings):
                        self.assertFalse(run_manifest.write_run_manifest(output, {"commit": None, "dirty": None}, "start", 2))
                    self.assertFalse(path.exists())
                    self.assertIn("could not be saved", warnings.getvalue())
                    self.assertNotIn("private path", warnings.getvalue())

    def test_both_experiments_write_manifest_before_fitting(self):
        """One repeat of the real pipeline, with synthetic input and no downloads."""
        for module, n_subjects in ((run_experiment, 30), (wisdm_replication, 44)):
            with self.subTest(experiment=module.__name__):
                self.check_synthetic_run(module, n_subjects)

    def check_synthetic_run(self, module, n_subjects):
        rng = np.random.default_rng(42)
        subjects = np.repeat(np.arange(1, n_subjects + 1), 24)
        labels = np.tile(np.repeat(np.arange(6), 4), n_subjects)
        features = 2.0 * np.eye(6)[labels] + rng.normal(0, 0.8, size=(len(labels), 6))
        data = {
            "X_train": features[:360], "X_test": features[360:],
            "y_train": labels[:360], "y_test": labels[360:],
            "subject_train": subjects[:360], "subject_test": subjects[360:],
            "input_hashes": {},
            "X": features, "y": labels, "subjects": subjects,
            "labels": [f"class_{index}" for index in range(6)],
            "feature_names": [f"feature_{index}" for index in range(6)],
            "source_sha256": "synthetic", "nested_source_sha256": "synthetic",
            "n_source_arff_files": n_subjects, "duplicate_aliases": [],
            "activity_subject_counts": {f"class_{index}": n_subjects for index in range(6)},
            "missing_activities_by_subject": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            initialize_repository(repository)
            output = repository / "results"
            output.mkdir()
            (output / "protocol.json").write_text("{}\n")
            git(repository, "add", "results/protocol.json")
            git(repository, "commit", "-qm", "Existing protocol")
            commit = git(repository, "rev-parse", "HEAD")
            original_select_models = module.select_models

            def select_models(*args):
                # A real fit must already have a manifest, not just final outputs.
                self.assertTrue((output / "run_manifest.json").exists())
                return original_select_models(*args)

            with patch.object(module, "HERE", repository), patch.object(module, "REPEAT_SEEDS", (42,)), patch.object(module, "load_data", return_value=data), patch.object(module, "select_models", side_effect=select_models), redirect_stdout(io.StringIO()):
                summary = module.run_experiment(repository / "unused_data", output)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["git"], {"commit": commit, "dirty": False})
            self.assertTrue(run_manifest.capture_git_state(repository)["dirty"])
            self.assertEqual(manifest["started_at_utc"], summary["started_at_utc"])
            self.assertEqual(summary["rows"]["predictions"], len(labels) * 2 * 4 * 2)
            self.assertEqual(manifest["threadpools"]["requested_limit"], 2)
            blas = [pool for pool in manifest["threadpools"]["libraries"] if pool["user_api"] == "blas"]
            self.assertTrue(all(pool["num_threads"] == 2 for pool in blas))


if __name__ == "__main__":
    unittest.main()
