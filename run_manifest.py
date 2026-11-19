"""Best-effort provenance for future experiment runs, without machine paths."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from threadpoolctl import threadpool_info


PACKAGES = ("numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "threadpoolctl")
POOL_FIELDS = ("user_api", "internal_api", "version", "num_threads", "threading_layer", "architecture")


def capture_git_state(repository: Path) -> dict:
    """Call before writing outputs so generated files do not change this snapshot."""
    state = {"commit": None, "dirty": None}

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=repository, check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()

    try:
        # A source archive inside another checkout must not inherit its identity.
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repository.resolve():
            return state
        state["commit"] = git("rev-parse", "--verify", "HEAD")
        state["dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return state


def _optional(read):
    try:
        return read()
    except Exception:
        # Metadata is diagnostic only. Never include exception text or paths.
        return None


def collect_run_manifest(git_state: dict, started_at_utc: str, thread_limit: int) -> dict:
    """Inspect libraries inside the experiment's active threadpool_limits context."""
    pools = _optional(threadpool_info)
    if pools is not None:
        pools = [
            {key: pool.get(key) for key in POOL_FIELDS}
            for pool in pools
        ]
    return {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "git": git_state,
        "python": {
            "version": _optional(platform.python_version),
            "implementation": _optional(platform.python_implementation),
        },
        "platform": {
            "system": _optional(platform.system),
            "release": _optional(platform.release),
            "machine": _optional(platform.machine),
        },
        "packages": {name: _optional(lambda name=name: metadata.version(name)) for name in PACKAGES},
        "threadpools": {"requested_limit": thread_limit, "libraries": pools},
    }


def write_run_manifest(output_dir: Path, git_state: dict, started_at_utc: str, thread_limit: int) -> bool:
    """Save a start-of-fitting snapshot; missing provenance must not stop a run."""
    path = output_dir / "run_manifest.json"
    try:
        manifest = collect_run_manifest(git_state, started_at_utc, thread_limit)
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        # Do not leave an older run's manifest beside this run's protocol.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        print("Warning: run_manifest.json could not be saved; provenance is unavailable for this run.", file=sys.stderr)
        return False
    return True
