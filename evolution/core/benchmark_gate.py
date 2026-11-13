"""Run a benchmark/test command against hermes-agent with evolved files applied.

The command runs in a throwaway git worktree (or a copy, if the repo is not a
git checkout), so the user's checkout is never modified. A zero exit status
passes. Point it at TBLite, a pytest subset, or any script you trust, e.g.
``--benchmark-cmd "python -m pytest tests/tools -q"``.
"""

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from evolution.core.constraints import ConstraintResult


def run_with_changes(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    timeout: int = 3600,
    workdir: Optional[Path] = None,
) -> ConstraintResult:
    tmp = Path(tempfile.mkdtemp(prefix="hermes-gate-", dir=workdir))
    tree = tmp / "tree"
    is_git = subprocess.run(["git", "-C", str(hermes_repo), "rev-parse", "--git-dir"], capture_output=True).returncode == 0
    try:
        if is_git:
            r = subprocess.run(["git", "-C", str(hermes_repo), "worktree", "add", "--detach", str(tree), "HEAD"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return ConstraintResult(False, "benchmark_gate", f"worktree failed: {r.stderr.strip()}")
        else:
            shutil.copytree(hermes_repo, tree, symlinks=True)
        for rel, content in files.items():
            (tree / rel).write_text(content)
        try:
            proc = subprocess.run(shlex.split(command), cwd=tree, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ConstraintResult(False, "benchmark_gate", f"timed out after {timeout}s: {command}")
        tail = (proc.stdout + proc.stderr)[-2000:]
        if proc.returncode == 0:
            return ConstraintResult(True, "benchmark_gate", f"passed: {command}", details=tail)
        return ConstraintResult(False, "benchmark_gate", f"exit {proc.returncode}: {command}", details=tail)
    finally:
        if is_git:
            subprocess.run(["git", "-C", str(hermes_repo), "worktree", "remove", "--force", str(tree)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
