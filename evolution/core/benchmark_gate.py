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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.core.benchmark_score import parse_score
from evolution.core.constraints import ConstraintResult


@dataclass
class BenchmarkComparison:
    passed: bool
    baseline_score: Optional[float]
    evolved_score: Optional[float]
    message: str


def benchmark_metrics(bench: Optional[BenchmarkComparison]) -> Optional[dict]:
    """Metrics-dict form of a comparison (None when no benchmark ran)."""
    if bench is None:
        return None
    return {"passed": bench.passed, "baseline_score": bench.baseline_score,
            "evolved_score": bench.evolved_score, "message": bench.message}


def _run_in_tree(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    timeout: int,
    workdir: Optional[Path],
) -> tuple[bool, str, ConstraintResult]:
    """Run ``command`` in a throwaway tree with ``files`` applied.

    Returns ``(exit_ok, output, result)``; ``output`` is the full stdout+stderr
    ("" on timeout or setup failure) and ``result`` is the gate verdict.
    """
    tmp = Path(tempfile.mkdtemp(prefix="hermes-gate-", dir=workdir))
    tree = tmp / "tree"
    is_git = subprocess.run(["git", "-C", str(hermes_repo), "rev-parse", "--git-dir"], capture_output=True).returncode == 0
    try:
        if is_git:
            r = subprocess.run(["git", "-C", str(hermes_repo), "worktree", "add", "--detach", str(tree), "HEAD"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return False, "", ConstraintResult(False, "benchmark_gate", f"worktree failed: {r.stderr.strip()}")
        else:
            shutil.copytree(hermes_repo, tree, symlinks=True)
        for rel, content in files.items():
            (tree / rel).write_text(content)
        try:
            proc = subprocess.run(shlex.split(command), cwd=tree, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "", ConstraintResult(False, "benchmark_gate", f"timed out after {timeout}s: {command}")
        output = proc.stdout + proc.stderr
        tail = output[-2000:]
        if proc.returncode == 0:
            return True, output, ConstraintResult(True, "benchmark_gate", f"passed: {command}", details=tail)
        return False, output, ConstraintResult(False, "benchmark_gate", f"exit {proc.returncode}: {command}", details=tail)
    finally:
        if is_git:
            subprocess.run(["git", "-C", str(hermes_repo), "worktree", "remove", "--force", str(tree)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def run_with_changes(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    timeout: int = 3600,
    workdir: Optional[Path] = None,
) -> ConstraintResult:
    return _run_in_tree(hermes_repo, files, command, timeout, workdir)[2]


def compare_benchmark(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    max_regression: float = 0.02,
    timeout: int = 3600,
) -> BenchmarkComparison:
    """Run ``command`` on the unchanged repo and with ``files`` applied; compare scores."""
    base_ok, base_out, _ = _run_in_tree(hermes_repo, {}, command, timeout, None)
    evo_ok, evo_out, _ = _run_in_tree(hermes_repo, files, command, timeout, None)
    base, evo = parse_score(base_out), parse_score(evo_out)
    if base is not None and evo is not None:
        return BenchmarkComparison(
            evo >= base - max_regression, base, evo,
            f"score {base:.4f} \u2192 {evo:.4f} (max regression {max_regression})",
        )
    return BenchmarkComparison(
        evo_ok or not base_ok, base, evo,
        f"scores unparseable (baseline exit ok={base_ok}, evolved exit ok={evo_ok}); compared exit codes",
    )
