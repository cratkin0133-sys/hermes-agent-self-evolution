"""Run a benchmark/test command against hermes-agent with evolved files applied.

The command runs in a throwaway git worktree (or a copy, if the repo is not a
git checkout), so the user's checkout is never modified. Point it at TBLite, a
pytest subset, or any script you trust, e.g.
``--benchmark-cmd "python -m pytest tests/tools -q"``.

``run_with_changes`` is a plain pass/fail gate: a zero exit status passes.

``compare_benchmark`` runs the command twice (baseline and evolved) and compares
scores parsed from the output (see ``evolution.core.benchmark_score``: a
``SCORE=<n>`` line, a JSON ``{"score": n}`` line, or a pytest summary pass
rate). The evolved score may drop by at most ``max_regression``. If no score can
be parsed, exit codes are compared instead (an already-failing baseline cannot
be regressed). A timeout or worktree/copy setup failure in either run makes the
result inconclusive, which fails the gate.
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


REGRESSION_EPSILON = 1e-9  # absorbs float error in "evolved >= baseline - max_regression"


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


@dataclass
class _TreeRun:
    exit_ok: bool
    output: str  # stdout + "\n" + stderr ("" on timeout or setup failure)
    result: ConstraintResult
    infra_error: Optional[str] = None  # set on timeout or setup failure (not a real exit)


def _run_in_tree(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    timeout: int,
    workdir: Optional[Path],
) -> _TreeRun:
    """Run ``command`` in a throwaway tree with ``files`` applied."""
    tmp = Path(tempfile.mkdtemp(prefix="hermes-gate-", dir=workdir))
    tree = tmp / "tree"
    is_git = subprocess.run(["git", "-C", str(hermes_repo), "rev-parse", "--git-dir"], capture_output=True).returncode == 0
    try:
        if is_git:
            r = subprocess.run(["git", "-C", str(hermes_repo), "worktree", "add", "--detach", str(tree), "HEAD"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return _infra(f"worktree failed: {r.stderr.strip()}")
        else:
            try:
                shutil.copytree(hermes_repo, tree, symlinks=True)
            except (OSError, shutil.Error) as e:
                return _infra(f"copy failed: {e}")
        try:
            for rel, content in files.items():
                (tree / rel).write_text(content)
        except OSError as e:
            return _infra(f"applying files failed: {e}")
        try:
            proc = subprocess.run(shlex.split(command), cwd=tree, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return _infra(f"timed out after {timeout}s: {command}")
        output = proc.stdout + "\n" + proc.stderr
        tail = output[-2000:]
        if proc.returncode == 0:
            return _TreeRun(True, output, ConstraintResult(True, "benchmark_gate", f"passed: {command}", details=tail))
        return _TreeRun(False, output,
                        ConstraintResult(False, "benchmark_gate", f"exit {proc.returncode}: {command}", details=tail))
    finally:
        if is_git:
            subprocess.run(["git", "-C", str(hermes_repo), "worktree", "remove", "--force", str(tree)], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _infra(reason: str) -> _TreeRun:
    return _TreeRun(False, "", ConstraintResult(False, "benchmark_gate", reason), infra_error=reason)


def run_with_changes(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    timeout: int = 3600,
    workdir: Optional[Path] = None,
) -> ConstraintResult:
    return _run_in_tree(hermes_repo, files, command, timeout, workdir).result


def compare_benchmark(
    hermes_repo: Path,
    files: dict[str, str],
    command: str,
    max_regression: float = 0.02,
    timeout: int = 3600,
) -> BenchmarkComparison:
    """Run ``command`` on the unchanged repo and with ``files`` applied; compare scores.

    A timeout or setup failure in either run fails the gate ("inconclusive: ...").
    """
    base_run = _run_in_tree(hermes_repo, {}, command, timeout, None)
    evo_run = _run_in_tree(hermes_repo, files, command, timeout, None)
    base, evo = parse_score(base_run.output), parse_score(evo_run.output)
    infra = [f"{name} run {run.infra_error}" for name, run in (("baseline", base_run), ("evolved", evo_run))
             if run.infra_error]
    if infra:
        return BenchmarkComparison(False, base, evo, "inconclusive: " + "; ".join(infra))
    base_ok, evo_ok = base_run.exit_ok, evo_run.exit_ok
    if base is not None and evo is not None:
        return BenchmarkComparison(
            evo >= base - max_regression - REGRESSION_EPSILON, base, evo,
            f"score {base:.4f} \u2192 {evo:.4f} (max regression {max_regression})",
        )
    return BenchmarkComparison(
        evo_ok or not base_ok, base, evo,
        f"scores unparseable (baseline exit ok={base_ok}, evolved exit ok={evo_ok}); compared exit codes",
    )
