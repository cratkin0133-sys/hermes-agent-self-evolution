"""Phase 4: evolve one hermes-agent tool file to fix a reproducible bug.

A small, MIT-licensed evolutionary loop in the spirit of Darwin Goedel
Machines (population + failure-driven LLM mutation + test-based fitness).
Darwinian Evolver is not used because its CLI only runs problems registered
inside its own (AGPL) package; see PLAN.md licensing notes.

Fitness of an organism (a version of the file):
  0.0                         if static guards fail (signature/registry/safety)
  0.6 * repro_passes + 0.4 * tests_pass  otherwise
Accepted only when repro passes, tests pass and guards hold.

Usage:
    python -m evolution.code.evolve_tool_code --tool read_file \\
        --repro "python -m pytest tests/tools/test_read_file_bug.py -q" \\
        --bug "read_file crashes on empty files" \\
        --test-cmd "python -m pytest tests/tools -q -x"
"""

import math
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from evolution.code.code_guards import check_guards
from evolution.code.mutator import apply_blocks, propose
from evolution.core.config import resolve_hermes_agent_path
from evolution.core.lm import make_lm
from evolution.core.pr_builder import create_branch, write_review_bundle

console = Console()


@dataclass
class Organism:
    source: str
    score: float = 0.0
    repro_ok: bool = False
    tests_ok: bool = False
    report: str = ""
    diagnosis: str = "baseline"
    parent: Optional[int] = None
    children: int = 0
    guard_problems: list[str] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return self.repro_ok and self.tests_ok and not self.guard_problems


def find_tool_file(repo: Path, tool: str) -> Optional[Path]:
    """File whose ``registry.register(name=<tool>, ...)`` call registers the tool (AST, any formatting)."""
    import ast
    for path in sorted((repo / "tools").rglob("*.py")):
        text = path.read_text(errors="ignore")
        if tool not in text or "registry.register" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "registry"):
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant) and kw.value.value == tool:
                        return path
    return None


class Workspace:
    """A detached git worktree (or copy) where candidate files are evaluated."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-evolve-code-"))
        self.tree = self.tmp / "tree"
        self.is_git = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-dir"],
                                     capture_output=True).returncode == 0
        if self.is_git:
            subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(self.tree), "HEAD"],
                           check=True, capture_output=True)
        else:
            shutil.copytree(repo, self.tree, symlinks=True)

    def run(self, command: str, timeout: int) -> tuple[bool, str]:
        try:
            p = subprocess.run(shlex.split(command), cwd=self.tree, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT after {timeout}s"
        return p.returncode == 0, (p.stdout + p.stderr)[-3000:]

    def close(self):
        if self.is_git:
            subprocess.run(["git", "-C", str(self.repo), "worktree", "remove", "--force", str(self.tree)], capture_output=True)
        shutil.rmtree(self.tmp, ignore_errors=True)


def evaluate(ws: Workspace, rel: str, baseline: str, org: Organism, repro: str, tests: str, timeout: int) -> Organism:
    org.guard_problems = check_guards(baseline, org.source) if org.source != baseline else []
    if org.guard_problems:
        org.score, org.report = 0.0, "Rejected by safety guards:\n" + "\n".join(org.guard_problems)
        return org
    (ws.tree / rel).write_text(org.source)
    org.repro_ok, repro_out = ws.run(repro, timeout)
    org.tests_ok, test_out = ws.run(tests, timeout) if org.repro_ok else (False, "(skipped: repro still failing)")
    org.score = 0.6 * org.repro_ok + 0.4 * org.tests_ok
    org.report = (f"Reproduction ({repro}): {'PASS' if org.repro_ok else 'FAIL'}\n{repro_out}\n\n"
                  f"Test suite ({tests}): {'PASS' if org.tests_ok else 'FAIL'}\n{test_out}")
    return org


def select_parent(pop: list[Organism], rng: random.Random, sharpness: float = 10.0) -> int:
    """Score-weighted (sigmoid around the 75th percentile) with a novelty bonus for few children."""
    scores = sorted(o.score for o in pop)
    mid = scores[int(0.75 * (len(scores) - 1))]
    weights = [1 / (1 + math.exp(-sharpness * (o.score - mid))) / (1 + o.children) for o in pop]
    return rng.choices(range(len(pop)), weights=weights)[0]


def evolve_tool_code(
    repro: str,
    bug: str,
    tool: Optional[str] = None,
    file: Optional[str] = None,
    test_cmd: str = "python -m pytest tests/ -q -x",
    iterations: int = 10,
    model: str = "openai/gpt-4.1",
    hermes_repo: Optional[str] = None,
    timeout: int = 1800,
    output_root: Path = Path("output"),
    create_pr_branch: bool = False,
    seed: int = 0,
) -> dict:
    repo = resolve_hermes_agent_path(hermes_repo)
    target = (repo / file) if file else find_tool_file(repo, tool or "")
    if not target or not target.exists():
        console.print(f"[red]✗ Could not find tool file for {tool or file}[/red]")
        sys.exit(1)
    rel = str(target.relative_to(repo))
    baseline = target.read_text()
    console.print(f"\n[bold cyan]🧬 Phase 4[/bold cyan] — Evolving {rel}\n  Bug: {bug}")

    lm = make_lm(model, temperature=0.8, max_tokens=16000)
    rng = random.Random(seed)
    ws = Workspace(repo)
    history: list[str] = []
    try:
        root = evaluate(ws, rel, baseline, Organism(baseline), repro, test_cmd, timeout)
        console.print(f"  Baseline: repro {'PASS' if root.repro_ok else 'FAIL'} (score {root.score:.2f})")
        if root.repro_ok:
            console.print("[yellow]⚠ Reproduction already passes on baseline — nothing to fix.[/yellow]")
            return {"phase": "tool_code", "accepted": False, "rejection_reasons": ["repro passes on baseline"]}
        pop = [root]          # breeding population: only guard-clean organisms
        evaluated = 1
        for it in range(1, iterations + 1):
            pi = select_parent(pop, rng)
            parent = pop[pi]
            parent.children += 1
            try:
                mut = propose(lm, rel, parent.source, bug, parent.report, history)
                child = Organism(apply_blocks(parent.source, mut.blocks), diagnosis=mut.diagnosis, parent=pi)
            except Exception as e:
                history.append(f"iteration {it}: mutation unusable ({e!r})")
                console.print(f"  [{it}] mutation unusable: {e!r}")
                continue
            evaluate(ws, rel, baseline, child, repro, test_cmd, timeout)
            evaluated += 1
            if not child.guard_problems:  # unsafe variants are logged, never bred from
                pop.append(child)
            outcome = ("guards: " + "; ".join(child.guard_problems)) if child.guard_problems else \
                f"repro {'PASS' if child.repro_ok else 'FAIL'}, tests {'PASS' if child.tests_ok else 'FAIL'}"
            history.append(f"iteration {it} (from #{pi}): {mut.diagnosis[:160]!r} → {outcome}")
            console.print(f"  [{it}] parent #{pi} → score {child.score:.2f} ({outcome})")
            if child.solved:
                break
    finally:
        ws.close()

    best = max(pop, key=lambda o: (o.solved, o.score))
    accepted = best.solved and best.source != baseline
    out_dir = output_root / "code" / datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "phase": "tool_code", "accepted": accepted, "file": rel, "bug": bug, "repro": repro, "test_cmd": test_cmd,
        "best_score": best.score, "repro_passes": best.repro_ok, "tests_pass": best.tests_ok,
        "organisms_evaluated": evaluated, "population_size": len(pop), "diagnosis": best.diagnosis, "model": model,
    }
    changes = {rel: (baseline, best.source)} if best.source != baseline else {}
    write_review_bundle(
        out_dir, f"Fix: {bug}",
        f"Evolved fix for `{rel}`.\n\n**Diagnosis:** {best.diagnosis}\n\n**Reproduction:** `{repro}` → "
        f"{'passes' if best.repro_ok else 'still fails'}. **Tests:** `{test_cmd}` → {'pass' if best.tests_ok else 'fail'}.\n\n"
        "Every line of evolved code needs human review before merge.",
        metrics, changes,
    )
    (out_dir / "evolution_log.txt").write_text("\n".join(history) + "\n")
    console.print(f"\n  Review bundle: {out_dir}/PR_BODY.md")
    if accepted:
        console.print("[bold green]✓ Fix found: reproduction and tests pass, guards hold[/bold green]")
        if create_pr_branch:
            branch = f"evolve/code-{Path(rel).stem}-{datetime.now():%Y%m%d-%H%M%S}"
            wt = create_branch(repo, branch, {rel: best.source}, f"Fix: {bug}", out_dir / "worktree")
            console.print(f"  Branch {branch} committed in {wt}" if wt else "  [yellow]not a git repo; no branch[/yellow]")
    else:
        console.print("[yellow]⚠ No accepted fix — see evolution_log.txt[/yellow]")
    metrics["output_dir"] = str(out_dir)
    return metrics


@click.command()
@click.option("--tool", default=None, help="Tool name; its file is found via registry.register(name=...)")
@click.option("--file", default=None, help="Or a repo-relative path to the file to evolve")
@click.option("--repro", required=True, help="Command that reproduces the bug: must FAIL now and PASS when fixed")
@click.option("--bug", required=True, help="Bug description (e.g. the GitHub issue text)")
@click.option("--test-cmd", default="python -m pytest tests/ -q -x", help="Regression suite; must pass")
@click.option("--iterations", default=10)
@click.option("--model", default="openai/gpt-4.1", help="Mutator model")
@click.option("--timeout", default=1800, help="Per-command timeout (s)")
@click.option("--hermes-repo", default=None)
@click.option("--create-branch", "create_pr_branch", is_flag=True)
def main(tool, file, repro, bug, test_cmd, iterations, model, timeout, hermes_repo, create_pr_branch):
    """Evolve a tool's code until a bug reproduction passes without breaking tests."""
    if not (tool or file):
        raise click.UsageError("Pass --tool or --file")
    evolve_tool_code(repro=repro, bug=bug, tool=tool, file=file, test_cmd=test_cmd, iterations=iterations,
                     model=model, hermes_repo=hermes_repo, timeout=timeout, create_pr_branch=create_pr_branch)


if __name__ == "__main__":
    main()
