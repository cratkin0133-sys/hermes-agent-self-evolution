"""Phase 2: evolve hermes-agent tool descriptions for better tool selection.

All selected descriptions are evolved *together*: every example is scored
against the full catalog, so one tool cannot improve by stealing another's
correct selections, and the holdout gate rejects any per-tool regression.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --tools read_file,search_files,terminal
    python -m evolution.tools.evolve_tool_descriptions --dataset datasets/tools/golden.jsonl --create-branch
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evolution.core.benchmark_gate import benchmark_metrics, compare_benchmark
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.lm import make_lm
from evolution.core.multi_component import Rollout, evaluate_candidate, mean, optimize_components, split
from evolution.core.pr_builder import create_branch, write_review_bundle
from evolution.core.source_patch import literal_changes
from evolution.tools.tool_descriptions import (
    ToolTask, discover_tool_descriptions, generate_tasks, make_selection_run, per_tool_accuracy,
)

console = Console()


def evolve_tool_descriptions(
    tools: Optional[list[str]] = None,
    iterations: int = 5,
    dataset_path: Optional[str] = None,
    per_tool: int = 8,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    output_root: Path = Path("output"),
    create_pr_branch: bool = False,
    regression_tolerance: float = 0.0,
    benchmark_cmd: Optional[str] = None,
    benchmark_max_regression: float = 0.02,
    benchmark_timeout: int = 3600,
    dry_run: bool = False,
) -> dict:
    repo = resolve_hermes_agent_path(hermes_repo)
    config = EvolutionConfig(hermes_agent_path=repo, optimizer_model=optimizer_model, eval_model=eval_model)
    console.print("\n[bold cyan]🧬 Phase 2[/bold cyan] — Evolving tool descriptions\n")

    found, skipped = discover_tool_descriptions(repo, tools)
    all_found, _ = discover_tool_descriptions(repo)
    for name, why in sorted(skipped.items()):
        console.print(f"  [yellow]skip[/yellow] {escape(name)}: {why}")
    if not found:
        console.print("[red]✗ No evolvable tool descriptions found[/red]")
        sys.exit(1)
    catalog_baseline = {n: lit.value for n, lit in all_found.items()}
    baseline = {n: lit.value for n, lit in found.items()}
    console.print(f"  Evolving {len(baseline)} descriptions: {', '.join(sorted(baseline))}")
    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return {"tools": sorted(baseline)}

    selector = make_lm(eval_model)
    if dataset_path:
        tasks = [t for t in ToolTask.load_jsonl(Path(dataset_path)) if t.tool in baseline or t.tool == "none"]
        console.print(f"  Loaded {len(tasks)} tasks from {dataset_path}")
    else:
        tasks = generate_tasks(baseline, make_lm(optimizer_model), per_tool=per_tool)
        save = Path("datasets") / "tools" / f"synthetic_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        ToolTask.save_jsonl(tasks, save)
        console.print(f"  Generated {len(tasks)} synthetic tasks → {save}")
    train, val, holdout = split(tasks)
    console.print(f"  Split: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

    limits = {n: max(config.max_tool_desc_size, len(v)) for n, v in catalog_baseline.items()}
    run = make_selection_run(selector, limits)

    def run_full_catalog(candidate: dict[str, str], ex: ToolTask) -> Rollout:
        return run({**catalog_baseline, **candidate}, ex)

    start = time.time()
    evolved_partial = optimize_components(
        baseline, train, val, run_full_catalog, reflection_model=optimizer_model,
        max_metric_calls=iterations * (len(val) + 3) + len(val),
    )
    evolved = {**catalog_baseline, **evolved_partial}
    elapsed = time.time() - start

    # ── Gates ────────────────────────────────────────────────────────────
    validator = ConstraintValidator(config)
    changed = {n for n in baseline if evolved[n] != baseline[n]}
    constraint_ok = True
    for n in sorted(changed):
        for c in validator.validate_all(evolved[n], "tool_description", baseline_text=baseline[n]):
            if not c.passed:
                constraint_ok = False
                console.print(f"  [red]✗ {n}: {c.constraint_name}: {c.message}[/red]")

    base_roll = evaluate_candidate(run_full_catalog, baseline, holdout)
    evo_roll = evaluate_candidate(run_full_catalog, evolved_partial, holdout)
    base_acc = per_tool_accuracy(holdout, base_roll)
    evo_acc = per_tool_accuracy(holdout, evo_roll)
    overall_b = mean([float(r.output == t.tool) for t, r in zip(holdout, base_roll)])
    overall_e = mean([float(r.output == t.tool) for t, r in zip(holdout, evo_roll)])
    regressions = {t: (base_acc[t], evo_acc[t]) for t in base_acc if evo_acc[t] < base_acc[t] - regression_tolerance}

    table = Table(title="Tool selection (holdout)")
    for col in ("Tool", "Baseline", "Evolved", "Desc chars"):
        table.add_column(col)
    for t in base_acc:
        size = f"{len(baseline[t])}→{len(evolved[t])}" if t in baseline else "-"
        color = "red" if t in regressions else ("green" if evo_acc[t] > base_acc[t] else "white")
        table.add_row(t, f"{base_acc[t]:.2f}", f"[{color}]{evo_acc[t]:.2f}[/{color}]", size)
    table.add_row("[bold]overall[/bold]", f"{overall_b:.3f}", f"{overall_e:.3f}", "")
    console.print(table)

    changes = literal_changes(repo, found, evolved_partial)
    reasons = []
    if not changed:
        reasons.append("no description changed")
    if not constraint_ok:
        reasons.append("constraint failure")
    if regressions:
        reasons.append(f"per-tool regression: {regressions}")
    if not overall_e > overall_b:
        reasons.append("no holdout improvement")

    # The benchmark is the most expensive gate: only run it when everything else passed.
    bench = None
    if benchmark_cmd and changes and not reasons:
        console.print(f"\n  Running benchmark gate: {escape(benchmark_cmd)}")
        bench = compare_benchmark(repo, {k: v[1] for k, v in changes.items()}, benchmark_cmd,
                                  max_regression=benchmark_max_regression, timeout=benchmark_timeout)
        console.print(f"  [{'green' if bench.passed else 'red'}]{escape(bench.message)}[/]")
        if not bench.passed:
            reasons.append("benchmark gate failed")

    accepted = not reasons
    out_dir = output_root / "tools" / datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "phase": "tool_descriptions", "accepted": accepted, "rejection_reasons": reasons,
        "baseline_accuracy": overall_b,
        "evolved_accuracy": overall_e, "improvement": overall_e - overall_b, "changed_tools": sorted(changed),
        "regressions": regressions,
        "benchmark": benchmark_metrics(bench), "per_tool_baseline": base_acc, "per_tool_evolved": evo_acc,
        "holdout_examples": len(holdout), "elapsed_seconds": round(elapsed, 1),
        "optimizer_model": optimizer_model, "eval_model": eval_model, "output_dir": str(out_dir),
    }

    write_review_bundle(
        out_dir, "Evolve tool descriptions",
        f"GEPA-evolved descriptions for {len(changed)} tool(s). Holdout selection accuracy "
        f"{overall_b:.3f} → {overall_e:.3f}. Schema structure and parameter descriptions unchanged.",
        metrics, changes,
    )
    console.print(f"\n  Review bundle: {out_dir}/PR_BODY.md")

    if not accepted:
        console.print(f"[yellow]⚠ Not accepted: {escape('; '.join(reasons))}[/yellow]")
    else:
        console.print(f"[bold green]✓ Accepted: accuracy {overall_e - overall_b:+.3f}[/bold green]")
        if create_pr_branch:
            branch = f"evolve/tools-{datetime.now():%Y%m%d-%H%M%S}"
            wt = create_branch(repo, branch, {k: v[1] for k, v in changes.items()},
                               "Evolve tool descriptions via GEPA", out_dir / "worktree")
            console.print(f"  Branch {branch} committed in {wt}" if wt else "  [yellow]hermes repo is not git; no branch[/yellow]")
    return metrics


@click.command()
@click.option("--tools", default=None, help="Comma-separated tool names (default: all literal descriptions)")
@click.option("--iterations", default=5, help="Optimization budget (≈ full validation passes)")
@click.option("--dataset", "dataset_path", default=None, help="Golden JSONL of {task, tool}")
@click.option("--per-tool", default=8, help="Synthetic tasks per tool")
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--hermes-repo", default=None)
@click.option("--create-branch", "create_pr_branch", is_flag=True, help="Commit accepted result on a branch in a hermes-agent worktree")
@click.option("--benchmark-cmd", default=None, help="Command run on baseline and evolved hermes-agent worktrees; score from SCORE=<n>, JSON {\"score\": n} or pytest summary")
@click.option("--benchmark-max-regression", default=0.02, type=float, help="Max allowed benchmark score drop vs baseline")
@click.option("--benchmark-timeout", default=3600, type=int, help="Seconds per benchmark run; a timeout fails the gate")
@click.option("--dry-run", is_flag=True)
def main(tools, iterations, dataset_path, per_tool, optimizer_model, eval_model, hermes_repo, create_pr_branch,
         benchmark_cmd, benchmark_max_regression, benchmark_timeout, dry_run):
    """Evolve tool descriptions with GEPA, gated on per-tool selection accuracy."""
    evolve_tool_descriptions(
        tools=tools.split(",") if tools else None, iterations=iterations, dataset_path=dataset_path,
        per_tool=per_tool, optimizer_model=optimizer_model, eval_model=eval_model,
        hermes_repo=hermes_repo, create_pr_branch=create_pr_branch, benchmark_cmd=benchmark_cmd,
        benchmark_max_regression=benchmark_max_regression, benchmark_timeout=benchmark_timeout, dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
