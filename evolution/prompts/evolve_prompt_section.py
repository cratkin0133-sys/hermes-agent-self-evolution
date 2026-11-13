"""Phase 3: evolve hermes-agent system-prompt sections against behavioural scenarios.

The highest-blast-radius text phase, so it is the strictest: every changed
section must stay within +20% / -50% of its size, keep any --must-keep phrases,
improve holdout behaviour without regressing any section, and (optionally) pass
a benchmark command run on a worktree with the change applied.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section SESSION_SEARCH_GUIDANCE
    python -m evolution.prompts.evolve_prompt_section --section "PLATFORM_HINTS[cli]" \\
        --benchmark-cmd "python -m pytest tests/agent -q" --create-branch
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

from evolution.core.benchmark_gate import run_with_changes
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.lm import make_lm
from evolution.core.multi_component import evaluate_candidate, mean, optimize_components, split
from evolution.core.pr_builder import create_branch, write_review_bundle
from evolution.core.source_patch import literal_changes
from evolution.prompts.prompt_sections import (
    Scenario, discover_sections, generate_scenarios, make_behavior_run, per_section_scores,
)

console = Console()


def evolve_prompt_sections(
    sections: Optional[list[str]] = None,
    iterations: int = 5,
    dataset_path: Optional[str] = None,
    per_section: int = 8,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: Optional[str] = None,
    must_keep: Optional[list[str]] = None,
    benchmark_cmd: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    output_root: Path = Path("output"),
    create_pr_branch: bool = False,
    min_improvement: float = 0.0,
    max_growth: float = 0.2,
    dry_run: bool = False,
) -> dict:
    repo = resolve_hermes_agent_path(hermes_repo)
    config = EvolutionConfig(hermes_agent_path=repo, optimizer_model=optimizer_model, eval_model=eval_model,
                             max_prompt_growth=max_growth)
    console.print("\n[bold cyan]🧬 Phase 3[/bold cyan] — Evolving system-prompt sections\n")

    found, skipped = discover_sections(repo, sections)
    for name, why in sorted(skipped.items()):
        console.print(f"  [yellow]skip[/yellow] {escape(name)}: {why}")
    if not found:
        console.print("[red]✗ No evolvable literal sections found[/red]")
        sys.exit(1)
    baseline = {k: lit.value for k, lit in found.items()}
    for k, v in baseline.items():
        console.print(f"  {escape(k)}: {len(v)} chars")
    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return {"sections": sorted(baseline)}

    agent = make_lm(eval_model)
    judge = make_lm(judge_model or optimizer_model)
    if dataset_path:
        scenarios = [s for s in Scenario.load_jsonl(Path(dataset_path)) if s.section in baseline]
        console.print(f"  Loaded {len(scenarios)} scenarios from {dataset_path}")
    else:
        scenarios = generate_scenarios(baseline, make_lm(optimizer_model), per_section=per_section)
        save = Path("datasets") / "prompts" / f"synthetic_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        Scenario.save_jsonl(scenarios, save)
        console.print(f"  Generated {len(scenarios)} scenarios → {save}")
    train, val, holdout = split(scenarios)
    console.print(f"  Split: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

    limits = {k: int(len(v) * (1 + config.max_prompt_growth)) for k, v in baseline.items()}
    run = make_behavior_run(agent, judge, limits)

    start = time.time()
    evolved = optimize_components(
        baseline, train, val, run, reflection_model=optimizer_model,
        max_metric_calls=iterations * (len(val) + 3) + len(val),
    )
    elapsed = time.time() - start

    validator = ConstraintValidator(config)
    changed = sorted(k for k in evolved if evolved[k] != baseline[k])
    failures: list[str] = []
    for k in changed:
        for c in validator.validate_all(evolved[k], "prompt_section", baseline_text=baseline[k]):
            if not c.passed:
                failures.append(f"{k}: {c.constraint_name}: {c.message}")
        for phrase in must_keep or []:
            if phrase.lower() in baseline[k].lower() and phrase.lower() not in evolved[k].lower():
                failures.append(f"{k}: must_keep: dropped {phrase!r}")
    for f in failures:
        console.print(f"  [red]✗ {escape(f)}[/red]")

    base_roll = evaluate_candidate(run, baseline, holdout)
    evo_roll = evaluate_candidate(run, evolved, holdout)
    base_sec, evo_sec = per_section_scores(holdout, base_roll), per_section_scores(holdout, evo_roll)
    overall_b, overall_e = mean([r.score for r in base_roll]), mean([r.score for r in evo_roll])
    regressions = {k: (base_sec[k], evo_sec[k]) for k in base_sec if evo_sec[k] < base_sec[k]}

    table = Table(title="Behaviour score (holdout)")
    for col in ("Section", "Baseline", "Evolved", "Chars"):
        table.add_column(col)
    for k in base_sec:
        color = "red" if k in regressions else ("green" if evo_sec[k] > base_sec[k] else "white")
        table.add_row(escape(k), f"{base_sec[k]:.2f}", f"[{color}]{evo_sec[k]:.2f}[/{color}]", f"{len(baseline[k])}→{len(evolved[k])}")
    table.add_row("[bold]overall[/bold]", f"{overall_b:.3f}", f"{overall_e:.3f}", "")
    console.print(table)

    changes = literal_changes(repo, found, evolved)
    bench = None
    if benchmark_cmd and changes and not failures:
        console.print(f"\n  Running benchmark gate: {benchmark_cmd}")
        bench = run_with_changes(repo, {k: v[1] for k, v in changes.items()}, benchmark_cmd)
        console.print(f"  [{'green' if bench.passed else 'red'}]{bench.message}[/]")

    reasons = []
    if not changed:
        reasons.append("no section changed")
    if failures:
        reasons.append("constraint failure")
    if regressions:
        reasons.append(f"section regression: {regressions}")
    if overall_e - overall_b <= min_improvement:
        reasons.append(f"improvement {overall_e - overall_b:+.3f} not above {min_improvement}")
    if bench is not None and not bench.passed:
        reasons.append("benchmark gate failed")
    accepted = not reasons

    out_dir = output_root / "prompts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {
        "phase": "prompt_sections", "accepted": accepted, "rejection_reasons": reasons,
        "baseline_score": overall_b, "evolved_score": overall_e, "improvement": overall_e - overall_b,
        "changed_sections": changed, "per_section_baseline": base_sec, "per_section_evolved": evo_sec,
        "constraint_failures": failures, "benchmark": bench.message if bench else None,
        "holdout_examples": len(holdout), "elapsed_seconds": round(elapsed, 1),
        "optimizer_model": optimizer_model, "eval_model": eval_model,
    }
    write_review_bundle(
        out_dir, "Evolve system-prompt sections",
        f"GEPA-evolved {', '.join(changed) or 'no sections'}. Holdout behaviour score "
        f"{overall_b:.3f} → {overall_e:.3f}. Review for tone/persona drift before merging.",
        metrics, changes,
    )
    console.print(f"\n  Review bundle: {out_dir}/PR_BODY.md")
    if accepted:
        console.print(f"[bold green]✓ Accepted: {overall_e - overall_b:+.3f}[/bold green]")
        if create_pr_branch:
            branch = f"evolve/prompts-{datetime.now():%Y%m%d-%H%M%S}"
            wt = create_branch(repo, branch, {k: v[1] for k, v in changes.items()},
                               "Evolve system-prompt sections via GEPA", out_dir / "worktree")
            console.print(f"  Branch {branch} committed in {wt}" if wt else "  [yellow]hermes repo is not git; no branch[/yellow]")
    else:
        console.print(f"[yellow]⚠ Not accepted: {'; '.join(reasons)}[/yellow]")
    metrics["output_dir"] = str(out_dir)
    return metrics


@click.command()
@click.option("--section", "sections", multiple=True, help="Section or PLATFORM_HINTS[key]; repeatable (default: all)")
@click.option("--iterations", default=5)
@click.option("--dataset", "dataset_path", default=None, help="Golden JSONL of {section, user_message, expected_behavior}")
@click.option("--per-section", default=8)
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model that plays the agent")
@click.option("--judge-model", default=None, help="Model that scores behaviour (default: optimizer model)")
@click.option("--must-keep", default=None, help="Comma-separated phrases that must survive evolution")
@click.option("--benchmark-cmd", default=None, help="Command run in a hermes-agent worktree with the change; must exit 0")
@click.option("--min-improvement", default=0.0, type=float)
@click.option("--max-growth", default=0.2, type=float, help="Max per-section growth (PLAN default 0.2 = +20%)")
@click.option("--hermes-repo", default=None)
@click.option("--create-branch", "create_pr_branch", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(sections, iterations, dataset_path, per_section, optimizer_model, eval_model, judge_model,
         must_keep, benchmark_cmd, min_improvement, max_growth, hermes_repo, create_pr_branch, dry_run):
    """Evolve system-prompt sections with GEPA under strict behavioural gates."""
    evolve_prompt_sections(
        sections=list(sections) or None, iterations=iterations, dataset_path=dataset_path,
        per_section=per_section, optimizer_model=optimizer_model, eval_model=eval_model,
        judge_model=judge_model, must_keep=must_keep.split(",") if must_keep else None,
        benchmark_cmd=benchmark_cmd, min_improvement=min_improvement, max_growth=max_growth, hermes_repo=hermes_repo,
        create_pr_branch=create_pr_branch, dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
