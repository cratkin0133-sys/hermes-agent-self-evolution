"""Phase 5: continuous self-improvement loop.

    python -m evolution.monitor.loop collect          # mine session history → usage/correction signals
    python -m evolution.monitor.loop triage           # rank what to optimize next
    python -m evolution.monitor.loop run              # collect → triage → evolve top target(s) → record
    python -m evolution.monitor.loop bench --cmd ...  # run a benchmark, record score; fail on regression vs last accepted run
    python -m evolution.monitor.loop history          # recent runs
    python -m evolution.monitor.loop cron             # print a crontab line for weekly unattended runs

Automates detection and optimization only. Accepted results become review
bundles (and, with --create-branch, local branches); a human still merges.
"""

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.core.benchmark_gate import REGRESSION_EPSILON
from evolution.core.benchmark_score import parse_score
from evolution.core.config import resolve_hermes_agent_path
from evolution.monitor.signals import compute_signals, load_messages, skill_names
from evolution.monitor.store import MetricsStore
from evolution.monitor.triage import rank

console = Console()
SOURCES = ("claude-code", "copilot", "hermes")


def collect(repo: Path, store: MetricsStore, messages: Optional[list[dict]] = None, tools: Optional[list[str]] = None):
    from evolution.tools.tool_descriptions import discover_tool_descriptions
    msgs = load_messages(SOURCES) if messages is None else messages
    tool_names = tools if tools is not None else sorted(discover_tool_descriptions(repo)[0])
    signals = compute_signals(msgs, skill_names(repo), "skill") + compute_signals(msgs, tool_names, "tool")
    store.record_signals(signals, source=",".join(SOURCES) if messages is None else "provided")
    return msgs, signals


def dispatch(phase: str, target: str, repo: Path, dry_run: bool, create_branch: bool,
             optimizer_model: str, eval_model: str, iterations: int) -> dict:
    """Run the right evolver for a target. Never raises: failures become rejected metrics."""
    try:
        if phase == "skill":
            from evolution.skills.evolve_skill import evolve
            m = evolve(skill_name=target, iterations=iterations, eval_source="synthetic",
                       optimizer_model=optimizer_model, eval_model=eval_model, hermes_repo=str(repo),
                       create_pr_branch=create_branch, dry_run=dry_run)
        elif phase == "tool":
            from evolution.tools.evolve_tool_descriptions import evolve_tool_descriptions
            m = evolve_tool_descriptions(tools=[target], iterations=iterations, optimizer_model=optimizer_model,
                                         eval_model=eval_model, hermes_repo=str(repo),
                                         create_pr_branch=create_branch, dry_run=dry_run)
        else:
            return {"accepted": False, "rejection_reasons": [f"unknown phase {phase}"]}
        return m or {"accepted": False}
    except SystemExit as e:
        return {"accepted": False, "rejection_reasons": [f"evolver exited ({e.code})"]}
    except Exception as e:
        return {"accepted": False, "rejection_reasons": [f"{type(e).__name__}: {e}"]}


def _signals_table(signals) -> Table:
    t = Table(title="Usage signals")
    for c in ("Phase", "Target", "Uses", "Corrections"):
        t.add_column(c)
    for s in sorted(signals, key=lambda s: (-s.uses, s.target))[:30]:
        t.add_row(s.phase, s.target, str(s.uses), str(s.corrections))
    return t


@click.group()
@click.option("--hermes-repo", default=None)
@click.option("--db", default=None, help="Metrics DB path (default ~/.hermes/self-evolution/metrics.db)")
@click.pass_context
def cli(ctx, hermes_repo, db):
    ctx.obj = {"repo": resolve_hermes_agent_path(hermes_repo), "store": MetricsStore(Path(db) if db else None)}


@cli.command("collect")
@click.pass_context
def collect_cmd(ctx):
    """Mine session history into usage/correction signals."""
    msgs, signals = collect(ctx.obj["repo"], ctx.obj["store"])
    console.print(f"Read {len(msgs)} messages; {len(signals)} targets with usage.")
    console.print(_signals_table(signals))


@cli.command("triage")
@click.option("--min-uses", default=3)
@click.option("--cooldown-days", default=14)
@click.pass_context
def triage_cmd(ctx, min_uses, cooldown_days):
    """Rank targets from the latest signals."""
    from evolution.monitor.signals import Signal
    sigs = [Signal(r["phase"], r["target"], r["uses"], r["corrections"]) for r in ctx.obj["store"].latest_signals()]
    _print_ranking(rank(sigs, ctx.obj["store"], min_uses, cooldown_days))


def _print_ranking(targets):
    t = Table(title="Triage")
    for c in ("#", "Phase", "Target", "Priority", "Why"):
        t.add_column(c)
    for i, x in enumerate(targets[:20], 1):
        t.add_row(str(i), x.phase, x.target, f"{x.priority:.3f}", x.reason)
    console.print(t if targets else "No targets meet the thresholds yet.")


@cli.command("run")
@click.option("--max-targets", default=1)
@click.option("--phases", default="skill,tool", help="Which phases the loop may run")
@click.option("--min-uses", default=3)
@click.option("--cooldown-days", default=14)
@click.option("--iterations", default=5)
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--create-branch", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Triage and validate setup without spending on optimization")
@click.pass_context
def run_cmd(ctx, max_targets, phases, min_uses, cooldown_days, iterations, optimizer_model, eval_model, create_branch, dry_run):
    """collect → triage → evolve the top target(s) → record results."""
    repo, store = ctx.obj["repo"], ctx.obj["store"]
    run_loop(repo, store, max_targets, phases.split(","), min_uses, cooldown_days, iterations,
             optimizer_model, eval_model, create_branch, dry_run)


def run_loop(repo, store, max_targets=1, phases=("skill", "tool"), min_uses=3, cooldown_days=14, iterations=5,
             optimizer_model="openai/gpt-4.1", eval_model="openai/gpt-4.1-mini", create_branch=False,
             dry_run=False, messages=None, tools=None) -> list[tuple]:
    _, signals = collect(repo, store, messages, tools)
    targets = [t for t in rank(signals, store, min_uses, cooldown_days) if t.phase in phases][:max_targets]
    _print_ranking(targets)
    results = []
    for t in targets:
        console.rule(f"{t.phase}: {t.target}")
        m = dispatch(t.phase, t.target, repo, dry_run, create_branch, optimizer_model, eval_model, iterations)
        if not dry_run:
            store.record_run(t.phase, t.target, m)
        results.append((t, m))
        console.print(f"  → {'ACCEPTED' if m.get('accepted') else 'not accepted'} "
                      f"{m.get('output_dir', '')} {'; '.join(m.get('rejection_reasons', []))}")
    return results


@cli.command("bench")
@click.option("--cmd", "command", required=True, help="Benchmark command run in the hermes-agent repo")
@click.option("--name", default="benchmark")
@click.option("--timeout", default=7200)
@click.option("--max-regression", default=0.02, type=float, help="Fail if the score drops more than this vs the last accepted scored run")
@click.pass_context
def bench_cmd(ctx, command, name, timeout, max_regression):
    """Run a benchmark/test command, record pass/fail and score, and flag score regressions."""
    store = ctx.obj["store"]
    prev_row = store.previous_scored_run("benchmark", name)
    prev = prev_row["evolved"] if prev_row else None
    score = None
    try:
        p = subprocess.run(shlex.split(command), cwd=ctx.obj["repo"], capture_output=True, text=True, timeout=timeout)
        output = p.stdout + "\n" + p.stderr
        ok, tail, score = p.returncode == 0, output[-2000:], parse_score(output)
    except subprocess.TimeoutExpired:
        ok, tail = False, f"timeout after {timeout}s"
    regressed = ok and score is not None and prev is not None and score < prev - max_regression - REGRESSION_EPSILON
    # A regressed run is recorded as not accepted, so it never becomes the next baseline.
    store.record_run("benchmark", name, {"accepted": ok and not regressed, "command": command,
                                         "output_tail": tail, "evolved_score": score})
    fmt = lambda v: "n/a" if v is None else f"{v:g}"
    console.print(f"{name}: {'PASS' if ok else 'FAIL'} score={fmt(score)} (prev {fmt(prev)})")
    if regressed:
        console.print(f"REGRESSION {prev:g} \u2192 {score:g}")
        ok = False
    sys.exit(0 if ok else 1)


@cli.command("history")
@click.option("--limit", default=20)
@click.pass_context
def history_cmd(ctx, limit):
    t = Table(title="Recent runs")
    for c in ("When", "Phase", "Target", "Accepted", "Baseline", "Evolved", "Output"):
        t.add_column(c)
    for r in ctx.obj["store"].runs(limit):
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        t.add_row(r["ts"], r["phase"], r["target"], "yes" if r["accepted"] else "no",
                  fmt(r["baseline"]), fmt(r["evolved"]), r["output_dir"] or "")
    console.print(t)


@cli.command("cron")
@click.option("--schedule", default="17 3 * * 0", help="cron schedule (default Sundays 03:17)")
@click.pass_context
def cron_cmd(ctx, schedule):
    """Print a crontab line for unattended weekly runs (install with `crontab -e`)."""
    here = Path(__file__).resolve().parents[2]
    log = Path.home() / ".hermes" / "self-evolution" / "loop.log"
    click.echo(f'{schedule} cd {here} && HERMES_AGENT_REPO={ctx.obj["repo"]} {sys.executable} '
               f'-m evolution.monitor.loop run --max-targets 1 --create-branch >> {log} 2>&1')


if __name__ == "__main__":
    cli()
