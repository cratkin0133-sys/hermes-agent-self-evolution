"""Phase 5: signals → triage → dispatch → record → cooldown."""

import sys
from pathlib import Path

import pytest

import evolution.monitor.loop as loop
from evolution.monitor.signals import Signal, compute_signals
from evolution.monitor.store import MetricsStore
from evolution.monitor.triage import rank


def msgs(session, *texts):
    return [{"session_id": session, "timestamp": i, "task_input": t} for i, t in enumerate(texts)]


MESSAGES = (
    msgs("a", "use the requesting-code-review skill on my diff", "no, that's not what I asked")
    + msgs("b", "run requesting code review please", "you missed the security check, try again")
    + msgs("c", "requesting-code-review on PR 12", "thanks, looks good")
    + msgs("d", "use systematic-debugging here", "great")
    + msgs("e", "systematic-debugging for this flaky test", "perfect")
    + msgs("f", "systematic-debugging again", "nice")
    + msgs("g", "read_file config.yaml", "ok")
)


def test_compute_signals_counts_uses_and_corrections():
    s = {x.target: x for x in compute_signals(MESSAGES, ["requesting-code-review", "systematic-debugging", "spike"], "skill")}
    assert (s["requesting-code-review"].uses, s["requesting-code-review"].corrections) == (3, 2)
    assert (s["systematic-debugging"].uses, s["systematic-debugging"].corrections) == (3, 0)
    assert "spike" not in s


def test_rank_prefers_corrected_targets_and_respects_thresholds(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    sigs = [Signal("skill", "a", 3, 2), Signal("skill", "b", 3, 0), Signal("skill", "c", 1, 1)]
    ranked = rank(sigs, store, min_uses=3)
    assert [t.target for t in ranked] == ["a", "b"]


@pytest.fixture
def repo(tmp_path):
    for name in ("requesting-code-review", "systematic-debugging"):
        d = tmp_path / "hermes" / "skills" / "dev" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\nbody\n")
    (tmp_path / "hermes" / "tools").mkdir()
    return tmp_path / "hermes"


def test_run_loop_evolves_top_target_records_it_and_then_cools_down(repo, tmp_path, monkeypatch):
    calls = []

    def fake_dispatch(phase, target, *a, **k):
        calls.append((phase, target))
        return {"accepted": True, "baseline_score": 0.4, "evolved_score": 0.7, "improvement": 0.3, "output_dir": "out/x"}

    monkeypatch.setattr(loop, "dispatch", fake_dispatch)
    store = MetricsStore(tmp_path / "m.db")
    loop.run_loop(repo, store, max_targets=1, phases=("skill",), messages=MESSAGES, tools=[])
    assert calls == [("skill", "requesting-code-review")]
    run = store.last_run("skill", "requesting-code-review")
    assert run["accepted"] == 1 and run["improvement"] == pytest.approx(0.3)

    loop.run_loop(repo, store, max_targets=1, phases=("skill",), messages=MESSAGES, tools=[])
    assert calls[-1] == ("skill", "systematic-debugging")  # first target is in cooldown


def test_dispatch_turns_evolver_exit_into_rejected_metrics(repo, monkeypatch):
    import evolution.skills.evolve_skill as es

    def boom(**kw):
        raise SystemExit(1)

    monkeypatch.setattr(es, "evolve", boom)
    m = loop.dispatch("skill", "x", repo, False, False, "m", "m", 1)
    assert m["accepted"] is False and "exited" in m["rejection_reasons"][0]


def _bench(tmp_path, score, extra=()):
    from click.testing import CliRunner
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    cmd = f'{sys.executable} -c "print(\'SCORE={score}\')"'
    return CliRunner().invoke(loop.cli, ["--hermes-repo", str(repo), "--db", str(tmp_path / "m.db"),
                                         "bench", "--cmd", cmd, "--name", "tb", *extra])


def test_bench_records_score_and_flags_regression(tmp_path):
    first = _bench(tmp_path, 0.9)
    assert first.exit_code == 0, first.output
    assert "PASS score=0.9" in first.output and "prev n/a" in first.output
    second = _bench(tmp_path, 0.5)
    assert second.exit_code == 1, second.output
    assert "REGRESSION" in second.output
    assert MetricsStore(tmp_path / "m.db").last_run("benchmark", "tb")["evolved"] == 0.5


def test_bench_small_drop_within_max_regression_passes(tmp_path):
    assert _bench(tmp_path, 0.9).exit_code == 0
    r = _bench(tmp_path, 0.89)
    assert r.exit_code == 0, r.output
    assert _bench(tmp_path, 0.5, ("--max-regression", "0.5")).exit_code == 0
