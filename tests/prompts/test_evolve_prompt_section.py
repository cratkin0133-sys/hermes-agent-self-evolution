"""Phase 3 end to end, offline: fake prompt_builder.py, scripted agent/judge/reflector."""

import json
import re
import shlex
import sys

import pytest

import evolution.core.multi_component as mc
import evolution.prompts.evolve_prompt_section as eps
from evolution.prompts.prompt_sections import assemble_system_prompt, discover_sections
from tests.fakes import fake_lm, xml_fields

BUILDER = '''\
DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent. "
    "Be direct and admit uncertainty."
)
MEMORY_GUIDANCE = build_memory_guidance(True)
SESSION_SEARCH_GUIDANCE = (
    "Use session_search when context from earlier work could help. Prefer specific queries "
    "(project names, file names, error strings) over vague ones, read the top hits before "
    "answering, and say so plainly when nothing relevant turns up rather than guessing."
)
PLATFORM_HINTS = {"cli": "Plain terminal: no markdown.", "telegram": "Telegram renders formatting."}
'''
TRIGGER = "On 'last time', run session_search first."


def agent(system, user):
    if "last time" in user:
        return "Calling session_search for your earlier work." if "'last time'" in (system or "") else "I don't have that context."
    return "Sure."


def judge(system, user):
    reply = re.search(r"<agent_reply>\n(.*?)\n</agent_reply>", user, re.S).group(1)
    ok = "session_search" in reply
    return xml_fields(score="1.0" if ok else "0.0",
                      feedback="Good." if ok else "Agent did not search past sessions after 'last time'; the prompt should say to.")


def reflector_good(system, user):
    cur = user.split("```")[1].strip()
    return f"```\n{cur} {TRIGGER}\n```" if cur.startswith("Use session_search") else f"```\n{cur}\n```"


def reflector_gutting(system, user):
    return "```\nOn 'last time' search.\n```"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(BUILDER)
    rows = [{"section": "SESSION_SEARCH_GUIDANCE", "user_message": f"Do it like last time for task {i}",
             "expected_behavior": "Searches past sessions before answering"} for i in range(10)]
    (tmp_path / "golden.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def _route(reflector):
    return lambda model, **kw: fake_lm(reflector if "reflect" in model else judge if "judge" in model else agent, model)


def _run(repo, tmp_path, monkeypatch, reflector, **kw):
    monkeypatch.setattr(eps, "make_lm", _route(reflector))
    monkeypatch.setattr(mc, "make_lm", _route(reflector))
    return eps.evolve_prompt_sections(
        sections=["SESSION_SEARCH_GUIDANCE"], iterations=3, dataset_path=str(repo / "golden.jsonl"),
        optimizer_model="openai/fake-reflect", eval_model="openai/fake-agent", judge_model="openai/fake-judge",
        hermes_repo=str(repo), output_root=tmp_path / "out", **kw,
    )


def test_phase3_accepts_targeted_improvement(repo, tmp_path, monkeypatch):
    m = _run(repo, tmp_path, monkeypatch, reflector_good)
    assert m["accepted"], m
    assert m["changed_sections"] == ["SESSION_SEARCH_GUIDANCE"]
    new_src = next((tmp_path / "out").glob("prompts/*/evolved/agent/prompt_builder.py")).read_text()
    ns = {"build_memory_guidance": lambda x: "mem"}
    exec(new_src, ns)
    assert ns["SESSION_SEARCH_GUIDANCE"].endswith(TRIGGER)
    assert ns["DEFAULT_AGENT_IDENTITY"] == "You are Hermes Agent. Be direct and admit uncertainty."


def test_phase3_rejects_gutted_section_even_if_score_rises(repo, tmp_path, monkeypatch):
    m = _run(repo, tmp_path, monkeypatch, reflector_gutting)
    assert not m["accepted"]
    assert any("shrink_limit" in f or "growth_limit" in f for f in m["constraint_failures"]), m


def _bench_cmd(evolved_score, base_score):
    code = ("import pathlib; t=pathlib.Path('agent/prompt_builder.py').read_text(); "
            f"print('SCORE={evolved_score}' if 'search first' in t else 'SCORE={base_score}')")
    return shlex.join([sys.executable, "-c", code])


def test_phase3_benchmark_score_drop_rejects(repo, tmp_path, monkeypatch):
    m = _run(repo, tmp_path, monkeypatch, reflector_good, benchmark_cmd=_bench_cmd(0.5, 0.9))
    assert not m["accepted"]
    assert "benchmark gate failed" in m["rejection_reasons"]
    assert m["benchmark"]["passed"] is False
    assert (m["benchmark"]["baseline_score"], m["benchmark"]["evolved_score"]) == (0.9, 0.5)


def test_phase3_benchmark_equal_scores_accepts(repo, tmp_path, monkeypatch):
    m = _run(repo, tmp_path, monkeypatch, reflector_good, benchmark_cmd=_bench_cmd(0.7, 0.7))
    assert m["accepted"], m
    assert m["benchmark"]["passed"] is True and m["benchmark"]["evolved_score"] == 0.7


def test_phase3_benchmark_regression_tolerance_is_configurable(repo, tmp_path, monkeypatch):
    m = _run(repo, tmp_path, monkeypatch, reflector_good, benchmark_cmd=_bench_cmd(0.69, 0.7),
             benchmark_max_regression=0.05)
    assert m["accepted"], m


def test_discovery_and_platform_assembly(repo):
    found, skipped = discover_sections(repo)
    assert "MEMORY_GUIDANCE" in skipped and "PLATFORM_HINTS[cli]" in found
    cand = {k: v.value for k, v in found.items()}
    sp = assemble_system_prompt(cand, "PLATFORM_HINTS[cli]")
    assert "no markdown" in sp and "Telegram" not in sp
