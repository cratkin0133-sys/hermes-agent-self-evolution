"""Judge-backed GEPA metric: scoring, fallback, and evolve() validation."""

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import LLMJudge, make_judge_metric, skill_gepa_metric
from evolution.skills.evolve_skill import evolve
from evolution.skills.skill_module import SkillModule
from tests.fakes import fake_lm, xml_fields


def _gold():
    return dspy.Example(task_input="do X", expected_behavior="run tests carefully").with_inputs("task_input")


def _judge(script):
    return LLMJudge(EvolutionConfig(hermes_agent_path=None), lm=fake_lm(script))


def test_judge_metric_uses_composite_and_feedback():
    calls = []

    def script(system, user):
        calls.append(user)
        return xml_fields(plan="ok", correctness="1.0", procedure_following="0.5",
                          conciseness="0.0", feedback="F")

    metric = make_judge_metric(_judge(script))
    res = metric(_gold(), dspy.Prediction(output="answer", skill_text="SK"))
    assert res.score == pytest.approx(0.65)
    assert res.feedback == "F"
    assert calls


def test_empty_output_scores_zero_without_judge_call():
    calls = []

    def script(system, user):
        calls.append(user)
        return xml_fields(plan="ok", correctness="1.0", procedure_following="1.0",
                          conciseness="1.0", feedback="F")

    metric = make_judge_metric(_judge(script))
    res = metric(_gold(), dspy.Prediction(output="   ", skill_text="SK"))
    assert res.score == 0.0
    assert res.feedback == "The response was empty. The skill must lead to a concrete answer."
    assert calls == []


def test_judge_failure_falls_back_to_heuristic():
    metric = make_judge_metric(_judge(lambda s, u: "garbage"))
    gold = _gold()
    pred = dspy.Prediction(output="run tests carefully", skill_text="SK")
    res = metric(gold, pred)
    expected = skill_gepa_metric(gold, pred)
    assert res.score == expected.score
    assert res.feedback.startswith("[judge unavailable:")
    assert res.feedback.endswith(expected.feedback)


def test_skill_module_forward_exposes_skill_text():
    lm = fake_lm(lambda s, u: xml_fields(plan="r", output="out"))
    with dspy.context(lm=lm):
        pred = SkillModule("BODY")(task_input="hi")
    assert pred.skill_text == "BODY"
    assert pred.output == "out"


def test_evolve_rejects_invalid_metric_before_any_work(tmp_path):
    skill_dir = tmp_path / "skills" / "cat" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\n\nBody\n")
    with pytest.raises(ValueError):
        evolve("demo", hermes_repo=str(tmp_path), metric="bogus", dry_run=False)
