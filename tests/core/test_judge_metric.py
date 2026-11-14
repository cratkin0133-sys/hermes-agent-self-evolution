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


GOOD = xml_fields(plan="ok", correctness="1.0", procedure_following="1.0", conciseness="1.0", feedback="F")


def test_judge_failure_after_first_call_falls_back_to_heuristic():
    replies = iter([GOOD])
    metric = make_judge_metric(_judge(lambda s, u: next(replies, "garbage")))
    gold = _gold()
    assert metric(gold, dspy.Prediction(output="first", skill_text="SK")).score == pytest.approx(1.0)
    assert metric.fallbacks == 0
    pred = dspy.Prediction(output="run tests carefully", skill_text="SK")
    res = metric(gold, pred)
    expected = skill_gepa_metric(gold, pred)
    assert res.score == expected.score
    assert res.feedback.startswith("[judge unavailable:")
    assert res.feedback.endswith(expected.feedback)
    assert metric.fallbacks == 1
    metric(gold, pred)
    assert metric.fallbacks == 2


def test_judge_failure_on_first_call_raises():
    metric = make_judge_metric(_judge(lambda s, u: "garbage"))
    with pytest.raises(RuntimeError, match=r"(?s)LLM judge failed on first call: .*--judge-model"):
        metric(_gold(), dspy.Prediction(output="answer", skill_text="SK"))
    assert metric.fallbacks == 0
    assert isinstance(metric.config_error, RuntimeError)
    # Once the first call failed, the metric keeps refusing instead of silently falling back.
    with pytest.raises(RuntimeError):
        metric(_gold(), dspy.Prediction(output="answer", skill_text="SK"))


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


# ── evolve()-level: fallback accounting and the holdout rule ──────────────

BODY = "Answer the user's question about the repository. Be precise and cite the files you used."
# Stay within default +20% body growth so constraint validation does not reject the stub evolution.
EVOLVED = BODY + " Run tests."


def _setup_evolve(tmp_path, monkeypatch, judge_script):
    import evolution.skills.evolve_skill as es
    from tests.fakes import ScriptedEngine

    repo = tmp_path / "hermes"
    skill_dir = repo / "skills" / "cat" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: demo\ndescription: d\n---\n\n{BODY}\n")
    ds = tmp_path / "ds"
    ds.mkdir()
    for split, n in (("train", 2), ("val", 1), ("holdout", 2)):
        rows = [{"task_input": f"{split} question {i}", "expected_behavior": "runs the tests"} for i in range(n)]
        (ds / f"{split}.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")

    def agent(system, user):
        return xml_fields(plan="p", output="good answer" if "Run tests" in (system or "") else "bad answer")

    real_lm = dspy.LM

    def make(model, **kw):
        return real_lm(model, engine=ScriptedEngine(judge_script if "judge" in model else agent),
                       cache=False, num_retries=0)

    class StubGEPA:
        def __init__(self, metric, **kw):
            self.metric = metric

        def compile(self, module, trainset, valset):
            evolved = SkillModule(EVOLVED)
            for ex in trainset:  # judge calls made "during optimization"
                try:
                    self.metric(ex, evolved(task_input=ex.task_input))
                except Exception:  # like dspy's Evaluate: metric errors become failure scores
                    pass
            return evolved

    monkeypatch.setattr(es.dspy, "LM", make)
    monkeypatch.setattr(es.dspy, "GEPA", StubGEPA)
    monkeypatch.chdir(tmp_path)
    return lambda: es.evolve("demo", hermes_repo=str(repo), eval_source="golden", dataset_path=str(ds),
                             optimizer_model="openai/fake-opt", eval_model="openai/fake-agent",
                             judge_model="openai/fake-judge", iterations=1)


def _scored(user):
    good = "good answer" in user
    return xml_fields(plan="ok", correctness="1.0" if good else "0.1", procedure_following="1.0" if good else "0.1",
                      conciseness="1.0", feedback="F")


def test_evolve_accepts_and_records_judge_metrics(tmp_path, monkeypatch):
    m = _setup_evolve(tmp_path, monkeypatch, lambda s, u: _scored(u))()
    assert m["metric"] == "judge" and m["judge_model"] == "openai/fake-judge"
    assert m["judge_fallbacks"] == 0
    assert m["accepted"], m
    assert m["rejection_reasons"] == []


def test_evolve_rejects_on_judge_fallback_during_holdout(tmp_path, monkeypatch):
    def judge(system, user):
        return "garbage" if "holdout question 1" in user else _scored(user)

    m = _setup_evolve(tmp_path, monkeypatch, judge)()
    assert m["judge_fallbacks"] >= 1
    assert not m["accepted"]
    assert "judge fallback during holdout scoring" in m["rejection_reasons"]


def test_evolve_raises_when_first_judge_call_fails(tmp_path, monkeypatch):
    run = _setup_evolve(tmp_path, monkeypatch, lambda s, u: "garbage")
    with pytest.raises(RuntimeError, match="LLM judge failed on first call"):
        run()


class _FlakyJudge:
    """Judge whose first ``fail_first`` score() calls raise, then succeed."""

    def __init__(self, fail_first):
        import threading
        self.fail_first = fail_first
        self.calls = 0
        self._lock = threading.Lock()

    def score(self, **kw):
        from evolution.core.fitness import FitnessScore
        with self._lock:
            self.calls += 1
            n = self.calls
        if n <= self.fail_first:
            raise ConnectionError("transient")
        return FitnessScore(correctness=1.0, procedure_following=1.0, conciseness=1.0, feedback="ok")


def test_transient_first_call_failure_is_retried_once():
    judge = _FlakyJudge(fail_first=1)
    metric = make_judge_metric(judge)
    res = metric(_gold(), dspy.Prediction(output="answer", skill_text="SK"))
    assert res.score == pytest.approx(1.0)
    assert metric.config_error is None and metric.fallbacks == 0
    assert judge.calls == 2


def test_first_call_failing_twice_is_a_config_error():
    judge = _FlakyJudge(fail_first=2)
    metric = make_judge_metric(judge)
    with pytest.raises(RuntimeError, match="LLM judge failed on first call"):
        metric(_gold(), dspy.Prediction(output="answer", skill_text="SK"))
    assert judge.calls == 2


def test_concurrent_calls_with_flaky_first_request_do_not_abort():
    from concurrent.futures import ThreadPoolExecutor
    judge = _FlakyJudge(fail_first=1)
    metric = make_judge_metric(judge)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: metric(_gold(), dspy.Prediction(output=f"a{i}", skill_text="SK")), range(16)))
    assert all(r.score == pytest.approx(1.0) for r in results)
    assert metric.config_error is None and metric.fallbacks == 0
