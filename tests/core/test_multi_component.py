"""GEPA over a dict of components, fully offline."""

from evolution.core.lm import as_reflection_lm
from evolution.core.multi_component import Rollout, optimize_components
from tests.fakes import fake_lm


def test_optimizer_improves_the_component_the_feedback_points_at():
    # Score is 1 when component "a" mentions the example's word; "b" is irrelevant.
    def run(cand, ex):
        hit = ex in cand["a"]
        return Rollout(inputs=ex, output=cand["a"], score=float(hit),
                       feedback="ok" if hit else f"missing word: {ex}", components=["a"])

    def reflect(system, user):
        words = sorted({line.split("missing word: ")[1].strip().strip('"\'}],') for line in user.splitlines() if "missing word: " in line})
        return "```\nbase " + " ".join(["alpha", "beta", "gamma"] if words else []) + "\n```"

    train = ["alpha", "beta", "gamma"]
    best = optimize_components({"a": "base", "b": "untouched"}, train, train, run,
                               reflection_model="unused", max_metric_calls=40,
                               reflection_lm=as_reflection_lm(fake_lm(reflect)))
    assert all(w in best["a"] for w in train)
    assert best["b"] == "untouched"
