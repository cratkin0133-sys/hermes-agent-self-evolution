"""GEPA requires a 5-arg metric returning score + textual feedback."""

import dspy
from evolution.core.fitness import skill_gepa_metric, skill_fitness_metric


def _ex():
    return dspy.Example(task_input="review PR", expected_behavior="check tests and security").with_inputs("task_input")


def test_gepa_metric_returns_score_and_feedback():
    r = skill_gepa_metric(_ex(), dspy.Prediction(output="I check tests"), None, "predictor.predict", None)
    assert 0.0 <= r.score <= 1.0
    assert isinstance(r.feedback, str) and r.feedback
    assert "security" in r.feedback  # names missing expected terms


def test_gepa_metric_empty_output_scores_zero():
    r = skill_gepa_metric(_ex(), dspy.Prediction(output=""), None, None, None)
    assert r.score == 0.0


def test_gepa_metric_score_matches_float_metric():
    pred = dspy.Prediction(output="check tests and security")
    assert skill_gepa_metric(_ex(), pred).score == skill_fitness_metric(_ex(), pred)
