import subprocess
import sys

import pytest

from evolution.core.benchmark_gate import BenchmarkComparison, compare_benchmark
from evolution.core.benchmark_score import parse_score


def test_score_line():
    assert parse_score("SCORE=0.87") == 0.87


def test_last_score_line_wins():
    assert parse_score("SCORE=0.1\nnoise\nscore: 0.9\n") == 0.9


def test_json_score():
    assert parse_score('{"score": 0.5, "n": 10}') == 0.5


def test_last_json_line_wins():
    assert parse_score('{"score": 0.1}\n{"score": 0.4}\n') == 0.4


def test_pytest_summary():
    assert parse_score("===== 8 passed, 2 failed in 1.2s =====") == pytest.approx(0.8)


def test_pytest_summary_errors():
    assert parse_score("3 passed, 1 error") == pytest.approx(0.75)


def test_no_score():
    assert parse_score("no score here") is None
    assert parse_score("1 error") is None


def test_score_beats_json_and_pytest():
    out = 'SCORE=0.3\n{"score": 0.9}\n5 passed\n'
    assert parse_score(out) == 0.3
    assert parse_score('SCORE=0.3\n{"score": 0.9}\n') == 0.3
    assert parse_score('{"score": 0.9}\n5 passed, 5 failed\n') == 0.9


def _repo(tmp_path, git=True):
    r = tmp_path / "repo"
    r.mkdir(parents=True)
    (r / "m.py").write_text("X = 8\n")
    if git:
        for c in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
            subprocess.run(["git", "-C", str(r), *c], check=True)
    return r


CMD = f'{sys.executable} -c "import m; print(f\'SCORE={{m.X/10}}\')"'


@pytest.mark.parametrize("git", [True, False])
def test_compare_scores(tmp_path, git):
    r = _repo(tmp_path, git)
    res = compare_benchmark(r, {"m.py": "X = 9\n"}, CMD)
    assert isinstance(res, BenchmarkComparison)
    assert res.passed and res.baseline_score == 0.8 and res.evolved_score == 0.9
    assert not compare_benchmark(r, {"m.py": "X = 7\n"}, CMD, max_regression=0.05).passed
    assert compare_benchmark(r, {"m.py": "X = 7\n"}, CMD, max_regression=0.2).passed
    assert (r / "m.py").read_text() == "X = 8\n"


def test_compare_exit_code_fallback(tmp_path):
    r = _repo(tmp_path)
    ok = f'{sys.executable} -c "print(1)"'
    res = compare_benchmark(r, {"m.py": "X = 1\n"}, ok)
    assert res.passed and res.baseline_score is None and "unparseable" in res.message
    bad = f'{sys.executable} -c "import m,sys; sys.exit(0 if m.X == 8 else 1)"'
    assert not compare_benchmark(r, {"m.py": "X = 1\n"}, bad).passed
    failing = f'{sys.executable} -c "import sys; sys.exit(1)"'
    assert compare_benchmark(r, {"m.py": "X = 1\n"}, failing).passed
