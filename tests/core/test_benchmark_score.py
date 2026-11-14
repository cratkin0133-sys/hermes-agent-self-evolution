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


def test_pytest_line_without_counts_is_skipped():
    out = "===== 8 passed, 2 failed in 1.2s =====\nupload failed\n"
    assert parse_score(out) == pytest.approx(0.8)


def test_last_pytest_summary_wins():
    out = "1 passed, 1 failed\nmore\n===== 3 passed, 1 failed =====\n"
    assert parse_score(out) == pytest.approx(0.75)


SLEEP = f'{sys.executable} -c "import time; time.sleep(30)"'


def test_both_runs_time_out_is_inconclusive(tmp_path):
    r = _repo(tmp_path)
    res = compare_benchmark(r, {"m.py": "X = 9\n"}, SLEEP, timeout=1)
    assert not res.passed
    assert res.message.startswith("inconclusive:")
    assert "baseline" in res.message and "timed out" in res.message


def test_one_run_times_out_fails(tmp_path):
    r = _repo(tmp_path)
    cmd = (f'{sys.executable} -c "import m, time; time.sleep(30 if m.X == 9 else 0); '
           f'print(\'SCORE=0.5\')"')
    res = compare_benchmark(r, {"m.py": "X = 9\n"}, cmd, timeout=2)
    assert not res.passed
    assert res.message.startswith("inconclusive:") and "evolved" in res.message


def test_setup_failure_is_inconclusive(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "-C", str(r), "init", "-q"], check=True)  # no HEAD: worktree add fails
    ok = f'{sys.executable} -c "print(1)"'
    res = compare_benchmark(r, {"m.py": "X = 9\n"}, ok)
    assert not res.passed and res.message.startswith("inconclusive:")
    assert "baseline" in res.message and "evolved" in res.message and "worktree failed" in res.message


def test_stdout_without_newline_and_stderr_still_parse(tmp_path):
    r = _repo(tmp_path)
    cmd = (f'{sys.executable} -c "import sys; sys.stdout.write(\'SCORE=0.7\'); '
           f'sys.stderr.write(\'warning text\')"')
    res = compare_benchmark(r, {"m.py": "X = 9\n"}, cmd)
    assert res.baseline_score == 0.7 and res.evolved_score == 0.7 and res.passed


def test_regression_epsilon(tmp_path):
    r = _repo(tmp_path)
    cmd = f'{sys.executable} -c "import m; print(\'SCORE=\' + (\'0.90\' if m.X == 8 else \'0.88\'))"'
    assert compare_benchmark(r, {"m.py": "X = 9\n"}, cmd, max_regression=0.02).passed
    # 0.7 >= 0.8 - 0.1 is False in raw floating point; the epsilon makes it pass.
    cmd2 = f'{sys.executable} -c "import m; print(\'SCORE=\' + (\'0.8\' if m.X == 8 else \'0.7\'))"'
    assert compare_benchmark(r, {"m.py": "X = 9\n"}, cmd2, max_regression=0.1).passed
