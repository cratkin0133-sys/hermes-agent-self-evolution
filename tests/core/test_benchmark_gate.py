import subprocess
import sys

from evolution.core.benchmark_gate import run_with_changes


def _repo(tmp_path, git: bool):
    r = tmp_path / "repo"
    r.mkdir(parents=True)
    (r / "m.py").write_text("X = 1\n")
    if git:
        for c in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
            subprocess.run(["git", "-C", str(r), *c], check=True)
    return r


def test_gate_sees_changes_and_leaves_repo_untouched(tmp_path):
    for git in (True, False):
        r = _repo(tmp_path / str(git), git)
        check = f'{sys.executable} -c "import m,sys; sys.exit(0 if m.X == 2 else 1)"'
        assert run_with_changes(r, {"m.py": "X = 2\n"}, check).passed
        assert not run_with_changes(r, {"m.py": "X = 3\n"}, check).passed
        assert (r / "m.py").read_text() == "X = 1\n"
