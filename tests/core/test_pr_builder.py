import subprocess

from evolution.core.pr_builder import create_branch, write_review_bundle


def test_bundle_and_branch(tmp_path):
    repo = tmp_path / "hermes"
    repo.mkdir()
    (repo / "f.py").write_text("X = 'a'\n")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)

    body = write_review_bundle(tmp_path / "out", "T", "S", {"score": 1.0}, {"f.py": ("X = 'a'\n", "X = 'b'\n")})
    assert "| score | 1.0 |" in body.read_text()
    assert "+X = 'b'" in (tmp_path / "out" / "changes.diff").read_text()

    wt = create_branch(repo, "evolve/test", {"f.py": "X = 'b'\n"}, "evolve", tmp_path / "wt")
    assert (wt / "f.py").read_text() == "X = 'b'\n"
    log = subprocess.run(["git", "-C", str(repo), "log", "evolve/test", "--oneline"], capture_output=True, text=True).stdout
    assert "evolve" in log
    assert (repo / "f.py").read_text() == "X = 'a'\n"  # user's checkout untouched
