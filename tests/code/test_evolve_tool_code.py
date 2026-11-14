"""Phase 4 end to end, offline: real git repo, real pytest, scripted mutator."""

import subprocess
import sys

import pytest

import evolution.code.evolve_tool_code as etc
from evolution.code.code_guards import check_guards
from evolution.code.mutator import apply_blocks, parse_mutation
from tests.fakes import fake_lm

TOOL = '''\
class _Registry:
    def register(self, **kw):
        pass


registry = _Registry()


def safe_div(a, b):
    """Divide a by b; return None when b is zero."""
    if not isinstance(a, (int, float)):
        raise TypeError("a must be a number")
    return a / b


registry.register(name="safe_div", handler=safe_div)
'''
TESTS = '''\
import pytest
from tools.calc_tools import safe_div

def test_divides():
    assert safe_div(6, 3) == 2

def test_rejects_non_numbers():
    with pytest.raises(TypeError):
        safe_div("6", 3)
'''

UNSAFE = '''The type check is in the way.
<<<<<<< SEARCH
    if not isinstance(a, (int, float)):
        raise TypeError("a must be a number")
    return a / b
=======
    return a / b if b else None
>>>>>>> REPLACE'''
GOOD = '''Division by zero is not handled; return None as documented.
<<<<<<< SEARCH
    return a / b
=======
    if b == 0:
        return None
    return a / b
>>>>>>> REPLACE'''


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "hermes"
    (r / "tools").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "tools" / "__init__.py").write_text("")
    (r / "tools" / "calc_tools.py").write_text(TOOL)
    (r / "tests" / "test_calc.py").write_text(TESTS)
    for c in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(["git", "-C", str(r), *c], check=True)
    return r


def test_phase4_rejects_unsafe_mutation_then_finds_fix(repo, tmp_path, monkeypatch):
    replies = iter([UNSAFE, GOOD])
    monkeypatch.setattr(etc, "make_lm", lambda model, **kw: fake_lm(lambda s, u: next(replies)))
    repro = f'{sys.executable} -c "from tools.calc_tools import safe_div; assert safe_div(1, 0) is None"'
    m = etc.evolve_tool_code(
        repro=repro, bug="safe_div raises ZeroDivisionError instead of returning None",
        tool="safe_div", test_cmd=f"{sys.executable} -m pytest tests -q -p no:cacheprovider",
        iterations=4, hermes_repo=str(repo), output_root=tmp_path / "out", create_pr_branch=True,
    )
    assert m["accepted"], m
    assert m["organisms_evaluated"] == 3  # baseline, unsafe (rejected), good
    log = next((tmp_path / "out").glob("code/*/evolution_log.txt")).read_text()
    assert "fewer raise statements" in log
    diff = next((tmp_path / "out").glob("code/*/changes.diff")).read_text()
    assert "+    if b == 0:" in diff
    assert (repo / "tools" / "calc_tools.py").read_text() == TOOL  # user checkout untouched
    branches = subprocess.run(["git", "-C", str(repo), "branch"], capture_output=True, text=True).stdout
    assert "evolve/code-calc_tools" in branches


def test_guards():
    assert check_guards(TOOL, TOOL) == []
    assert any("signature changed" in p for p in check_guards(TOOL, TOOL.replace("def safe_div(a, b)", "def safe_div(a, b, c=1)")))
    assert any("registry" in p for p in check_guards(TOOL, TOOL.replace('name="safe_div"', 'name="div"')))
    assert any("syntax" in p for p in check_guards(TOOL, TOOL + "\ndef broken(:\n"))
    assert check_guards(TOOL, TOOL + "\n\ndef helper(x):\n    return x\n") == []


def test_apply_blocks_requires_unique_exact_match():
    m = parse_mutation(GOOD)
    assert len(m.blocks) == 1 and m.diagnosis.startswith("Division")
    assert "if b == 0" in apply_blocks(TOOL, m.blocks)
    with pytest.raises(ValueError):
        apply_blocks(TOOL, [("not in file", "x")])
