"""Phase 2 end to end, offline: a fake hermes repo, scripted selector + reflector."""

import json
import re

import pytest

import evolution.core.multi_component as mc
import evolution.tools.evolve_tool_descriptions as etd
from evolution.tools.tool_descriptions import normalize_choice
from tests.fakes import fake_lm, xml_fields

TOOLS_SRC = '''\
READ_SCHEMA = {"name": "read_file", "description": "Read a file from disk with line numbers.", "parameters": {}}
TERM_SCHEMA = {"name": "terminal", "description": "Run any shell command, e.g. to build, test, or print file contents.", "parameters": {}}
'''


def selector(system, user):
    # Picks the first tool (alphabetical) whose description mentions the task's first word.
    task = re.search(r"<task>\n(.*?)\n</task>", user, re.S).group(1)
    word = task.split()[0].lower()
    tools = re.findall(r"^- (\w+): (.*)$", user, re.M)
    pick = next((n for n, d in sorted(tools) if word in d.lower()), "terminal")
    return xml_fields(tool_name=pick)


def reflector(system, user):
    current = user.split("```")[1].strip()
    if current.startswith("Read a file"):
        current += " Use it to show file contents instead of cat."
    return f"```\n{current}\n```"


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "file_tools.py").write_text(TOOLS_SRC)
    tasks = [{"task": f"contents of file{i}.py please", "tool": "read_file"} for i in range(14)]
    tasks += [{"task": f"build project {i}", "tool": "terminal"} for i in range(6)]
    ds = tmp_path / "golden.jsonl"
    ds.write_text("\n".join(json.dumps(t) for t in tasks))
    return tmp_path, ds


def test_phase2_evolves_descriptions_and_passes_gates(fake_repo, tmp_path, monkeypatch):
    repo, ds = fake_repo
    route = lambda model, **kw: fake_lm(reflector if "reflect" in model else selector, model)
    monkeypatch.setattr(etd, "make_lm", route)
    monkeypatch.setattr(mc, "make_lm", route)

    m = etd.evolve_tool_descriptions(
        iterations=4, dataset_path=str(ds), optimizer_model="openai/fake-reflect",
        eval_model="openai/fake-select", hermes_repo=str(repo), output_root=tmp_path / "out",
    )
    assert m["accepted"], m
    assert m["evolved_accuracy"] > m["baseline_accuracy"]
    assert m["changed_tools"] == ["read_file"]
    assert not m["regressions"]
    evolved_src = (tmp_path / "out").glob("tools/*/evolved/tools/file_tools.py").__next__().read_text()
    ns = {}
    exec(evolved_src, ns)
    assert "instead of cat" in ns["READ_SCHEMA"]["description"]
    assert ns["TERM_SCHEMA"]["description"].startswith("Run any shell command")


def test_normalize_choice_prefers_exact_and_longest():
    names = ["read_file", "file", "terminal"]
    assert normalize_choice("`read_file`", names) == "read_file"
    assert normalize_choice("I would use read_file here", names) == "read_file"
    assert normalize_choice("None", names) == "none"
