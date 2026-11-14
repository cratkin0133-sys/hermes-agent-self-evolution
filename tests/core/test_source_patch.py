import ast
from pathlib import Path

import pytest

from evolution.core.source_patch import find_schema_descriptions, find_string_constants, replace_literals

SRC = '''\
IDENTITY = (
    "You are Hermes. "
    "Be direct."
)
BUILT = build("x")
HINTS = {"cli": "No markdown.", "telegram": "Use formatting ✨."}

READ_SCHEMA = {
    "name": "read_file",
    "description": "Read a file. Use instead of cat.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path"}}},
}
SEARCH_SCHEMA = {"name": "search_files", "description": "Search. " + NOTE, "parameters": {}}
'''


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text(SRC)
    return p


def test_finds_constants_and_dict_entries(src):
    found, skipped = find_string_constants(src, ["IDENTITY", "BUILT", "HINTS", "MISSING"])
    assert found["IDENTITY"].value == "You are Hermes. Be direct."
    assert found["HINTS[telegram]"].value == "Use formatting ✨."
    assert "BUILT" in skipped and "MISSING" in skipped


def test_finds_literal_schema_descriptions_and_skips_computed(src):
    found, skipped = find_schema_descriptions(src)
    assert found["read_file"].value == "Read a file. Use instead of cat."
    assert "search_files" in skipped


def test_replace_round_trips_and_preserves_rest(src):
    consts, _ = find_string_constants(src, ["IDENTITY", "HINTS"])
    schemas, _ = find_schema_descriptions(src)
    new = replace_literals(src.read_text(), {
        consts["IDENTITY"]: "You are Hermes.\nBe direct and honest.",
        consts["HINTS[telegram]"]: 'Use "rich" ✨ formatting.',
        schemas["read_file"]: "Read a text file with line numbers.",
    })
    ns = {"build": lambda x: x, "NOTE": "n"}
    exec(compile(new, "m", "exec"), ns)
    assert ns["IDENTITY"] == "You are Hermes.\nBe direct and honest."
    assert ns["HINTS"] == {"cli": "No markdown.", "telegram": 'Use "rich" ✨ formatting.'}
    assert ns["READ_SCHEMA"]["description"] == "Read a text file with line numbers."
    assert ns["READ_SCHEMA"]["parameters"]["properties"]["path"]["description"] == "Path"
    assert 'BUILT = build("x")' in new


def test_multiline_rewrite_keeps_comments_and_parens(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('X = (\n    "alpha "\n    # keep me\n    "beta"\n)\n')
    lit = find_string_constants(p, ["X"])[0]["X"]
    new = replace_literals(p.read_text(), {lit: "alpha beta " + "gamma " * 30})
    assert "# keep me" in new
    assert "((" not in new and new.count("(") == 1
    assert max(len(l) for l in new.splitlines()) <= 100
    ns = {}
    exec(new, ns)
    assert ns["X"] == "alpha beta " + "gamma " * 30
