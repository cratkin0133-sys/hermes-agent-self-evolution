"""Locate and rewrite string literals in Python source without executing it.

Used by Phase 2 (tool schema descriptions) and Phase 3 (prompt-section
constants). Only *literal* strings are handled: an implicit concatenation of
literals counts (the parser folds it into one constant), but f-strings, calls,
or ``"a" + VAR`` expressions are reported as not evolvable rather than guessed.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class StringLiteral:
    """A string constant and its exact location in a source file."""
    file: Path
    key: str            # e.g. "MEMORY_GUIDANCE", "PLATFORM_HINTS[cli]", "read_file"
    value: str
    lineno: int         # 1-based, as reported by ast
    col: int            # UTF-8 byte offset
    end_lineno: int
    end_col: int

    @property
    def multiline_source(self) -> bool:
        return self.end_lineno > self.lineno


def _is_str(node: Optional[ast.AST]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _lit(path: Path, key: str, node: ast.Constant) -> StringLiteral:
    return StringLiteral(path, key, node.value, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _module_assigns(tree: ast.Module) -> Iterable[tuple[str, ast.AST]]:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            yield stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            yield stmt.target.id, stmt.value


def find_string_constants(path: Path, names: Optional[Iterable[str]] = None) -> tuple[dict[str, StringLiteral], dict[str, str]]:
    """Module-level ``NAME = "..."`` constants and ``NAME = {"k": "..."}`` dict entries.

    Returns (found, skipped) where skipped maps a requested name to the reason
    it cannot be evolved (e.g. it is built by a function call).
    """
    wanted = set(names) if names is not None else None
    tree = ast.parse(Path(path).read_text())
    found: dict[str, StringLiteral] = {}
    skipped: dict[str, str] = {}
    for name, value in _module_assigns(tree):
        if wanted is not None and name not in wanted:
            continue
        if _is_str(value):
            found[name] = _lit(path, name, value)
        elif isinstance(value, ast.Dict):
            entries = [(k.value, v) for k, v in zip(value.keys, value.values) if _is_str(k) and _is_str(v)]
            for k, v in entries:
                found[f"{name}[{k}]"] = _lit(path, f"{name}[{k}]", v)
            if not entries and wanted is not None:
                skipped[name] = "dict has no literal string values"
        elif wanted is not None:
            skipped[name] = f"not a string literal ({type(value).__name__})"
    if wanted is not None:
        for name in wanted - {k.split("[")[0] for k in found} - set(skipped):
            skipped[name] = "not defined at module level"
    return found, skipped


def find_schema_descriptions(path: Path) -> tuple[dict[str, StringLiteral], dict[str, str]]:
    """Tool schemas: module-level dict literals with literal ``"name"`` and ``"description"``.

    Returns (found keyed by tool name, skipped tool name -> reason).
    """
    tree = ast.parse(Path(path).read_text())
    found: dict[str, StringLiteral] = {}
    skipped: dict[str, str] = {}
    for _, value in _module_assigns(tree):
        if not isinstance(value, ast.Dict):
            continue
        fields = {k.value: v for k, v in zip(value.keys, value.values) if _is_str(k)}
        if not _is_str(fields.get("name")) or "description" not in fields:
            continue
        tool = fields["name"].value
        desc = fields["description"]
        if _is_str(desc):
            found[tool] = _lit(path, tool, desc)
        else:
            skipped[tool] = "description is computed, not a literal"
    return found, skipped


def format_literal(value: str, multiline: bool, indent: str = "    ") -> str:
    """Render a Python string literal. Multi-line sources become a parenthesized
    implicit concatenation, one chunk per line, so diffs stay readable."""
    if not multiline:
        return json.dumps(value, ensure_ascii=False)
    chunks = value.splitlines(keepends=True) or [""]
    body = "\n".join(f"{indent}{json.dumps(c, ensure_ascii=False)}" for c in chunks)
    return f"(\n{body}\n)"


def replace_literals(source: str, replacements: dict[StringLiteral, str]) -> str:
    """Replace each literal's exact source span with a new literal of ``value``.

    Verifies by re-parsing that every replaced constant now holds the new value.
    """
    data = source.encode()
    starts = [0]
    for line in data.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def off(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    spans = sorted(replacements.items(), key=lambda kv: off(kv[0].lineno, kv[0].col), reverse=True)
    for lit, new_value in spans:
        a, b = off(lit.lineno, lit.col), off(lit.end_lineno, lit.end_col)
        line_start = data.rfind(b"\n", 0, a) + 1
        indent = " " * (len(data[line_start:a].decode()) - len(data[line_start:a].decode().lstrip()) + 4)
        data = data[:a] + format_literal(new_value, lit.multiline_source, indent).encode() + data[b:]
    new_source = data.decode()

    tree = ast.parse(new_source)
    values = {n.value for n in ast.walk(tree) if _is_str(n)}
    missing = [lit.key for lit, v in replacements.items() if v not in values]
    if missing:
        raise ValueError(f"Rewrite verification failed for: {missing}")
    return new_source
