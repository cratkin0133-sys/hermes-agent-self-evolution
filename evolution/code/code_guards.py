"""Static safety guards for evolved tool code (PLAN Phase 4 constraints).

Rejects a mutation that changes an existing function signature, touches a
``registry.register(...)`` call, or reduces error handling / safety checks.
New helper functions are allowed.
"""

import ast
from collections import Counter


def _qualname_defs(tree: ast.AST) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    out[name] = child
                walk(child, name + ".")
    walk(tree)
    return out


def signatures(source: str) -> dict[str, str]:
    sigs = {}
    for name, fn in _qualname_defs(ast.parse(source)).items():
        ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
        prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
        sigs[name] = f"{prefix}({ast.unparse(fn.args)}){ret}"
    return sigs


def register_calls(source: str) -> list[str]:
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "registry":
                calls.append(ast.unparse(node))
    return sorted(calls)


def safety_counts(source: str) -> Counter:
    kinds = {ast.Try: "try", ast.ExceptHandler: "except", ast.Raise: "raise", ast.Assert: "assert"}
    if hasattr(ast, "TryStar"):
        kinds[ast.TryStar] = "try"
    c = Counter()
    for node in ast.walk(ast.parse(source)):
        for k, label in kinds.items():
            if isinstance(node, k):
                c[label] += 1
    return c


def check_guards(old: str, new: str) -> list[str]:
    """Return a list of violations (empty = safe)."""
    try:
        ast.parse(new)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    problems = []
    old_sigs, new_sigs = signatures(old), signatures(new)
    for name, sig in old_sigs.items():
        if name not in new_sigs:
            problems.append(f"function removed: {name}")
        elif new_sigs[name] != sig:
            problems.append(f"signature changed: {name}{sig} -> {name}{new_sigs[name]}")
    if register_calls(old) != register_calls(new):
        problems.append("registry.register(...) calls changed")
    oc, nc = safety_counts(old), safety_counts(new)
    for k in ("try", "except", "raise", "assert"):
        if nc[k] < oc[k]:
            problems.append(f"fewer {k} statements: {oc[k]} -> {nc[k]}")
    return problems
