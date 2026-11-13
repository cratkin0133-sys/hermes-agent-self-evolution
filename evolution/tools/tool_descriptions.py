"""Discover tool descriptions in hermes-agent and score tool selection."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import dspy

from evolution.core.multi_component import Rollout
from evolution.core.source_patch import StringLiteral, find_schema_descriptions

NO_TOOL = "none"


def discover_tool_descriptions(
    hermes_repo: Path, only: Optional[Iterable[str]] = None,
) -> tuple[dict[str, StringLiteral], dict[str, str]]:
    """Find literal top-level tool descriptions under ``tools/``.

    Parameter descriptions and schema structure stay frozen; only the top-level
    ``description`` text is evolvable.
    """
    wanted = set(only) if only else None
    found: dict[str, StringLiteral] = {}
    skipped: dict[str, str] = {}
    for path in sorted((hermes_repo / "tools").rglob("*.py")):
        if "test" in path.name:
            continue
        try:
            f, s = find_schema_descriptions(path)
        except SyntaxError:
            continue
        for name, lit in f.items():
            if wanted is None or name in wanted:
                found.setdefault(name, lit)
        skipped.update({k: v for k, v in s.items() if wanted is None or k in wanted})
    if wanted:
        for name in wanted - set(found) - set(skipped):
            skipped[name] = "no literal schema found under tools/"
    return found, skipped


@dataclass
class ToolTask:
    task: str
    tool: str  # correct tool name, or NO_TOOL

    @classmethod
    def load_jsonl(cls, path: Path) -> list["ToolTask"]:
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return [cls(task=r["task"], tool=r["tool"]) for r in rows]

    @staticmethod
    def save_jsonl(tasks: list["ToolTask"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(asdict(t)) for t in tasks) + "\n")


def catalog(candidate: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in sorted(candidate.items()))


class SelectTool(dspy.Signature):
    """You are an AI agent choosing which tool to call. Read the tool catalog and
    pick the single best tool for the task. If the task needs no tool, answer "none".
    Answer with the exact tool name only."""
    task: str = dspy.InputField()
    tool_catalog: str = dspy.InputField()
    tool_name: str = dspy.OutputField(desc="Exact tool name from the catalog, or none")


class GenerateToolTasks(dspy.Signature):
    """Write realistic user requests for an AI coding agent where the target tool is
    clearly the best choice among the catalog. Include some requests where a
    different, more generic tool (e.g. a shell) could also work but the target tool
    is better. One request per line, no numbering."""
    tool_catalog: str = dspy.InputField()
    target_tool: str = dspy.InputField()
    count: int = dspy.InputField()
    requests: list[str] = dspy.OutputField()


def normalize_choice(raw: str, names: Iterable[str]) -> str:
    text = (raw or "").strip().strip("`'\" .").lower()
    names = list(names)
    if text in (n.lower() for n in names):
        return next(n for n in names if n.lower() == text)
    for n in sorted(names, key=len, reverse=True):  # longest match wins (read_file vs file)
        if re.search(rf"\b{re.escape(n.lower())}\b", text):
            return n
    return NO_TOOL if NO_TOOL in text or not text else text


def make_selection_run(selector_lm, limits: dict[str, int]):
    """Build run(candidate, ToolTask) -> Rollout. ``limits`` caps each description."""
    predict = dspy.Predict(SelectTool)

    def run(candidate: dict[str, str], ex: ToolTask) -> Rollout:
        with dspy.context(lm=selector_lm):
            try:
                picked = normalize_choice(predict(task=ex.task, tool_catalog=catalog(candidate)).tool_name, candidate)
            except Exception as e:  # unparseable reply counts as a miss
                picked = f"<error: {type(e).__name__}>"
        correct = picked == ex.tool
        score = 1.0 if correct else 0.0
        if correct:
            feedback = f"Correct: chose {ex.tool}."
            comps = [ex.tool] if ex.tool in candidate else []
        else:
            feedback = (
                f"Wrong tool. Task: {ex.task!r}. Correct: {ex.tool}. Agent chose: {picked}. "
                f"Make the {ex.tool} description clearly claim tasks like this"
                + (f", and make the {picked} description say when NOT to use it." if picked in candidate else ".")
            )
            comps = [c for c in (ex.tool, picked) if c in candidate]
        over = {t: len(candidate[t]) for t in comps if len(candidate[t]) > limits.get(t, 10**9)}
        if over:
            score *= 0.5
            feedback += " Too long: " + ", ".join(f"{t} is {n} chars (max {limits[t]})" for t, n in over.items()) + "."
        return Rollout(inputs=ex.task, output=picked, score=score, feedback=feedback, components=comps)

    return run


def generate_tasks(candidate: dict[str, str], gen_lm, per_tool: int = 8, no_tool: int = 4) -> list[ToolTask]:
    gen = dspy.Predict(GenerateToolTasks)
    cat = catalog(candidate)
    tasks: list[ToolTask] = []
    with dspy.context(lm=gen_lm):
        for tool in sorted(candidate):
            reqs = gen(tool_catalog=cat, target_tool=tool, count=per_tool).requests
            tasks += [ToolTask(r.strip(), tool) for r in reqs[:per_tool] if r and r.strip()]
    tasks += [ToolTask(t, NO_TOOL) for t in [
        "What's the difference between a list and a tuple in Python?",
        "Thanks, that's all for now.",
        "Explain what a race condition is in one paragraph.",
        "Suggest a good name for a CLI that converts CSV to JSON.",
    ][:no_tool]]
    return tasks


def per_tool_accuracy(tasks: list[ToolTask], rollouts: list[Rollout]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for t, r in zip(tasks, rollouts):
        by.setdefault(t.tool, []).append(r.output == t.tool)
    return {k: sum(v) / len(v) for k, v in sorted(by.items())}
