"""Discover hermes-agent system-prompt sections and score agent behaviour."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import dspy

from evolution.core.lm import complete
from evolution.core.multi_component import Rollout
from evolution.core.source_patch import StringLiteral, find_string_constants

PROMPT_BUILDER = Path("agent") / "prompt_builder.py"

# PLAN.md Phase 3: the hand-written, evolvable sections. User data, the
# auto-generated skills index, and project context files are never evolved.
EVOLVABLE_SECTIONS = (
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "PLATFORM_HINTS",
)


def discover_sections(
    hermes_repo: Path, requested: Optional[Iterable[str]] = None,
) -> tuple[dict[str, StringLiteral], dict[str, str]]:
    """Return evolvable literal sections keyed like ``SKILLS_GUIDANCE`` or ``PLATFORM_HINTS[cli]``.

    ``requested`` may name whole sections or single platform entries.
    """
    requested = list(requested or EVOLVABLE_SECTIONS)
    bases = {r.split("[")[0] for r in requested}
    bad = bases - set(EVOLVABLE_SECTIONS)
    if bad:
        raise ValueError(f"Not an evolvable section: {sorted(bad)} (choose from {EVOLVABLE_SECTIONS})")
    found, skipped = find_string_constants(hermes_repo / PROMPT_BUILDER, bases)
    keep = {k: v for k, v in found.items() if k in requested or k.split("[")[0] in requested}
    for r in requested:
        if "[" in r and r not in keep:
            skipped[r] = "no literal entry with that key"
    return keep, skipped


@dataclass
class Scenario:
    section: str          # component this scenario tests, e.g. "SESSION_SEARCH_GUIDANCE"
    user_message: str
    expected_behavior: str

    @classmethod
    def load_jsonl(cls, path: Path) -> list["Scenario"]:
        return [cls(**json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]

    @staticmethod
    def save_jsonl(items: list["Scenario"], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(asdict(s)) for s in items) + "\n")


def assemble_system_prompt(candidate: dict[str, str], scenario_section: str) -> str:
    """Shared sections + only the platform hint the scenario runs on (as hermes does)."""
    parts = [v for k, v in candidate.items() if not k.startswith("PLATFORM_HINTS[")]
    if scenario_section.startswith("PLATFORM_HINTS[") and scenario_section in candidate:
        parts.append(candidate[scenario_section])
    return "\n\n".join(parts)


class JudgeBehavior(dspy.Signature):
    """Judge whether an AI agent's reply shows the expected behaviour. Score 0.0-1.0
    and give specific feedback about what in the agent's *system prompt* would need
    to change for it to behave as expected."""
    user_message: str = dspy.InputField()
    expected_behavior: str = dspy.InputField()
    agent_reply: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0.0-1.0")
    feedback: str = dspy.OutputField()


class GenerateScenarios(dspy.Signature):
    """Write test scenarios for one section of an AI agent's system prompt. Each
    scenario is a user message that should trigger the behaviour the section asks
    for, plus a one-sentence rubric of the expected behaviour."""
    section_name: str = dspy.InputField()
    section_text: str = dspy.InputField()
    count: int = dspy.InputField()
    scenarios: list[dict[str, str]] = dspy.OutputField(desc='[{"user_message": ..., "expected_behavior": ...}]')


def make_behavior_run(agent_lm, judge_lm, limits: dict[str, int]):
    judge = dspy.Predict(JudgeBehavior)

    def run(candidate: dict[str, str], sc: Scenario) -> Rollout:
        system = assemble_system_prompt(candidate, sc.section)
        try:
            reply = complete(agent_lm, sc.user_message, system=system)
        except Exception as e:
            reply = f"<error: {type(e).__name__}>"
        with dspy.context(lm=judge_lm):
            try:
                verdict = judge(user_message=sc.user_message, expected_behavior=sc.expected_behavior, agent_reply=reply)
                score = min(1.0, max(0.0, float(verdict.score)))
                feedback = str(verdict.feedback)
            except Exception as e:
                score, feedback = 0.0, f"Judge failed ({type(e).__name__}); treat as failure."
        comps = [sc.section] if sc.section in candidate else []
        over = {c: len(candidate[c]) for c in comps if len(candidate[c]) > limits.get(c, 10**9)}
        if over:
            score *= 0.5
            feedback += " Section too long: " + ", ".join(f"{c} {n} chars (max {limits[c]})" for c, n in over.items()) + "."
        return Rollout(
            inputs=f"[{sc.section}] user: {sc.user_message}\nexpected: {sc.expected_behavior}",
            output=reply, score=score, feedback=feedback, components=comps,
        )

    return run


def generate_scenarios(candidate: dict[str, str], gen_lm, per_section: int = 8) -> list[Scenario]:
    gen = dspy.Predict(GenerateScenarios)
    out: list[Scenario] = []
    with dspy.context(lm=gen_lm):
        for name, text in candidate.items():
            for s in gen(section_name=name, section_text=text, count=per_section).scenarios[:per_section]:
                if s.get("user_message") and s.get("expected_behavior"):
                    out.append(Scenario(name, s["user_message"], s["expected_behavior"]))
    return out


def per_section_scores(scenarios: list[Scenario], rollouts: list[Rollout]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for s, r in zip(scenarios, rollouts):
        by.setdefault(s.section, []).append(r.score)
    return {k: sum(v) / len(v) for k, v in sorted(by.items())}
