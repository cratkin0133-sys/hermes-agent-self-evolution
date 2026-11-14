"""Turn session history into per-skill / per-tool usage and correction counts.

Heuristic by design (no LLM calls, runs unattended): a *use* is a user message
that names the target; a *correction* is a use followed, in the same session,
by a user message that sounds like pushback ("no", "that's wrong", "use X
instead", ...). Good enough to rank what to look at; not a success metric.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CORRECTION = re.compile(
    r"^\s*(no|nope|wrong|stop)\b|that'?s (wrong|not (it|right|what))|not what i (asked|wanted|meant)"
    r"|(didn'?t|doesn'?t|does not|did not) work|try again|you (forgot|missed|ignored)|\binstead\b",
    re.I,
)


@dataclass
class Signal:
    phase: str      # "skill" | "tool"
    target: str
    uses: int
    corrections: int

    @property
    def correction_rate(self) -> float:
        return self.corrections / self.uses if self.uses else 0.0


def skill_names(hermes_repo: Path) -> list[str]:
    return sorted({p.parent.name for p in (hermes_repo / "skills").rglob("SKILL.md")})


def _pattern(name: str) -> re.Pattern:
    alts = {name, name.replace("-", " "), name.replace("_", " ")}
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True)) + r")(?![\w-])", re.I)


def compute_signals(messages: list[dict], targets: Iterable[str], phase: str) -> list[Signal]:
    """``messages`` use the importer schema: task_input, session_id, timestamp."""
    sessions: dict[str, list[dict]] = {}
    for m in messages:
        sessions.setdefault(m.get("session_id") or "_", []).append(m)
    for msgs in sessions.values():
        msgs.sort(key=lambda m: m.get("timestamp") or 0)
    out = []
    for target in targets:
        pat = _pattern(target)
        uses = corrections = 0
        for msgs in sessions.values():
            for i, m in enumerate(msgs):
                if pat.search(m.get("task_input", "")):
                    uses += 1
                    nxt = msgs[i + 1]["task_input"] if i + 1 < len(msgs) else ""
                    if CORRECTION.search(nxt):
                        corrections += 1
        if uses:
            out.append(Signal(phase, target, uses, corrections))
    return out


def load_messages(sources: Iterable[str]) -> list[dict]:
    from evolution.core.external_importers import ClaudeCodeImporter, CopilotImporter, HermesSessionImporter
    importers = {"claude-code": ClaudeCodeImporter, "copilot": CopilotImporter, "hermes": HermesSessionImporter}
    msgs: list[dict] = []
    for s in sources:
        try:
            msgs += importers[s].extract_messages()
        except Exception:
            continue  # a missing/corrupt history source should not stop the loop
    return msgs
