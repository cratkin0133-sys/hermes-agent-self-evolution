"""Rank optimization targets by (potential improvement × usage frequency)."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from evolution.monitor.signals import Signal


@dataclass
class Target:
    phase: str
    target: str
    priority: float
    reason: str


def rank(signals: list[Signal], store, min_uses: int = 3, cooldown_days: int = 14) -> list[Target]:
    """Higher correction rate and more use → higher priority.

    The correction rate is smoothed (Laplace) so one unlucky use does not
    dominate; usage enters logarithmically so a popular target that is fine
    does not outrank a moderately used one that keeps getting corrected.
    Targets evolved (accepted or not) within ``cooldown_days`` are skipped.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=cooldown_days)
    ranked = []
    for s in signals:
        if s.uses < min_uses:
            continue
        last = store.last_run(s.phase, s.target) if store else None
        if last and datetime.fromisoformat(last["ts"]) > cutoff:
            continue
        rate = (s.corrections + 1) / (s.uses + 2)
        priority = rate * math.log1p(s.uses)
        ranked.append(Target(s.phase, s.target, round(priority, 4),
                             f"{s.uses} uses, {s.corrections} followed by a correction ({s.correction_rate:.0%})"))
    return sorted(ranked, key=lambda t: t.priority, reverse=True)
