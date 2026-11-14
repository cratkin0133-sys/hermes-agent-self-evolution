"""Extract a numeric score from benchmark command output."""

import json
import re
from typing import Optional

_SCORE_RE = re.compile(r"^\s*SCORE\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def parse_score(output: str) -> Optional[float]:
    """Return a score from ``output``, or None.

    Rules in priority order: (a) last ``SCORE=<float>`` line, (b) last line that
    is a JSON object with a numeric ``"score"``, (c) pytest summary pass ratio.
    """
    lines = output.splitlines()

    for line in reversed(lines):
        m = _SCORE_RE.match(line)
        if m:
            return float(m.group(1))

    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            score = obj.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return float(score)

    for line in reversed(lines):
        if "passed" in line or "failed" in line:
            def count(pattern: str) -> int:
                m = re.search(pattern, line)
                return int(m.group(1)) if m else 0

            passed = count(r"(\d+) passed")
            total = passed + count(r"(\d+) failed") + count(r"(\d+) errors?")
            return passed / total if total > 0 else None
    return None
