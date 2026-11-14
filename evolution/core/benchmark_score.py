"""Extract a numeric score from benchmark command output."""

import json
import re
from typing import Optional

_SCORE_RE = re.compile(r"^\s*SCORE\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
_PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed)")


def parse_score(output: str) -> Optional[float]:
    """Return a score from ``output``, or None.

    Rules in priority order: (a) last ``SCORE=<float>`` line, (b) last line that
    is a JSON object with a numeric ``"score"``, (c) pytest summary pass ratio from the last line that holds a
    ``<n> passed`` or ``<n> failed`` count.
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
        if not _PYTEST_COUNT_RE.search(line):
            continue  # e.g. "upload failed": mentions failed but is no summary

        def count(pattern: str) -> int:
            m = re.search(pattern, line)
            return int(m.group(1)) if m else 0

        passed = count(r"(\d+) passed")
        total = passed + count(r"(\d+) failed") + count(r"(\d+) errors?")
        return passed / total if total > 0 else None
    return None
