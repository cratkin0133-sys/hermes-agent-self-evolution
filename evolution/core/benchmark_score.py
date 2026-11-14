"""Extract a numeric score from benchmark command output."""

import json
import re
from typing import Optional

_SCORE_RE = re.compile(r"^\s*SCORE\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
_COUNT = r"\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)"
# A whole pytest summary: comma-separated "<n> <word>" counts, optionally "in 1.2s" / "in 61.02s (0:01:01)".
_PYTEST_SUMMARY_RE = re.compile(rf"^{_COUNT}(?:, {_COUNT})*(?: in \d+(?:\.\d+)?s(?: \([^)]*\))?)?$")
_PASS_FAIL_RE = re.compile(r"\d+ (?:passed|failed)\b")


def parse_score(output: str) -> Optional[float]:
    """Return a score from ``output``, or None.

    Rules in priority order: (a) last ``SCORE=<float>`` line, (b) last line that
    is a JSON object with a numeric ``"score"``, (c) pytest summary pass ratio
    from the last line that, stripped of ``=`` and whitespace, consists only of
    comma-separated ``<n> <word>`` counts (optionally ``in <t>s``) and has a
    passed or failed count.
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
        core = line.strip().strip("=").strip()
        if not (_PYTEST_SUMMARY_RE.match(core) and _PASS_FAIL_RE.search(core)):
            continue  # e.g. "upload failed" or "3 failed attempts": not a pytest summary

        def count(pattern: str) -> int:
            m = re.search(pattern, line)
            return int(m.group(1)) if m else 0

        passed = count(r"(\d+) passed")
        total = passed + count(r"(\d+) failed") + count(r"(\d+) errors?")
        return passed / total if total > 0 else None
    return None
