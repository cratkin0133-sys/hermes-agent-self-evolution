"""LLM mutator: diagnose a failure and propose SEARCH/REPLACE edits to one file."""

import re
from dataclasses import dataclass

from evolution.core.lm import complete

SYSTEM = """You are a senior Python engineer fixing a bug in one file of the Hermes Agent tools package.
Rules:
- Make the smallest change that fixes the failure. Do not refactor unrelated code.
- Never change an existing function's signature, never edit registry.register(...) calls,
  never remove error handling, validation or safety checks.
- Reply with a short diagnosis, then one or more edit blocks in exactly this format:

<<<<<<< SEARCH
exact lines copied from the current file
=======
replacement lines
>>>>>>> REPLACE

SEARCH text must match the current file exactly (including indentation) and be unique."""

BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.S)


@dataclass
class Mutation:
    diagnosis: str
    blocks: list[tuple[str, str]]


def parse_mutation(text: str) -> Mutation:
    blocks = [(s, r) for s, r in BLOCK.findall(text)]
    diagnosis = text.split("<<<<<<< SEARCH")[0].strip()
    return Mutation(diagnosis=diagnosis[:2000], blocks=blocks)


def apply_blocks(source: str, blocks: list[tuple[str, str]]) -> str:
    if not blocks:
        raise ValueError("no edit blocks in reply")
    for search, replace in blocks:
        n = source.count(search)
        if n != 1:
            raise ValueError(f"SEARCH block matched {n} times (must be exactly 1): {search[:80]!r}")
        source = source.replace(search, replace, 1)
    return source


def build_prompt(rel_path: str, source: str, bug: str, report: str, history: list[str]) -> str:
    hist = "\n".join(f"- {h}" for h in history[-6:]) or "- (none yet)"
    return (
        f"## Bug\n{bug}\n\n## Latest evaluation of the current version\n{report}\n\n"
        f"## Earlier attempts (learn from these)\n{hist}\n\n## Current file: {rel_path}\n```python\n{source}\n```"
    )


def propose(lm, rel_path: str, source: str, bug: str, report: str, history: list[str]) -> Mutation:
    return parse_mutation(complete(lm, build_prompt(rel_path, source, bug, report, history), system=SYSTEM))
