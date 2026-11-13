"""Thin helpers over dspy.LM so every phase builds and calls models one way.

Tests swap ``make_lm`` (or pass ``engine=``) to run fully offline.
"""

from typing import Callable, Optional

import dspy
from dspy.lm15 import Request
from dspy._vendor.lm15 import types as _t


def make_lm(model: str, **kwargs) -> dspy.LM:
    """Build a dspy.LM. Single seam for tests and future provider tweaks."""
    return dspy.LM(model, **kwargs)


def complete(lm: dspy.LM, user: str, system: Optional[str] = None) -> str:
    """One chat completion with an optional real system message; returns text."""
    req = Request(
        model=lm.model,
        system=system,
        messages=(_t.Message(role="user", parts=(_t.TextPart(text=user),)),),
    )
    resp = lm(req)
    return "".join(getattr(p, "text", "") for p in resp.message.parts).strip()


def as_reflection_lm(lm: dspy.LM) -> Callable[[object], str]:
    """Adapt a dspy.LM to gepa's LanguageModel protocol: prompt (str|messages) -> str."""

    def _call(prompt) -> str:
        if isinstance(prompt, str):
            return complete(lm, prompt)
        system = "\n".join(m["content"] for m in prompt if m.get("role") == "system") or None
        user = "\n\n".join(m["content"] for m in prompt if m.get("role") != "system")
        return complete(lm, user, system=system)

    return _call
