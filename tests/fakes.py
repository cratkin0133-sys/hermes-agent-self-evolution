"""Offline scripted LMs for tests. ``script(system, user) -> str`` decides each reply."""

from typing import Callable, Optional

import dspy
from dspy._vendor.lm15 import types as _t

Script = Callable[[Optional[str], str], str]


def _text_of(parts) -> str:
    return "\n".join(getattr(p, "text", "") or "" for p in parts or ())


class ScriptedEngine:
    def __init__(self, script: Script):
        self.script = script
        self.calls: list[tuple[Optional[str], str]] = []

    def complete(self, req):
        system = req.system if isinstance(req.system, str) else (_text_of(req.system) or None)
        user = "\n".join(_text_of(m.parts) for m in req.messages)
        self.calls.append((system, user))
        out = self.script(system, user)
        return _t.Response(
            id="fake", model=req.model,
            message=_t.Message(role="assistant", parts=(_t.TextPart(text=out),)),
            finish_reason="stop", usage=_t.Usage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def fake_lm(script: Script, model: str = "openai/fake") -> dspy.LM:
    lm = dspy.LM(model, engine=ScriptedEngine(script), cache=False, num_retries=0)
    return lm


def xml_fields(**fields: str) -> str:
    """Reply in DSPy 3.5's default XML field format."""
    return "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in fields.items())
