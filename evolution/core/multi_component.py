"""Evolve several named text components *together* with GEPA.

Phase 1 wraps one skill as a DSPy predictor. Phases 2 and 3 instead have many
texts that only make sense as a set (all tool descriptions compete for the same
selections; all prompt sections form one system prompt). The ``gepa`` library
supports that natively: a candidate is ``dict[name -> text]`` and an adapter
decides how a whole candidate is scored and which examples inform each
component's rewrite. This module is that adapter, kept generic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import gepa
from gepa.core.adapter import EvaluationBatch

from evolution.core.lm import as_reflection_lm, make_lm


@dataclass
class Rollout:
    """Result of running one example under one candidate."""
    inputs: str                     # what the example asked (shown to the reflection LM)
    output: str                     # what the system did
    score: float                    # 0..1
    feedback: str                   # why — this is what GEPA learns from
    components: list[str] = field(default_factory=list)  # components this example informs; [] = all


RunFn = Callable[[dict[str, str], Any], Rollout]


class CallbackAdapter:
    """GEPAAdapter that delegates scoring to a plain ``run(candidate, example)`` function."""

    def __init__(self, run: RunFn):
        self.run = run
        # gepa's protocol attribute; None lets gepa use its default proposer.
        self.propose_new_texts = None

    def evaluate(self, batch: list, candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        rollouts = [self.run(candidate, ex) for ex in batch]
        return EvaluationBatch(
            outputs=[r.output for r in rollouts],
            scores=[r.score for r in rollouts],
            trajectories=rollouts if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rollouts: list[Rollout] = eval_batch.trajectories or []
        out = {}
        for comp in components_to_update:
            relevant = [r for r in rollouts if not r.components or comp in r.components] or rollouts
            out[comp] = [
                {"Inputs": r.inputs, "Generated Outputs": r.output, "Feedback": r.feedback}
                for r in relevant
            ]
        return out


def evaluate_candidate(run: RunFn, candidate: dict[str, str], examples: list) -> list[Rollout]:
    return [run(candidate, ex) for ex in examples]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def optimize_components(
    seed: dict[str, str],
    trainset: list,
    valset: list,
    run: RunFn,
    reflection_model: str,
    max_metric_calls: int,
    run_dir: Optional[Path] = None,
    reflection_lm=None,
) -> dict[str, str]:
    """Run gepa.optimize and return the best candidate found (seed if none better)."""
    lm = reflection_lm or as_reflection_lm(make_lm(reflection_model, temperature=1.0, max_tokens=16000))
    result = gepa.optimize(
        seed_candidate=dict(seed),
        trainset=trainset,
        valset=valset,
        adapter=CallbackAdapter(run),
        reflection_lm=lm,
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir) if run_dir else None,
        display_progress_bar=False,
        raise_on_exception=False,
    )
    return dict(result.best_candidate)


def split(items: list, ratios=(0.6, 0.2, 0.2), seed: int = 0) -> tuple[list, list, list]:
    """Deterministic shuffle + train/val/holdout split; every split gets >= 1 item when possible."""
    import random
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n - n_train > 1 else 0
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]
