# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill requesting-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill requesting-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | GEPA (multi-component) | ✅ Implemented |
| **Phase 3** | System prompt sections | GEPA (multi-component) | ✅ Implemented |
| **Phase 4** | Tool implementation code | Built-in evolutionary loop | ✅ Implemented |
| **Phase 5** | Continuous improvement loop | Monitor + triage + scheduler | ✅ Implemented |

Every phase writes a review bundle (`output/<phase>/<timestamp>/PR_BODY.md`,
`changes.diff`, `metrics.json`, evolved files). With `--create-branch`, an accepted
result is also committed on a new branch in a **git worktree** of hermes-agent; your
checkout is never modified and nothing is pushed.

### Phase 1 — skills

```bash
python -m evolution.skills.evolve_skill --skill requesting-code-review --iterations 10
python -m evolution.skills.evolve_skill --skill arxiv --metric heuristic       # no judge calls
python -m evolution.skills.evolve_skill --skill arxiv --judge-model openai/gpt-4.1-mini
```
`--metric judge` (default) scores every rollout with an LLM judge (correctness, procedure
following, conciseness, plus written feedback that GEPA reflects on); `--metric heuristic`
uses keyword overlap with the expected behaviour instead. The judge defaults to the
**optimizer model** (`--optimizer-model`), so it is one strong-model call per scored
example during optimization and holdout: pass a cheaper `--judge-model` to cut cost. If the
judge's first call fails, the run stops with an error (check `--judge-model` and its API
key). Later judge failures fall back to the heuristic for that example and are counted in
`metrics.json` as `judge_fallbacks`; a run that needed a fallback while scoring the holdout
set is not accepted.

### Benchmark score gate (Phases 2 and 3)

`--benchmark-cmd` runs a command twice in throwaway worktrees of hermes-agent: once
unchanged (baseline) and once with the evolved files applied. A score is read from the
combined stdout/stderr, first match wins:
1. the last `SCORE=<n>` (or `SCORE: <n>`) line;
2. the last line that is a JSON object with a numeric `"score"`, e.g. `{"score": 0.83}`;
3. the last pytest summary line with a count (`8 passed, 2 failed` → 0.8; errors count as failures).

The gate fails when the evolved score is more than `--benchmark-max-regression` below the
baseline. If neither run prints a score, exit codes are compared (an already-failing
baseline cannot be regressed). A run that exceeds `--benchmark-timeout` seconds (default
3600) or whose worktree cannot be set up makes the result **inconclusive, which fails the
gate**. The benchmark is the most expensive gate, so it only runs once every other gate
has passed.

### Phase 2 — tool descriptions

```bash
python -m evolution.tools.evolve_tool_descriptions \
    --tools read_file,write_file,patch --iterations 5          # synthetic tasks
python -m evolution.tools.evolve_tool_descriptions \
    --dataset my_tasks.jsonl --create-branch                     # {"task": ..., "tool": ...}
```
All selected descriptions evolve **together** against the full catalog. Accepted only if
holdout selection accuracy improves, **no tool's** accuracy drops, and each description
stays ≤500 chars (or its current size, if already longer). Only literal `description`
strings are evolved; computed ones (e.g. `"..." + NOTE`) are reported and skipped, and
schema structure and parameter descriptions are frozen. Optional benchmark gate:
`--benchmark-cmd`, `--benchmark-max-regression` (default 0.02), `--benchmark-timeout`
(see above).

### Phase 3 — system prompt sections

```bash
python -m evolution.prompts.evolve_prompt_section --section SESSION_SEARCH_GUIDANCE
python -m evolution.prompts.evolve_prompt_section --section "PLATFORM_HINTS[cli]" \
    --must-keep "plain text" --benchmark-cmd "python -m pytest tests/agent -q"
```
Evolvable: `DEFAULT_AGENT_IDENTITY`, `SESSION_SEARCH_GUIDANCE`, `SKILLS_GUIDANCE`,
`PLATFORM_HINTS[...]` (`MEMORY_GUIDANCE` is built by a function call, so it is skipped).
An agent model replies under the assembled system prompt and a judge model scores each
behavioural scenario. Gates: ±size limits (`--max-growth`, default +20%; shrink floor
50%), `--must-keep` phrases, no per-section regression, and the optional benchmark gate:
`--benchmark-cmd`, `--benchmark-max-regression` (default 0: no score drop allowed),
`--benchmark-timeout` (see above).

### Phase 4 — tool code

```bash
python -m evolution.code.evolve_tool_code --tool read_file \
    --bug "read_file crashes on empty files" \
    --repro "python -m pytest tests/tools/test_empty_file_bug.py -q" \
    --test-cmd "python -m pytest tests/tools -q -x" --iterations 10
```
`--repro` must fail now and pass once fixed. An LLM proposes SEARCH/REPLACE edits from
the latest failure output; candidates are scored in a detached worktree. Static guards
reject any change to existing function signatures or `registry.register(...)` calls,
or any reduction in `try`/`except`/`raise`/`assert`.

Instead of `--bug`, take the bug text from a GitHub issue:
```bash
python -m evolution.code.evolve_tool_code --tool read_file --bug-issue 123 \
    --issue-repo NousResearch/hermes-agent --repro "python -m pytest tests/tools/test_bug.py -q"
```
`--issue-repo` defaults to `NousResearch/hermes-agent`; set `GITHUB_TOKEN` for private
repos or higher rate limits. The issue title, body and URL are used, and the review bundle
links the issue.

**Issue text is untrusted input.** Anyone can open an issue, and its text steers code that
an LLM writes and that is then **executed on your machine** (repro and test commands run
the candidate file). Protections:
- only comments by the repo's owner, members and collaborators are used; `--all-comments`
  includes everyone's;
- the mutator prompt wraps the bug in a block marked as untrusted data, not instructions;
- repro/test commands run without `GITHUB_TOKEN`, `GH_TOKEN` or any `*_API_KEY` variable;
- static guards and human review of every diff before merge.

Residual risk: the issue body itself (written by whoever opened it) still reaches the LLM,
and prompt-injection defences are not watertight; the candidate code runs with your user's
file-system and network access, and other secrets in your environment or on disk are
reachable. Use `--bug-issue` only for issues you have read, and prefer a sandboxed
machine or container.

### Phase 5 — continuous loop

```bash
python -m evolution.monitor.loop collect     # usage + correction signals from session history
python -m evolution.monitor.loop triage      # ranked targets
python -m evolution.monitor.loop run --max-targets 1 --create-branch
python -m evolution.monitor.loop bench --cmd "python -m pytest tests -q"   # record trend
python -m evolution.monitor.loop cron        # prints a weekly crontab line
```
`bench` records the command's exit status and score (same score convention as the
benchmark gate above) and exits 1 when the command fails or the score dropped more than
`--max-regression` (default 0.02) below the baseline. The baseline is the **last accepted
scored run**: a regressed or failed run is recorded as not accepted, so it never becomes
the new baseline.
Signals and runs are stored locally in `~/.hermes/self-evolution/metrics.db`. The loop
automates detection and optimization only; a human reviews and merges.

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| Built-in evolutionary loop (`evolution/code`) | Population + failure-driven LLM mutation + test fitness, inspired by [Darwin Gödel Machines](https://arxiv.org/abs/2505.22954). [Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver) is not used: its CLI only runs problems registered inside its own AGPL package. | MIT |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars, prompt sections ≤+20%; nothing may shrink below 50% of its original size
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
