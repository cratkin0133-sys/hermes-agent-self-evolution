# Plan: LLM-judge scoring, benchmark score gates, GitHub issue input

**Spec:** `PLAN.md` (sections "Benchmarks as Fitness Signals", "Constraints & Guardrails",
Phase 1 / Phase 4 breakdowns). This plan closes three gaps left after Phases 1–5 landed.

## Context

Repository: `hermes-agent-self-evolution` (Python ≥3.10, DSPy 3.5, gepa 0.1.4). It evolves
text and code in a separate `hermes-agent` checkout. Relevant existing modules:

- `evolution/core/fitness.py` — `FitnessScore` (weighted `composite`), `LLMJudge` (a
  `dspy.ChainOfThought(JudgeSignature)` with outputs `correctness`, `procedure_following`,
  `conciseness`, `feedback`), `skill_fitness_metric` (keyword-overlap heuristic, returns
  float), `skill_gepa_metric` (heuristic score + text feedback, returns
  `dspy.Prediction(score, feedback)`), `_parse_score`.
- `evolution/skills/skill_module.py` — `SkillModule`; the skill body is the predictor's
  instructions; `skill_text` property reads it back; `forward(task_input)` returns
  `dspy.Prediction(output=...)`.
- `evolution/skills/evolve_skill.py` — Phase 1 `evolve(...)` + click `main`. GEPA uses
  `skill_gepa_metric`; holdout scoring uses `skill_fitness_metric`.
- `evolution/core/benchmark_gate.py` — `run_with_changes(hermes_repo, files, command,
  timeout=3600, workdir=None) -> ConstraintResult`: runs `command` in a throwaway worktree
  with `files` applied; pass = exit 0.
- `evolution/tools/evolve_tool_descriptions.py` (Phase 2), `evolution/prompts/evolve_prompt_section.py`
  (Phase 3, already has `--benchmark-cmd` using `run_with_changes`),
  `evolution/code/evolve_tool_code.py` (Phase 4, `--bug` text required),
  `evolution/monitor/loop.py` (Phase 5; `bench` subcommand records pass/fail via
  `MetricsStore.record_run`), `evolution/monitor/store.py`.
- Tests use offline scripted LMs: `tests/fakes.py` provides `fake_lm(script, model)` where
  `script(system, user) -> str`, and `xml_fields(**fields)` which renders DSPy 3.5's default
  XML reply format. A `dspy.ChainOfThought` predictor additionally expects a `plan` field.

## Global Constraints

1. All tests run offline: no network access, no API keys. LLM calls in tests use
   `tests.fakes.fake_lm`; HTTP in tests is replaced by injection or monkeypatching.
2. Use DSPy 3.5 APIs only. For raw completions use `evolution.core.lm.complete` /
   `make_lm`; never call `lm(messages=...)` (removed in DSPy 3.5).
3. Never modify the hermes-agent checkout in place; anything that runs code against it
   goes through a throwaway worktree/copy (existing helpers do this).
4. Backwards compatible CLIs: existing flags keep working with their current meaning.
   New defaults are only those this plan states.
5. The full existing suite (`python -m pytest -q`, currently 176 tests) must still pass
   after each task, plus the task's new tests.
6. One or more commits per task. Every commit message ends with exactly these two lines:
   `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
   `Claude-Session: https://claude.ai/code/session_01FN1FRphEt4Wnwg7KLJPKXm`
7. Do not add dependencies to `pyproject.toml` (stdlib `urllib`/`json`/`re` suffice).

---

### Task 1: LLM-judge metric for Phase 1

**Goal:** Phase 1 currently optimizes a keyword-overlap heuristic, which teaches GEPA to
repeat keywords. Score with the existing `LLMJudge` instead, keeping the heuristic as an
option and as a fallback.

**Files:** `evolution/core/fitness.py`, `evolution/skills/skill_module.py`,
`evolution/skills/evolve_skill.py`, new `tests/core/test_judge_metric.py`.

**Changes:**

1. `LLMJudge.__init__(self, config: EvolutionConfig, lm: Optional[dspy.LM] = None)`. Store
   `lm`. In `score(...)`, use `self.lm` if set, otherwise keep the current behaviour
   (`dspy.LM(self.config.eval_model)`). No other behaviour change to `score`.
2. `SkillModule.forward` returns `dspy.Prediction(output=result.output, skill_text=self.skill_text)`
   so metrics can see the skill text under evaluation.
3. New function in `fitness.py`:

   ```python
   def make_judge_metric(judge: LLMJudge, max_size: Optional[int] = None):
       """GEPA-compatible metric backed by LLMJudge.

       Returns metric(gold, pred, trace=None, pred_name=None, pred_trace=None)
       -> dspy.Prediction(score=float, feedback=str).
       """
   ```
   Behaviour of the returned metric:
   - Empty/whitespace `pred.output` → `score=0.0`, feedback
     `"The response was empty. The skill must lead to a concrete answer."` (no judge call).
   - Otherwise call `judge.score(task_input=gold.task_input, expected_behavior=gold.expected_behavior,
     agent_output=pred.output, skill_text=getattr(pred, "skill_text", ""),
     artifact_size=len(skill_text) if max_size else None, max_size=max_size)` and return
     `score = fitness.composite`, `feedback = fitness.feedback`.
   - If the judge raises any `Exception`, fall back to `skill_gepa_metric(gold, pred)` and
     prefix its feedback with `"[judge unavailable: <ExceptionTypeName>] "`.
4. `evolve()` gains parameters `metric: str = "judge"` and `judge_model: Optional[str] = None`
   (default: `optimizer_model`). When `metric == "judge"`: build
   `LLMJudge(config, lm=dspy.LM(judge_model or optimizer_model))`, `gepa_metric =
   make_judge_metric(judge, max_size=config.max_skill_size)`, and holdout scoring uses
   `gepa_metric(ex, pred).score`. When `metric == "heuristic"`: today's behaviour exactly
   (`skill_gepa_metric` for GEPA, `skill_fitness_metric` for holdout). Any other value →
   `ValueError`. Record `"metric"` and `"judge_model"` (None for heuristic) in the metrics dict.
5. CLI: `--metric` (`click.Choice(["judge", "heuristic"])`, default `judge`) and
   `--judge-model` (default None), passed through to `evolve()`.

**Tests** (`tests/core/test_judge_metric.py`), using `fake_lm` + `xml_fields`; the judge is a
ChainOfThought so the fake reply must include `plan`:
- Judge replying correctness=1.0, procedure_following=0.5, conciseness=0.0, feedback="F"
  → metric score == 0.5*1.0 + 0.3*0.5 + 0.2*0.0 = 0.65 (approx), feedback == "F".
- Empty output → score 0.0, judge never called (fake records no call).
- Judge LM whose reply is unparseable (e.g. returns `"garbage"`) → result equals the
  heuristic score from `skill_gepa_metric` and feedback starts with `"[judge unavailable:"`.
- `SkillModule("BODY")` forward under a fake LM returns a prediction with
  `skill_text == "BODY"`.
- `evolve(..., metric="bogus", dry_run=False)` path: calling with an invalid metric raises
  `ValueError` before any LLM call (validate early, right after loading the skill is fine;
  a fake hermes repo with one SKILL.md in `tmp_path` suffices).

---

### Task 2: Benchmark score parsing and baseline-vs-evolved comparison

**Goal:** PLAN.md gates on benchmark *score regression* (e.g. "TBLite within 2%"), not exit
codes. hermes-agent no longer ships TBLite/YC-Bench, so support any command whose output
reports a score.

**Files:** new `evolution/core/benchmark_score.py`, `evolution/core/benchmark_gate.py`,
new `tests/core/test_benchmark_score.py`.

**Changes:**

1. `evolution/core/benchmark_score.py`:

   ```python
   def parse_score(output: str) -> Optional[float]:
   ```
   Rules, first match wins, scanning the whole output:
   - (a) The **last** line matching regex `^\s*SCORE\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$`
     (case-insensitive) → that float.
   - (b) Else the **last** line that parses as a JSON object containing a numeric `"score"`
     key → that value as float.
   - (c) Else a pytest summary: find the last line containing `passed` or `failed`; extract
     integers for `(\d+) passed`, `(\d+) failed`, `(\d+) errors?`; if total > 0 return
     `passed / (passed + failed + errors)`.
   - Else `None`.
   Values from (a)/(b) are returned as-is (not clamped).

2. In `benchmark_gate.py` add:

   ```python
   @dataclass
   class BenchmarkComparison:
       passed: bool
       baseline_score: Optional[float]
       evolved_score: Optional[float]
       message: str

   def compare_benchmark(hermes_repo: Path, files: dict[str, str], command: str,
                         max_regression: float = 0.02, timeout: int = 3600) -> BenchmarkComparison:
   ```
   Runs `command` twice in throwaway trees: once with no file changes (baseline), once with
   `files` applied (evolved). Refactor so both runs reuse one private helper that returns
   `(exit_ok: bool, output: str)`; `run_with_changes` keeps its exact signature and
   behaviour, implemented on that helper. Decision:
   - Both scores parse → `passed = evolved >= baseline - max_regression`; message
     `f"score {baseline:.4f} → {evolved:.4f} (max regression {max_regression})"`.
   - Either score fails to parse → fall back to exit codes: `passed = evolved_exit_ok or
     not baseline_exit_ok` (an already-failing baseline cannot be regressed by exit code),
     message says scores were unparseable and which exit codes were seen.
   - Timeouts count as exit failure with `None` score.

**Tests** (`tests/core/test_benchmark_score.py`):
- `parse_score` on: `"SCORE=0.87"` → 0.87; output with two SCORE lines → last;
  `'{"score": 0.5, "n": 10}'` → 0.5; pytest summary `"===== 8 passed, 2 failed in 1.2s ====="`
  → 0.8; `"3 passed, 1 error"` → 0.75; `"no score here"` → None; SCORE line beats JSON when
  both are present.
- `compare_benchmark` on a tiny git repo (built in `tmp_path`, like
  `tests/core/test_benchmark_gate.py` does) with a module `m.py` containing `X = <n>` and a
  command `f'{sys.executable} -c "import m; print(f\'SCORE={m.X/10}\')"'`:
  baseline X=8 → evolved X=9 passes; evolved X=7 with max_regression 0.05 fails; with
  max_regression 0.2 passes. A command printing no score that exits 0 for both → passes
  via exit-code fallback. The repo file on disk is unchanged afterwards.

---

### Task 3: Wire score gates into Phases 2, 3 and the loop's `bench`

**Depends on:** Task 2 (`compare_benchmark`, `parse_score`).

**Files:** `evolution/tools/evolve_tool_descriptions.py`, `evolution/prompts/evolve_prompt_section.py`,
`evolution/monitor/loop.py`, `evolution/monitor/store.py` (only if needed), tests in
`tests/tools/`, `tests/prompts/`, `tests/monitor/`.

**Changes:**

1. **Phase 3** (`evolve_prompt_sections`): replace the `run_with_changes` call with
   `compare_benchmark(repo, {rel: new_src}, benchmark_cmd, max_regression=benchmark_max_regression)`.
   New parameter `benchmark_max_regression: float = 0.0` (PLAN: zero tolerance for system
   prompts) and CLI `--benchmark-max-regression` (float, default 0.0). Metrics key
   `"benchmark"` becomes a dict `{"passed", "baseline_score", "evolved_score", "message"}`
   (or None when no command). Rejection reason text stays `"benchmark gate failed"`.
2. **Phase 2** (`evolve_tool_descriptions`): add the same gate. New parameters
   `benchmark_cmd: Optional[str] = None`, `benchmark_max_regression: float = 0.02` (PLAN:
   "TBLite within 2%"); CLI `--benchmark-cmd`, `--benchmark-max-regression`. Run only when
   there are changes and constraints passed; a failed comparison makes the result not
   accepted with reason `"benchmark gate failed"`. Same `"benchmark"` metrics dict.
3. **Loop `bench`**: after running, `score = parse_score(output)`; record the run with
   metrics `{"accepted": ok, "command": ..., "output_tail": ..., "evolved_score": score}`
   (so `MetricsStore` stores it in the `evolved` column). New option `--max-regression`
   (float, default 0.02): look up the **previous** run for the same `--name` *before*
   recording the new one; if both previous and current scores exist and
   `current < previous - max_regression`, print `REGRESSION <prev> → <cur>` and exit 1 even
   if the command exited 0. Print `"<name>: PASS|FAIL score=<cur or n/a> (prev <p or n/a>)"`.

**Tests:**
- Phase 3: extend `tests/prompts/test_evolve_prompt_section.py` (reuse its fixtures/route) —
  with a `benchmark_cmd` that prints `SCORE=` lower for the evolved prompt than the
  baseline, the run is not accepted and `"benchmark gate failed"` is in
  `rejection_reasons`; with a command that prints equal scores it is accepted. The fake
  repo there is not a git repo; `compare_benchmark` must work on non-git repos (it copies).
  A simple command: `f'{sys.executable} -c "import pathlib; t=pathlib.Path(\'agent/prompt_builder.py\').read_text(); print(\'SCORE=0.5\' if \'last time\' in t else \'SCORE=0.9\')"'`
  (adapt to the fixture's actual evolved text).
- Phase 2: analogous test in `tests/tools/test_evolve_tool_descriptions.py`.
- Loop: invoke the click CLI with `click.testing.CliRunner` using `--db` in `tmp_path` and a
  `--hermes-repo` pointing at a `tmp_path` dir: first `bench --cmd "<print SCORE=0.9>"`
  exits 0; then `bench --cmd "<print SCORE=0.5>"` exits 1 and prints `REGRESSION`; the
  store's latest benchmark row has `evolved == 0.5`.

---

### Task 4: `--bug-issue` for Phase 4 (fetch bug text from GitHub)

**Goal:** PLAN.md's Phase 4 invocation is `--bug-issue 742`. Fetch the issue (title, body,
first comments) from the GitHub REST API and use it as the bug description.

**Files:** new `evolution/code/github_issues.py`, `evolution/code/evolve_tool_code.py`,
new `tests/code/test_github_issues.py`, extend `tests/code/test_evolve_tool_code.py`.

**Changes:**

1. `evolution/code/github_issues.py`:

   ```python
   @dataclass
   class Issue:
       repo: str          # "owner/name"
       number: int
       title: str
       body: str
       url: str           # html_url
       comments: list[str]

   def fetch_issue(repo: str, number: int, token: Optional[str] = None,
                   max_comments: int = 10, timeout: int = 20, opener=None) -> Issue:
   def issue_to_bug_text(issue: Issue, max_chars: int = 6000) -> str:
   ```
   - `fetch_issue` GETs `https://api.github.com/repos/{repo}/issues/{number}` and, when the
     issue's `comments` count > 0, `.../issues/{number}/comments?per_page={max_comments}`.
     Headers: `Accept: application/vnd.github+json`, `User-Agent: hermes-agent-self-evolution`,
     and `Authorization: Bearer <token>` when a token is given. `token` defaults to the
     `GITHUB_TOKEN` environment variable when the argument is None. `opener` defaults to
     `urllib.request.urlopen` (called as `opener(request, timeout=timeout)`); tests inject a fake.
   - If the payload has a `"pull_request"` key, raise `ValueError("#N is a pull request, not an issue")`.
   - HTTP errors raise `RuntimeError` with a clear message including the status code;
     404 → message mentions the repo may be private or the number wrong and that
     `GITHUB_TOKEN` can be set.
   - Null `body` becomes `""`.
   - `issue_to_bug_text` → `f"{title} ({url})\n\n{body}"` followed by each comment as
     `"\n\n--- comment ---\n" + text`, truncated to `max_chars` total (truncate the end,
     append `"\n[truncated]"` when truncated).
2. `evolve_tool_code.py`:
   - CLI: `--bug` becomes optional; new `--bug-issue` (int) and `--issue-repo` (default
     `"NousResearch/hermes-agent"`). Exactly one of `--bug` / `--bug-issue` is required
     (raise `click.UsageError` otherwise).
   - `evolve_tool_code(...)` gains `bug_issue: Optional[int] = None`,
     `issue_repo: str = "NousResearch/hermes-agent"`, and `bug` becomes
     `Optional[str] = None`. When `bug_issue` is set: fetch, `bug = issue_to_bug_text(issue)`,
     and add `"issue_url": issue.url` to metrics; the review bundle title becomes
     `f"Fix #{number}: {issue.title}"` and the summary includes `f"Fixes {issue.url}"`.
     Fetch failures print a red error and `sys.exit(1)` (consistent with other errors).
     If neither is set → `ValueError`.

**Tests:**
- `tests/code/test_github_issues.py` with a fake opener returning objects that support
  `.read()` and context-manager use, serving JSON for the issue and comments URLs; assert
  headers (User-Agent, Accept, Authorization only when token given), parsed fields, comment
  fetch skipped when `comments == 0`, PR payload → ValueError, HTTP 404 (raise
  `urllib.error.HTTPError`) → RuntimeError mentioning `GITHUB_TOKEN`, truncation behaviour of
  `issue_to_bug_text`. Use `monkeypatch.delenv("GITHUB_TOKEN", raising=False)` where needed.
- Extend `tests/code/test_evolve_tool_code.py`: run the existing fixture with
  `bug_issue=12` and `bug=None`, monkeypatching `evolution.code.evolve_tool_code.fetch_issue`
  (import it into that module by name) to return an `Issue`; assert accepted, metrics
  `issue_url`, and that `PR_BODY.md` contains `Fixes <url>`.
- CLI: `CliRunner` with neither `--bug` nor `--bug-issue` → non-zero exit with usage error.
