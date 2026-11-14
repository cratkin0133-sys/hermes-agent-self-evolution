"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig

# Latin tokens only — Python's \w also glues unspaced CJK into one giant token.
_LATIN_WORD_RE = re.compile(r"[a-zA-Z0-9_]{4,}")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?|[,;.]")
_UNIT_SPLIT_RE = re.compile(r"(?:\r?\n|[.!;?。！？]+[ \t]*)")

# Grammatical negation applies to the next content head, not the whole clause,
# so "do not skip the inspection" does not look like "do not inspect".
_ASSERT_NEGATIONS = frozenset({
    "not", "never", "nor", "neither", "dont", "doesnt", "didnt", "wont",
    "wouldnt", "cant", "cannot", "couldnt", "mustnt", "shouldnt", "isnt",
    "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "avoid", "refrain",
    "no", "none", "nothing", "nobody", "nowhere",
})
_WITHOUT = "without"
_RESET_WORDS = frozenset({
    "but", "however", "although", "though", "except",
})
_TRANSPARENT = frozenset({
    "thoroughly", "carefully", "fully", "properly", "simply", "merely",
    "really", "quickly", "directly", "explicitly", "correctly", "completely",
    "successfully", "first", "still", "even", "ever", "immediately",
    "longer", "already", "subsequently",
})
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with",
    "from", "by", "at", "as", "into", "that", "this", "these", "those",
    "it", "its", "be", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "you", "your", "we", "our", "they", "their",
    "when", "while", "if", "then", "than", "so", "such", "via", "per",
    "each", "other", "also", "only", "just", "please", "always", "must",
    "should", "shall", "can", "will", "would", "could", "may", "might",
    "have", "has", "had",
})
# Same stem, opposite sign. A swap counts only when the original sign is gone.
_RELATIONS = {
    "before": ("before", 1),
    "after": ("before", -1),
    "above": ("above", 1),
    "below": ("above", -1),
    "earlier": ("earlier", 1),
    "later": ("earlier", -1),
    "more": ("more", 1),
    "less": ("more", -1),
    "fewer": ("more", -1),
    "higher": ("higher", 1),
    "lower": ("higher", -1),
    "increase": ("increase", 1),
    "decrease": ("increase", -1),
    "enable": ("enable", 1),
    "disable": ("enable", -1),
    "allow": ("allow", 1),
    "deny": ("allow", -1),
    "disallow": ("allow", -1),
    "include": ("include", 1),
    "exclude": ("include", -1),
    "accept": ("accept", 1),
    "reject": ("accept", -1),
    "require": ("require", 1),
    "required": ("require", 1),
    "forbid": ("require", -1),
    "forbidden": ("require", -1),
    "start": ("start", 1),
    "stop": ("start", -1),
    "add": ("add", 1),
    "remove": ("add", -1),
    "open": ("open", 1),
    "close": ("open", -1),
}
_Signed = tuple[str, int, str]  # stem, sign, "word" | "relation"

_CJK_NEG_RE = re.compile(
    "しないでください|してはいけません|しない|しません|できない|ありません|ではない|"
    "ません|ない|禁止|不要|不可|不能|没有|沒有|勿|别|別|不"
)
_CJK_NEG_RESTORE = {
    "しないでください": "してください",
    "してはいけません": "してください",
    "しない": "する",
    "しません": "します",
    "できない": "できる",
    "ありません": "あります",
    "ではない": "である",
}


def _is_cjk_char(ch: str) -> bool:
    return (
        ("\u4e00" <= ch <= "\u9fff")
        or ("\u3040" <= ch <= "\u30ff")
        or ("\uac00" <= ch <= "\ud7a3")
    )


def _stem(word: str) -> str:
    """Light suffix fold so approve/approving and review/reviewing align."""
    stripped = False
    if word.endswith("ies") and len(word) > 5:
        word = word[:-3] + "y"
        stripped = True
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
        stripped = True
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
        stripped = True
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
        stripped = True
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        word = word[:-1]
        stripped = True
    if stripped and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    if word.endswith("ion") and len(word) > 6:
        word = word[:-3]
    elif word.endswith("ment") and len(word) > 6:
        word = word[:-4]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def _relation_for(word: str) -> Optional[tuple[str, int]]:
    candidates = [word]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)]
            candidates.append(base)
            if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
                candidates.append(base)
            candidates.append(base + "e")
    for candidate in candidates:
        found = _RELATIONS.get(candidate)
        if found:
            return found
    return None


def _signed_tokens(text: str) -> list[_Signed]:
    """Content tokens tagged with polarity and relation direction.

    Negation marks the next content head only. ``without`` under an open
    negation ("do not approve without inspecting") restores that head to
    positive, which keeps an equivalent paraphrase from looking reversed.
    """
    pending: Optional[int] = None
    negation_open = False
    tokens: list[_Signed] = []
    normalized = text.replace("\u2019", "'").replace("\u2018", "'")
    for raw in _LATIN_TOKEN_RE.findall(normalized):
        if raw in ",;.":
            pending = None
            negation_open = False
            continue
        word = raw.lower().replace("'", "")
        if word in _RESET_WORDS:
            pending = None
            negation_open = False
            continue
        if word in _ASSERT_NEGATIONS:
            pending = -1
            negation_open = True
            continue
        if word == _WITHOUT:
            pending = 1 if negation_open else -1
            negation_open = False
            continue
        if word in _STOPWORDS or word in _TRANSPARENT or word.isdigit():
            continue
        relation = _relation_for(word)
        sign = 1 if pending is None else pending
        pending = None
        if relation:
            canon, rel_sign = relation
            tokens.append((canon, sign * rel_sign, "relation"))
        else:
            tokens.append((_stem(word), sign, "word"))
    return tokens


def _split_units(text: str) -> list[str]:
    return [part.strip() for part in _UNIT_SPLIT_RE.split(text) if part.strip()]


def _token_recall(baseline: list[_Signed], evolved: list[_Signed]) -> float:
    if not baseline:
        return 0.0
    pools: dict[tuple[str, str], int] = {}
    for stem, _sign, kind in evolved:
        key = (stem, kind)
        pools[key] = pools.get(key, 0) + 1
    matched = 0
    for stem, _sign, kind in baseline:
        key = (stem, kind)
        if pools.get(key, 0) > 0:
            pools[key] -= 1
            matched += 1
    return matched / len(baseline)


def _early_content_negated(baseline: list[_Signed], evolved: list[_Signed]) -> bool:
    """A content word just after the leading action flipped sign."""
    words = [tok for tok in baseline if tok[2] == "word"]
    for tok in words[1:3]:
        if any(
            evolved_tok[0] == tok[0]
            and evolved_tok[2] == "word"
            and evolved_tok[1] != tok[1]
            for evolved_tok in evolved
        ):
            return True
    return False


def _lead_word_negated(baseline: list[_Signed], evolved: list[_Signed]) -> bool:
    lead = next((tok for tok in baseline if tok[2] == "word"), None)
    if lead is None:
        return False
    return any(
        tok[0] == lead[0] and tok[2] == "word" and tok[1] != lead[1]
        for tok in evolved
    )


def _relation_inverted(baseline: list[_Signed], evolved: list[_Signed]) -> bool:
    """True when a relation stem lost its original sign and gained the opposite."""
    def counts(tokens: list[_Signed]) -> dict[tuple[str, int], int]:
        found: dict[tuple[str, int], int] = {}
        for stem, sign, kind in tokens:
            if kind != "relation":
                continue
            key = (stem, sign)
            found[key] = found.get(key, 0) + 1
        return found

    base_counts = counts(baseline)
    evolved_counts = counts(evolved)
    for (stem, sign), count in base_counts.items():
        if evolved_counts.get((stem, sign), 0) >= count:
            continue
        if evolved_counts.get((stem, -sign), 0) > base_counts.get((stem, -sign), 0):
            return True
    return False


def _latin_reversal_reason(baseline: str, evolved: str) -> Optional[str]:
    for base_unit in _split_units(baseline):
        base_tokens = _signed_tokens(base_unit)
        if len(base_tokens) < 2:
            continue
        for evolved_unit in _split_units(evolved):
            evolved_tokens = _signed_tokens(evolved_unit)
            recall = _token_recall(base_tokens, evolved_tokens)
            if recall < 0.75:
                continue
            # The leading action, or a near-copy's next content word ("run no checks").
            if _lead_word_negated(base_tokens, evolved_tokens) or (
                recall >= 0.9 and _early_content_negated(base_tokens, evolved_tokens)
            ):
                return "a baseline directive was negated"
            if _relation_inverted(base_tokens, evolved_tokens):
                return "a baseline relation was inverted"
    return None


def _cjk_normalized(text: str) -> tuple[str, int]:
    count = 0

    def restore(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return _CJK_NEG_RESTORE.get(match.group(0), "")

    normalized = _CJK_NEG_RE.sub(restore, text)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized, count


def _cjk_reversal_reason(baseline: str, evolved: str) -> Optional[str]:
    """Near-copy CJK sentences that only insert or remove a negation."""
    for base_unit in _split_units(baseline):
        if sum(1 for ch in base_unit if _is_cjk_char(ch)) < 8:
            continue
        base_norm, base_count = _cjk_normalized(base_unit)
        if len(base_norm) < 8:
            continue
        for evolved_unit in _split_units(evolved):
            evolved_norm, evolved_count = _cjk_normalized(evolved_unit)
            if base_count == evolved_count or len(evolved_norm) < 8:
                continue
            if SequenceMatcher(None, base_norm, evolved_norm).ratio() >= 0.92:
                return "a baseline directive was negated"
    return None


def _instruction_reversal(baseline: str, evolved: str) -> Optional[str]:
    """Return why ``evolved`` reverses ``baseline``, or None when it does not.

    Word overlap cannot see this: a contradictory rewrite keeps the same
    vocabulary and only flips negation or a relation such as before/after.
    """
    return _latin_reversal_reason(baseline, evolved) or _cjk_reversal_reason(baseline, evolved)


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type, baseline_text))

        # 2. Growth, shrinkage, and semantic preservation (if baseline provided).
        # For skills, compare markdown bodies so YAML frontmatter cannot mask
        # a collapsed or drifted body.
        if baseline_text:
            evolved_cmp = self._comparison_text(artifact_text, artifact_type)
            baseline_cmp = self._comparison_text(baseline_text, artifact_type)
            results.append(self._check_growth(evolved_cmp, baseline_cmp, artifact_type))
            results.append(self._check_shrink(evolved_cmp, baseline_cmp))
            results.append(self._check_semantic_preservation(evolved_cmp, baseline_cmp))

        # 3. Non-empty. For skills, judge the markdown body: a reassembled
        # file always starts with YAML frontmatter.
        results.append(
            self._check_non_empty(self._non_empty_text(artifact_text, artifact_type))
        )

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                # Extract failure summary
                last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
                return ConstraintResult(
                    passed=False,
                    constraint_name="test_suite",
                    message="Test suite failed",
                    details="\n".join(last_lines),
                )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str, baseline_text: Optional[str] = None) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size  # Default

        # Descriptions that already exceed the cap upstream may keep (not grow
        # past) their size; the growth check bounds them further.
        if baseline_text and artifact_type in ("tool_description", "param_description"):
            limit = max(limit, len(baseline_text))

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        if not baseline.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message="Baseline empty — growth limit skipped (repair from empty body)",
            )

        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        # Short descriptions are bounded by their absolute cap (PLAN: <=500 / <=200
        # chars), not by relative growth — a 40-char description may need to double.
        caps = {
            "tool_description": self.config.max_tool_desc_size,
            "param_description": self.config.max_param_desc_size,
        }
        if artifact_type in caps and len(text) <= caps[artifact_type]:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth {growth:+.1%} within absolute cap ({len(text)}/{caps[artifact_type]} chars)",
            )

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_limit",
            message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
        )

    def _check_shrink(self, text: str, baseline: str) -> ConstraintResult:
        """Guard against an optimizer gutting the artifact to game the metric."""
        if not baseline.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="shrink_limit",
                message="Baseline empty — shrink limit skipped (repair from empty body)",
            )

        ratio = len(text) / max(1, len(baseline))
        floor = self.config.min_size_ratio
        if ratio >= floor:
            return ConstraintResult(
                passed=True,
                constraint_name="shrink_limit",
                message=f"Size retained: {ratio:.0%} of baseline (min {floor:.0%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="shrink_limit",
            message=f"Shrank to {ratio:.0%} of baseline (min {floor:.0%}) — likely lost content",
        )

    def _check_semantic_preservation(self, text: str, baseline: str) -> ConstraintResult:
        """Reject drifted rewrites and rewrites that reverse baseline instructions.

        Term and bigram overlap must stay above the configured minimum. Overlap
        alone accepts a negated near-copy, so a directive whose leading action
        is negated, or whose relation (before/after, accept/reject, and similar)
        is swapped, fails even when the vocabulary is unchanged.
        """
        if not baseline.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message="Baseline empty — semantic check skipped",
            )
        if not text.strip():
            return ConstraintResult(
                passed=False,
                constraint_name="semantic_preservation",
                message="Evolved text empty while baseline has content",
            )

        reversal = _instruction_reversal(baseline, text)
        if reversal:
            return ConstraintResult(
                passed=False,
                constraint_name="semantic_preservation",
                message=f"Instruction reversal: {reversal}",
            )

        minimum = self.config.min_semantic_overlap
        overlap, metric = self._semantic_overlap_ratio(text, baseline)
        if metric == "unsupported":
            return ConstraintResult(
                passed=False,
                constraint_name="semantic_preservation",
                message="Baseline has no comparable alphanumeric content",
            )

        if overlap >= minimum:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message=f"Semantic overlap OK ({metric}): {overlap:.1%} (min {minimum:.1%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="semantic_preservation",
            message=(
                f"Semantic drift ({metric}): overlap {overlap:.1%} below minimum {minimum:.1%}"
            ),
        )

    @classmethod
    def _has_cjk_content(cls, text: str, min_chars: int = 8) -> bool:
        """True when the text has enough CJK to require bigram preservation."""
        return sum(1 for c in text if cls._is_cjk_char(c)) >= min_chars

    @staticmethod
    def _is_cjk_char(ch: str) -> bool:
        return _is_cjk_char(ch)

    @classmethod
    def _semantic_bigrams(cls, text: str) -> set[str]:
        """Adjacent CJK pairs only — Latin must not inflate bigram recall."""
        chars = [c for c in text.lower() if cls._is_cjk_char(c)]
        if len(chars) < 2:
            return set()
        return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}

    def _semantic_overlap_ratio(self, text: str, baseline: str) -> tuple[float, str]:
        """Return (overlap_ratio, metric_name). Uses the strictest applicable metric."""
        scores: list[float] = []
        parts: list[str] = []

        baseline_words = set(_LATIN_WORD_RE.findall(baseline.lower()))
        evolved_words = set(_LATIN_WORD_RE.findall(text.lower()))
        if baseline_words:
            scores.append(len(baseline_words & evolved_words) / len(baseline_words))
            parts.append("term")

        if self._has_cjk_content(baseline):
            baseline_bigrams = self._semantic_bigrams(baseline)
            evolved_bigrams = self._semantic_bigrams(text)
            if baseline_bigrams:
                scores.append(len(baseline_bigrams & evolved_bigrams) / len(baseline_bigrams))
                parts.append("bigram")

        if not scores:
            baseline_chars = {c for c in baseline.lower() if c.isalnum()}
            evolved_chars = {c for c in text.lower() if c.isalnum()}
            if not baseline_chars:
                return 0.0, "unsupported"
            scores.append(len(baseline_chars & evolved_chars) / len(baseline_chars))
            parts.append("character")

        return min(scores), "+".join(parts)

    @staticmethod
    def _skill_markdown_body(text: str) -> str:
        """Return skill markdown body after a line-bounded YAML frontmatter block."""
        lines = text.strip().splitlines()
        if not lines or lines[0].strip() != "---":
            return text
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1 :])
        return text

    def _non_empty_text(self, text: str, artifact_type: str) -> str:
        """Markdown body for skills; the original text for other artifacts."""
        if artifact_type != "skill":
            return text
        return self._skill_markdown_body(text)

    def _comparison_text(self, text: str, artifact_type: str) -> str:
        """Text used for size-delta and semantic checks against a baseline."""
        return self._non_empty_text(text, artifact_type)

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="non_empty",
            message="Artifact is empty",
        )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        missing = []
        if not has_frontmatter:
            missing.append("YAML frontmatter (---)")
        if not has_name:
            missing.append("name field")
        if not has_description:
            missing.append("description field")
        return ConstraintResult(
            passed=False,
            constraint_name="skill_structure",
            message=f"Skill missing: {', '.join(missing)}",
        )
