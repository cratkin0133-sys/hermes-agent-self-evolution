"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import re
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig

_WORD_RE = re.compile(r"[a-zA-Z0-9_]{4,}")


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
        """Reject variants that drift away from the baseline's significant terms.

        Uses significant-word recall (words of length ≥ 4) against the baseline.
        Catches catastrophic rewrites and near-empty collapses that still have
        a few leftover tokens — the README's semantic-preservation guardrail.
        """
        baseline_words = set(_WORD_RE.findall(baseline.lower()))
        if not baseline_words:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message="No significant baseline terms to compare",
            )

        evolved_words = set(_WORD_RE.findall(text.lower()))
        overlap = len(baseline_words & evolved_words) / len(baseline_words)
        minimum = self.config.min_semantic_overlap

        if overlap >= minimum:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message=f"Semantic overlap OK: {overlap:.1%} (min {minimum:.1%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="semantic_preservation",
            message=(
                f"Semantic drift: overlap {overlap:.1%} below minimum {minimum:.1%} "
                f"({len(baseline_words & evolved_words)}/{len(baseline_words)} terms retained)"
            ),
        )

    def _non_empty_text(self, text: str, artifact_type: str) -> str:
        """Markdown body for skills; the original text for other artifacts."""
        if artifact_type != "skill":
            return text
        stripped = text.strip()
        if not stripped.startswith("---"):
            return text
        parts = stripped.split("---", 2)
        if len(parts) < 3:
            return text
        return parts[2]

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
