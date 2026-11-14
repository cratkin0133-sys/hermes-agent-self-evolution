"""Tests for constraint validators."""

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


def _unspaced(start: int, length: int) -> str:
    """Code points from an unspaced block. No natural-language text is stored."""
    return "".join(chr(start + offset) for offset in range(length))


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed


class TestGrowthConstraints:
    def test_acceptable_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # 10% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_excessive_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1300  # 30% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert not result.passed

    def test_repair_from_empty_baseline_skips_growth_cap(self, validator):
        baseline = "   \n"
        evolved = "x" * 500
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed
        assert "skipped" in result.message

    def test_moderate_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller — within min_size_ratio
        result = validator._check_shrink(evolved, baseline)
        assert result.passed

    def test_catastrophic_shrinkage_fails(self, validator):
        """Bug-hunt gap: 8.3k → 97 char collapse must not pass the gates."""
        baseline = "x" * 8300
        evolved = "x" * 97
        result = validator._check_shrink(evolved, baseline)
        assert not result.passed
        assert "Shrank to" in result.message


class TestSkillBodyExtraction:
    def test_thematic_break_in_body_is_preserved(self, validator):
        """Markdown --- in the body must not truncate comparison text."""
        skill = (
            "---\nname: x\ndescription: y\n---\n\n"
            "# Section A\n\ncontent alpha\n\n---\n\n# Section B\n\ncontent beta\n"
        )
        body = validator._skill_markdown_body(skill)
        assert "content alpha" in body
        assert "content beta" in body
        assert body.count("---") >= 1


class TestSemanticPreservation:
    def test_retained_terms_pass(self, validator):
        baseline = "Review pull requests for security bugs and test coverage gaps"
        evolved = (
            "When reviewing pull requests, check security bugs carefully "
            "and report test coverage gaps with concrete examples"
        )
        result = validator._check_semantic_preservation(evolved, baseline)
        assert result.passed

    def test_unrelated_rewrite_fails(self, validator):
        baseline = "Review pull requests for security bugs and test coverage gaps"
        evolved = "Bake sourdough bread with careful fermentation timing notes"
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "Semantic drift" in result.message

    def test_non_latin_baseline_uses_character_overlap(self, validator):
        baseline = _unspaced(0x4E00, 12)
        unrelated = _unspaced(0x5000, 12)
        result = validator._check_semantic_preservation(unrelated, baseline)
        assert not result.passed
        assert "bigram" in result.message

    def test_unspaced_minor_rewrite_passes_bigram_overlap(self, validator):
        baseline = _unspaced(0x4E00, 16)
        evolved = baseline + _unspaced(0x5100, 4)
        result = validator._check_semantic_preservation(evolved, baseline)
        assert result.passed
        assert "bigram" in result.message

    def test_stripped_unspaced_text_fails_despite_unchanged_english(self, validator):
        baseline = (
            _unspaced(0x4E00, 16)
            + "\n\nAlways inspect security vulnerabilities and test coverage before approving."
        )
        gutted = "Always inspect security vulnerabilities and test coverage before approving."
        result = validator._check_semantic_preservation(gutted, baseline)
        assert not result.passed
        assert "bigram" in result.message

    def test_mixed_baseline_rejects_english_drift(self, validator):
        baseline = (
            _unspaced(0x4E00, 10)
            + "\n\nAlways inspect security vulnerabilities and test coverage before approving."
        )
        drifted = (
            _unspaced(0x4E00, 10)
            + "\n\nBake sourdough bread with fermentation timing notes."
        )
        result = validator._check_semantic_preservation(drifted, baseline)
        assert not result.passed
        assert "term" in result.message

    def test_negated_near_copy_fails_despite_word_overlap(self, validator):
        """Same vocabulary with the instruction reversed must not deploy."""
        baseline = (
            "Always inspect security vulnerabilities and test coverage "
            "before approving the pull request."
        )
        evolved = (
            "Never inspect security vulnerabilities and test coverage "
            "before approving the pull request."
        )
        overlap, _metric = validator._semantic_overlap_ratio(evolved, baseline)
        assert overlap >= validator.config.min_semantic_overlap
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "Instruction reversal" in result.message
        assert "negated" in result.message

    def test_must_not_and_contraction_reverse_the_directive(self, validator):
        baseline = "You must verify tests cover the new behavior and edge cases."
        must_not = "You must not verify tests cover the new behavior and edge cases."
        result = validator._check_semantic_preservation(must_not, baseline)
        assert not result.passed
        assert "negated" in result.message

        baseline = "Inspect security vulnerabilities before approving the pull request."
        contracted = "Don't inspect security vulnerabilities before approving the pull request."
        result = validator._check_semantic_preservation(contracted, baseline)
        assert not result.passed
        assert "negated" in result.message

    def test_relation_swap_fails_despite_shared_vocabulary(self, validator):
        baseline = "Inspect security vulnerabilities before approving the pull request."
        evolved = "Inspect security vulnerabilities after approving the pull request."
        overlap, _metric = validator._semantic_overlap_ratio(evolved, baseline)
        assert overlap >= validator.config.min_semantic_overlap
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "inverted" in result.message

        baseline = "Accept security fixes that include regression tests before merging."
        evolved = "Reject security fixes that include regression tests before merging."
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "Instruction reversal" in result.message

    def test_one_reversed_step_fails_the_skill(self, validator):
        baseline = (
            "Read the pull request description and linked issues.\n"
            "Inspect security-sensitive changes for injection risks.\n"
            "Verify tests cover the new behavior and edge cases.\n"
        )
        evolved = (
            "Read the pull request description and linked issues.\n"
            "Inspect security-sensitive changes for injection risks.\n"
            "Do not verify tests cover the new behavior and edge cases.\n"
        )
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "negated" in result.message

    def test_caution_and_double_negation_paraphrase_still_pass(self, validator):
        baseline = "Always inspect security vulnerabilities before approving the pull request."
        caution = (
            "Always inspect security vulnerabilities before approving the pull request. "
            "Do not skip the security vulnerabilities inspection."
        )
        assert validator._check_semantic_preservation(caution, baseline).passed

        paraphrase = (
            "Do not approve the pull request without first inspecting security vulnerabilities."
        )
        result = validator._check_semantic_preservation(paraphrase, baseline)
        assert result.passed

        elaborated = (
            "After deploying, inspect security vulnerabilities before approving the pull request."
        )
        assert validator._check_semantic_preservation(elaborated, baseline).passed

    def test_without_and_no_determiner_reverse_the_directive(self, validator):
        baseline = "Always inspect security vulnerabilities before approving the pull request."
        skipped = "Approve the pull request without first inspecting security vulnerabilities."
        result = validator._check_semantic_preservation(skipped, baseline)
        assert not result.passed
        assert "negated" in result.message

        baseline = "Run checks before approving the deployment."
        evolved = "Run no checks before approving the deployment."
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "negated" in result.message

    def test_appended_negated_reminder_does_not_fail_unchanged_directive(self, validator):
        baseline = "Inspect security vulnerabilities before approving the pull request."
        evolved = (
            "Inspect security vulnerabilities before approving the pull request. "
            "Do not inspect security vulnerabilities before approving the pull request."
        )
        result = validator._check_semantic_preservation(evolved, baseline)
        assert result.passed

    def test_no_issues_phrase_does_not_trigger_false_reversal(self, validator):
        baseline = "Report issues and test gaps before approving the pull request."
        evolved = (
            "Report issues with no regressions and test gaps before approving the pull request."
        )
        result = validator._check_semantic_preservation(evolved, baseline)
        assert result.passed

    def test_coordinated_negation_rejects_listed_verbs(self, validator):
        baseline = "Read the pull request, inspect security changes, and approve when ready."
        evolved = "Do not read the pull request, inspect security changes, or approve when ready."
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed
        assert "negated" in result.message

    def test_validate_all_rejects_negated_skill_body(self, validator):
        baseline = (
            "---\nname: review\ndescription: Review changes\n---\n\n"
            "Always inspect security vulnerabilities and test coverage "
            "before approving the pull request.\n"
        )
        evolved = (
            "---\nname: review\ndescription: Review changes\n---\n\n"
            "Never inspect security vulnerabilities and test coverage "
            "before approving the pull request.\n"
        )
        results = {
            r.constraint_name: r
            for r in validator.validate_all(evolved, "skill", baseline_text=baseline)
        }
        assert results["shrink_limit"].passed is True
        assert results["semantic_preservation"].passed is False
        assert "Instruction reversal" in results["semantic_preservation"].message


class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nContent here"
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results)

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

    def test_frontmatter_only_skill_fails_non_empty(self, validator):
        """YAML frontmatter must not satisfy non_empty when the body collapsed."""
        skill = "---\nname: test\ndescription: Test skill\n---\n\n   \n"
        results = {r.constraint_name: r.passed for r in validator.validate_all(skill, "skill")}
        assert results["skill_structure"] is True
        assert results["non_empty"] is False

    def test_collapsed_skill_fails_against_baseline(self, validator):
        """Frontmatter + tiny body must fail size-delta and semantic gates."""
        baseline_body = (
            "# Code review\n\n"
            "1. Read the pull request description and linked issues\n"
            "2. Inspect security-sensitive changes for injection risks\n"
            "3. Verify tests cover the new behavior and edge cases\n"
            "4. Summarize findings with severity and file references\n"
        ) * 40  # ~8k chars of meaningful procedure
        baseline = (
            "---\nname: requesting-code-review\n"
            "description: Request a thorough code review\n---\n\n"
            + baseline_body
        )
        collapsed = (
            "---\nname: requesting-code-review\n"
            "description: Request a thorough code review\n---\n\n"
            "Be brief.\n"
        )
        results = {
            r.constraint_name: r
            for r in validator.validate_all(collapsed, "skill", baseline_text=baseline)
        }
        assert results["non_empty"].passed is True  # tiny body is non-empty
        assert results["shrink_limit"].passed is False
        assert results["semantic_preservation"].passed is False

    def test_bloated_frontmatter_cannot_mask_body_shrink(self, validator):
        """Body-only comparison must fail even when YAML padding dominates file size."""
        yaml_pad = "meta: " + ("padding " * 400) + "\n"
        front = f"---\nname: requesting-code-review\ndescription: x\n{yaml_pad}---\n\n"
        baseline = front + ("step\n" * 200)
        collapsed = front + "be brief\n"
        results = {
            r.constraint_name: r.passed
            for r in validator.validate_all(collapsed, "skill", baseline_text=baseline)
        }
        assert results["shrink_limit"] is False
        assert results["semantic_preservation"] is False


def test_reassembled_skill_passes_structure_check():
    """Regression: evolve_skill must validate the reassembled file, not the bare body."""
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator
    from evolution.skills.skill_module import reassemble_skill
    full = reassemble_skill("name: x\ndescription: y", "# Body\n1. step")
    v = ConstraintValidator(EvolutionConfig(hermes_agent_path=None))
    results = {r.constraint_name: r.passed for r in v.validate_all(full, "skill", baseline_text=full)}
    assert results["skill_structure"] is True


def test_shrink_guard_rejects_gutted_artifact():
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator
    v = ConstraintValidator(EvolutionConfig(hermes_agent_path=None))
    res = {r.constraint_name: r.passed for r in v.validate_all("tiny", "skill", baseline_text="x" * 100)}
    assert res["shrink_limit"] is False


def test_tool_description_already_over_cap_may_keep_its_size():
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator
    v = ConstraintValidator(EvolutionConfig(hermes_agent_path=None))
    base = "d" * 700
    res = {r.constraint_name: r.passed for r in v.validate_all("e" * 690, "tool_description", baseline_text=base)}
    assert res["size_limit"] and res["growth_limit"] and res["shrink_limit"]


def test_short_tool_description_may_grow_up_to_cap_but_not_beyond_relative_limit_when_over_cap():
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator
    v = ConstraintValidator(EvolutionConfig(hermes_agent_path=None))
    grow = {r.constraint_name: r.passed for r in v.validate_all("x" * 85, "tool_description", baseline_text="x" * 40)}
    assert grow["growth_limit"] and grow["size_limit"]
    bloat = {r.constraint_name: r.passed for r in v.validate_all("x" * 900, "tool_description", baseline_text="x" * 700)}
    assert not bloat["growth_limit"]
