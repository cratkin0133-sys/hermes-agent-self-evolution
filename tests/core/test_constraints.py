"""Tests for constraint validators."""

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


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

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed


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
