"""Tests for skill module loading and parsing."""

import pytest
from pathlib import Path
from evolution.skills.skill_module import load_skill, reassemble_skill


SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
metadata:
  hermes:
    tags: [testing]
---

# Test Skill — Testing Things

## When to Use
Use this when you need to test things.

## Procedure
1. First, do the thing
2. Then, verify it worked
3. Report results

## Pitfalls
- Don't forget to check edge cases
"""


class TestLoadSkill:
    def test_parses_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["name"] == "test-skill"
        assert skill["description"] == "A skill for testing things"
        assert "version: 1.0.0" in skill["frontmatter"]

    def test_parses_body(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert "# Test Skill" in skill["body"]
        assert "## Procedure" in skill["body"]
        assert "Don't forget" in skill["body"]

    def test_raw_contains_everything(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["raw"] == SAMPLE_SKILL

    def test_path_is_stored(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["path"] == skill_file


class TestReassembleSkill:
    def test_roundtrip(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        reassembled = reassemble_skill(skill["frontmatter"], skill["body"])
        assert "---" in reassembled
        assert "name: test-skill" in reassembled
        assert "# Test Skill" in reassembled

    def test_preserves_frontmatter(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        body = "# My Skill\nDo the thing."
        result = reassemble_skill(frontmatter, body)

        assert result.startswith("---\n")
        assert "name: my-skill" in result
        assert "# My Skill" in result

    def test_evolved_body_replaces_original(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        evolved_body = "# EVOLVED\nNew and improved procedure."
        result = reassemble_skill(frontmatter, evolved_body)

        assert "EVOLVED" in result
        assert "New and improved" in result


class TestSkillModuleOptimizable:
    """The skill body must live where DSPy optimizers can mutate it."""

    def test_skill_text_is_predictor_instructions(self):
        from evolution.skills.skill_module import SkillModule
        m = SkillModule("# My Skill\n1. do it")
        preds = list(m.named_predictors())
        assert len(preds) == 1
        assert preds[0][1].signature.instructions == "# My Skill\n1. do it"
        assert m.skill_text == "# My Skill\n1. do it"

    def test_skill_text_reflects_optimizer_mutation(self):
        from evolution.skills.skill_module import SkillModule
        m = SkillModule("original")
        _, pred = next(iter(m.named_predictors()))
        # Simulate what GEPA/MIPROv2 do: replace the predictor's instructions
        pred.signature = pred.signature.with_instructions("evolved")
        assert m.skill_text == "evolved"

    def test_forward_does_not_pass_skill_as_input(self):
        from evolution.skills.skill_module import SkillModule
        m = SkillModule("x")
        _, pred = next(iter(m.named_predictors()))
        assert "skill_instructions" not in pred.signature.input_fields
        assert "task_input" in pred.signature.input_fields
