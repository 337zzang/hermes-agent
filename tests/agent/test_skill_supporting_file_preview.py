from pathlib import Path

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool


def _build_message(tmp_path: Path, monkeypatch, supporting: list[str]) -> str:
    skill_dir = tmp_path / "large-skill"
    skill_dir.mkdir()
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    loaded_skill = {
        "content": "# Large Skill\n\nUse the routed references only.",
        "linked_files": {"references": supporting},
    }
    return skill_commands._build_skill_message(
        loaded_skill,
        skill_dir,
        "[Skill active.]",
    )


def test_small_supporting_catalog_is_embedded_in_full(tmp_path, monkeypatch):
    message = _build_message(
        tmp_path,
        monkeypatch,
        ["references/one.md", "scripts/run.py"],
    )

    assert "references/one.md" in message
    assert "scripts/run.py" in message
    assert "additional files omitted" not in message


def test_large_supporting_catalog_uses_bounded_preview(tmp_path, monkeypatch):
    limit = skill_commands._SKILL_SUPPORTING_FILE_PREVIEW_LIMIT
    supporting = [f"references/item-{index:02d}.md" for index in range(limit + 8)]
    message = _build_message(tmp_path, monkeypatch, supporting)

    for path in supporting[:limit]:
        assert path in message
    for path in supporting[limit:]:
        assert path not in message
    assert "8 additional files omitted from this activation preview" in message
    assert 'skill_view(name="large-skill") to inspect the complete catalog' in message


def test_supporting_catalog_deduplicates_before_counting(tmp_path, monkeypatch):
    repeated = ["references/one.md", "references/one.md", "scripts/run.py"]
    message = _build_message(tmp_path, monkeypatch, repeated)

    assert message.count("- references/one.md  ->") == 1
    assert "additional files omitted" not in message
