from __future__ import annotations

import uuid
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_cli(session_id: str, enabled: bool):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli.config = {"goals": {"auto_start": {"enabled": enabled}, "max_turns": 7}}
    cli._goal_manager = None
    return cli


def test_cli_auto_start_disabled_is_noop(hermes_home):
    from hermes_cli.goals import GoalManager
    cli = _make_cli(f"cli-auto-off-{uuid.uuid4().hex}", False)

    assert cli._maybe_auto_start_goal_for_input("Fix the failing tests") is False
    assert GoalManager(cli.session_id).state is None


def test_cli_auto_start_sets_goal_for_agentic_message(hermes_home):
    cli = _make_cli(f"cli-auto-on-{uuid.uuid4().hex}", True)

    assert cli._maybe_auto_start_goal_for_input(
        "Implement the retry workflow and verify tests"
    ) is True
    state = cli._get_goal_manager().state
    assert state is not None
    assert state.status == "active"
    assert state.goal == "Implement the retry workflow and verify tests"


def test_cli_auto_start_skips_slash_simple_question_and_existing_goal(hermes_home):
    cli = _make_cli(f"cli-auto-existing-{uuid.uuid4().hex}", True)
    cli._get_goal_manager().set("existing goal")

    assert cli._maybe_auto_start_goal_for_input("/goal status") is False
    assert cli._maybe_auto_start_goal_for_input("What time is it?") is False
    assert cli._maybe_auto_start_goal_for_input("Fix the failing tests") is False
    assert cli._get_goal_manager().state.goal == "existing goal"
