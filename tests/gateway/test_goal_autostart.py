from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakeSessionEntry:
    def __init__(self, session_id: str):
        self.session_id = session_id


class _FakeSessionStore:
    def __init__(self, session_id: str):
        self.entry = _FakeSessionEntry(session_id)

    def get_or_create_session(self, source):
        return self.entry


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


def _make_runner(session_id: str, enabled: bool):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(goals={"auto_start": {"enabled": enabled}, "max_turns": 7})
    runner.session_store = _FakeSessionStore(session_id)
    return runner


def _make_event(text: str, *, internal: bool = False):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-gateway-auto",
            chat_type="channel",
            user_id="user-gateway-auto",
        ),
        message_id="msg-gateway-auto",
        internal=internal,
    )


def test_gateway_auto_start_disabled_is_noop(hermes_home):
    from hermes_cli.goals import GoalManager
    sid = f"sid-gateway-auto-off-{uuid.uuid4().hex}"
    runner = _make_runner(sid, False)

    assert runner._maybe_auto_start_goal_for_event(
        _make_event("Debug the worker crash and verify the fix")
    ) is False
    assert GoalManager(sid).state is None


def test_gateway_auto_start_sets_goal_for_agentic_message(hermes_home):
    sid = f"sid-gateway-auto-on-{uuid.uuid4().hex}"
    runner = _make_runner(sid, True)

    assert runner._maybe_auto_start_goal_for_event(
        _make_event("Debug the worker crash and verify the fix")
    ) is True
    state = runner._get_goal_manager_for_event(_make_event("noop"))[0].state
    assert state is not None
    assert state.status == "active"
    assert state.goal == "Debug the worker crash and verify the fix"


def test_gateway_auto_start_skips_commands_internal_and_existing_goal(hermes_home):
    sid = f"sid-gateway-auto-existing-{uuid.uuid4().hex}"
    runner = _make_runner(sid, True)
    mgr, _ = runner._get_goal_manager_for_event(_make_event("noop"))
    mgr.set("existing goal")

    assert runner._maybe_auto_start_goal_for_event(_make_event("/status")) is False
    assert runner._maybe_auto_start_goal_for_event(
        _make_event("Fix the failing tests", internal=True)
    ) is False
    assert runner._maybe_auto_start_goal_for_event(
        _make_event("Fix the failing tests")
    ) is False
    assert mgr.state.goal == "existing goal"
