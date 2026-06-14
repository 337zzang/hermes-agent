import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals


class _FakeSessionEntry:
    session_id = "sid-gateway-goal-config"


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:goal-config"


@pytest.mark.asyncio
async def test_gateway_goal_uses_goals_max_turns_from_full_config(tmp_path, monkeypatch):
    """Gateway /goal should honor top-level goals.max_turns from config.yaml."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal ship the benchmark",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-config",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "⊙ Goal set (7-turn budget): ship the benchmark" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 7
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_budget_flag_overrides_max_turns(tmp_path, monkeypatch):
    """`/goal --budget N <text>` overrides the global goals.max_turns per goal."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal --budget 3 ship the benchmark",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-budget",
    )

    response = await GatewayRunner._handle_goal_command(runner, event)

    try:
        assert "3-turn budget" in response
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.max_turns == 3
        assert state.goal == "ship the benchmark"
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_goal_resume_extend_adds_budget_keeps_progress(tmp_path, monkeypatch):
    """`/goal resume extend N` adds N turns and keeps progress (no reset)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 10\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    # Seed a paused goal that already spent 7 of 10 turns.
    seed = goals.GoalManager("sid-gateway-goal-config")
    seed.set("do x", max_turns=10)
    seed.state.turns_used = 7
    goals.save_goal("sid-gateway-goal-config", seed.state)
    seed.pause()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/goal resume extend 5",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-goal-resume",
    )

    await GatewayRunner._handle_goal_command(runner, event)

    try:
        state = goals.GoalManager("sid-gateway-goal-config").state
        assert state is not None
        assert state.status == "active"
        assert state.max_turns == 15  # 10 + 5
        assert state.turns_used == 7  # progress kept (no reset)
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_gateway_subgoal_on_paused_goal_notes_resume(tmp_path, monkeypatch):
    """Adding /subgoal while the goal is paused tells the user it applies after resume."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("goals:\n  max_turns: 10\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    seed = goals.GoalManager("sid-gateway-goal-config")
    seed.set("do x")
    seed.pause()  # paused — has_goal() is still True, but not active

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/subgoal also handle errors",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-goal-config",
            chat_type="channel",
            user_id="user-goal-config",
        ),
        message_id="msg-subgoal-paused",
    )

    response = await GatewayRunner._handle_subgoal_command(runner, event)

    try:
        assert "Added subgoal" in response
        assert "resume" in response.lower()  # paused → applies after /goal resume
    finally:
        goals._DB_CACHE.clear()
