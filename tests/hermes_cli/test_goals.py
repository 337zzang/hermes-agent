"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_judge_response
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, reason, _pf, wait = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert verdict == "done"
        assert reason == "all good"
        assert wait is None



    def test_json_in_markdown_fence(self):
        from hermes_cli.goals import _parse_judge_response

        raw = '```json\n{"done": true, "reason": "done"}\n```'
        verdict, reason, _pf, _w = _parse_judge_response(raw)
        assert verdict == "done"
        assert "done" in reason

    def test_json_embedded_in_prose(self):
        """Some models prefix reasoning before emitting JSON — we extract it."""
        from hermes_cli.goals import _parse_judge_response

        raw = 'Looking at this... the agent says X. Verdict: {"done": false, "reason": "partial"}'
        verdict, reason, _pf, _w = _parse_judge_response(raw)
        assert verdict == "continue"
        assert reason == "partial"

    def test_json_with_braces_inside_reason_string(self):
        """Braces inside the reason string must not truncate extraction (regression).

        The old non-greedy ``\\{.*?\\}`` regex stopped at the first ``}`` — here
        the ``}`` after ``{x}`` — yielding invalid JSON and a false parse failure.
        """
        from hermes_cli.goals import _parse_judge_response

        raw = 'Reasoning first. Verdict: {"done": false, "reason": "need {x} fixed first"}'
        done, reason, parse_failed = _parse_judge_response(raw)
        assert done is False
        assert reason == "need {x} fixed first"
        assert parse_failed is False

    def test_json_with_nested_object_in_prose(self):
        """A nested object after prose must be extracted whole, not truncated."""
        from hermes_cli.goals import _parse_judge_response

        raw = 'Thinking... Verdict: {"done": true, "reason": "ok", "meta": {"score": 1}}'
        done, reason, parse_failed = _parse_judge_response(raw)
        assert done is True
        assert reason == "ok"
        assert parse_failed is False

    def test_string_done_values(self):
        from hermes_cli.goals import _parse_judge_response

        for s in ("true", "yes", "done", "1"):
            verdict, _, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert verdict == "done"
        for s in ("false", "no", "not yet"):
            verdict, _, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert verdict == "continue"

    def test_new_verdict_shape(self):
        """The explicit {"verdict": ...} shape is honored."""
        from hermes_cli.goals import _parse_judge_response

        v, _, _, _ = _parse_judge_response('{"verdict": "done", "reason": "r"}')
        assert v == "done"
        v, _, _, _ = _parse_judge_response('{"verdict": "continue", "reason": "r"}')
        assert v == "continue"

    def test_wait_verdict_with_pid(self):
        from hermes_cli.goals import _parse_judge_response

        v, reason, pf, wait = _parse_judge_response(
            '{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI running"}'
        )
        assert v == "wait"
        assert pf is False
        assert wait == {"pid": 4242}
        assert reason == "CI running"




# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:


    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("boom"),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_judge_says_done(self):
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": true, "reason": "achieved"}'))]
            ),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"

    def test_reasoning_only_response_uses_reasoning_fallback(self):
        """content=None with the verdict in a reasoning field still parses.

        Reasoning models (DeepSeek-R1, Qwen-QwQ, ...) can return content=None
        with the text in reasoning_content; without the fallback the judge sees
        an empty body and mis-counts it as a parse failure.
        """
        from hermes_cli import goals

        fake_client = MagicMock()
        msg = MagicMock(
            content=None,
            reasoning=None,
            reasoning_content='{"done": true, "reason": "done via reasoning"}',
            reasoning_details=None,
        )
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, parse_failed = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "done via reasoning"
        assert parse_failed is False

    def test_judge_timeout_resolved_from_config(self):
        """auxiliary.goal_judge.timeout flows through to the judge API call."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"done": false, "reason": "x"}'))]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ), patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"goal_judge": {"timeout": 7.5}}},
        ):
            goals.judge_goal("goal", "response")
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["timeout"] == 7.5

    def test_judge_says_continue(self):
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": false, "reason": "not yet"}'))]
            ),
        ):
            verdict, reason, _, _wd, _tf = goals.judge_goal("goal", "agent response")
        assert verdict == "continue"
        assert reason == "not yet"


class TestGoalAutoStartPolicy:
    def test_auto_start_config_is_opt_in(self):
        from hermes_cli.goals import is_goal_auto_start_enabled

        assert is_goal_auto_start_enabled({}) is False
        assert is_goal_auto_start_enabled({"goals": {"auto_start": {"enabled": False}}}) is False
        assert is_goal_auto_start_enabled({"goals": {"auto_start": {"enabled": True}}}) is True

    def test_auto_start_text_classifier_is_conservative(self):
        from hermes_cli.goals import should_auto_start_goal_from_text

        assert should_auto_start_goal_from_text("Fix the failing tests") is True
        assert should_auto_start_goal_from_text("구현하고 검증해줘") is True
        assert should_auto_start_goal_from_text("/goal status") is False
        assert should_auto_start_goal_from_text("[Continuing toward your standing goal]\nGoal: x") is False
        assert should_auto_start_goal_from_text("What time is it?") is False
        assert should_auto_start_goal_from_text("yes") is False


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert state.max_turns == 5
        assert state.turns_used == 0
        assert mgr.is_active()
        assert "active" in mgr.status_line().lower()
        assert "port the thing" in mgr.status_line()






