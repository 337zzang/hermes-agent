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


def _sg(text: str, status: str = "pending") -> dict:
    return {"text": text, "status": status}


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

    def test_judge_requests_json_object_response_format(self):
        """judge_goal asks for a structured JSON response when supported."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"done": false, "reason": "x"}'))]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            goals.judge_goal("goal", "response")
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs.get("response_format") == {"type": "json_object"}

    def test_judge_falls_back_when_response_format_rejected(self):
        """A provider that rejects response_format → retry once without it
        (the freeform-JSON parser still handles the reply)."""
        from hermes_cli import goals

        fake_client = MagicMock()
        good = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"done": true, "reason": "ok"}'))]
        )
        fake_client.chat.completions.create.side_effect = [
            ValueError("response_format is not supported by this model"),
            good,
        ]
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "response")
        assert verdict == "done"
        assert reason == "ok"
        assert fake_client.chat.completions.create.call_count == 2
        # The fallback retry must not carry response_format.
        assert "response_format" not in fake_client.chat.completions.create.call_args_list[1].kwargs

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

    def test_judge_includes_recent_responses_when_provided(self, hermes_home):
        from hermes_cli import goals

        captured = {}
        fake_client = MagicMock()

        def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": false, "reason": "x"}'))]
            )

        fake_client.chat.completions.create.side_effect = _create
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            goals.judge_goal(
                "ship it",
                "current turn",
                recent_responses=["prior turn", "current turn"],
            )

        user_msg = next(
            m["content"] for m in captured["messages"] if m["role"] == "user"
        )
        assert "Recent assistant progress" in user_msg
        assert "prior turn" in user_msg
        assert "current turn" in user_msg
        assert "Agent's most recent response" in user_msg

    def test_judge_recent_responses_are_capped_and_truncated(self, hermes_home):
        from hermes_cli import goals

        captured = {}
        fake_client = MagicMock()

        def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": false, "reason": "x"}'))]
            )

        fake_client.chat.completions.create.side_effect = _create
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            goals.judge_goal(
                "ship it",
                "current",
                recent_responses=["drop me", "keep 1", "k" * 2000, "keep 3"],
            )

        user_msg = next(
            m["content"] for m in captured["messages"] if m["role"] == "user"
        )
        assert "drop me" not in user_msg
        assert "keep 1" in user_msg
        assert "keep 3" in user_msg
        assert "… [truncated]" in user_msg

    def test_judge_history_with_subgoals_includes_pending_and_done(self, hermes_home):
        from hermes_cli import goals

        captured = {}
        fake_client = MagicMock()

        def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"done": true, "reason": "ok"}'))]
            )

        fake_client.chat.completions.create.side_effect = _create
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            goals.judge_goal(
                "ship it",
                "current",
                recent_responses=["prior"],
                subgoals=[_sg("already", "done"), _sg("remaining")],
            )

        user_msg = next(
            m["content"] for m in captured["messages"] if m["role"] == "user"
        )
        assert "Recent assistant progress" in user_msg
        assert "Pending criteria" in user_msg
        assert "2. [ ] remaining" in user_msg
        assert "Already satisfied" in user_msg
        assert "1. [✓] already" in user_msg


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

    def test_auto_start_rejects_non_string_and_empty(self):
        from hermes_cli.goals import should_auto_start_goal_from_text

        assert should_auto_start_goal_from_text(None) is False
        assert should_auto_start_goal_from_text(123) is False
        assert should_auto_start_goal_from_text(["fix it"]) is False
        assert should_auto_start_goal_from_text("") is False
        assert should_auto_start_goal_from_text("   ") is False

    def test_auto_start_rejects_short_acks_all_variants(self):
        """Every ack token (case-insensitive, EN + KO) is rejected."""
        from hermes_cli.goals import should_auto_start_goal_from_text

        for ack in ("yes", "YES", "Ok", "네", "예", "응", "아니요", "승인", "확인", "approve", "deny"):
            assert should_auto_start_goal_from_text(ack) is False, ack

    def test_auto_start_verb_question_still_triggers(self):
        """A question that contains an agentic verb still auto-starts — the
        short-question guard only suppresses verb-LESS short questions."""
        from hermes_cli.goals import should_auto_start_goal_from_text

        assert should_auto_start_goal_from_text("Can you fix the login bug?") is True

    def test_auto_start_verbless_question_rejected_regardless_of_length(self):
        """Verb-less questions never auto-start; the 160-char threshold does not
        flip the result (the agentic-verb check dominates) — guards against a
        behavior change if that branch is ever made significant."""
        from hermes_cli.goals import should_auto_start_goal_from_text

        short_q = "a" * 158 + "?"   # 159 chars, no agentic verb
        long_q = "a" * 159 + "?"    # 160 chars, no agentic verb
        assert len(short_q) == 159 and len(long_q) == 160
        assert should_auto_start_goal_from_text(short_q) is False
        assert should_auto_start_goal_from_text(long_q) is False

    def test_auto_start_korean_question_with_verb_is_known_quirk(self):
        """Current behavior: a Korean question containing an agentic verb
        ('확인') auto-starts even though it reads as a question. This is a known
        quirk the config-driven classifier (plan P14) is meant to address; the
        test pins today's behavior so that future change is intentional."""
        from hermes_cli.goals import should_auto_start_goal_from_text

        assert should_auto_start_goal_from_text("버그 가능성 확인 가능?") is True


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

    def test_set_auto_decompose_default_off(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-auto-decompose-off")
        with patch.object(
            goals,
            "_decompose_goal_into_subgoals",
            side_effect=AssertionError("auto-decompose should be off by default"),
        ):
            state = mgr.set("port the thing")
        assert state.subgoals == []

    def test_set_auto_decompose_on_populates_pending_subgoals(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-auto-decompose-on")
        with patch.object(
            goals,
            "_decompose_goal_into_subgoals",
            return_value=["write tests", "update docs", "verify"],
        ):
            state = mgr.set("port the thing", auto_decompose=True)
        assert state.subgoals == [
            _sg("write tests"),
            _sg("update docs"),
            _sg("verify"),
        ]

    def test_set_decompose_flag_cleans_goal_and_populates_subgoals(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-auto-decompose-flag")
        with patch.object(
            goals,
            "_decompose_goal_into_subgoals",
            return_value=["one", "two", "three"],
        ):
            state = mgr.set("ship the feature --decompose")
        assert state.goal == "ship the feature"
        assert state.subgoals == [_sg("one"), _sg("two"), _sg("three")]

    def test_set_auto_decompose_failure_fails_open(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-auto-decompose-fail")
        with patch.object(
            goals,
            "_decompose_goal_into_subgoals",
            side_effect=RuntimeError("judge down"),
        ):
            state = mgr.set("ship the feature", auto_decompose=True)
        assert state.subgoals == []

    def test_set_rejects_empty(self, hermes_home):
        from hermes_cli.goals import GoalManager





    def test_resume_keep_budget_preserves_turns(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="resume-keep")
        mgr.set("do x", max_turns=10)
        mgr.state.turns_used = 6
        mgr.pause()
        resumed = mgr.resume(reset_budget=False)
        assert resumed.turns_used == 6
        assert resumed.status == "active"

    def test_resume_extend_turns_adds_budget_and_keeps_progress(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="resume-ext")
        mgr.set("do x", max_turns=10)
        mgr.state.turns_used = 8
        mgr.pause()
        resumed = mgr.resume(extend_turns=5)
        assert resumed.max_turns == 15
        assert resumed.turns_used == 8  # progress kept when extending

    def test_bare_resume_resets_budget(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="resume-reset")
        mgr.set("do x", max_turns=10)
        mgr.state.turns_used = 6
        mgr.pause()
        resumed = mgr.resume()
        assert resumed.turns_used == 0

    def test_clear(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-5")
        mgr.set("goal")
        mgr.clear()
        assert mgr.state is None
        assert not mgr.is_active()

    def test_persistence_across_managers(self, hermes_home):
        """Key invariant: a second manager on the same session sees the goal.

        This is what makes /resume work — each session rebinds its
        GoalManager and picks up the saved state.
        """
        from hermes_cli.goals import GoalManager

        mgr1 = GoalManager(session_id="persist-sid")
        mgr1.set("do the thing")

        mgr2 = GoalManager(session_id="persist-sid")
        assert mgr2.state is not None
        assert mgr2.state.goal == "do the thing"
        assert mgr2.is_active()

    def test_evaluate_after_turn_done(self, hermes_home):
        """Judge says done → status=done, no continuation."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-1")
        mgr.set("ship it")

        with patch.object(goals, "judge_goal", return_value=("done", "shipped", False, None, False)):
            decision = mgr.evaluate_after_turn("I shipped the feature.")

        assert decision["verdict"] == "done"
        assert decision["should_continue"] is False
        assert decision["continuation_prompt"] is None
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_continue_under_budget(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-2", default_max_turns=5)
        mgr.set("a long goal")

        with patch.object(goals, "judge_goal", return_value=("continue", "more work", False, None, False)):
            decision = mgr.evaluate_after_turn("made some progress")

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert decision["continuation_prompt"] is not None
        assert "a long goal" in decision["continuation_prompt"]
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1

    def test_evaluate_after_turn_budget_exhausted(self, hermes_home):
        """When turn budget hits ceiling, auto-pause instead of continuing."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-3", default_max_turns=2)
        mgr.set("hard goal")

        with patch.object(goals, "judge_goal", return_value=("continue", "not yet", False, None, False)):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.turns_used == 1
            assert mgr.state.status == "active"

            d2 = mgr.evaluate_after_turn("step 2")
            # turns_used is now 2 which equals max_turns → paused
            assert d2["should_continue"] is False
            assert mgr.state.status == "paused"
            assert mgr.state.turns_used == 2
            assert "budget" in (mgr.state.paused_reason or "").lower()

    def test_evaluate_after_turn_inactive(self, hermes_home):
        """evaluate_after_turn is a no-op when goal isn't active."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="eval-sid-4")
        d = mgr.evaluate_after_turn("anything")
        assert d["verdict"] == "inactive"
        assert d["should_continue"] is False

        mgr.set("a goal")
        mgr.pause()
        d2 = mgr.evaluate_after_turn("anything")
        assert d2["verdict"] == "inactive"
        assert d2["should_continue"] is False

    def test_continuation_prompt_shape(self, hermes_home):
        """The continuation prompt must include the goal text verbatim —
        and must be safe to inject as a user-role message (prompt-cache
        invariants: no system-prompt mutation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-sid")
        mgr.set("port goal command to hermes")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "port goal command to hermes" in prompt
        assert prompt.strip()  # non-empty
        # No judge feedback yet (goal just set) → no feedback line.
        assert "Last review feedback" not in prompt

    def test_continuation_prompt_includes_last_judge_reason(self, hermes_home):
        """The continuation carries the judge's last feedback so the next turn
        knows what was missing (mirrors the kanban continuation)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="cont-reason")
        mgr.set("ship the feature")
        mgr.state.last_reason = "tests are still failing"
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "Last review feedback: tests are still failing" in prompt
        assert "ship the feature" in prompt


# ──────────────────────────────────────────────────────────────────────
# Smoke: CommandDef is wired
# ──────────────────────────────────────────────────────────────────────


def test_goal_command_in_registry():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("goal")
    assert cmd is not None
    assert cmd.name == "goal"


def test_goal_command_dispatches_in_cli_registry_helpers():
    """goal shows up in autocomplete / help categories alongside other Session cmds."""
    from hermes_cli.commands import COMMANDS, COMMANDS_BY_CATEGORY

    assert "/goal" in COMMANDS
    session_cmds = COMMANDS_BY_CATEGORY.get("Session", {})
    assert "/goal" in session_cmds


# ──────────────────────────────────────────────────────────────────────
# Auto-pause on consecutive judge parse failures
# ──────────────────────────────────────────────────────────────────────


class TestJudgeParseFailureAutoPause:
    """Regression: weak judge models (e.g. deepseek-v4-flash) that return
    empty strings or non-JSON prose must auto-pause the loop after N turns
    instead of burning the whole turn budget."""

    def test_parse_response_flags_empty_as_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, reason, parse_failed, _w = _parse_judge_response("")
        assert verdict == "continue"
        assert parse_failed is True
        assert "empty" in reason.lower()

    def test_parse_response_flags_non_json_as_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, reason, parse_failed, _w = _parse_judge_response(
            "Let me analyze whether the goal is fully satisfied based on the agent's response..."
        )
        assert verdict == "continue"
        assert parse_failed is True
        assert "not json" in reason.lower()

    def test_parse_response_clean_json_is_not_parse_failure(self):
        from hermes_cli.goals import _parse_judge_response

        verdict, _, parse_failed, _w = _parse_judge_response(
            '{"done": false, "reason": "more work"}'
        )
        assert verdict == "continue"
        assert parse_failed is False

    def test_api_error_does_not_count_as_parse_failure(self):
        """Transient network/API errors must not trip the auto-pause guard."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=RuntimeError("connection reset"),
        ):
            verdict, _, parse_failed, _wd, transport_failed = goals.judge_goal(
                "goal", "response"
            )
        assert verdict == "continue"
        assert parse_failed is False
        assert transport_failed is True

    def test_empty_judge_reply_flagged_as_parse_failure(self):
        """End-to-end: judge returns empty content → parse_failed=True."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=""))]),
        ):
            verdict, _, parse_failed, _wd, _tf = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert parse_failed is True

    def test_auto_pause_after_three_consecutive_parse_failures(self, hermes_home):
        """N=3 consecutive parse failures → auto-pause with config pointer."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES

        assert DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES == 3
        mgr = GoalManager(session_id="parse-fail-sid-1", default_max_turns=20)
        mgr.set("do a thing")

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge returned empty response", True, None, False)
        ):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 1

            d2 = mgr.evaluate_after_turn("step 2")
            assert d2["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 2

            d3 = mgr.evaluate_after_turn("step 3")
            assert d3["should_continue"] is False
            assert d3["status"] == "paused"
            assert mgr.state.consecutive_parse_failures == 3
            # Message points at the config surface so the user can fix it.
            assert "auxiliary" in d3["message"]
            assert "goal_judge" in d3["message"]
            assert "config.yaml" in d3["message"]

    def test_parse_failure_counter_resets_on_good_reply(self, hermes_home):
        """A single good judge reply resets the counter — transient flakes don't pause."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-2", default_max_turns=20)
        mgr.set("another goal")

        # Two parse failures…
        with patch.object(
            goals, "judge_goal", return_value=("continue", "not json", True, None, False)
        ):
            mgr.evaluate_after_turn("step 1")
            mgr.evaluate_after_turn("step 2")
            assert mgr.state.consecutive_parse_failures == 2

        # …then one clean reply resets the counter.
        with patch.object(
            goals, "judge_goal", return_value=("continue", "making progress", False, None, False)
        ):
            d = mgr.evaluate_after_turn("step 3")
            assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0

    def test_transport_failures_do_not_increment_parse_counter(self, hermes_home):
        """Transport failures use their own counter and a good reply resets both."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-3", default_max_turns=20)
        mgr.set("goal")
        assert mgr.state is not None

        with patch.object(
            goals,
            "judge_goal",
            return_value=(
                "continue",
                "judge error: RuntimeError",
                False,
                None,
                True,
            ),
        ):
            for _ in range(2):
                d = mgr.evaluate_after_turn("still going")
                assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0
            assert mgr.state.consecutive_transport_failures == 2
            assert mgr.state.status == "active"

        with patch.object(
            goals,
            "judge_goal",
            return_value=("continue", "making progress", False, None, False),
        ):
            d = mgr.evaluate_after_turn("recovered")

        assert d["should_continue"] is True
        assert mgr.state.consecutive_parse_failures == 0
        assert mgr.state.consecutive_transport_failures == 0

    def test_consecutive_parse_failures_persists_across_goalmanager_reloads(
        self, hermes_home
    ):
        """The counter must be durable so cross-session resumes see it."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, load_goal

        mgr = GoalManager(session_id="parse-fail-sid-4", default_max_turns=20)
        mgr.set("persistent goal")

        with patch.object(
            goals, "judge_goal", return_value=("continue", "empty", True, None, False)
        ):
            mgr.evaluate_after_turn("r")
            mgr.evaluate_after_turn("r")

        reloaded = load_goal("parse-fail-sid-4")
        assert reloaded is not None
        assert reloaded.consecutive_parse_failures == 2


# ──────────────────────────────────────────────────────────────────────
# /subgoal — user-added criteria
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateSubgoalsBackcompat:
    def test_old_state_meta_row_loads_without_subgoals(self):
        """A goal serialized BEFORE the subgoals field existed must
        round-trip with an empty list, not crash."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "consecutive_parse_failures": 0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "do a thing"
        assert state.subgoals == []

    def test_subgoals_round_trip(self):
        from hermes_cli.goals import GoalState
        state = GoalState(goal="g", subgoals=["a", "b", "c"])
        rt = GoalState.from_json(state.to_json())
        assert rt.subgoals == [_sg("a"), _sg("b"), _sg("c")]

    def test_legacy_string_subgoals_promote_to_pending_dicts(self):
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 0,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "subgoals": ["old str sub"],
        })
        state = GoalState.from_json(legacy)
        assert state.subgoals == [_sg("old str sub")]

    def test_dict_subgoals_round_trip_status(self):
        from hermes_cli.goals import GoalState

        state = GoalState(goal="g", subgoals=[_sg("done one", "done"), _sg("todo")])
        rt = GoalState.from_json(state.to_json())
        assert rt.subgoals == [_sg("done one", "done"), _sg("todo")]
        assert "[✓] done one" in rt.render_subgoals_block()
        assert "[ ] todo" in rt.render_subgoals_block()


class TestMigrateGoalToSession:
    """migrate_goal_to_session carries a /goal from a parent session to its
    compression continuation child (#33618). load_goal does a flat
    per-session lookup with no lineage walk, so without migration an active
    goal silently dies when compression rotates session_id."""

    def test_migrates_active_goal_to_child(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("parent-sid", GoalState(goal="ship the feature"))
        assert migrate_goal_to_session("parent-sid", "child-sid", reason="compression") is True
        child = load_goal("child-sid")
        assert child is not None and child.goal == "ship the feature"
        # Parent row archived (cleared) so only the child is active.
        parent = load_goal("parent-sid")
        assert parent is not None and parent.status == "cleared"

    def test_no_goal_to_migrate_returns_false(self, hermes_home):
        from hermes_cli.goals import migrate_goal_to_session, load_goal
        assert migrate_goal_to_session("empty-parent", "child2") is False
        assert load_goal("child2") is None

    def test_does_not_clobber_existing_child_goal(self, hermes_home):
        from hermes_cli.goals import save_goal, load_goal, migrate_goal_to_session, GoalState
        save_goal("p3", GoalState(goal="parent goal"))
        save_goal("c3", GoalState(goal="child already has one"))
        assert migrate_goal_to_session("p3", "c3") is False
        assert load_goal("c3").goal == "child already has one"

    def test_same_id_is_noop(self, hermes_home):
        from hermes_cli.goals import save_goal, migrate_goal_to_session, GoalState
        save_goal("same", GoalState(goal="g"))
        assert migrate_goal_to_session("same", "same") is False

    def test_cleared_goal_not_migrated(self, hermes_home):
        from hermes_cli.goals import save_goal, clear_goal, migrate_goal_to_session, load_goal, GoalState
        save_goal("p4", GoalState(goal="done already"))
        clear_goal("p4")
        assert migrate_goal_to_session("p4", "c4") is False
        assert load_goal("c4") is None


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == [_sg("use bullet points")]

    def test_add_subgoal_requires_active_goal(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-noactive")
        with pytest.raises(RuntimeError):
            mgr.add_subgoal("oops")

    def test_add_empty_subgoal_rejected(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-empty")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.add_subgoal("   ")

    def test_remove_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-remove")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")
        mgr.add_subgoal("third")
        removed = mgr.remove_subgoal(2)
        assert removed == "second"
        assert mgr.state.subgoals == [_sg("first"), _sg("third")]

    def test_mark_subgoal_done(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-done")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")
        marked = mgr.mark_subgoal_done(2)
        assert marked == "second"
        assert mgr.state.subgoals == [_sg("first"), _sg("second", "done")]
        assert "[✓] second" in mgr.render_subgoals()

    def test_remove_subgoal_out_of_range(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-oob")
        mgr.set("g")
        mgr.add_subgoal("only")
        with pytest.raises(IndexError):
            mgr.remove_subgoal(5)
        with pytest.raises(IndexError):
            mgr.remove_subgoal(0)

    def test_clear_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-clear")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        prev = mgr.clear_subgoals()
        assert prev == 2
        assert mgr.state.subgoals == []

    def test_subgoals_persist_across_reloads(self, hermes_home):
        """Subgoals stored in SessionDB survive a fresh GoalManager."""
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-persist")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")

        mgr2 = GoalManager(session_id="sub-persist")
        assert mgr2.state.subgoals == [_sg("first"), _sg("second")]


class TestContinuationPromptWithSubgoals:
    def test_empty_subgoals_uses_original_template(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-empty")
        mgr.set("ship the feature")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" not in prompt

    def test_with_subgoals_includes_them(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-with")
        mgr.set("ship the feature")
        mgr.add_subgoal("write tests")
        mgr.add_subgoal("update docs")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" in prompt
        assert "1. [ ] write tests" in prompt
        assert "2. [ ] update docs" in prompt


class TestJudgeGoalWithSubgoals:
    def test_judge_uses_subgoals_template_when_provided(self, hermes_home):
        """judge_goal switches templates when subgoals is non-empty.

        We don't actually call the model — we patch the aux client to
        capture the prompt that would be sent.
        """
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "all done"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            verdict, reason, parse_failed, _wd, _tf = goals.judge_goal(
                "ship the feature",
                "ok shipped",
                subgoals=["write tests", "update docs"],
            )

        # The aux client was called with a prompt that includes the subgoals.
        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" in user_msg
        assert "1. [ ] write tests" in user_msg
        assert "2. [ ] update docs" in user_msg
        assert "PENDING" in user_msg
        assert verdict == "done"

    def test_judge_uses_original_template_when_no_subgoals(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "ok"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        def _fake_call_llm(**kwargs):
            captured.update(kwargs)
            return _FakeResp()

        with patch("agent.auxiliary_client.call_llm", side_effect=_fake_call_llm):
            goals.judge_goal("ship it", "done", subgoals=None)

        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" not in user_msg
        assert "ship it" in user_msg


class TestStatusLineSubgoalCount:
    def test_status_line_no_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-empty")
        mgr.set("ship it")
        line = mgr.status_line()
        assert "ship it" in line
        assert "subgoal" not in line.lower()

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line


# ──────────────────────────────────────────────────────────────────────
# Wait barrier — parking the goal loop on a background process
# ──────────────────────────────────────────────────────────────────────


class TestWaitBarrier:
    """The /goal wait barrier parks the loop on a live PID and resumes when
    the process exits, without burning turns or calling the judge."""

    @staticmethod
    def _spawn_sleeper():
        """Start a short-lived child process; return its Popen handle."""
        import subprocess
        import sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    @staticmethod
    def _dead_pid():
        """A PID that is essentially guaranteed not to be running."""
        return 2_000_000_000

    def test_wait_on_requires_active_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="wb-noactive")
        with pytest.raises(RuntimeError):
            mgr.wait_on(12345)

    def test_wait_on_rejects_bad_pid(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="wb-badpid")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.wait_on(0)

    def test_parked_on_live_pid_does_not_continue_or_judge(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-live")
            mgr.set("ship it", max_turns=5)
            mgr.wait_on(proc.pid, reason="CI green")
            assert mgr.is_waiting() is True

            # The judge must NOT be called while parked, and no turn is burned.
            judge = MagicMock(return_value=("continue", "x", False, None, False))
            with patch.object(goals, "judge_goal", judge):
                decision = mgr.evaluate_after_turn("still waiting on CI")

            judge.assert_not_called()
            assert decision["verdict"] == "waiting"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.turns_used == 0  # no turn consumed while parked
            assert "CI green" in decision["message"]
            assert mgr.state.status == "active"  # still active, just parked
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_barrier_auto_clears_when_process_exits_and_loop_resumes(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        mgr = GoalManager(session_id="wb-exit")
        mgr.set("ship it", max_turns=5)
        mgr.wait_on(proc.pid, reason="build")
        assert mgr.is_waiting() is True

        # Kill the process — barrier should auto-clear and judging resumes.
        proc.terminate()
        proc.wait(timeout=10)

        assert mgr.is_waiting() is False  # lazy auto-clear
        assert mgr.state.waiting_on_pid is None

        with patch.object(goals, "judge_goal", return_value=("continue", "more", False, None, False)):
            decision = mgr.evaluate_after_turn("process finished, here are results")

        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.turns_used == 1  # now a turn IS consumed

    def test_dead_pid_never_parks(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="wb-dead")
        mgr.set("g", max_turns=5)
        mgr.wait_on(self._dead_pid(), reason="already-dead")
        # is_waiting clears the stale barrier immediately.
        assert mgr.is_waiting() is False

        with patch.object(goals, "judge_goal", return_value=("continue", "go", False, None, False)):
            decision = mgr.evaluate_after_turn("response")
        assert decision["should_continue"] is True

    def test_stop_waiting_clears_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-stop")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            assert mgr.is_waiting() is True
            assert mgr.stop_waiting() is True
            assert mgr.state.waiting_on_pid is None
            assert mgr.is_waiting() is False
            assert mgr.stop_waiting() is False  # idempotent
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_pause_and_resume_clear_barrier(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-pause")
            mgr.set("g")
            mgr.wait_on(proc.pid)
            mgr.pause()
            assert mgr.state.waiting_on_pid is None

            mgr.resume()
            assert mgr.state.waiting_on_pid is None
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_barrier_persists_and_reloads(self, hermes_home):
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="wb-persist")
            mgr.set("g")
            mgr.wait_on(proc.pid, reason="deploy")

            # Fresh manager loads the persisted barrier.
            mgr2 = GoalManager(session_id="wb-persist")
            assert mgr2.state.waiting_on_pid == proc.pid
            assert mgr2.state.waiting_reason == "deploy"
            assert mgr2.is_waiting() is True
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_old_state_row_loads_without_barrier_fields(self, hermes_home):
        """Backwards-compat: a state_meta row written before the barrier
        existed must load with no barrier."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "old goal",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
        })
        st = GoalState.from_json(legacy)
        assert st.goal == "old goal"
        assert st.waiting_on_pid is None
        assert st.waiting_reason is None
        assert st.waiting_since == 0.0
        assert st.waiting_until == 0.0


# ──────────────────────────────────────────────────────────────────────
# Judge-driven auto-wait — the judge parks the loop on its own
# ──────────────────────────────────────────────────────────────────────


class TestJudgeDrivenWait:
    """The judge returns a `wait` verdict (given live background-process
    context) and the loop parks automatically — no manual /goal wait."""

    @staticmethod
    def _spawn_sleeper():
        import subprocess, sys
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_judge_wait_pid_parks_loop(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        proc = self._spawn_sleeper()
        try:
            mgr = GoalManager(session_id="jw-pid", default_max_turns=10)
            mgr.set("ship the PR")
            # Judge sees the running process and says wait-on-pid.
            with patch.object(
                goals, "judge_goal",
                return_value=("wait", "CI watcher still running", False, {"pid": proc.pid}, False),
            ):
                decision = mgr.evaluate_after_turn(
                    "Pushed the PR, watching CI.",
                    background_processes=[{
                        "pid": proc.pid, "command": "wait_for_pr_green.sh",
                        "status": "running", "uptime_seconds": 12,
                    }],
                )
            assert decision["verdict"] == "wait"
            assert decision["should_continue"] is False
            assert decision["continuation_prompt"] is None
            assert mgr.state.waiting_on_pid == proc.pid
            assert mgr.is_waiting() is True

            # Next turn while still parked: judge must NOT be called again.
            judge = MagicMock()
            with patch.object(goals, "judge_goal", judge):
                d2 = mgr.evaluate_after_turn("still going")
            judge.assert_not_called()
            assert d2["verdict"] == "waiting"
            assert d2["should_continue"] is False
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    def test_judge_wait_seconds_parks_loop(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-secs", default_max_turns=10)
        mgr.set("retry after backoff")
        with patch.object(
            goals, "judge_goal",
            return_value=("wait", "rate limited", False, {"seconds": 120}, False),
        ):
            decision = mgr.evaluate_after_turn("Hit a 429, backing off.")
        assert decision["verdict"] == "wait"
        assert decision["should_continue"] is False
        assert mgr.state.waiting_until > 0
        assert mgr.state.waiting_on_pid is None
        assert mgr.is_waiting() is True

    def test_time_barrier_clears_after_deadline(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-deadline")
        mgr.set("g")
        mgr.wait_for_seconds(120, reason="backoff")
        assert mgr.is_waiting() is True
        # Force the deadline into the past → barrier auto-clears.
        mgr.state.waiting_until = time.time() - 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_until == 0.0

    def test_continue_verdict_still_continues_with_background(self, hermes_home):
        """A running process present but judge says continue → normal loop."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="jw-cont", default_max_turns=10)
        mgr.set("do work")
        with patch.object(
            goals, "judge_goal",
            return_value=("continue", "more to do", False, None, False),
        ):
            decision = mgr.evaluate_after_turn(
                "made progress",
                background_processes=[{"pid": 999999, "command": "x", "status": "running"}],
            )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
        assert mgr.state.waiting_on_pid is None


# ──────────────────────────────────────────────────────────────────────
# Session/trigger barrier — wait on a process's OWN trigger, not just exit
# ──────────────────────────────────────────────────────────────────────


class TestSessionTriggerBarrier:
    """The session barrier (wait_on_session) releases when a process's own
    trigger fires — a watch_patterns match mid-run (process may never exit)
    OR exit — not only on PID exit. CI-safe: uses synthetic registry session
    objects, no real child processes."""

    @staticmethod
    def _inject(sid, *, watch_patterns=None, exited=False):
        import time as _t
        from tools.process_registry import process_registry, ProcessSession
        s = ProcessSession(id=sid, command="watcher.sh", task_id="t",
                           session_key="", cwd="/tmp", started_at=_t.time())
        if watch_patterns:
            s.watch_patterns = list(watch_patterns)
        s.exited = exited
        if exited:
            process_registry._finished[sid] = s
        else:
            process_registry._running[sid] = s
        return s, process_registry

    def test_registry_is_session_waiting_running_unmatched(self, hermes_home):
        s, reg = self._inject("proc_t1", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t1") is True

    def test_registry_releases_on_watch_match_while_alive(self, hermes_home):
        s, reg = self._inject("proc_t2", watch_patterns=["READY"])
        assert reg.is_session_waiting("proc_t2") is True
        s._watch_hits = 1  # what _check_watch_patterns sets on a match
        # Released even though the process is STILL running (never exited).
        assert s.exited is False
        assert reg.is_session_waiting("proc_t2") is False

    def test_registry_releases_on_exit_plain_session(self, hermes_home):
        s, reg = self._inject("proc_t3")  # no watch pattern
        assert reg.is_session_waiting("proc_t3") is True
        s.exited = True
        assert reg.is_session_waiting("proc_t3") is False

    def test_registry_unknown_session_never_waits(self, hermes_home):
        from tools.process_registry import process_registry
        assert process_registry.is_session_waiting("proc_does_not_exist") is False

    def test_goal_parks_on_session_and_releases_on_trigger(self, hermes_home):
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        s, reg = self._inject("proc_t4", watch_patterns=["BUILD SUCCESSFUL"])
        mgr = GoalManager(session_id="st-goal", default_max_turns=10)
        mgr.set("wait for the build to succeed")
        with patch.object(
            goals, "judge_goal",
            return_value=("wait", "blocked on build", False, {"session_id": "proc_t4"}, False),
        ):
            decision = mgr.evaluate_after_turn(
                "Started the build watcher.",
                background_processes=[{
                    "session_id": "proc_t4", "pid": 4242, "command": "watcher.sh",
                    "status": "running", "watch_patterns": ["BUILD SUCCESSFUL"],
                    "watch_hit": False,
                }],
            )
        assert decision["verdict"] == "wait"
        assert mgr.state.waiting_on_session == "proc_t4"
        assert mgr.is_waiting() is True

        # Judge must NOT be called again while parked.
        judge = MagicMock()
        with patch.object(goals, "judge_goal", judge):
            d2 = mgr.evaluate_after_turn("still building")
        judge.assert_not_called()
        assert d2["should_continue"] is False

        # Trigger fires mid-run (process still alive) → barrier releases.
        s._watch_hits = 1
        assert mgr.is_waiting() is False
        assert mgr.state.waiting_on_session is None

        # Loop resumes with a real judge verdict.
        with patch.object(goals, "judge_goal",
                          return_value=("continue", "build done", False, None, False)):
            d3 = mgr.evaluate_after_turn("build succeeded")
        assert d3["should_continue"] is True

    def test_wait_on_session_validation(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="st-val")
        # No active goal → RuntimeError
        try:
            mgr.wait_on_session("proc_x")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        mgr.set("g")
        try:
            mgr.wait_on_session("")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_session_directive_parsed_from_judge(self, hermes_home):
        from hermes_cli.goals import _parse_judge_response
        v, _, pf, wd = _parse_judge_response(
            '{"verdict": "wait", "wait_on_session": "proc_abc", "reason": "r"}'
        )
        assert v == "wait"
        assert pf is False
        assert wd == {"session_id": "proc_abc"}

    def test_old_state_loads_without_session_field(self, hermes_home):
        from hermes_cli.goals import GoalState
        st = GoalState.from_json(json.dumps({
            "goal": "g", "status": "active", "turns_used": 0, "max_turns": 20,
        }))
        assert st.waiting_on_session is None


# ──────────────────────────────────────────────────────────────────────
# Completion contract (Codex-inspired structured goals)
# ──────────────────────────────────────────────────────────────────────


class TestParseContract:
    def test_plain_goal_no_contract(self):
        from hermes_cli.goals import parse_contract

        headline, contract = parse_contract("Migrate auth to JWT")
        assert headline == "Migrate auth to JWT"
        assert contract.is_empty()

    def test_incidental_colon_not_treated_as_field(self):
        from hermes_cli.goals import parse_contract

        # "Fix bug:" — "fix bug" is not a known alias, so the whole line
        # stays the headline and no contract field is populated.
        headline, contract = parse_contract("Fix bug: the parser drops trailing commas")
        assert headline == "Fix bug: the parser drops trailing commas"
        assert contract.is_empty()

    def test_inline_fields_parsed(self):
        from hermes_cli.goals import parse_contract

        text = (
            "Migrate auth to JWT\n"
            "verify: the auth test suite passes\n"
            "constraints: keep the /login response shape unchanged\n"
            "boundaries: only touch services/auth and its tests\n"
            "stop when: a schema change needs product sign-off"
        )
        headline, contract = parse_contract(text)
        assert headline == "Migrate auth to JWT"
        assert contract.verification == "the auth test suite passes"
        assert contract.constraints == "keep the /login response shape unchanged"
        assert contract.boundaries == "only touch services/auth and its tests"
        assert contract.stop_when == "a schema change needs product sign-off"
        assert not contract.is_empty()

    def test_alias_variants(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("Goal\nverified by: tests green\npreserve: public API")
        assert c.verification == "tests green"
        assert c.constraints == "public API"

    def test_multiple_lines_same_field_joined(self):
        from hermes_cli.goals import parse_contract

        _, c = parse_contract("G\nconstraints: a\nconstraints: b")
        assert c.constraints == "a b"


class TestGoalContractSerialization:
    def test_roundtrip_with_contract(self):
        from hermes_cli.goals import GoalState, GoalContract

        state = GoalState(
            goal="ship it",
            contract=GoalContract(
                verification="pytest passes",
                constraints="don't break the API",
            ),
        )
        restored = GoalState.from_json(state.to_json())
        assert restored.goal == "ship it"
        assert restored.contract.verification == "pytest passes"
        assert restored.contract.constraints == "don't break the API"
        assert restored.has_contract()

    def test_old_row_without_contract_loads_clean(self):
        # A state_meta row written before this feature has no "contract" key.
        from hermes_cli.goals import GoalState

        legacy = '{"goal": "old goal", "status": "active", "turns_used": 2}'
        state = GoalState.from_json(legacy)
        assert state.goal == "old goal"
        assert state.turns_used == 2
        assert state.contract.is_empty()
        assert not state.has_contract()

    def test_render_block_omits_empty_fields(self):
        from hermes_cli.goals import GoalContract

        block = GoalContract(outcome="X", verification="Y").render_block()
        assert "Outcome: X" in block
        assert "Verification: Y" in block
        assert "Constraints" not in block


class TestGoalManagerContract:
    def test_set_with_contract(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-set")
        mgr.set("ship it", contract=GoalContract(verification="tests pass"))
        assert mgr.has_contract()
        assert "contract" in mgr.status_line()

    def test_set_without_contract_no_marker(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="c-none")
        mgr.set("ship it")
        assert not mgr.has_contract()
        assert "contract" not in mgr.status_line()

    def test_continuation_prompt_includes_contract(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-cont")
        mgr.set("ship it", contract=GoalContract(verification="run pytest"))
        prompt = mgr.next_continuation_prompt()
        assert "Completion contract" in prompt
        assert "run pytest" in prompt
        assert "concrete evidence" in prompt

    def test_set_contract_after_the_fact(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        mgr = GoalManager(session_id="c-after")
        mgr.set("ship it")
        assert not mgr.has_contract()
        mgr.set_contract(GoalContract(verification="x"))
        assert mgr.has_contract()
        # Survives reload.
        from hermes_cli.goals import GoalManager as GM2
        assert GM2(session_id="c-after").has_contract()

    def test_persistence_roundtrip(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalContract

        GoalManager(session_id="c-persist").set(
            "ship it", contract=GoalContract(outcome="O", verification="V")
        )
        reloaded = GoalManager(session_id="c-persist")
        assert reloaded.state.contract.outcome == "O"
        assert reloaded.state.contract.verification == "V"


class TestJudgeWithContract:
    def _fake_call_llm(self, captured, content='{"done": false, "reason": "more"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_uses_contract_template(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._fake_call_llm(captured)):
            goals.judge_goal(
                "ship it", "I think it's done",
                contract=GoalContract(verification="pytest -q passes"),
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        assert "completion contract" in user_msg.lower()
        assert "pytest -q passes" in user_msg
        assert "concrete evidence" in user_msg

    def test_contract_plus_subgoals_combine(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._fake_call_llm(captured)):
            goals.judge_goal(
                "ship it", "done",
                subgoals=["write changelog"],
                contract=GoalContract(verification="pytest passes"),
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        assert "pytest passes" in user_msg
        assert "write changelog" in user_msg


class TestDraftContract:
    def test_draft_parses_json(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        class _FakeMsg:
            content = (
                '{"outcome": "auth on JWT", "verification": "auth suite green", '
                '"constraints": "no API change", "boundaries": "services/auth", '
                '"stop_when": "schema change needed"}'
            )
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_FakeResp()):
            contract = goals.draft_contract("Migrate auth to JWT")
        assert contract is not None
        assert contract.outcome == "auth on JWT"
        assert contract.verification == "auth suite green"
        assert not contract.is_empty()

    def test_draft_returns_none_on_bad_json(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        class _FakeMsg:
            content = "I cannot produce JSON, sorry"
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_FakeResp()):
            assert goals.draft_contract("anything") is None

    def test_draft_returns_none_when_no_client(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        with patch("agent.auxiliary_client.call_llm",
                   side_effect=RuntimeError("No LLM provider configured")):
            assert goals.draft_contract("anything") is None


# ──────────────────────────────────────────────────────────────────────
# Compose: completion contract + wait barrier in one judge call
# ──────────────────────────────────────────────────────────────────────


class TestContractAndBackgroundCompose:
    """A contract goal blocked on a background process must surface BOTH
    the contract block and the background-process list to the judge, so it
    can return either done (evidence met) or wait (parked on the poller)."""

    def _capture_call_llm(self, captured, content='{"verdict": "wait", "wait_on_pid": 4242, "reason": "CI still running"}'):
        """judge_goal routes through call_llm (#35566) — capture its kwargs."""
        class _FakeMsg:
            pass
        _FakeMsg.content = content
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]

        def _fake(**kwargs):
            captured.update(kwargs)
            return _FakeResp()
        return _fake

    def test_judge_prompt_carries_contract_and_background(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        bg = [{
            "session_id": "ci-watch", "pid": 4242, "status": "running",
            "command": "wait_for_pr_green.sh 50501", "trigger": "exit",
        }]
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._capture_call_llm(captured)):
            verdict, reason, parse_failed, wait_directive, _tf = goals.judge_goal(
                "ship the PR",
                "I pushed and started the CI watcher; waiting on it now.",
                contract=GoalContract(verification="PR CI goes green"),
                background_processes=bg,
            )
        user_msg = next(
            (m["content"] for m in (captured.get("messages") or []) if m["role"] == "user"), ""
        )
        # Both surfaces present in one prompt.
        assert "completion contract" in user_msg.lower()
        assert "PR CI goes green" in user_msg
        assert "Background processes" in user_msg
        assert "4242" in user_msg
        # The judge can return a wait verdict on a contract goal.
        assert verdict == "wait"
        assert wait_directive and wait_directive.get("pid") == 4242

    def test_contract_goal_can_still_complete_on_evidence(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals
        from hermes_cli.goals import GoalContract

        captured = {}
        bg = [{"session_id": "ci", "pid": 4242, "status": "running", "command": "ci", "trigger": "exit"}]
        with patch("agent.auxiliary_client.call_llm",
                   side_effect=self._capture_call_llm(
                       captured,
                       content='{"verdict": "done", "reason": "CI is green, evidence shown"}',
                   )):
            verdict, reason, parse_failed, wait_directive, _tf = goals.judge_goal(
                "ship the PR",
                "CI finished: 30 passed, 0 failed. Done.",
                contract=GoalContract(verification="PR CI goes green"),
                background_processes=bg,
            )
        assert verdict == "done"
        assert wait_directive is None


# ──────────────────────────────────────────────────────────────────────
# Pure config/string helpers — fallback guards (incl. the documented
# reasoning-model truncation regression that max_tokens defends against).
# ──────────────────────────────────────────────────────────────────────


class TestPureHelpers:
    def test_truncate_under_limit_unchanged(self):
        from hermes_cli.goals import _truncate

        assert _truncate("short", 100) == "short"
        assert _truncate("", 100) == ""

    def test_truncate_over_limit_appends_marker(self):
        from hermes_cli.goals import _truncate

        out = _truncate("x" * 50, 10)
        assert out.startswith("x" * 10)
        assert out.endswith("… [truncated]")

    def test_goal_judge_max_tokens_from_config(self):
        from hermes_cli import goals

        with patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"goal_judge": {"max_tokens": 1234}}},
        ):
            assert goals._goal_judge_max_tokens() == 1234

    def test_goal_judge_max_tokens_falls_back_on_bad_value(self):
        from hermes_cli import goals

        for bad in (0, -5, "nope", None):
            with patch(
                "hermes_cli.config.load_config",
                return_value={"auxiliary": {"goal_judge": {"max_tokens": bad}}},
            ):
                assert goals._goal_judge_max_tokens() == goals.DEFAULT_JUDGE_MAX_TOKENS
        with patch("hermes_cli.config.load_config", return_value={}):
            assert goals._goal_judge_max_tokens() == goals.DEFAULT_JUDGE_MAX_TOKENS

    def test_goal_judge_timeout_from_config(self):
        from hermes_cli import goals

        with patch(
            "hermes_cli.config.load_config",
            return_value={"auxiliary": {"goal_judge": {"timeout": 12.5}}},
        ):
            assert goals._goal_judge_timeout() == 12.5

    def test_goal_judge_timeout_falls_back_on_bad_value(self):
        from hermes_cli import goals

        for bad in (0, -1, "nope", None):
            with patch(
                "hermes_cli.config.load_config",
                return_value={"auxiliary": {"goal_judge": {"timeout": bad}}},
            ):
                assert goals._goal_judge_timeout() == goals.DEFAULT_JUDGE_TIMEOUT
        with patch("hermes_cli.config.load_config", return_value={}):
            assert goals._goal_judge_timeout() == goals.DEFAULT_JUDGE_TIMEOUT

    def test_judge_history_turns_from_config_defaults_and_clamps(self):
        from hermes_cli.goals import goal_judge_history_turns_from_config

        assert goal_judge_history_turns_from_config({"goals": {}}) == 3
        assert goal_judge_history_turns_from_config({"goals": {"judge_history_turns": 2}}) == 2
        assert goal_judge_history_turns_from_config({"goals": {"judge_history_turns": 99}}) == 3
        assert goal_judge_history_turns_from_config({"goals": {"judge_history_turns": 0}}) == 0

    def test_extract_recent_assistant_responses(self):
        from hermes_cli.goals import extract_recent_assistant_responses

        messages = [
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "middle"}],
            },
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "new"},
        ]
        assert extract_recent_assistant_responses(messages, limit=2) == ["middle", "new"]


class TestParseGoalBudgetFlag:
    def test_no_flag_returns_full_text(self):
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("ship the feature") == (None, "ship the feature")

    def test_budget_flag_parsed(self):
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("--budget 50 ship it") == (50, "ship it")

    def test_turns_alias_parsed(self):
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("--turns 5 do the thing") == (5, "do the thing")

    def test_invalid_budget_value_left_intact(self):
        """A non-int after the flag is not a budget — leave the text untouched."""
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("--budget abc do x") == (None, "--budget abc do x")

    def test_non_positive_budget_left_intact(self):
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("--budget 0 do x") == (None, "--budget 0 do x")

    def test_flag_only_without_text(self):
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("--budget 50") == (50, "")

    def test_flag_not_at_start_is_ignored(self):
        """Only a leading flag counts — a stray --budget inside the goal stays."""
        from hermes_cli.goals import parse_goal_budget_flag

        assert parse_goal_budget_flag("fix the --budget parser") == (
            None,
            "fix the --budget parser",
        )


class TestParseGoalDecomposeFlag:
    def test_no_flag_returns_text(self):
        from hermes_cli.goals import parse_goal_decompose_flag

        assert parse_goal_decompose_flag("ship the feature") == (
            False,
            "ship the feature",
        )

    def test_leading_flag(self):
        from hermes_cli.goals import parse_goal_decompose_flag

        assert parse_goal_decompose_flag("--decompose ship the feature") == (
            True,
            "ship the feature",
        )

    def test_trailing_flag(self):
        from hermes_cli.goals import parse_goal_decompose_flag

        assert parse_goal_decompose_flag("ship the feature --decompose") == (
            True,
            "ship the feature",
        )


class TestParseResumeFlags:
    def test_bare_resets(self):
        from hermes_cli.goals import parse_resume_flags

        assert parse_resume_flags("") == (True, None)

    def test_keep_budget(self):
        from hermes_cli.goals import parse_resume_flags

        assert parse_resume_flags("--keep-budget") == (False, None)

    def test_extend(self):
        from hermes_cli.goals import parse_resume_flags

        assert parse_resume_flags("extend 5") == (False, 5)

    def test_invalid_extend_falls_back_to_reset(self):
        from hermes_cli.goals import parse_resume_flags

        assert parse_resume_flags("extend abc") == (True, None)
        assert parse_resume_flags("extend") == (True, None)

    def test_unknown_flag_falls_back_to_reset(self):
        from hermes_cli.goals import parse_resume_flags

        assert parse_resume_flags("--whatever") == (True, None)


class TestGoalStatePersistence:
    """Round-trip + fail-safe guards for the /resume persistence backbone."""

    def test_goalstate_full_roundtrip(self):
        from hermes_cli.goals import GoalState

        st = GoalState(
            goal="ship it",
            status="paused",
            turns_used=4,
            max_turns=12,
            created_at=111.0,
            last_turn_at=222.0,
            last_verdict="continue",
            last_reason="needs tests",
            paused_reason="turn budget exhausted (12/12)",
            consecutive_parse_failures=2,
            subgoals=["a", "b"],
        )
        assert GoalState.from_json(st.to_json()) == st

    def test_load_goal_returns_none_on_corrupt_row(self, hermes_home):
        from hermes_cli import goals

        db = goals._get_session_db()
        assert db is not None
        db.set_meta(goals._meta_key("corrupt-sid"), "{not valid json")
        assert goals.load_goal("corrupt-sid") is None

    def test_module_clear_goal_marks_cleared(self, hermes_home):
        from hermes_cli import goals

        goals.save_goal("clr-sid", goals.GoalState(goal="do x"))
        goals.clear_goal("clr-sid")
        loaded = goals.load_goal("clr-sid")
        assert loaded is not None
        assert loaded.status == "cleared"
