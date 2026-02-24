"""Tests for the state machine engine."""

import asyncio
import pytest
from src.bot.state_machine import BotState, BotContext, BotStats, StateMachine


class TestBotStats:
    def test_initial_values(self):
        stats = BotStats()
        assert stats.monuments_visited == 0
        assert stats.errors == 0
        assert stats.runtime_seconds >= 0

    def test_to_dict(self):
        stats = BotStats()
        stats.battles_fought = 5
        d = stats.to_dict()
        assert d["battles_fought"] == 5
        assert "runtime_seconds" in d


class TestBotContext:
    def test_log_action(self):
        ctx = BotContext()
        ctx.log_action("Test action")
        assert len(ctx.action_log) == 1
        assert ctx.action_log[0]["message"] == "Test action"

    def test_log_truncation(self):
        ctx = BotContext()
        for i in range(250):
            ctx.log_action(f"Action {i}")
        assert len(ctx.action_log) == 200


class TestStateMachine:
    def test_register_handler(self):
        sm = StateMachine({})
        async def dummy(ctx, cfg):
            return BotState.STOPPED
        sm.register_handler(BotState.INITIALIZING, dummy)
        assert BotState.INITIALIZING in sm._handlers

    def test_pause_resume(self):
        sm = StateMachine({})
        sm._running = True
        sm.pause()
        assert sm.is_paused
        assert sm.state == BotState.PAUSED

        sm.resume()
        assert not sm.is_paused
        assert sm.state == BotState.OPENING_MINIMAP

    def test_stop(self):
        sm = StateMachine({})
        sm._running = True
        sm.stop()
        assert not sm.is_running
        assert sm.state == BotState.STOPPED

    def test_get_status(self):
        sm = StateMachine({})
        status = sm.get_status()
        assert "state" in status
        assert "stats" in status
        assert "running" in status

    def test_check_limits_consecutive_errors(self):
        sm = StateMachine({"bot": {"max_consecutive_errors": 3}})
        sm.context.stats.consecutive_errors = 3
        assert sm._check_limits() is True

    def test_check_limits_ok(self):
        sm = StateMachine({"bot": {"max_consecutive_errors": 5, "max_total_errors": 20}})
        sm.context.stats.consecutive_errors = 1
        assert sm._check_limits() is False


class TestStateMachineRun:
    @pytest.mark.asyncio
    async def test_runs_and_stops(self):
        sm = StateMachine({"bot": {}})
        call_count = 0

        async def handler(ctx, cfg):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                sm.stop()
                return BotState.STOPPED
            return BotState.INITIALIZING

        sm.register_handler(BotState.INITIALIZING, handler)
        sm.register_handler(BotState.STOPPED, handler)

        await sm.run()
        assert call_count >= 3
        assert sm.state == BotState.STOPPED
