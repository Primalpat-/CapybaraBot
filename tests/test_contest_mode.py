"""Tests for contest mode logic."""

import time
import pytest

from src.bot.state_machine import BotContext, BotState, MonumentRecord
from src.bot.states import StateHandlers


def _make_handlers():
    """Create a minimal StateHandlers-like object for testing helper methods."""
    obj = object.__new__(StateHandlers)
    obj._defeat_counts = {}
    obj._unbeatable_players = set()
    obj._last_unbeatable_decay = time.time()
    obj._visited_slots = set()
    obj._event_logger = None
    return obj


def _make_config(**contest_overrides):
    cfg = {
        "contest": {
            "flip_velocity_threshold": 2.0,
            "recent_flip_seconds": 120,
            "poll_interval_seconds": 18,
            "max_duration_seconds": 300,
            "stable_seconds": 120,
            "power_vulnerability_threshold": 5000,
        },
        "bot": {
            "max_beatable_defender_power": 0,
            "max_defeats_before_skip": 2,
        },
        "persistence": {
            "post_capture_watch_seconds": 300,
            "recheck_interval_seconds": 900,
        },
    }
    cfg["contest"].update(contest_overrides)
    return cfg


class TestShouldEnterContest:
    def test_after_capture(self):
        h = _make_handlers()
        ctx = BotContext()
        ctx.monument_tracker[1].captured_at = time.time() - 5  # captured 5s ago
        assert h._should_enter_contest(1, ctx, _make_config()) is True

    def test_high_velocity(self):
        h = _make_handlers()
        ctx = BotContext()
        ctx.monument_tracker[1].flip_velocity = 3.0
        assert h._should_enter_contest(1, ctx, _make_config()) is True

    def test_recent_flip(self):
        h = _make_handlers()
        ctx = BotContext()
        ctx.monument_tracker[1].last_flip_time = time.time() - 60  # 60s ago
        assert h._should_enter_contest(1, ctx, _make_config()) is True

    def test_vulnerable_friendly(self):
        h = _make_handlers()
        ctx = BotContext()
        rec = ctx.monument_tracker[1]
        rec.last_status = "friendly"
        rec.garrison_power = 2000
        rec.last_checked = time.time() - 10
        assert h._should_enter_contest(1, ctx, _make_config()) is True

    def test_stable_monument(self):
        """Stable monument with no indicators should NOT enter contest."""
        h = _make_handlers()
        ctx = BotContext()
        rec = ctx.monument_tracker[1]
        rec.last_status = "friendly"
        rec.garrison_power = 10000
        rec.flip_velocity = 0.0
        rec.last_flip_time = 0.0
        rec.captured_at = 0.0
        rec.last_checked = time.time() - 10
        assert h._should_enter_contest(1, ctx, _make_config()) is False

    def test_old_capture_no_contest(self):
        """Captured long ago should NOT trigger."""
        h = _make_handlers()
        ctx = BotContext()
        ctx.monument_tracker[1].captured_at = time.time() - 600  # 10 min ago
        assert h._should_enter_contest(1, ctx, _make_config()) is False

    def test_invalid_slot(self):
        h = _make_handlers()
        ctx = BotContext()
        assert h._should_enter_contest(99, ctx, _make_config()) is False


class TestDefeatTracking:
    def test_single_defeat_not_unbeatable(self):
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        h._record_defeat("Alice", ctx, cfg)
        assert "Alice" not in h._unbeatable_players
        assert h._defeat_counts["Alice"] == 1

    def test_two_defeats_becomes_unbeatable(self):
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        h._record_defeat("Alice", ctx, cfg)
        h._record_defeat("Alice", ctx, cfg)
        assert "Alice" in h._unbeatable_players

    def test_victory_resets_defeat_count(self):
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        h._record_defeat("Alice", ctx, cfg)
        h._record_victory("Alice")
        assert "Alice" not in h._defeat_counts
        assert "Alice" not in h._unbeatable_players

    def test_decay_clears_all(self):
        h = _make_handlers()
        h._defeat_counts = {"Alice": 2, "Bob": 1}
        h._unbeatable_players = {"Alice"}
        h._last_unbeatable_decay = time.time() - 2000  # well past 1800s
        h._decay_unbeatable_list()
        assert len(h._unbeatable_players) == 0
        assert len(h._defeat_counts) == 0

    def test_no_decay_too_soon(self):
        h = _make_handlers()
        h._defeat_counts = {"Alice": 2}
        h._unbeatable_players = {"Alice"}
        h._last_unbeatable_decay = time.time() - 10  # only 10s ago
        h._decay_unbeatable_list()
        assert "Alice" in h._unbeatable_players


class TestCanBeatDefender:
    def test_unknown_defender_always_try(self):
        from src.vision.parser import DefenderInfo
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        d = DefenderInfo(slot=1, status="active", name="", power=0)
        assert h._can_beat_defender(d, ctx, cfg) is True

    def test_unbeatable_player(self):
        from src.vision.parser import DefenderInfo
        h = _make_handlers()
        h._unbeatable_players.add("Alice")
        ctx = BotContext()
        cfg = _make_config()
        d = DefenderInfo(slot=1, status="active", name="Alice", power=5000)
        assert h._can_beat_defender(d, ctx, cfg) is False

    def test_power_above_max(self):
        from src.vision.parser import DefenderInfo
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        cfg["bot"]["max_beatable_defender_power"] = 3000
        d = DefenderInfo(slot=1, status="active", name="Bob", power=5000)
        assert h._can_beat_defender(d, ctx, cfg) is False

    def test_power_below_max(self):
        from src.vision.parser import DefenderInfo
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        cfg["bot"]["max_beatable_defender_power"] = 10000
        d = DefenderInfo(slot=1, status="active", name="Bob", power=5000)
        assert h._can_beat_defender(d, ctx, cfg) is True

    def test_power_zero_means_no_limit(self):
        from src.vision.parser import DefenderInfo
        h = _make_handlers()
        ctx = BotContext()
        cfg = _make_config()
        cfg["bot"]["max_beatable_defender_power"] = 0  # disabled
        d = DefenderInfo(slot=1, status="active", name="Bob", power=999999)
        assert h._can_beat_defender(d, ctx, cfg) is True
