"""Tests for the persistence module."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.bot.persistence import (
    CumulativeStats,
    EventLogger,
    PeriodicSaver,
    load_cumulative_stats,
    load_monument_tracker,
    save_cumulative_stats,
    save_monument_tracker,
)
from src.bot.state_machine import BotStats, MonumentRecord


# ---------------------------------------------------------------------------
# Monument tracker save/load
# ---------------------------------------------------------------------------

class TestMonumentTracker:
    def test_roundtrip(self, tmp_path):
        """Save and load monument tracker preserving all fields."""
        tracker = {i: MonumentRecord(slot=i) for i in range(1, 5)}
        tracker[1].last_status = "enemy"
        tracker[1].check_count = 5
        tracker[1].flipped_to_enemy = 2
        tracker[1].flipped_to_friendly = 1
        tracker[1].last_flip_time = 1000.0
        tracker[1].last_flip_from = "friendly"
        tracker[1].last_flip_to = "enemy"
        tracker[1].captured_at = 900.0
        tracker[1].times_captured = 3
        tracker[1].consecutive_enemy_checks = 4

        with patch("src.bot.persistence.MONUMENT_FILE", tmp_path / "tracker.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                save_monument_tracker(tracker)
                loaded = load_monument_tracker()

        rec = loaded[1]
        assert rec.last_status == "enemy"
        assert rec.check_count == 5
        assert rec.flipped_to_enemy == 2
        assert rec.flipped_to_friendly == 1
        assert rec.last_flip_time == 1000.0
        assert rec.last_flip_from == "friendly"
        assert rec.last_flip_to == "enemy"
        assert rec.captured_at == 900.0
        assert rec.times_captured == 3
        assert rec.consecutive_enemy_checks == 4

    def test_backwards_compat_old_json(self, tmp_path):
        """Old JSON files missing new fields should load with defaults."""
        old_data = {
            "1": {"slot": 1, "last_checked": 100.0, "last_status": "enemy",
                   "owner_name": "Bob", "check_count": 3, "flipped_to_enemy": 1},
            "2": {"slot": 2},
        }
        tracker_file = tmp_path / "tracker.json"
        tracker_file.write_text(json.dumps(old_data))

        with patch("src.bot.persistence.MONUMENT_FILE", tracker_file):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                loaded = load_monument_tracker()

        assert loaded[1].flipped_to_enemy == 1
        assert loaded[1].flipped_to_friendly == 0  # default
        assert loaded[1].captured_at == 0.0  # default
        assert loaded[1].consecutive_enemy_checks == 0  # default
        # Slots 3 and 4 should be default records
        assert loaded[3].last_status == "unknown"
        assert loaded[4].check_count == 0

    def test_load_missing_file(self, tmp_path):
        """Loading when file doesn't exist returns default tracker."""
        with patch("src.bot.persistence.MONUMENT_FILE", tmp_path / "nope.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                loaded = load_monument_tracker()
        assert len(loaded) == 4
        for i in range(1, 5):
            assert loaded[i].slot == i
            assert loaded[i].last_status == "unknown"


# ---------------------------------------------------------------------------
# Cumulative stats
# ---------------------------------------------------------------------------

class TestCumulativeStats:
    def test_roundtrip(self, tmp_path):
        """Save cumulative stats and load them back."""
        session = BotStats()
        session.monuments_visited = 10
        session.battles_fought = 5
        session.battles_won = 3
        session.defeats = 2
        session.monuments_captured = 1
        session.api_calls = 20
        session.total_cost = 0.05

        existing = CumulativeStats()

        with patch("src.bot.persistence.CUMULATIVE_FILE", tmp_path / "cumulative.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                result = save_cumulative_stats(session, existing)
                loaded = load_cumulative_stats()

        assert loaded.total_sessions == 1
        assert loaded.monuments_visited == 10
        assert loaded.battles_fought == 5
        assert loaded.battles_won == 3
        assert loaded.defeats == 2
        assert loaded.monuments_captured == 1
        assert loaded.api_calls == 20
        assert loaded.first_session != ""

    def test_accumulation(self, tmp_path):
        """Multiple sessions accumulate correctly."""
        s1 = BotStats()
        s1.battles_won = 5
        s1.defeats = 1

        s2 = BotStats()
        s2.battles_won = 3
        s2.defeats = 2

        existing = CumulativeStats()

        with patch("src.bot.persistence.CUMULATIVE_FILE", tmp_path / "cumulative.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                save_cumulative_stats(s1, existing)
                save_cumulative_stats(s2, existing)
                loaded = load_cumulative_stats()

        assert loaded.total_sessions == 2
        assert loaded.battles_won == 8
        assert loaded.defeats == 3

    def test_load_missing_file(self, tmp_path):
        """Loading when file doesn't exist returns fresh stats."""
        with patch("src.bot.persistence.CUMULATIVE_FILE", tmp_path / "nope.json"):
            loaded = load_cumulative_stats()
        assert loaded.total_sessions == 0


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------

class TestEventLogger:
    def test_writes_valid_jsonl(self, tmp_path):
        """EventLogger writes parseable JSON lines."""
        events_file = tmp_path / "events.jsonl"
        with patch("src.bot.persistence.EVENTS_FILE", events_file):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                el = EventLogger()
                el._path = events_file
                el.log("session_start")
                el.log("battle_won", slot=1, defender="Bob")
                el.log("monument_captured", slot=2)

        lines = events_file.read_text().strip().splitlines()
        assert len(lines) == 3

        e1 = json.loads(lines[0])
        assert e1["event"] == "session_start"
        assert "timestamp" in e1

        e2 = json.loads(lines[1])
        assert e2["event"] == "battle_won"
        assert e2["slot"] == 1
        assert e2["defender"] == "Bob"

        e3 = json.loads(lines[2])
        assert e3["event"] == "monument_captured"
        assert e3["slot"] == 2


# ---------------------------------------------------------------------------
# PeriodicSaver
# ---------------------------------------------------------------------------

class TestPeriodicSaver:
    def test_fires_at_interval(self):
        """PeriodicSaver only fires after the interval has elapsed."""
        call_count = 0

        def save_fn():
            nonlocal call_count
            call_count += 1

        saver = PeriodicSaver(interval_seconds=0.1)

        # First call always fires (last_save is 0)
        saver.maybe_save(save_fn)
        assert call_count == 1

        # Immediate second call should NOT fire
        saver.maybe_save(save_fn)
        assert call_count == 1

        # After interval, should fire again
        time.sleep(0.15)
        saver.maybe_save(save_fn)
        assert call_count == 2


# ---------------------------------------------------------------------------
# Score monument slot
# ---------------------------------------------------------------------------

class TestScoreMonumentSlot:
    """Test the _score_monument_slot method from StateHandlers."""

    def _make_config(self, **overrides):
        cfg = {
            "persistence": {
                "post_capture_watch_seconds": 300,
                "recheck_interval_seconds": 900,
                "well_defended_threshold": 0,
            }
        }
        cfg["persistence"].update(overrides)
        return cfg

    def _make_handlers(self):
        """Create a minimal StateHandlers-like object with just the scoring method."""
        from src.bot.states import StateHandlers
        # We only need the _score_monument_slot method, so use __new__ to skip __init__
        obj = object.__new__(StateHandlers)
        return obj

    def test_tier0_recently_captured(self):
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, captured_at=time.time() - 10)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 0

    def test_tier1_contested(self):
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, flipped_to_enemy=3)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 1

    def test_tier2_default(self):
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 2

    def test_tier3_well_defended(self):
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(
            slot=1,
            consecutive_enemy_checks=5,
            flipped_to_enemy=0,
            last_checked=time.time() - 60,  # checked recently
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 3

    def test_tier3_becomes_tier2_when_stale(self):
        """Well-defended slot that hasn't been checked in a long time should be tier 2."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(
            slot=1,
            consecutive_enemy_checks=5,
            flipped_to_enemy=0,
            last_checked=time.time() - 1000,  # older than recheck_interval
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 2


# ---------------------------------------------------------------------------
# BotStats.defeats
# ---------------------------------------------------------------------------

class TestBotStatsDefeats:
    def test_defeats_field_exists(self):
        stats = BotStats()
        assert stats.defeats == 0
        stats.defeats = 5
        assert stats.defeats == 5

    def test_defeats_in_to_dict(self):
        stats = BotStats()
        stats.defeats = 3
        d = stats.to_dict()
        assert "defeats" in d
        assert d["defeats"] == 3
