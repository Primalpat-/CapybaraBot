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
        tracker[1].garrison_power = 15000
        tracker[1].defender_powers = [5000, 4000, 6000]
        tracker[1].defender_names = ["Alice", "Bob", "Carol"]
        tracker[1].flip_velocity = 2.5
        tracker[1].flip_history = [
            {"time": 900.0, "from": "enemy", "to": "friendly"},
            {"time": 1000.0, "from": "friendly", "to": "enemy"},
        ]

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
        assert rec.garrison_power == 15000
        assert rec.defender_powers == [5000, 4000, 6000]
        assert rec.defender_names == ["Alice", "Bob", "Carol"]
        assert rec.flip_velocity == 2.5
        assert len(rec.flip_history) == 2
        assert rec.flip_history[0]["from"] == "enemy"
        assert rec.flip_history[1]["to"] == "enemy"

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
        # New fields should get defaults from old JSON
        assert loaded[1].garrison_power == -1
        assert loaded[1].defender_powers == []
        assert loaded[1].defender_names == []
        assert loaded[1].flip_velocity == 0.0
        assert loaded[1].flip_history == []
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
        session.defeats = 2
        session.monuments_captured = 1
        session.vision_calls = 20
        session.ocr_reads = 15

        existing = CumulativeStats()

        with patch("src.bot.persistence.CUMULATIVE_FILE", tmp_path / "cumulative.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                result = save_cumulative_stats(session, existing)
                loaded = load_cumulative_stats()

        assert loaded.total_sessions == 1
        assert loaded.monuments_visited == 10
        assert loaded.battles_fought == 5
        assert loaded.defeats == 2
        assert loaded.monuments_captured == 1
        assert loaded.vision_calls == 20
        assert loaded.ocr_reads == 15
        assert loaded.first_session != ""

    def test_accumulation(self, tmp_path):
        """Multiple sessions accumulate correctly."""
        s1 = BotStats()
        s1.vision_calls = 5
        s1.defeats = 1

        s2 = BotStats()
        s2.vision_calls = 3
        s2.defeats = 2

        existing = CumulativeStats()

        with patch("src.bot.persistence.CUMULATIVE_FILE", tmp_path / "cumulative.json"):
            with patch("src.bot.persistence.DATA_DIR", tmp_path):
                save_cumulative_stats(s1, existing)
                save_cumulative_stats(s2, existing)
                loaded = load_cumulative_stats()

        assert loaded.total_sessions == 2
        assert loaded.vision_calls == 8
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
            },
            "contest": {
                "flip_velocity_threshold": 2.0,
                "power_vulnerability_threshold": 5000,
            },
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

    def test_tier1_high_flip_velocity(self):
        """High flip velocity → tier 1 (actively contested)."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, flip_velocity=3.0)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 1

    def test_tier1_vulnerable_friendly(self):
        """Friendly with low garrison → tier 1."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, last_status="friendly", garrison_count=1)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 1

    def test_tier2_enemy(self):
        """Enemy monument → tier 2."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, last_status="enemy")}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 2

    def test_tier3_default(self):
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 3

    def test_tier4_well_defended(self):
        """Fully garrisoned, high power, no flips → tier 4 (skip until stale)."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(
            slot=1,
            last_status="friendly",
            flip_velocity=0.0,
            garrison_count=3,
            garrison_power=10000,
            last_checked=time.time() - 60,  # checked recently
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 4

    def test_tier4_becomes_tier3_when_stale(self):
        """Well-defended slot that hasn't been checked in a long time → tier 3."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(
            slot=1,
            last_status="friendly",
            flip_velocity=0.0,
            garrison_count=3,
            garrison_power=10000,
            last_checked=time.time() - 1000,  # older than recheck_interval
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 3

    def test_tier4_only_highest_power(self):
        """Only the highest-power friendly monument gets SAFE; weaker ones stay CHECK."""
        h = self._make_handlers()
        cfg = self._make_config()
        now = time.time()
        tracker = {
            1: MonumentRecord(
                slot=1, last_status="friendly", flip_velocity=0.0,
                garrison_count=3, garrison_power=40_000_000,
                last_checked=now - 60,
            ),
            2: MonumentRecord(
                slot=2, last_status="friendly", flip_velocity=0.0,
                garrison_count=3, garrison_power=15_000_000,
                last_checked=now - 60,
            ),
        }
        tier1, _ = h._score_monument_slot(1, tracker, cfg)
        tier2, _ = h._score_monument_slot(2, tracker, cfg)
        assert tier1 == 4  # strongest → SAFE
        assert tier2 == 3  # weaker → CHECK

    def test_tier3_lower_power_checked_first(self):
        """Within Tier 3, lower power gets a lower tiebreaker (checked sooner)."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {
            1: MonumentRecord(slot=1, garrison_power=20_000_000),
            2: MonumentRecord(slot=2, garrison_power=5_000_000),
        }
        _, tie1 = h._score_monument_slot(1, tracker, cfg)
        _, tie2 = h._score_monument_slot(2, tracker, cfg)
        assert tie2 < tie1  # lower power = lower tiebreaker = checked first

    def test_flip_velocity_decays_over_time(self):
        """Flip velocity decays with time — monument drops from urgent after ~30 min."""
        h = self._make_handlers()
        cfg = self._make_config()
        now = time.time()

        # Recent flip (5 min ago) — should still be urgent
        tracker = {1: MonumentRecord(
            slot=1, flip_velocity=3.0, last_flip_time=now - 300,
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 1  # still urgent

        # Old flip (1 hour ago) — velocity decays well below threshold
        tracker = {1: MonumentRecord(
            slot=1, flip_velocity=3.0, last_flip_time=now - 3600,
        )}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 3  # decayed to default/check

    def test_flip_velocity_no_last_flip_time_no_decay(self):
        """If last_flip_time is 0, no decay is applied (backwards compat)."""
        h = self._make_handlers()
        cfg = self._make_config()
        tracker = {1: MonumentRecord(slot=1, flip_velocity=3.0, last_flip_time=0.0)}
        tier, _ = h._score_monument_slot(1, tracker, cfg)
        assert tier == 1  # uses raw velocity, still urgent


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
