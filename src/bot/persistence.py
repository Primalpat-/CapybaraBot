"""Persistence layer — saves monument data, cumulative stats, and event logs across sessions."""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.bot.state_machine import MonumentRecord

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MONUMENT_FILE = DATA_DIR / "monument_tracker.json"
CUMULATIVE_FILE = DATA_DIR / "cumulative_stats.json"
EVENTS_FILE = DATA_DIR / "events.jsonl"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CumulativeStats
# ---------------------------------------------------------------------------

@dataclass
class CumulativeStats:
    total_sessions: int = 0
    total_runtime_seconds: float = 0.0
    monuments_visited: int = 0
    battles_fought: int = 0
    battles_won: int = 0
    defeats: int = 0
    monuments_captured: int = 0
    api_calls: int = 0
    total_cost: float = 0.0
    first_session: str = ""
    last_session: str = ""


def load_cumulative_stats() -> CumulativeStats:
    if not CUMULATIVE_FILE.exists():
        return CumulativeStats()
    try:
        data = json.loads(CUMULATIVE_FILE.read_text(encoding="utf-8"))
        return CumulativeStats(**{k: v for k, v in data.items() if k in CumulativeStats.__dataclass_fields__})
    except Exception:
        logger.warning("Failed to load cumulative stats, starting fresh", exc_info=True)
        return CumulativeStats()


def save_cumulative_stats(session_stats, existing: CumulativeStats) -> CumulativeStats:
    """Merge a session's BotStats into the cumulative totals and persist."""
    now = datetime.now(timezone.utc).isoformat()
    existing.total_sessions += 1
    existing.total_runtime_seconds += session_stats.runtime_seconds
    existing.monuments_visited += session_stats.monuments_visited
    existing.battles_fought += session_stats.battles_fought
    existing.battles_won += session_stats.battles_won
    existing.defeats += session_stats.defeats
    existing.monuments_captured += session_stats.monuments_captured
    existing.api_calls += session_stats.api_calls
    existing.total_cost += session_stats.total_cost
    if not existing.first_session:
        existing.first_session = now
    existing.last_session = now

    _ensure_data_dir()
    _atomic_write(CUMULATIVE_FILE, json.dumps(asdict(existing), indent=2))
    logger.info("Cumulative stats saved")
    return existing


# ---------------------------------------------------------------------------
# Monument tracker save/load
# ---------------------------------------------------------------------------

def save_monument_tracker(tracker: dict[int, MonumentRecord]) -> None:
    """Atomically save monument tracker to JSON."""
    data = {}
    for slot, rec in tracker.items():
        data[str(slot)] = {
            "slot": rec.slot,
            "last_checked": rec.last_checked,
            "last_status": rec.last_status,
            "owner_name": rec.owner_name,
            "check_count": rec.check_count,
            "flipped_to_enemy": rec.flipped_to_enemy,
            "flipped_to_friendly": rec.flipped_to_friendly,
            "last_flip_time": rec.last_flip_time,
            "last_flip_from": rec.last_flip_from,
            "last_flip_to": rec.last_flip_to,
            "captured_at": rec.captured_at,
            "times_captured": rec.times_captured,
            "consecutive_enemy_checks": rec.consecutive_enemy_checks,
            "garrison_count": rec.garrison_count,
            "garrison_power": rec.garrison_power,
            "defender_powers": rec.defender_powers,
            "defender_names": rec.defender_names,
            "flip_velocity": rec.flip_velocity,
        }
    _ensure_data_dir()
    _atomic_write(MONUMENT_FILE, json.dumps(data, indent=2))
    logger.debug("Monument tracker saved")


def load_monument_tracker() -> dict[int, MonumentRecord]:
    """Load monument tracker from JSON, merging with default slots 1-4."""
    defaults = {i: MonumentRecord(slot=i) for i in range(1, 5)}
    if not MONUMENT_FILE.exists():
        return defaults
    try:
        raw = json.loads(MONUMENT_FILE.read_text(encoding="utf-8"))
        for key, vals in raw.items():
            slot = int(key)
            if slot in defaults:
                rec = defaults[slot]
                rec.last_checked = vals.get("last_checked", 0.0)
                rec.last_status = vals.get("last_status", "unknown")
                rec.owner_name = vals.get("owner_name", "")
                rec.check_count = vals.get("check_count", 0)
                rec.flipped_to_enemy = vals.get("flipped_to_enemy", 0)
                rec.flipped_to_friendly = vals.get("flipped_to_friendly", 0)
                rec.last_flip_time = vals.get("last_flip_time", 0.0)
                rec.last_flip_from = vals.get("last_flip_from", "")
                rec.last_flip_to = vals.get("last_flip_to", "")
                rec.captured_at = vals.get("captured_at", 0.0)
                rec.times_captured = vals.get("times_captured", 0)
                rec.consecutive_enemy_checks = vals.get("consecutive_enemy_checks", 0)
                rec.garrison_count = vals.get("garrison_count", -1)
                rec.garrison_power = vals.get("garrison_power", -1)
                rec.defender_powers = vals.get("defender_powers", [])
                rec.defender_names = vals.get("defender_names", [])
                rec.flip_velocity = vals.get("flip_velocity", 0.0)
        logger.info("Monument tracker loaded from disk")
    except Exception:
        logger.warning("Failed to load monument tracker, using defaults", exc_info=True)
    return defaults


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------

class EventLogger:
    """Appends structured events as JSON lines to data/events.jsonl."""

    def __init__(self) -> None:
        _ensure_data_dir()
        self._path = EVENTS_FILE

    def log(self, event: str, **data) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write event log", exc_info=True)


# ---------------------------------------------------------------------------
# PeriodicSaver
# ---------------------------------------------------------------------------

class PeriodicSaver:
    """Calls a save function at most once per interval."""

    def __init__(self, interval_seconds: float = 60.0) -> None:
        self._interval = interval_seconds
        self._last_save: float = 0.0

    def maybe_save(self, save_fn) -> None:
        now = time.time()
        if now - self._last_save >= self._interval:
            try:
                save_fn()
                self._last_save = now
            except Exception:
                logger.debug("Periodic save failed", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    """Write to a temp file then replace, to avoid partial writes."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
