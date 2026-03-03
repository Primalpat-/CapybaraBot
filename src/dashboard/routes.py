"""Dashboard API routes."""

import base64
import json
import logging
from dataclasses import asdict

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from src.bot.persistence import EVENTS_FILE, load_cumulative_stats, save_monument_tracker
from src.dashboard.app import get_state_machine
from src.utils.logging_config import dashboard_handler
from src.vision.ocr_reader import FACTION_NAMES

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Whitelist of editable config keys: (section, key)
EDITABLE_KEYS = {
    ("bot", "faction"),
    ("timing", "after_tap"),
    ("timing", "screen_transition"),
    ("timing", "battle_poll_interval"),
    ("timing", "jitter_factor"),
    ("timing", "loading_wait"),
    ("contest", "poll_interval_seconds"),
    ("contest", "max_duration_seconds"),
    ("contest", "stable_seconds"),
    ("persistence", "recheck_interval_seconds"),
    ("persistence", "check_recheck_interval_seconds"),
}

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/api/status")
async def get_status():
    """Get current bot status."""
    sm = get_state_machine()
    if sm is None:
        return JSONResponse({"error": "Bot not initialized"}, status_code=503)
    return sm.get_status()


@router.get("/api/screenshot")
async def get_screenshot():
    """Get the latest screenshot as base64 PNG."""
    sm = get_state_machine()
    if sm is None or sm.context.last_screenshot is None:
        return JSONResponse({"image": None})

    b64 = base64.b64encode(sm.context.last_screenshot).decode("utf-8")
    return {"image": b64}


@router.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    entries = dashboard_handler.get_entries(limit)
    return {"logs": entries}


@router.post("/api/pause")
async def pause_bot():
    """Pause the bot."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")
    sm.pause()
    return {"status": "paused"}


@router.post("/api/resume")
async def resume_bot():
    """Resume the bot."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")
    sm.resume()
    return {"status": "resumed"}


@router.post("/api/stop")
async def stop_bot():
    """Stop the bot."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")
    sm.stop()
    return {"status": "stopped"}


@router.get("/api/vision")
async def get_vision_stats():
    """Get vision API usage stats."""
    sm = get_state_machine()
    if sm is None:
        return JSONResponse({"error": "Bot not initialized"}, status_code=503)
    return sm.context.stats.to_dict()


@router.get("/api/events")
async def get_events(
    limit: int = Query(100, ge=1, le=1000),
    event_type: str | None = None,
):
    """Get recent events from the event log, newest first."""
    if not EVENTS_FILE.exists():
        return {"events": []}
    try:
        lines = EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines()
        events = []
        for line in reversed(lines):
            if not line.strip():
                continue
            entry = json.loads(line)
            if event_type and entry.get("event") != event_type:
                continue
            events.append(entry)
            if len(events) >= limit:
                break
        return {"events": events}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/clear-flips")
async def clear_flips():
    """Clear flip history from all monument slots and save immediately."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")
    for rec in sm.context.monument_tracker.values():
        rec.flip_history = []
        rec.flip_velocity = 0.0
        rec.flipped_to_enemy = 0
        rec.flipped_to_friendly = 0
        rec.last_flip_time = 0.0
        rec.last_flip_from = ""
        rec.last_flip_to = ""
    save_monument_tracker(sm.context.monument_tracker)
    logger.info("Flip history cleared via dashboard")
    return {"status": "flips_cleared"}


@router.get("/api/cumulative")
async def get_cumulative():
    """Get cumulative stats across all sessions."""
    stats = load_cumulative_stats()
    return asdict(stats)


@router.get("/api/config")
async def get_config():
    """Return the editable config subset."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")
    cfg = sm.config
    return {
        "bot": {"faction": cfg.get("bot", {}).get("faction", "")},
        "timing": {
            "after_tap": cfg.get("timing", {}).get("after_tap", 0.8),
            "screen_transition": cfg.get("timing", {}).get("screen_transition", 2.0),
            "battle_poll_interval": cfg.get("timing", {}).get("battle_poll_interval", 3.0),
            "jitter_factor": cfg.get("timing", {}).get("jitter_factor", 0.3),
            "loading_wait": cfg.get("timing", {}).get("loading_wait", 5.0),
        },
        "contest": {
            "poll_interval_seconds": cfg.get("contest", {}).get("poll_interval_seconds", 18),
            "max_duration_seconds": cfg.get("contest", {}).get("max_duration_seconds", 300),
            "stable_seconds": cfg.get("contest", {}).get("stable_seconds", 120),
        },
        "persistence": {
            "recheck_interval_seconds": cfg.get("persistence", {}).get("recheck_interval_seconds", 900),
            "check_recheck_interval_seconds": cfg.get("persistence", {}).get("check_recheck_interval_seconds", 600),
        },
    }


@router.post("/api/config")
async def update_config(body: dict):
    """Validate, merge into live config, and persist to config.yaml."""
    sm = get_state_machine()
    if sm is None:
        raise HTTPException(503, "Bot not initialized")

    errors = []
    updates: list[tuple[str, str, object]] = []  # (section, key, value)

    for section, values in body.items():
        if not isinstance(values, dict):
            errors.append(f"Expected object for section '{section}'")
            continue
        for key, value in values.items():
            if (section, key) not in EDITABLE_KEYS:
                errors.append(f"Key '{section}.{key}' is not editable")
                continue
            # Validate faction
            if section == "bot" and key == "faction":
                if not isinstance(value, str) or value.lower() not in FACTION_NAMES:
                    errors.append(f"Invalid faction: '{value}'. Must be one of: {', '.join(n.title() for n in FACTION_NAMES)}")
                    continue
            else:
                # Numeric validation
                if not isinstance(value, (int, float)):
                    errors.append(f"'{section}.{key}' must be a number")
                    continue
                if value < 0:
                    errors.append(f"'{section}.{key}' must be non-negative")
                    continue
            updates.append((section, key, value))

    if errors:
        raise HTTPException(422, detail=errors)

    if not updates:
        return {"status": "no_changes"}

    # Merge into live config in-place
    cfg = sm.config
    for section, key, value in updates:
        if section not in cfg:
            cfg[section] = {}
        cfg[section][key] = value

    # Persist to disk
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        logger.info(f"Config updated via dashboard: {[(s, k) for s, k, _ in updates]}")
    except Exception as e:
        logger.error(f"Failed to write config: {e}")
        raise HTTPException(500, f"Config applied in-memory but failed to save: {e}")

    return {"status": "saved", "updated": [f"{s}.{k}" for s, k, _ in updates]}
