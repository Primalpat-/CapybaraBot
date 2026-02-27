"""Dashboard API routes."""

import base64
import json
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from src.bot.persistence import EVENTS_FILE, load_cumulative_stats, save_monument_tracker
from src.dashboard.app import get_state_machine
from src.utils.logging_config import dashboard_handler

logger = logging.getLogger(__name__)

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
