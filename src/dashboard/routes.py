"""Dashboard API routes."""

import base64
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

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
