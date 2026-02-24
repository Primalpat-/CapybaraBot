"""FastAPI dashboard application with shared bot state."""

import base64

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.bot.state_machine import StateMachine

app = FastAPI(title="CapybaraBot Dashboard")

# Shared references — set by main.py before starting
_state_machine: StateMachine | None = None


def set_state_machine(sm: StateMachine) -> None:
    global _state_machine
    _state_machine = sm


def get_state_machine() -> StateMachine | None:
    return _state_machine


# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Import routes to register them
from src.dashboard.routes import router  # noqa: E402
app.include_router(router)
