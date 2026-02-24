# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CapybaraBot is an automated bot for "Capybara Go" (Alien Minefield mode) running in BlueStacks. It uses ADB for input/capture, Claude Vision API for screen understanding, and a finite state machine to navigate the game loop: open minimap → pick red monument → navigate → attack → capture → repeat.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot (requires BlueStacks + ADB + ANTHROPIC_API_KEY in .env)
python -m src.main

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_calibration.py -v

# Run a single test
python -m pytest tests/test_calibration.py::TestNeedsCalibration::test_all_needed_initially -v

# Calibration diagnostic tool
python -m scripts.diagnose_calibration
```

## Architecture

**Entry point**: `src/main.py` — wires ADB, Vision, Bot, Dashboard together and runs the state machine + FastAPI dashboard concurrently.

**State machine** (`src/bot/state_machine.py`): Drives the bot through 15 states. Each state has a handler registered in `src/main.py:register_all_handlers()`. Handlers are async coroutines with signature `async (BotContext, config) -> BotState`. The machine runs in a loop with 0.5s tick interval, stuck detection (60s timeout), and error limits.

**State handlers** (`src/bot/states.py`): The `StateHandlers` class holds all handler methods. `handle_initializing` is the universal entry point — it identifies the current screen via Vision API and routes to the correct state. The bot can start on any screen. See the routing table in the `handle_initializing` docstring.

**When adding a new screen type:**
1. Add visual cues + enum value to `identify_screen` in `config/prompts.yaml`
2. Add routing entry in `handle_initializing` (`src/bot/states.py`)
3. Add calibration elements if the screen has tappable UI (`src/bot/calibration.py` — `SCREEN_ELEMENTS` and `ELEMENT_DESCRIPTIONS`)
4. Add detection in mid-flow handlers (like `logged_out` is checked in `handle_opening_minimap`, `handle_approaching_monument`, etc.)

**Calibration** (`src/bot/calibration.py`): UI element positions are discovered via Vision API and stored as percentages (0-100). Persisted to `config/calibrated_coords.json`. Falls back to hardcoded pixel values in `config/config.yaml`. Cleared on every bot startup. The `_calibrate_for_screen` method in states.py handles the on-demand calibration flow.

**Vision** (`src/vision/`): Screenshots go through `VisionClient` → Claude API. Responses are cached with perceptual image hashing (30s TTL). The minimap uses OpenCV color detection (`minimap_detector.py`) instead of Vision API to save costs. All Vision prompts live in `config/prompts.yaml`. Parsers in `parser.py` convert JSON responses to typed dataclasses.

**ADB** (`src/adb/`): Screenshots captured via `adb exec-out screencap -p | base64` to avoid Windows binary corruption. Input uses `adb shell input tap/swipe/keyevent`. All commands are async.

**Dashboard** (`src/dashboard/`): FastAPI app on port 8080. REST endpoints for status, screenshots, logs, and bot control (pause/resume/stop).

## Key Patterns

- **Coordinate systems**: Vision returns percentages (0-100), ADB taps use pixels. Conversion: `pixel = percent / 100 * screen_dimension`. If Vision returns values >100, they're auto-converted from pixel to percent in `_calibrate_for_screen`.
- **Timing**: All waits use `wait(base, jitter_factor, label)` from `src/utils/timing.py`. Jitter prevents detection.
- **Unbeatable players**: After a defeat, the opponent is added to `_unbeatable_players`. Before attacking, defenders are checked against this set.
- **Reconnection flow**: `handle_reconnecting` checks what screen we're on and skips to the right step (logged_out → home_screen → mode_select → game).
- **Error recovery**: Multi-strategy — identify screen and route, press back, or restart from minimap.

## Config Files

- `config/config.yaml`: ADB connection, screen dimensions, coordinates, vision model, timing delays, bot limits, dashboard settings
- `config/prompts.yaml`: All Vision API prompt templates (identify_screen, check_monument, check_battle, calibrate_elements, etc.)
- `.env`: `ANTHROPIC_API_KEY` (required)

## Test Structure

92 tests across 7 files covering: state machine transitions, vision cache, coordinate calibration, minimap color detection, response parsing, and monument selection strategy. No external services needed — all tests use mocks/synthetic data.
