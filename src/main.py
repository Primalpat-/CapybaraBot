"""Entry point — wires ADB, Vision, Bot, and Dashboard together."""

import asyncio
import logging
import signal
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv

from src.adb.capture import ScreenCapture
from src.adb.connection import ADBConnection
from src.adb.input import ADBInput
from src.bot.actions import BotActions
from src.bot.state_machine import BotState, StateMachine
from src.bot.states import StateHandlers
from src.dashboard.app import app, set_state_machine
from src.utils.logging_config import setup_logging
from src.vision.cache import VisionCache
from src.vision.client import VisionClient

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_components(config: dict):
    """Instantiate all components from config."""
    adb_cfg = config.get("adb", {})
    connection = ADBConnection(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=adb_cfg.get("port", 5555),
        connect_timeout=adb_cfg.get("connect_timeout", 10),
        command_timeout=adb_cfg.get("command_timeout", 30),
        reconnect_attempts=adb_cfg.get("reconnect_attempts", 3),
        reconnect_delay=adb_cfg.get("reconnect_delay", 2.0),
    )

    capture = ScreenCapture(connection)

    input_cfg = config.get("input", {})
    adb_input = ADBInput(
        connection,
        tap_jitter=input_cfg.get("tap_jitter", 5),
        swipe_duration_min=input_cfg.get("swipe_duration_min", 200),
        swipe_duration_max=input_cfg.get("swipe_duration_max", 400),
    )

    vision_cfg = config.get("vision", {})
    vision = VisionClient(
        model=vision_cfg.get("model", "claude-sonnet-4-6"),
        max_tokens=vision_cfg.get("max_tokens", 1024),
        max_image_dimension=vision_cfg.get("max_image_dimension", 1024),
        temperature=vision_cfg.get("temperature", 0.0),
        input_token_cost_per_million=vision_cfg.get("input_token_cost_per_million", 3.0),
        output_token_cost_per_million=vision_cfg.get("output_token_cost_per_million", 15.0),
    )

    cache = VisionCache(ttl=30.0, max_entries=100)

    actions = BotActions(adb_input, config)

    state_machine = StateMachine(config)

    handlers = StateHandlers(capture, adb_input, vision, cache, actions, config)

    return connection, state_machine, handlers, vision


def register_all_handlers(sm: StateMachine, handlers: StateHandlers) -> None:
    """Register all state handlers on the state machine."""
    sm.register_handler(BotState.INITIALIZING, handlers.handle_initializing)
    sm.register_handler(BotState.OPENING_MINIMAP, handlers.handle_opening_minimap)
    sm.register_handler(BotState.READING_MINIMAP, handlers.handle_reading_minimap)
    sm.register_handler(BotState.NAVIGATING, handlers.handle_navigating)
    sm.register_handler(BotState.CHECKING_MONUMENT, handlers.handle_checking_monument)
    sm.register_handler(BotState.ATTACKING, handlers.handle_attacking)
    sm.register_handler(BotState.SKIPPING_BATTLE, handlers.handle_skipping_battle)
    sm.register_handler(BotState.POST_BATTLE, handlers.handle_post_battle)
    sm.register_handler(BotState.REFRESHING_POPUP, handlers.handle_refreshing_popup)
    sm.register_handler(BotState.IDLE, handlers.handle_idle)
    sm.register_handler(BotState.ERROR_RECOVERY, handlers.handle_error_recovery)
    sm.register_handler(BotState.PAUSED, handlers.handle_paused)
    sm.register_handler(BotState.STOPPED, handlers.handle_stopped)


async def run_dashboard(config: dict) -> None:
    """Run the FastAPI dashboard."""
    dash_cfg = config.get("dashboard", {})
    server_config = uvicorn.Config(
        app,
        host=dash_cfg.get("host", "0.0.0.0"),
        port=dash_cfg.get("port", 8080),
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def main() -> None:
    load_dotenv()
    setup_logging()

    config = load_config()
    logger.info("Configuration loaded")

    connection, state_machine, handlers, vision = build_components(config)
    register_all_handlers(state_machine, handlers)

    # Wire dashboard to bot state
    set_state_machine(state_machine)

    # Connect to ADB
    logger.info("Connecting to BlueStacks ADB...")
    if not await connection.connect_with_retry():
        logger.error("Failed to connect to ADB. Is BlueStacks running?")
        return

    device = await connection.get_device_info()
    if device:
        logger.info(f"Device: {device.serial} ({device.model or 'unknown model'})")

    # Run bot and dashboard concurrently
    logger.info("Starting bot loop and dashboard...")

    bot_task = asyncio.create_task(state_machine.run())
    dashboard_task = asyncio.create_task(run_dashboard(config))

    # Handle shutdown
    def handle_shutdown():
        logger.info("Shutdown signal received")
        state_machine.stop()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
    except Exception:
        pass

    try:
        await asyncio.gather(bot_task, dashboard_task, return_exceptions=True)
    except KeyboardInterrupt:
        handle_shutdown()

    # Print final stats
    usage = vision.get_usage_summary()
    stats = state_machine.context.stats.to_dict()
    logger.info(f"Final stats: {stats}")
    logger.info(f"Vision API usage: {usage}")

    await connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
