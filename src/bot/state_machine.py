"""Finite State Machine engine for the bot."""

import asyncio
import logging
import time
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BotPausedInterrupt(Exception):
    """Raised inside a handler when the bot is paused mid-execution."""
    pass


class BotState(Enum):
    INITIALIZING = auto()
    OPENING_MINIMAP = auto()
    READING_MINIMAP = auto()
    NAVIGATING = auto()
    APPROACHING_MONUMENT = auto()
    CHECKING_MONUMENT = auto()
    ATTACKING = auto()
    SKIPPING_BATTLE = auto()
    POST_BATTLE = auto()
    REFRESHING_POPUP = auto()
    RECONNECTING = auto()
    IDLE = auto()
    ERROR_RECOVERY = auto()
    STAGNATION_RECOVERY = auto()
    PAUSED = auto()
    STOPPED = auto()


@dataclass
class BotStats:
    monuments_visited: int = 0
    battles_fought: int = 0
    battles_won: int = 0
    defeats: int = 0
    monuments_captured: int = 0
    api_calls: int = 0
    total_cost: float = 0.0
    errors: int = 0
    consecutive_errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def runtime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "monuments_visited": self.monuments_visited,
            "battles_fought": self.battles_fought,
            "battles_won": self.battles_won,
            "defeats": self.defeats,
            "monuments_captured": self.monuments_captured,
            "api_calls": self.api_calls,
            "total_cost": round(self.total_cost, 4),
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "runtime_seconds": round(self.runtime_seconds, 1),
        }


@dataclass
class MonumentRecord:
    """Tracking data for a single minimap monument slot."""
    slot: int
    last_checked: float = 0.0          # timestamp of last popup check
    last_status: str = "unknown"       # "friendly", "enemy", "unknown"
    owner_name: str = ""               # defender/owner name from popup
    check_count: int = 0               # total times popup was opened
    flipped_to_enemy: int = 0          # times status changed TO enemy
    flipped_to_friendly: int = 0       # times flipped TO friendly
    last_flip_time: float = 0.0        # timestamp of most recent flip
    last_flip_from: str = ""           # status before flip
    last_flip_to: str = ""             # status after flip
    captured_at: float = 0.0           # when we last captured this slot
    times_captured: int = 0            # total captures by us
    consecutive_enemy_checks: int = 0  # consecutive checks where status=enemy


@dataclass
class BotContext:
    """Shared context passed between state handlers."""
    stats: BotStats = field(default_factory=BotStats)
    current_target: dict | None = None
    minimap_data: object | None = None
    monument_info: object | None = None
    last_screenshot: bytes | None = None
    state_enter_time: float = field(default_factory=time.time)
    error_message: str = ""
    action_log: list = field(default_factory=list)
    monument_tracker: dict[int, MonumentRecord] = field(default_factory=lambda: {
        i: MonumentRecord(slot=i) for i in range(1, 5)
    })
    last_progress_time: float = field(default_factory=time.time)
    stagnation_recovery_attempts: int = 0
    last_stagnation_recovery_time: float = 0.0

    def log_action(self, message: str) -> None:
        entry = {"time": time.time(), "message": message}
        self.action_log.append(entry)
        # Keep last 200 entries
        if len(self.action_log) > 200:
            self.action_log = self.action_log[-200:]
        logger.info(f"[Action] {message}")


class StateMachine:
    """Drives the bot through states using registered handlers."""

    def __init__(self, config: dict):
        self.config = config
        self.state = BotState.INITIALIZING
        self.context = BotContext()
        self._handlers: dict[BotState, object] = {}
        self._running = False
        self._paused = False
        self._tick_interval = 0.5  # seconds between ticks
        self._on_tick: object | None = None  # callback(context) called each tick

    def register_handler(self, state: BotState, handler) -> None:
        """Register a handler function for a state.

        Handler signature: async (context, config) -> BotState
        """
        self._handlers[state] = handler

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        if self._running and not self._paused:
            self._paused = True
            self.state = BotState.PAUSED
            self.context.log_action("Bot paused")
            logger.info("Bot paused")

    def resume(self) -> None:
        if self._paused:
            self._paused = False
            self.state = BotState.OPENING_MINIMAP
            self.context.log_action("Bot resumed")
            logger.info("Bot resumed")

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self.state = BotState.STOPPED
        self.context.log_action("Bot stopped")
        logger.info("Bot stopped")

    def _check_stuck(self) -> bool:
        """Check if we've been in the current state too long."""
        timeout = self.config.get("bot", {}).get("stuck_timeout", 60)
        elapsed = time.time() - self.context.state_enter_time
        if elapsed > timeout and self.state not in (
            BotState.PAUSED, BotState.STOPPED, BotState.IDLE, BotState.RECONNECTING,
            BotState.STAGNATION_RECOVERY,
        ):
            logger.warning(
                f"Stuck in {self.state.name} for {elapsed:.1f}s (timeout={timeout}s)"
            )
            return True
        return False

    def _check_limits(self) -> bool:
        """Check if we've hit any configured limits."""
        bot_cfg = self.config.get("bot", {})

        max_consecutive = bot_cfg.get("max_consecutive_errors", 5)
        if self.context.stats.consecutive_errors >= max_consecutive:
            logger.error(f"Max consecutive errors reached ({max_consecutive})")
            return True

        max_total = bot_cfg.get("max_total_errors", 20)
        if self.context.stats.errors >= max_total:
            logger.error(f"Max total errors reached ({max_total})")
            return True

        max_monuments = bot_cfg.get("max_monuments", 0)
        if max_monuments > 0 and self.context.stats.monuments_visited >= max_monuments:
            logger.info(f"Monument limit reached ({max_monuments})")
            return True

        return False

    def _check_stagnation(self) -> bool:
        """Check if the bot has gone too long without meaningful progress.

        Returns True if stagnation is detected and recovery should be attempted.
        Auto-pauses the bot if max recovery attempts have been exhausted.
        """
        exempt = (
            BotState.PAUSED, BotState.STOPPED, BotState.IDLE,
            BotState.STAGNATION_RECOVERY, BotState.RECONNECTING,
        )
        if self.state in exempt:
            return False

        bot_cfg = self.config.get("bot", {})
        timeout = bot_cfg.get("stagnation_timeout_seconds", 1800)
        elapsed = time.time() - self.context.last_progress_time
        if elapsed <= timeout:
            return False

        # Rate-limit recovery attempts
        interval = bot_cfg.get("recovery_interval_seconds", 120)
        since_last = time.time() - self.context.last_stagnation_recovery_time
        if self.context.last_stagnation_recovery_time > 0 and since_last < interval:
            return False

        # Check if we've exhausted recovery attempts
        max_attempts = bot_cfg.get("max_recovery_attempts", 5)
        if self.context.stagnation_recovery_attempts >= max_attempts:
            logger.warning(
                f"Stagnation: {max_attempts} recovery attempts exhausted — auto-pausing bot"
            )
            self.context.log_action(
                f"Auto-pausing: no progress for {elapsed:.0f}s and "
                f"{max_attempts} recovery attempts failed"
            )
            self.pause()
            return False

        logger.warning(
            f"Stagnation detected: no progress for {elapsed:.0f}s "
            f"(attempt {self.context.stagnation_recovery_attempts + 1}/{max_attempts})"
        )
        return True

    async def run(self) -> None:
        """Main bot loop."""
        self._running = True
        logger.info("State machine starting...")
        self.context.log_action("Bot started")

        while self._running:
            if self._paused:
                await asyncio.sleep(1)
                continue

            if self.state == BotState.STOPPED:
                break

            if self._check_limits():
                self.stop()
                break

            if self._check_stuck():
                self.context.stats.errors += 1
                self.context.stats.consecutive_errors += 1
                self.context.error_message = f"Stuck in {self.state.name}"
                self.state = BotState.ERROR_RECOVERY
                self.context.state_enter_time = time.time()

            if self._check_stagnation():
                self.context.stagnation_recovery_attempts += 1
                self.context.last_stagnation_recovery_time = time.time()
                self.state = BotState.STAGNATION_RECOVERY
                self.context.state_enter_time = time.time()

            handler = self._handlers.get(self.state)
            if handler is None:
                logger.error(f"No handler for state {self.state.name}")
                self.stop()
                break

            try:
                old_state = self.state
                next_state = await handler(self.context, self.config)

                if next_state != old_state:
                    logger.info(f"Transition: {old_state.name} → {next_state.name}")
                    self.context.state_enter_time = time.time()
                    self.state = next_state

                    # Reset consecutive errors on successful transition
                    if next_state != BotState.ERROR_RECOVERY:
                        self.context.stats.consecutive_errors = 0

            except BotPausedInterrupt:
                # Handler was interrupted by pause — state already set to
                # PAUSED by pause(), just log and let the loop continue to
                # the _paused check at the top.
                self.context.log_action("Paused mid-action")

            except Exception as e:
                logger.exception(f"Error in {self.state.name}: {e}")
                self.context.stats.errors += 1
                self.context.stats.consecutive_errors += 1
                self.context.error_message = str(e)
                self.state = BotState.ERROR_RECOVERY
                self.context.state_enter_time = time.time()

            if self._on_tick:
                try:
                    self._on_tick(self.context)
                except Exception:
                    logger.debug("on_tick callback error", exc_info=True)

            await asyncio.sleep(self._tick_interval)

        self.context.log_action("Bot loop ended")
        logger.info("State machine stopped.")

    def get_status(self) -> dict:
        """Get current bot status for the dashboard."""
        return {
            "state": self.state.name,
            "running": self._running,
            "paused": self._paused,
            "stats": self.context.stats.to_dict(),
            "current_target": self.context.current_target,
            "error_message": self.context.error_message,
            "last_progress_time": self.context.last_progress_time,
            "stagnation_recovery_attempts": self.context.stagnation_recovery_attempts,
            "action_log": self.context.action_log[-50:],
            "monuments": {
                slot: {
                    "slot": rec.slot,
                    "last_checked": rec.last_checked,
                    "last_status": rec.last_status,
                    "owner_name": rec.owner_name,
                    "check_count": rec.check_count,
                    "flipped_to_enemy": rec.flipped_to_enemy,
                    "flipped_to_friendly": rec.flipped_to_friendly,
                    "last_flip_time": rec.last_flip_time,
                    "captured_at": rec.captured_at,
                    "times_captured": rec.times_captured,
                    "consecutive_enemy_checks": rec.consecutive_enemy_checks,
                }
                for slot, rec in self.context.monument_tracker.items()
            },
        }
