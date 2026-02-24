"""Compound ADB actions used by state handlers."""

import logging

from src.adb.input import ADBInput
from src.utils.timing import wait

logger = logging.getLogger(__name__)


class BotActions:
    """High-level actions composed of multiple ADB inputs."""

    def __init__(self, adb_input: ADBInput, config: dict):
        self.input = adb_input
        self.coords = config.get("coordinates", {})
        self.timing = config.get("timing", {})
        self.jitter = self.timing.get("jitter_factor", 0.3)

    async def open_minimap(self) -> None:
        """Tap the minimap button to open the minimap overlay."""
        c = self.coords.get("minimap_button", {})
        logger.info("Opening minimap")
        await self.input.tap(c.get("x", 980), c.get("y", 180))
        await wait(self.timing.get("minimap_open", 1.5), self.jitter, "minimap open")

    async def tap_monument_on_minimap(self, x_percent: float, y_percent: float,
                                      screen_width: int, screen_height: int) -> None:
        """Tap a monument position on the minimap (percentage → pixel)."""
        x = int(x_percent / 100 * screen_width)
        y = int(y_percent / 100 * screen_height)
        logger.info(f"Tapping monument at ({x}, {y})")
        await self.input.tap(x, y)
        await wait(self.timing.get("screen_transition", 2.0), self.jitter, "monument tap")

    async def tap_action_button(self) -> None:
        """Tap the main action button on a monument popup."""
        c = self.coords.get("action_button", {})
        logger.info("Tapping action button")
        await self.input.tap(c.get("x", 540), c.get("y", 1600))
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "action tap")

    async def tap_skip_battle(self) -> None:
        """Tap the skip/speed-up battle button."""
        c = self.coords.get("skip_battle", {})
        logger.info("Tapping skip battle")
        await self.input.tap(c.get("x", 540), c.get("y", 1750))
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "skip battle")

    async def close_popup(self) -> None:
        """Close the current popup/overlay."""
        c = self.coords.get("close_popup", {})
        logger.info("Closing popup")
        await self.input.tap(c.get("x", 950), c.get("y", 400))
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "close popup")

    async def press_back(self) -> None:
        """Press the Android back button."""
        logger.info("Pressing back")
        await self.input.back()
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "back press")

    async def refresh_popup(self) -> None:
        """Close and reopen a monument popup to refresh its state."""
        logger.info("Refreshing popup (close + reopen)")
        await self.close_popup()
        await wait(self.timing.get("monument_popup_wait", 1.5), self.jitter, "popup refresh")
        # Tap the same location to reopen — caller sets the coordinates
        # This is a simplification; the actual re-tap is done by the state handler
