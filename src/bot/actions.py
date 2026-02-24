"""Compound ADB actions used by state handlers."""

import logging

from src.adb.input import ADBInput
from src.bot.calibration import CoordinateCalibrator
from src.utils.timing import wait

logger = logging.getLogger(__name__)


class BotActions:
    """High-level actions composed of multiple ADB inputs."""

    def __init__(self, adb_input: ADBInput, config: dict, calibrator: CoordinateCalibrator):
        self.input = adb_input
        self.calibrator = calibrator
        self.timing = config.get("timing", {})
        self.jitter = self.timing.get("jitter_factor", 0.3)

    async def open_minimap(self) -> None:
        """Tap the minimap button to open the minimap overlay."""
        x, y = self.calibrator.get_pixel("minimap_button")
        logger.info(f"Opening minimap at ({x}, {y})")
        await self.input.tap(x, y)
        await wait(self.timing.get("minimap_open", 1.5), self.jitter, "minimap open")

    async def tap_monument_on_minimap(self, x_percent: float, y_percent: float,
                                      screen_width: int, screen_height: int) -> None:
        """Tap a monument position on the minimap (percentage → pixel)."""
        x = int(x_percent / 100 * screen_width)
        y = int(y_percent / 100 * screen_height)
        logger.info(f"Tapping monument at ({x}, {y}) [{x_percent:.1f}%, {y_percent:.1f}%]")
        await self.input.tap(x, y)
        await wait(self.timing.get("screen_transition", 2.0), self.jitter, "monument tap")

    async def tap_action_button(self) -> None:
        """Tap the main action button on a monument popup."""
        x, y = self.calibrator.get_pixel("action_button")
        logger.info(f"Tapping action button at ({x}, {y})")
        await self.input.tap(x, y)
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "action tap")

    async def tap_skip_battle(self) -> None:
        """Tap the skip/speed-up battle button."""
        x, y = self.calibrator.get_pixel("skip_battle")
        logger.info(f"Tapping skip battle at ({x}, {y})")
        await self.input.tap(x, y)
        await wait(self.timing.get("after_tap", 0.8), self.jitter, "skip battle")

    async def close_popup(self) -> None:
        """Close the current popup/overlay."""
        x, y = self.calibrator.get_pixel("close_popup")
        logger.info(f"Closing popup at ({x}, {y})")
        await self.input.tap(x, y)
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
