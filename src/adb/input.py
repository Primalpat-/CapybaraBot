"""ADB input actions: tap, swipe, back with humanized jitter."""

import asyncio
import logging
import random

from src.adb.connection import ADBConnection

logger = logging.getLogger(__name__)


class ADBInput:
    """Sends touch/key input to the emulator via ADB."""

    def __init__(self, connection: ADBConnection,
                 tap_jitter: int = 5,
                 swipe_duration_min: int = 200,
                 swipe_duration_max: int = 400):
        self.connection = connection
        self.tap_jitter = tap_jitter
        self.swipe_duration_min = swipe_duration_min
        self.swipe_duration_max = swipe_duration_max

    def _jitter(self, value: int, jitter: int | None = None) -> int:
        """Add random jitter to a coordinate."""
        j = jitter if jitter is not None else self.tap_jitter
        return value + random.randint(-j, j)

    async def tap(self, x: int, y: int, jitter: int | None = None) -> None:
        """Tap at (x, y) with optional jitter."""
        jx = self._jitter(x, jitter)
        jy = self._jitter(y, jitter)
        logger.debug(f"Tap ({x},{y}) → jittered ({jx},{jy})")
        await self.connection.run_adb(
            "shell", "input", "tap", str(jx), str(jy)
        )

    async def swipe(self, x1: int, y1: int, x2: int, y2: int,
                    duration_ms: int | None = None) -> None:
        """Swipe from (x1,y1) to (x2,y2) with humanized duration."""
        if duration_ms is None:
            duration_ms = random.randint(self.swipe_duration_min, self.swipe_duration_max)

        # Add jitter to start and end points
        jx1 = self._jitter(x1)
        jy1 = self._jitter(y1)
        jx2 = self._jitter(x2)
        jy2 = self._jitter(y2)

        logger.debug(
            f"Swipe ({x1},{y1})→({x2},{y2}) → "
            f"jittered ({jx1},{jy1})→({jx2},{jy2}) duration={duration_ms}ms"
        )
        await self.connection.run_adb(
            "shell", "input", "swipe",
            str(jx1), str(jy1), str(jx2), str(jy2), str(duration_ms)
        )

    async def back(self) -> None:
        """Press the Android back button."""
        logger.debug("Pressing BACK")
        await self.connection.run_adb("shell", "input", "keyevent", "4")

    async def home(self) -> None:
        """Press the Android home button."""
        logger.debug("Pressing HOME")
        await self.connection.run_adb("shell", "input", "keyevent", "3")

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        """Long press at (x, y) by swiping to the same position."""
        jx = self._jitter(x)
        jy = self._jitter(y)
        logger.debug(f"Long press ({x},{y}) → ({jx},{jy}) duration={duration_ms}ms")
        await self.connection.run_adb(
            "shell", "input", "swipe",
            str(jx), str(jy), str(jx), str(jy), str(duration_ms)
        )
