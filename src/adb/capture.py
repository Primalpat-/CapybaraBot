"""Screenshot capture via ADB with Windows binary fix."""

import asyncio
import base64
import logging
import time
from pathlib import Path

from PIL import Image
import io

from src.adb.connection import ADBConnection

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Captures screenshots from the emulator via ADB.

    Uses base64 piping to avoid Windows binary corruption of PNG bytes.
    """

    def __init__(self, connection: ADBConnection, save_dir: str = "screenshots"):
        self.connection = connection
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_capture_time: float = 0
        self._capture_count: int = 0

    async def capture(self) -> bytes:
        """Capture a screenshot and return raw PNG bytes.

        Uses exec-out with base64 encoding to avoid Windows binary corruption.
        """
        if not self.connection.connected:
            raise ConnectionError("ADB not connected")

        # Pipe screencap through base64 on-device to avoid binary corruption
        stdout, stderr, rc = await self.connection.run_adb(
            "exec-out", "screencap -p | base64"
        )

        if rc != 0:
            raise RuntimeError(f"Screenshot failed (rc={rc}): {stderr.strip()}")

        # Decode base64 to get PNG bytes
        b64_data = stdout.strip().replace("\r\n", "").replace("\n", "")
        try:
            png_bytes = base64.b64decode(b64_data)
        except Exception as e:
            raise RuntimeError(f"Failed to decode screenshot base64: {e}")

        # Validate it's a valid PNG
        if not png_bytes.startswith(b"\x89PNG"):
            raise RuntimeError("Captured data is not a valid PNG")

        self._last_capture_time = time.time()
        self._capture_count += 1
        logger.debug(f"Screenshot captured ({len(png_bytes)} bytes)")
        return png_bytes

    async def capture_pil(self) -> Image.Image:
        """Capture and return as a PIL Image."""
        png_bytes = await self.capture()
        return Image.open(io.BytesIO(png_bytes))

    async def capture_and_save(self, filename: str | None = None) -> Path:
        """Capture a screenshot and save to disk."""
        png_bytes = await self.capture()

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{self._capture_count}.png"

        filepath = self.save_dir / filename
        filepath.write_bytes(png_bytes)
        logger.info(f"Screenshot saved: {filepath}")
        return filepath

    @property
    def last_capture_time(self) -> float:
        return self._last_capture_time

    @property
    def capture_count(self) -> int:
        return self._capture_count
