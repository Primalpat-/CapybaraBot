"""ADB connection management for BlueStacks emulator."""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    serial: str
    state: str
    model: str | None = None


class ADBConnection:
    """Manages ADB connection to BlueStacks."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555,
                 connect_timeout: int = 10, command_timeout: int = 30,
                 reconnect_attempts: int = 3, reconnect_delay: float = 2.0):
        self.host = host
        self.port = port
        self.serial = f"{host}:{port}"
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def run_adb(self, *args: str, timeout: int | None = None) -> tuple[str, str, int]:
        """Run an ADB command and return (stdout, stderr, returncode)."""
        timeout = timeout or self.command_timeout
        cmd = ["adb", "-s", self.serial, *args]
        logger.debug(f"ADB: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"ADB command timed out after {timeout}s: {' '.join(cmd)}")

        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            proc.returncode,
        )

    async def connect(self) -> bool:
        """Connect to the ADB device."""
        logger.info(f"Connecting to ADB device at {self.serial}...")

        stdout, stderr, rc = await self.run_adb(
            "connect", self.serial, timeout=self.connect_timeout
        )
        output = stdout + stderr

        if "connected" in output.lower():
            self._connected = True
            logger.info(f"Connected to {self.serial}")
            return True

        logger.error(f"Failed to connect: {output.strip()}")
        self._connected = False
        return False

    async def connect_with_retry(self) -> bool:
        """Connect with retry logic."""
        for attempt in range(1, self.reconnect_attempts + 1):
            if await self.connect():
                return True
            if attempt < self.reconnect_attempts:
                logger.warning(
                    f"Connection attempt {attempt}/{self.reconnect_attempts} failed, "
                    f"retrying in {self.reconnect_delay}s..."
                )
                await asyncio.sleep(self.reconnect_delay)

        logger.error("All connection attempts exhausted.")
        return False

    async def disconnect(self) -> None:
        """Disconnect from the ADB device."""
        await self.run_adb("disconnect", self.serial)
        self._connected = False
        logger.info(f"Disconnected from {self.serial}")

    async def health_check(self) -> bool:
        """Verify the device is still responsive."""
        try:
            stdout, _, rc = await self.run_adb("shell", "echo", "ok", timeout=5)
            alive = rc == 0 and "ok" in stdout
            if not alive:
                self._connected = False
            return alive
        except (TimeoutError, Exception) as e:
            logger.warning(f"Health check failed: {e}")
            self._connected = False
            return False

    async def ensure_connected(self) -> bool:
        """Ensure we have a live connection, reconnecting if necessary."""
        if self._connected and await self.health_check():
            return True
        logger.info("Connection lost, attempting reconnect...")
        return await self.connect_with_retry()

    async def get_device_info(self) -> DeviceInfo | None:
        """Get information about the connected device."""
        stdout, _, rc = await self.run_adb("devices", "-l")
        if rc != 0:
            return None

        for line in stdout.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and self.serial in parts[0]:
                model = None
                for part in parts[2:]:
                    if part.startswith("model:"):
                        model = part.split(":", 1)[1]
                return DeviceInfo(
                    serial=parts[0], state=parts[1], model=model
                )
        return None
