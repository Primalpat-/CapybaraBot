"""Humanized delays with configurable jitter."""

import asyncio
import random
import logging

logger = logging.getLogger(__name__)


def humanize(base_delay: float, jitter_factor: float = 0.3) -> float:
    """Apply random jitter to a base delay.

    Args:
        base_delay: Base delay in seconds.
        jitter_factor: Fraction of base_delay for jitter range (±).

    Returns:
        Jittered delay, always >= 0.1s.
    """
    jitter = base_delay * jitter_factor * random.uniform(-1, 1)
    return max(0.1, base_delay + jitter)


async def wait(base_delay: float, jitter_factor: float = 0.3,
               label: str = "",
               interrupt_check: "callable | None" = None) -> float:
    """Async sleep with humanized delay.

    Args:
        interrupt_check: Optional callable returning True when the wait
            should be cut short (e.g. bot paused).  When provided the
            sleep is broken into 1-second chunks so the check is polled
            frequently.

    Returns the actual delay used, or -1 if interrupted.
    """
    delay = humanize(base_delay, jitter_factor)
    if label:
        logger.debug(f"Waiting {delay:.2f}s ({label})")

    if interrupt_check is None:
        await asyncio.sleep(delay)
    else:
        remaining = delay
        while remaining > 0:
            chunk = min(remaining, 1.0)
            await asyncio.sleep(chunk)
            remaining -= chunk
            if interrupt_check():
                logger.debug(f"Wait interrupted ({label})")
                return -1

    return delay


async def wait_fixed(delay: float, label: str = "") -> None:
    """Async sleep with exact delay (no jitter)."""
    if label:
        logger.debug(f"Waiting {delay:.2f}s fixed ({label})")
    await asyncio.sleep(delay)
