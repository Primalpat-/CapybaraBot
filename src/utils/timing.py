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
               label: str = "") -> float:
    """Async sleep with humanized delay.

    Returns the actual delay used.
    """
    delay = humanize(base_delay, jitter_factor)
    if label:
        logger.debug(f"Waiting {delay:.2f}s ({label})")
    await asyncio.sleep(delay)
    return delay


async def wait_fixed(delay: float, label: str = "") -> None:
    """Async sleep with exact delay (no jitter)."""
    if label:
        logger.debug(f"Waiting {delay:.2f}s fixed ({label})")
    await asyncio.sleep(delay)
