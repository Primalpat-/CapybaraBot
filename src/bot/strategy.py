"""Monument selection strategy."""

import logging
import math

from src.vision.parser import MinimapReading, MonumentPosition, PlayerPosition

logger = logging.getLogger(__name__)


def distance(p1_x: float, p1_y: float, p2_x: float, p2_y: float) -> float:
    """Euclidean distance between two percentage-coordinate points."""
    return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)


def select_next_monument(
    minimap: MinimapReading,
    visited_ids: set[int] | None = None,
) -> MonumentPosition | None:
    """Select the next monument to visit.

    Strategy:
    1. Filter out already-visited and friendly monuments.
    2. Prioritize enemy monuments over neutral/unknown.
    3. Among candidates, pick the nearest one to the player.
    """
    if visited_ids is None:
        visited_ids = set()

    player = minimap.player_position
    if player is None:
        # Default to center if player position unknown
        player = PlayerPosition(x_percent=50, y_percent=50)

    candidates = [
        m for m in minimap.monuments
        if m.id not in visited_ids and m.likely_type != "friendly"
    ]

    if not candidates:
        # If all non-friendly are visited, try any unvisited
        candidates = [
            m for m in minimap.monuments
            if m.id not in visited_ids
        ]

    if not candidates:
        logger.info("No unvisited monuments remaining")
        return None

    # Sort: enemy first, then by distance
    def score(m: MonumentPosition) -> tuple[int, float]:
        type_priority = 0 if m.likely_type == "enemy" else 1
        dist = distance(player.x_percent, player.y_percent,
                        m.x_percent, m.y_percent)
        return (type_priority, dist)

    candidates.sort(key=score)
    chosen = candidates[0]

    logger.info(
        f"Selected monument {chosen.id} ({chosen.likely_type}) at "
        f"({chosen.x_percent:.1f}%, {chosen.y_percent:.1f}%)"
    )
    return chosen
