"""Auto-calibration of UI element coordinates via Vision discovery."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps screen types to the UI elements visible on that screen.
# For "minimap", we calibrate two GRID REFERENCE points and derive all 4 slots.
SCREEN_ELEMENTS: dict[str, list[str]] = {
    "main_map": ["minimap_button"],
    "minimap": ["minimap_square_topleft", "minimap_square_bottomright", "minimap_close"],
    "arrived_at_monument": ["world_monument"],
    "monument_popup": ["action_button", "close_popup"],
    "battle_active": ["skip_battle"],
    "logged_out": ["restart_button"],
    "battle_result": ["ok_button"],
    "home_screen": ["star_trek_button"],
    "mode_select": ["alien_minefield_button"],
    "occupy_prompt": ["occupy_cancel_button"],
}

# Human-readable descriptions used in the Vision calibration prompt
ELEMENT_DESCRIPTIONS: dict[str, str] = {
    "minimap_button": (
        "The small purple/dark square with a magnifying glass icon inside it, "
        "located at the bottom-right corner of the minimap widget in the top-right "
        "area of the screen. The minimap widget shows a tiny overview of the game "
        "map, and this purple magnifying glass button is at its bottom-right corner "
        "next to the 'Tier X Mine' label. Return the center of this purple "
        "magnifying glass button."
    ),
    "minimap_square_topleft": (
        "The TOP-LEFT colored square in the 2x2 grid of colored squares "
        "on the minimap overlay. Each square is a large, clearly colored region "
        "(either red or blue). Return the CENTER of this top-left square."
    ),
    "minimap_square_bottomright": (
        "The BOTTOM-RIGHT colored square in the 2x2 grid of colored squares "
        "on the minimap overlay. Each square is a large, clearly colored region "
        "(either red or blue). Return the CENTER of this bottom-right square."
    ),
    "world_monument": (
        "The monument structure in the 3D game world that the player's capybara "
        "character is standing on or next to. This is the tappable object near the "
        "center of the screen. Return its center position."
    ),
    "action_button": (
        "The main action button at the bottom of the monument popup — "
        "a large, prominent button labeled 'Attack', 'Claim', 'Visit', or similar. "
        "Return the center of the TEXT on this button, not the bottom edge."
    ),
    "close_popup": (
        "The close/X button on the monument popup — a small X icon "
        "in the upper-right corner of the popup overlay. "
        "Return the exact center of the X icon itself."
    ),
    "minimap_close": (
        "The small circular X button at the bottom-center of the minimap overlay. "
        "This button closes the minimap. Return the exact center of the X icon."
    ),
    "skip_battle": (
        "The green 'Skip' button on the RIGHT side of the bottom area of the "
        "battle screen. There are three buttons in a row on the bottom-right: "
        "the green 'Skip' button (with >> icon and the word 'Skip'), then a "
        "speed button (labeled x1, x2, x3, or x4 — may be grey, yellow, or blue), "
        "then a bar chart button. The 'Skip' button is the LEFTMOST of these "
        "three, but all three are on the RIGHT half of the screen. Return the "
        "center of the word 'Skip' on this green button."
    ),
    "ok_button": (
        "The large yellow/orange 'OK' button at the bottom-center of the "
        "Victory or Defeat results screen. Return the center of the 'OK' text."
    ),
    "restart_button": (
        "The large yellow 'Restart' button on the 'Tips' popup/dialog. "
        "This button appears when the user has been logged in on another device."
    ),
    "star_trek_button": (
        "The 'Star Trek' button on the home screen. It is in the RIGHT side of "
        "a horizontal row of icon buttons (Homestead, 10 days, Travel, Star, Star Trek) "
        "located ABOVE the bottom navigation bar. It has a golden star/crown icon "
        "with the text 'Star Trek' written below it. It is to the RIGHT of the 'Star' "
        "button and ABOVE the 'Events' button. Return the center of this icon."
    ),
    "alien_minefield_button": (
        "The 'Alien Minefield' icon/button on the mode selection screen. "
        "It is one of several game mode icons displayed on this screen."
    ),
    "occupy_cancel_button": (
        "The pink/red 'Cancel' button on the LEFT side of the 'Tips' popup. "
        "The popup asks about occupying a monument. There are two buttons: "
        "'Cancel' (pink/red, on the left) and 'OK' (yellow/orange, on the right). "
        "Return the center of the 'Cancel' button text."
    ),
}

# Number of monument slots on the minimap
NUM_MONUMENT_SLOTS = 4

_CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "config" / "calibrated_coords.json"


@dataclass
class CalibratedCoordinate:
    name: str
    x_percent: float
    y_percent: float
    screen_width: int
    screen_height: int
    confidence: float
    discovered_at: str

    def to_pixel(self, w: int, h: int) -> tuple[int, int]:
        """Convert percentage coordinates to pixel values for the given dimensions."""
        return (int(self.x_percent / 100 * w), int(self.y_percent / 100 * h))


class CoordinateCalibrator:
    """Stores, persists, and resolves UI element coordinates.

    Calibrated positions (percentages) take priority.  Falls back to the
    hardcoded pixel values in config.yaml when no calibration exists yet.
    """

    def __init__(self, config: dict):
        self._config_coords: dict = config.get("coordinates", {})
        self._screen_w: int = config.get("screen", {}).get("width", 1080)
        self._screen_h: int = config.get("screen", {}).get("height", 1920)
        self._calibrated: dict[str, CalibratedCoordinate] = {}
        self._load_persisted()

    # ── public API ──────────────────────────────────────────────

    def get_pixel(self, name: str) -> tuple[int, int]:
        """Return (x, y) pixel coordinates for a named element.

        Uses calibrated percentage if available, otherwise falls back to
        config.yaml pixel values.
        """
        if name in self._calibrated:
            return self._calibrated[name].to_pixel(self._screen_w, self._screen_h)

        # Fallback to config.yaml
        c = self._config_coords.get(name, {})
        return (c.get("x", 0), c.get("y", 0))

    def is_calibrated(self, name: str) -> bool:
        """Check if a specific element has been calibrated."""
        return name in self._calibrated

    def needs_calibration(self, screen_type: str) -> list[str]:
        """Return element names that still need calibration for *screen_type*."""
        elements = SCREEN_ELEMENTS.get(screen_type, [])
        return [e for e in elements if e not in self._calibrated]

    def store(self, name: str, x_percent: float, y_percent: float, confidence: float) -> None:
        """Store a calibrated coordinate (clamped to 0-100)."""
        x_percent = max(0.0, min(100.0, x_percent))
        y_percent = max(0.0, min(100.0, y_percent))

        self._calibrated[name] = CalibratedCoordinate(
            name=name,
            x_percent=x_percent,
            y_percent=y_percent,
            screen_width=self._screen_w,
            screen_height=self._screen_h,
            confidence=confidence,
            discovered_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            f"Calibrated {name}: ({x_percent:.1f}%, {y_percent:.1f}%) "
            f"→ {self._calibrated[name].to_pixel(self._screen_w, self._screen_h)} "
            f"(confidence={confidence:.2f})"
        )

    def derive_minimap_slots(self) -> bool:
        """Derive all 4 monument slot positions from the two grid reference points.

        Uses minimap_square_topleft (slot 1) and minimap_square_bottomright (slot 4)
        to compute slots 2 and 3 assuming a 2x2 grid.

        Returns True if derivation succeeded.
        """
        tl = self._calibrated.get("minimap_square_topleft")
        br = self._calibrated.get("minimap_square_bottomright")
        if not tl or not br:
            return False

        # Top-left = slot 1, bottom-right = slot 4
        # Slot 2 = top-right, slot 3 = bottom-left
        confidence = min(tl.confidence, br.confidence)

        self.store("monument_slot_1", tl.x_percent, tl.y_percent, confidence)
        self.store("monument_slot_2", br.x_percent, tl.y_percent, confidence)  # top-right
        self.store("monument_slot_3", tl.x_percent, br.y_percent, confidence)  # bottom-left
        self.store("monument_slot_4", br.x_percent, br.y_percent, confidence)

        logger.info(
            f"Derived 4 minimap slots from grid corners: "
            f"TL=({tl.x_percent:.1f}%, {tl.y_percent:.1f}%) "
            f"BR=({br.x_percent:.1f}%, {br.y_percent:.1f}%)"
        )
        return True

    def clear_all(self) -> None:
        """Clear all calibrated coordinates and delete the persisted file."""
        self._calibrated.clear()
        self._delete_persisted()
        logger.info("Cleared all calibrated coordinates")

    def invalidate(self, name: str) -> None:
        """Remove a single calibrated element, forcing re-calibration next time."""
        if name in self._calibrated:
            logger.info(f"Invalidating calibration for {name}")
            del self._calibrated[name]

    def set_screen_dimensions(self, w: int, h: int) -> None:
        """Update screen dimensions. Invalidates calibration if they changed."""
        if w == self._screen_w and h == self._screen_h:
            return

        logger.info(
            f"Screen dimensions changed from {self._screen_w}x{self._screen_h} "
            f"to {w}x{h} — invalidating calibration cache"
        )
        self._screen_w = w
        self._screen_h = h
        self._calibrated.clear()
        self._delete_persisted()

    def save(self) -> None:
        """Persist calibrated coordinates to JSON."""
        data = {
            "screen_width": self._screen_w,
            "screen_height": self._screen_h,
            "coordinates": {
                name: asdict(coord) for name, coord in self._calibrated.items()
            },
        }
        _CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CALIBRATION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug(f"Saved {len(self._calibrated)} calibrated coordinates")

    # ── internals ───────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """Load previously calibrated coordinates from JSON, if dimensions match."""
        if not _CALIBRATION_FILE.exists():
            return

        try:
            data = json.loads(_CALIBRATION_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load calibration file: {e}")
            return

        saved_w = data.get("screen_width", 0)
        saved_h = data.get("screen_height", 0)
        if saved_w != self._screen_w or saved_h != self._screen_h:
            logger.info(
                f"Calibration file dimensions ({saved_w}x{saved_h}) don't match "
                f"current ({self._screen_w}x{self._screen_h}) — ignoring"
            )
            return

        for name, coord_data in data.get("coordinates", {}).items():
            self._calibrated[name] = CalibratedCoordinate(**coord_data)

        logger.info(f"Loaded {len(self._calibrated)} calibrated coordinates from cache")

    def _delete_persisted(self) -> None:
        """Remove the calibration file."""
        if _CALIBRATION_FILE.exists():
            _CALIBRATION_FILE.unlink()
            logger.debug("Deleted stale calibration file")
