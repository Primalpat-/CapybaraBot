"""Parse Vision API JSON responses into typed dataclasses."""

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScreenIdentification:
    screen_type: str
    confidence: float
    details: str
    timer: str = ""  # hibernation countdown e.g. "00:35:31"


@dataclass
class MonumentPosition:
    id: int
    x_percent: float
    y_percent: float
    appearance: str
    likely_type: str


@dataclass
class PlayerPosition:
    x_percent: float
    y_percent: float


@dataclass
class MinimapReading:
    monuments: list[MonumentPosition]
    player_position: PlayerPosition | None
    total_monuments_visible: int


@dataclass
class DefenderInfo:
    slot: int
    status: str  # "active", "defeated", "empty"
    name: str = ""


@dataclass
class ActionButton:
    visible: bool
    text: str
    action_type: str = ""


@dataclass
class MonumentInfo:
    ownership: str
    is_friendly: bool | None
    monument_name: str
    defenders: list[DefenderInfo]
    all_defenders_defeated: bool
    action_button: ActionButton
    ownership_text: str = ""


@dataclass
class NavigationCheck:
    arrived: bool
    monument_popup_visible: bool
    screen_type: str
    details: str


@dataclass
class WorldMonumentLocation:
    found: bool
    x_percent: float
    y_percent: float
    confidence: float
    details: str


@dataclass
class MinimapColors:
    """Color of each monument slot on the minimap (1-indexed)."""
    slot_colors: dict[int, str]  # {1: "red", 2: "blue", ...}
    details: str


@dataclass
class BattleCheck:
    battle_state: str  # "in_progress", "victory", "defeat", "results_screen"
    skip_button_visible: bool
    continue_button_visible: bool
    details: str
    opponent_name: str = ""


@dataclass
class CalibratedElement:
    name: str
    x_percent: float
    y_percent: float
    confidence: float


@dataclass
class CalibrationResult:
    elements: list[CalibratedElement]
    screen_description: str


@dataclass
class PostBattleInfo:
    monument_captured: bool
    remaining_defenders: int | None
    all_defenders_defeated: bool
    next_action_available: str
    action_button: ActionButton


@dataclass
class RecoveryGuidance:
    diagnosis: str
    suggested_action: str  # tap, back, wait, launch_app, give_up
    tap_x_percent: float
    tap_y_percent: float
    tap_description: str
    confidence: float


@dataclass
class DailyPopupCheck:
    popup_visible: bool
    do_not_show_found: bool
    do_not_show_x: float
    do_not_show_y: float
    close_found: bool
    close_x: float
    close_y: float
    details: str


def _extract_json(text: str) -> dict:
    """Extract JSON from a response that might contain markdown fences or extra text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def parse_screen_identification(text: str) -> ScreenIdentification:
    data = _extract_json(text)
    return ScreenIdentification(
        screen_type=data.get("screen_type", "unknown"),
        confidence=float(data.get("confidence", 0)),
        details=data.get("details", ""),
        timer=data.get("timer", ""),
    )


def parse_timer_seconds(timer_str: str) -> int | None:
    """Parse a 'HH:MM:SS' or 'MM:SS' countdown string into total seconds.

    Returns None if the string can't be parsed or contains negative values
    (the game displays negative timers as a glitch when the timer has expired).
    """
    if not timer_str:
        return None
    # Negative timers are a game display bug — treat as expired/broken
    if "-" in timer_str:
        return None
    match = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", timer_str)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    match = re.match(r"(\d{1,2}):(\d{2})", timer_str)
    if match:
        m, s = int(match.group(1)), int(match.group(2))
        return m * 60 + s
    return None


def parse_minimap_reading(text: str) -> MinimapReading:
    data = _extract_json(text)

    monuments = []
    for m in data.get("monuments", []):
        monuments.append(MonumentPosition(
            id=m.get("id", 0),
            x_percent=float(m.get("x_percent", 0)),
            y_percent=float(m.get("y_percent", 0)),
            appearance=m.get("appearance", ""),
            likely_type=m.get("likely_type", "unknown"),
        ))

    pp = data.get("player_position")
    player_pos = None
    if pp:
        player_pos = PlayerPosition(
            x_percent=float(pp.get("x_percent", 0)),
            y_percent=float(pp.get("y_percent", 0)),
        )

    return MinimapReading(
        monuments=monuments,
        player_position=player_pos,
        total_monuments_visible=int(data.get("total_monuments_visible", len(monuments))),
    )


def parse_monument_info(text: str) -> MonumentInfo:
    data = _extract_json(text)

    defenders = []
    for d in data.get("defenders", []):
        defenders.append(DefenderInfo(
            slot=int(d.get("slot", 0)),
            status=d.get("status", "unknown"),
            name=d.get("name", ""),
        ))

    ab = data.get("action_button", {})
    action_button = ActionButton(
        visible=bool(ab.get("visible", False)),
        text=ab.get("text", ""),
        action_type=ab.get("action_type", "unknown").lower(),
    )

    is_friendly = data.get("is_friendly")
    if is_friendly is not None:
        is_friendly = bool(is_friendly)

    return MonumentInfo(
        ownership=data.get("ownership", "unknown"),
        is_friendly=is_friendly,
        monument_name=data.get("monument_name", ""),
        defenders=defenders,
        all_defenders_defeated=bool(data.get("all_defenders_defeated", False)),
        action_button=action_button,
        ownership_text=data.get("ownership_text", ""),
    )


def parse_navigation_check(text: str) -> NavigationCheck:
    data = _extract_json(text)
    return NavigationCheck(
        arrived=bool(data.get("arrived", False)),
        monument_popup_visible=bool(data.get("monument_popup_visible", False)),
        screen_type=data.get("screen_type", "unknown"),
        details=data.get("details", ""),
    )


def parse_world_monument_location(text: str) -> WorldMonumentLocation:
    data = _extract_json(text)
    return WorldMonumentLocation(
        found=bool(data.get("found", False)),
        x_percent=float(data.get("x_percent", 50)),
        y_percent=float(data.get("y_percent", 50)),
        confidence=float(data.get("confidence", 0)),
        details=data.get("details", ""),
    )


def parse_minimap_colors(text: str) -> MinimapColors:
    data = _extract_json(text)
    slot_colors = {}
    for sq in data.get("squares", []):
        slot = int(sq.get("slot", 0))
        color = sq.get("color", "unknown").lower()
        if slot > 0:
            slot_colors[slot] = color
    return MinimapColors(
        slot_colors=slot_colors,
        details=data.get("details", ""),
    )


def parse_battle_check(text: str) -> BattleCheck:
    data = _extract_json(text)
    return BattleCheck(
        battle_state=data.get("battle_state", "unknown"),
        skip_button_visible=bool(data.get("skip_button_visible", False)),
        continue_button_visible=bool(data.get("continue_button_visible", False)),
        details=data.get("details", ""),
        opponent_name=data.get("opponent_name", ""),
    )


def parse_calibration_result(text: str) -> CalibrationResult:
    data = _extract_json(text)

    elements = []
    for el in data.get("elements", []):
        elements.append(CalibratedElement(
            name=el.get("name", ""),
            x_percent=float(el.get("x_percent", 0)),
            y_percent=float(el.get("y_percent", 0)),
            confidence=float(el.get("confidence", 0)),
        ))

    return CalibrationResult(
        elements=elements,
        screen_description=data.get("screen_description", ""),
    )


def parse_post_battle(text: str) -> PostBattleInfo:
    data = _extract_json(text)

    ab = data.get("action_button", {})
    action_button = ActionButton(
        visible=bool(ab.get("visible", False)),
        text=ab.get("text", ""),
    )

    remaining = data.get("remaining_defenders")
    if remaining is not None:
        remaining = int(remaining)

    return PostBattleInfo(
        monument_captured=bool(data.get("monument_captured", False)),
        remaining_defenders=remaining,
        all_defenders_defeated=bool(data.get("all_defenders_defeated", False)),
        next_action_available=data.get("next_action_available", "unknown"),
        action_button=action_button,
    )


def parse_recovery_guidance(text: str) -> RecoveryGuidance:
    data = _extract_json(text)
    tap = data.get("tap_target", {})
    return RecoveryGuidance(
        diagnosis=data.get("diagnosis", ""),
        suggested_action=data.get("suggested_action", "give_up"),
        tap_x_percent=float(tap.get("x_percent", 50)),
        tap_y_percent=float(tap.get("y_percent", 50)),
        tap_description=tap.get("description", ""),
        confidence=float(data.get("confidence", 0)),
    )


def parse_daily_popup_check(text: str) -> DailyPopupCheck:
    data = _extract_json(text)
    dns = data.get("do_not_show_text", {})
    close = data.get("close_button", {})
    return DailyPopupCheck(
        popup_visible=bool(data.get("popup_visible", False)),
        do_not_show_found=bool(dns.get("found", False)),
        do_not_show_x=float(dns.get("x_percent", 50)),
        do_not_show_y=float(dns.get("y_percent", 50)),
        close_found=bool(close.get("found", False)),
        close_x=float(close.get("x_percent", 50)),
        close_y=float(close.get("y_percent", 50)),
        details=data.get("details", ""),
    )
