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


@dataclass
class NavigationCheck:
    arrived: bool
    monument_popup_visible: bool
    screen_type: str
    details: str


@dataclass
class BattleCheck:
    battle_state: str  # "in_progress", "victory", "defeat", "results_screen"
    skip_button_visible: bool
    continue_button_visible: bool
    details: str


@dataclass
class PostBattleInfo:
    monument_captured: bool
    remaining_defenders: int | None
    all_defenders_defeated: bool
    next_action_available: str
    action_button: ActionButton


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
    )


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
        ))

    ab = data.get("action_button", {})
    action_button = ActionButton(
        visible=bool(ab.get("visible", False)),
        text=ab.get("text", ""),
        action_type=ab.get("action_type", "unknown"),
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
    )


def parse_navigation_check(text: str) -> NavigationCheck:
    data = _extract_json(text)
    return NavigationCheck(
        arrived=bool(data.get("arrived", False)),
        monument_popup_visible=bool(data.get("monument_popup_visible", False)),
        screen_type=data.get("screen_type", "unknown"),
        details=data.get("details", ""),
    )


def parse_battle_check(text: str) -> BattleCheck:
    data = _extract_json(text)
    return BattleCheck(
        battle_state=data.get("battle_state", "unknown"),
        skip_button_visible=bool(data.get("skip_button_visible", False)),
        continue_button_visible=bool(data.get("continue_button_visible", False)),
        details=data.get("details", ""),
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
