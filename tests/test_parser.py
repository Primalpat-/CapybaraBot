"""Tests for vision response parser."""

import pytest
from src.vision.parser import (
    _extract_json,
    parse_screen_identification,
    parse_minimap_colors,
    parse_monument_info,
    parse_navigation_check,
    parse_world_monument_location,
    parse_battle_check,
    parse_post_battle,
    parse_calibration_result,
    parse_timer_seconds,
)


class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_surrounded_by_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")


class TestParseScreenIdentification:
    def test_basic(self):
        text = '{"screen_type": "minimap", "confidence": 0.95, "details": "Minimap overlay visible"}'
        result = parse_screen_identification(text)
        assert result.screen_type == "minimap"
        assert result.confidence == 0.95
        assert "Minimap" in result.details


class TestParseMinimapColors:
    def test_all_red(self):
        text = """{
            "squares": [
                {"slot": 1, "color": "red"},
                {"slot": 2, "color": "red"},
                {"slot": 3, "color": "red"},
                {"slot": 4, "color": "red"}
            ],
            "details": "all enemy"
        }"""
        result = parse_minimap_colors(text)
        assert len(result.slot_colors) == 4
        assert all(c == "red" for c in result.slot_colors.values())

    def test_mixed(self):
        text = """{
            "squares": [
                {"slot": 1, "color": "blue"},
                {"slot": 2, "color": "red"},
                {"slot": 3, "color": "blue"},
                {"slot": 4, "color": "red"}
            ],
            "details": "mixed"
        }"""
        result = parse_minimap_colors(text)
        assert result.slot_colors[1] == "blue"
        assert result.slot_colors[2] == "red"
        assert result.slot_colors[3] == "blue"
        assert result.slot_colors[4] == "red"

    def test_all_blue(self):
        text = '{"squares": [{"slot": 1, "color": "blue"}, {"slot": 2, "color": "blue"}, {"slot": 3, "color": "blue"}, {"slot": 4, "color": "blue"}], "details": "all friendly"}'
        result = parse_minimap_colors(text)
        assert all(c == "blue" for c in result.slot_colors.values())

    def test_empty_squares(self):
        text = '{"squares": [], "details": "no squares"}'
        result = parse_minimap_colors(text)
        assert len(result.slot_colors) == 0

    def test_case_insensitive(self):
        text = '{"squares": [{"slot": 1, "color": "RED"}, {"slot": 2, "color": "Blue"}], "details": ""}'
        result = parse_minimap_colors(text)
        assert result.slot_colors[1] == "red"
        assert result.slot_colors[2] == "blue"


class TestParseMonumentInfo:
    def test_enemy_monument(self):
        text = """{
            "ownership": "enemy",
            "is_friendly": false,
            "monument_name": "Tower Alpha",
            "defenders": [
                {"slot": 1, "status": "active"},
                {"slot": 2, "status": "defeated"}
            ],
            "all_defenders_defeated": false,
            "action_button": {
                "visible": true,
                "text": "Attack",
                "action_type": "attack"
            }
        }"""
        result = parse_monument_info(text)
        assert result.ownership == "enemy"
        assert result.is_friendly is False
        assert result.monument_name == "Tower Alpha"
        assert len(result.defenders) == 2
        assert result.defenders[0].status == "active"
        assert result.action_button.visible is True
        assert result.action_button.action_type == "attack"

    def test_friendly_monument(self):
        text = '{"ownership": "player", "is_friendly": true, "monument_name": "", "defenders": [], "all_defenders_defeated": true, "action_button": {"visible": false, "text": ""}}'
        result = parse_monument_info(text)
        assert result.is_friendly is True
        assert result.all_defenders_defeated is True


class TestParseNavigationCheck:
    def test_arrived(self):
        text = '{"arrived": true, "monument_popup_visible": true, "screen_type": "monument_popup", "details": "popup visible"}'
        result = parse_navigation_check(text)
        assert result.arrived is True
        assert result.monument_popup_visible is True

    def test_still_navigating(self):
        text = '{"arrived": false, "monument_popup_visible": false, "screen_type": "navigating", "details": "still moving"}'
        result = parse_navigation_check(text)
        assert result.arrived is False


class TestParseWorldMonumentLocation:
    def test_found(self):
        text = '{"found": true, "x_percent": 48.5, "y_percent": 55.2, "confidence": 0.9, "details": "stone monument"}'
        result = parse_world_monument_location(text)
        assert result.found is True
        assert result.x_percent == 48.5
        assert result.y_percent == 55.2
        assert result.confidence == 0.9

    def test_not_found(self):
        text = '{"found": false, "x_percent": 50, "y_percent": 50, "confidence": 0, "details": "no monument visible"}'
        result = parse_world_monument_location(text)
        assert result.found is False
        assert result.confidence == 0


class TestParseBattleCheck:
    def test_victory(self):
        text = '{"battle_state": "victory", "skip_button_visible": false, "continue_button_visible": true, "details": "won"}'
        result = parse_battle_check(text)
        assert result.battle_state == "victory"
        assert result.continue_button_visible is True

    def test_in_progress(self):
        text = '{"battle_state": "in_progress", "skip_button_visible": true, "continue_button_visible": false, "details": "fighting"}'
        result = parse_battle_check(text)
        assert result.battle_state == "in_progress"
        assert result.skip_button_visible is True


class TestParsePostBattle:
    def test_captured(self):
        text = '{"monument_captured": true, "remaining_defenders": 0, "all_defenders_defeated": true, "next_action_available": "claim", "action_button": {"visible": true, "text": "Claim"}}'
        result = parse_post_battle(text)
        assert result.monument_captured is True
        assert result.remaining_defenders == 0
        assert result.all_defenders_defeated is True

    def test_more_defenders(self):
        text = '{"monument_captured": false, "remaining_defenders": 2, "all_defenders_defeated": false, "next_action_available": "attack_next_defender", "action_button": {"visible": true, "text": "Attack"}}'
        result = parse_post_battle(text)
        assert result.monument_captured is False
        assert result.remaining_defenders == 2
        assert result.next_action_available == "attack_next_defender"


class TestParseCalibrationResult:
    def test_basic(self):
        text = """{
            "elements": [
                {"name": "minimap_button", "x_percent": 90.5, "y_percent": 9.4, "confidence": 0.95}
            ],
            "screen_description": "Main game screen with minimap icon"
        }"""
        result = parse_calibration_result(text)
        assert len(result.elements) == 1
        assert result.elements[0].name == "minimap_button"
        assert result.elements[0].x_percent == 90.5
        assert result.elements[0].y_percent == 9.4
        assert result.elements[0].confidence == 0.95
        assert "minimap" in result.screen_description.lower()

    def test_multiple_elements(self):
        text = """{
            "elements": [
                {"name": "action_button", "x_percent": 50.0, "y_percent": 83.3, "confidence": 0.9},
                {"name": "close_popup", "x_percent": 88.0, "y_percent": 20.8, "confidence": 0.85}
            ],
            "screen_description": "Monument popup"
        }"""
        result = parse_calibration_result(text)
        assert len(result.elements) == 2
        assert result.elements[0].name == "action_button"
        assert result.elements[1].name == "close_popup"

    def test_zero_confidence_element(self):
        text = """{
            "elements": [
                {"name": "skip_battle", "x_percent": 0, "y_percent": 0, "confidence": 0}
            ],
            "screen_description": "Could not find element"
        }"""
        result = parse_calibration_result(text)
        assert len(result.elements) == 1
        assert result.elements[0].confidence == 0

    def test_empty_elements(self):
        text = '{"elements": [], "screen_description": "empty screen"}'
        result = parse_calibration_result(text)
        assert len(result.elements) == 0

    def test_markdown_fenced(self):
        text = '```json\n{"elements": [{"name": "minimap_button", "x_percent": 91, "y_percent": 10, "confidence": 0.8}], "screen_description": "main map"}\n```'
        result = parse_calibration_result(text)
        assert len(result.elements) == 1
        assert result.elements[0].name == "minimap_button"


class TestParseTimerSeconds:
    def test_hhmmss(self):
        assert parse_timer_seconds("00:35:31") == 35 * 60 + 31

    def test_hhmmss_with_hours(self):
        assert parse_timer_seconds("1:02:03") == 3600 + 120 + 3

    def test_mmss(self):
        assert parse_timer_seconds("05:30") == 5 * 60 + 30

    def test_empty_string(self):
        assert parse_timer_seconds("") is None

    def test_garbage(self):
        assert parse_timer_seconds("no timer here") is None

    def test_negative_timer_returns_none(self):
        assert parse_timer_seconds("-15:-07:-22") is None

    def test_negative_mmss_returns_none(self):
        assert parse_timer_seconds("-05:-30") is None

    def test_screen_identification_timer(self):
        text = '{"screen_type": "hibernation", "confidence": 0.95, "details": "Sleeping capybara", "timer": "00:35:31"}'
        result = parse_screen_identification(text)
        assert result.screen_type == "hibernation"
        assert result.timer == "00:35:31"

    def test_screen_identification_no_timer(self):
        text = '{"screen_type": "main_map", "confidence": 0.9, "details": "Game world"}'
        result = parse_screen_identification(text)
        assert result.timer == ""
