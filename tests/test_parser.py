"""Tests for vision response parser."""

import pytest
from src.vision.parser import (
    _extract_json,
    parse_screen_identification,
    parse_minimap_reading,
    parse_monument_info,
    parse_navigation_check,
    parse_battle_check,
    parse_post_battle,
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


class TestParseMinimapReading:
    def test_with_monuments(self):
        text = """{
            "monuments": [
                {"id": 1, "x_percent": 30, "y_percent": 40, "appearance": "red icon", "likely_type": "enemy"},
                {"id": 2, "x_percent": 70, "y_percent": 60, "appearance": "blue icon", "likely_type": "friendly"}
            ],
            "player_position": {"x_percent": 50, "y_percent": 50},
            "total_monuments_visible": 2
        }"""
        result = parse_minimap_reading(text)
        assert len(result.monuments) == 2
        assert result.monuments[0].likely_type == "enemy"
        assert result.player_position.x_percent == 50
        assert result.total_monuments_visible == 2

    def test_empty_monuments(self):
        text = '{"monuments": [], "player_position": null, "total_monuments_visible": 0}'
        result = parse_minimap_reading(text)
        assert len(result.monuments) == 0
        assert result.player_position is None


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
