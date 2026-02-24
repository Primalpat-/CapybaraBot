"""Tests for monument selection strategy."""

import pytest
from src.bot.strategy import select_next_monument, distance
from src.vision.parser import MinimapReading, MonumentPosition, PlayerPosition


class TestDistance:
    def test_same_point(self):
        assert distance(50, 50, 50, 50) == 0

    def test_known_distance(self):
        d = distance(0, 0, 3, 4)
        assert abs(d - 5.0) < 0.001


class TestSelectNextMonument:
    def _make_minimap(self, monuments, player_x=50, player_y=50):
        return MinimapReading(
            monuments=monuments,
            player_position=PlayerPosition(x_percent=player_x, y_percent=player_y),
            total_monuments_visible=len(monuments),
        )

    def test_prefers_enemy_over_neutral(self):
        monuments = [
            MonumentPosition(id=1, x_percent=30, y_percent=30, appearance="", likely_type="neutral"),
            MonumentPosition(id=2, x_percent=80, y_percent=80, appearance="", likely_type="enemy"),
        ]
        minimap = self._make_minimap(monuments)
        result = select_next_monument(minimap)
        assert result.id == 2

    def test_prefers_nearest_enemy(self):
        monuments = [
            MonumentPosition(id=1, x_percent=90, y_percent=90, appearance="", likely_type="enemy"),
            MonumentPosition(id=2, x_percent=55, y_percent=55, appearance="", likely_type="enemy"),
        ]
        minimap = self._make_minimap(monuments)
        result = select_next_monument(minimap)
        assert result.id == 2

    def test_skips_friendly(self):
        monuments = [
            MonumentPosition(id=1, x_percent=51, y_percent=51, appearance="", likely_type="friendly"),
            MonumentPosition(id=2, x_percent=80, y_percent=80, appearance="", likely_type="enemy"),
        ]
        minimap = self._make_minimap(monuments)
        result = select_next_monument(minimap)
        assert result.id == 2

    def test_skips_visited(self):
        monuments = [
            MonumentPosition(id=1, x_percent=51, y_percent=51, appearance="", likely_type="enemy"),
            MonumentPosition(id=2, x_percent=80, y_percent=80, appearance="", likely_type="enemy"),
        ]
        minimap = self._make_minimap(monuments)
        result = select_next_monument(minimap, visited_ids={1})
        assert result.id == 2

    def test_returns_none_when_all_visited(self):
        monuments = [
            MonumentPosition(id=1, x_percent=30, y_percent=30, appearance="", likely_type="enemy"),
        ]
        minimap = self._make_minimap(monuments)
        result = select_next_monument(minimap, visited_ids={1})
        assert result is None

    def test_empty_minimap(self):
        minimap = self._make_minimap([])
        result = select_next_monument(minimap)
        assert result is None
