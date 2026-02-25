"""Tests for coordinate calibration system."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.bot.calibration import (
    CoordinateCalibrator,
    CalibratedCoordinate,
    SCREEN_ELEMENTS,
    _CALIBRATION_FILE,
)


@pytest.fixture
def config():
    return {
        "screen": {"width": 1080, "height": 1920},
        "coordinates": {
            "minimap_button": {"x": 980, "y": 180},
            "action_button": {"x": 540, "y": 1600},
            "close_popup": {"x": 950, "y": 400},
            "skip_battle": {"x": 540, "y": 1750},
        },
    }


@pytest.fixture
def calibrator(config, tmp_path, monkeypatch):
    """Create a calibrator that uses a temp file for persistence."""
    temp_file = tmp_path / "calibrated_coords.json"
    monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)
    return CoordinateCalibrator(config)


class TestCalibratedCoordinate:
    def test_to_pixel(self):
        coord = CalibratedCoordinate(
            name="test", x_percent=50.0, y_percent=25.0,
            screen_width=1080, screen_height=1920,
            confidence=0.9, discovered_at="2024-01-01T00:00:00",
        )
        assert coord.to_pixel(1080, 1920) == (540, 480)

    def test_to_pixel_different_resolution(self):
        coord = CalibratedCoordinate(
            name="test", x_percent=50.0, y_percent=50.0,
            screen_width=1080, screen_height=1920,
            confidence=0.9, discovered_at="2024-01-01T00:00:00",
        )
        # Percentages are resolution-independent
        assert coord.to_pixel(720, 1280) == (360, 640)

    def test_to_pixel_boundary(self):
        coord = CalibratedCoordinate(
            name="test", x_percent=100.0, y_percent=100.0,
            screen_width=1080, screen_height=1920,
            confidence=0.9, discovered_at="2024-01-01T00:00:00",
        )
        assert coord.to_pixel(1080, 1920) == (1080, 1920)


class TestGetPixel:
    def test_fallback_to_config(self, calibrator):
        """Without calibration, returns config.yaml values."""
        assert calibrator.get_pixel("minimap_button") == (980, 180)
        assert calibrator.get_pixel("action_button") == (540, 1600)

    def test_calibrated_overrides_config(self, calibrator):
        """After calibration, returns calibrated values."""
        calibrator.store("minimap_button", 90.0, 10.0, 0.95)
        x, y = calibrator.get_pixel("minimap_button")
        assert x == int(90.0 / 100 * 1080)
        assert y == int(10.0 / 100 * 1920)

    def test_unknown_element_returns_zero(self, calibrator):
        """Unknown element without config entry returns (0, 0)."""
        assert calibrator.get_pixel("nonexistent") == (0, 0)


class TestNeedsCalibration:
    def test_all_needed_initially(self, calibrator):
        needed = calibrator.needs_calibration("main_map")
        assert needed == ["minimap_button"]

    def test_none_needed_after_calibration(self, calibrator):
        calibrator.store("minimap_button", 90.0, 10.0, 0.95)
        assert calibrator.needs_calibration("main_map") == []

    def test_partial_calibration(self, calibrator):
        calibrator.store("action_button", 50.0, 83.0, 0.9)
        needed = calibrator.needs_calibration("monument_popup")
        assert needed == ["close_popup"]

    def test_unknown_screen_type(self, calibrator):
        assert calibrator.needs_calibration("nonexistent_screen") == []

    def test_minimap_grid_refs(self, calibrator):
        needed = calibrator.needs_calibration("minimap")
        assert len(needed) == 3
        assert "minimap_square_topleft" in needed
        assert "minimap_square_bottomright" in needed
        assert "minimap_close" in needed

    def test_minimap_grid_refs_partial(self, calibrator):
        calibrator.store("minimap_square_topleft", 25.0, 35.0, 0.9)
        needed = calibrator.needs_calibration("minimap")
        assert "minimap_square_bottomright" in needed
        assert "minimap_close" in needed

    def test_arrived_at_monument(self, calibrator):
        needed = calibrator.needs_calibration("arrived_at_monument")
        assert needed == ["world_monument"]


class TestStore:
    def test_clamps_coordinates(self, calibrator):
        calibrator.store("test", 150.0, -20.0, 0.8)
        coord = calibrator._calibrated["test"]
        assert coord.x_percent == 100.0
        assert coord.y_percent == 0.0

    def test_stores_with_metadata(self, calibrator):
        calibrator.store("minimap_button", 90.5, 9.4, 0.95)
        coord = calibrator._calibrated["minimap_button"]
        assert coord.name == "minimap_button"
        assert coord.screen_width == 1080
        assert coord.screen_height == 1920
        assert coord.confidence == 0.95
        assert coord.discovered_at  # non-empty timestamp


class TestDeriveMiniapSlots:
    def test_derives_all_four(self, calibrator):
        calibrator.store("minimap_square_topleft", 25.0, 30.0, 0.9)
        calibrator.store("minimap_square_bottomright", 75.0, 70.0, 0.85)
        assert calibrator.derive_minimap_slots() is True

        # Slot 1 = top-left
        assert calibrator._calibrated["monument_slot_1"].x_percent == 25.0
        assert calibrator._calibrated["monument_slot_1"].y_percent == 30.0
        # Slot 2 = top-right
        assert calibrator._calibrated["monument_slot_2"].x_percent == 75.0
        assert calibrator._calibrated["monument_slot_2"].y_percent == 30.0
        # Slot 3 = bottom-left
        assert calibrator._calibrated["monument_slot_3"].x_percent == 25.0
        assert calibrator._calibrated["monument_slot_3"].y_percent == 70.0
        # Slot 4 = bottom-right
        assert calibrator._calibrated["monument_slot_4"].x_percent == 75.0
        assert calibrator._calibrated["monument_slot_4"].y_percent == 70.0

    def test_returns_false_without_refs(self, calibrator):
        assert calibrator.derive_minimap_slots() is False

    def test_returns_false_with_partial_refs(self, calibrator):
        calibrator.store("minimap_square_topleft", 25.0, 30.0, 0.9)
        assert calibrator.derive_minimap_slots() is False

    def test_pixel_conversion(self, calibrator):
        calibrator.store("minimap_square_topleft", 25.0, 30.0, 0.9)
        calibrator.store("minimap_square_bottomright", 75.0, 70.0, 0.9)
        calibrator.derive_minimap_slots()

        # Slot 2 (top-right): 75% of 1080 = 810, 30% of 1920 = 576
        assert calibrator.get_pixel("monument_slot_2") == (810, 576)


class TestPersistence:
    def test_save_and_load(self, config, tmp_path, monkeypatch):
        temp_file = tmp_path / "calibrated_coords.json"
        monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)

        # Save
        cal1 = CoordinateCalibrator(config)
        cal1.store("minimap_button", 90.0, 10.0, 0.95)
        cal1.store("action_button", 50.0, 83.0, 0.9)
        cal1.save()

        assert temp_file.exists()

        # Load in a new instance
        cal2 = CoordinateCalibrator(config)
        assert "minimap_button" in cal2._calibrated
        assert "action_button" in cal2._calibrated
        assert cal2.get_pixel("minimap_button") == cal1.get_pixel("minimap_button")

    def test_dimension_mismatch_ignores_cache(self, config, tmp_path, monkeypatch):
        temp_file = tmp_path / "calibrated_coords.json"
        monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)

        # Save with 1080x1920
        cal1 = CoordinateCalibrator(config)
        cal1.store("minimap_button", 90.0, 10.0, 0.95)
        cal1.save()

        # Load with different dimensions
        config2 = {**config, "screen": {"width": 720, "height": 1280}}
        cal2 = CoordinateCalibrator(config2)
        assert "minimap_button" not in cal2._calibrated

    def test_corrupt_file_handled(self, config, tmp_path, monkeypatch):
        temp_file = tmp_path / "calibrated_coords.json"
        temp_file.write_text("not valid json", encoding="utf-8")
        monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)

        # Should not raise
        cal = CoordinateCalibrator(config)
        assert len(cal._calibrated) == 0

    def test_missing_file_handled(self, config, tmp_path, monkeypatch):
        temp_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)

        cal = CoordinateCalibrator(config)
        assert len(cal._calibrated) == 0


class TestSetScreenDimensions:
    def test_same_dimensions_noop(self, calibrator):
        calibrator.store("minimap_button", 90.0, 10.0, 0.95)
        calibrator.set_screen_dimensions(1080, 1920)
        assert "minimap_button" in calibrator._calibrated

    def test_different_dimensions_clears_cache(self, calibrator):
        calibrator.store("minimap_button", 90.0, 10.0, 0.95)
        calibrator.set_screen_dimensions(720, 1280)
        assert len(calibrator._calibrated) == 0

    def test_different_dimensions_deletes_file(self, config, tmp_path, monkeypatch):
        temp_file = tmp_path / "calibrated_coords.json"
        monkeypatch.setattr("src.bot.calibration._CALIBRATION_FILE", temp_file)

        cal = CoordinateCalibrator(config)
        cal.store("minimap_button", 90.0, 10.0, 0.95)
        cal.save()
        assert temp_file.exists()

        cal.set_screen_dimensions(720, 1280)
        assert not temp_file.exists()

    def test_updates_internal_dimensions(self, calibrator):
        calibrator.set_screen_dimensions(720, 1280)
        assert calibrator._screen_w == 720
        assert calibrator._screen_h == 1280
