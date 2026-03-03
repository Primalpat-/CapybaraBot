"""Tests for the OCR reader module."""

import numpy as np
import pytest

from src.vision.ocr_reader import (
    _clean_name,
    _crop_region,
    _detect_text_color,
    _extract_power_number,
    _is_name_text,
    _is_power_text,
    OCRMonumentReading,
    OCRDefenderReading,
)


class TestExtractPowerNumber:
    def test_plain_number(self):
        assert _extract_power_number("12345") == 12345

    def test_comma_formatted(self):
        assert _extract_power_number("12,345") == 12345

    def test_large_comma_formatted(self):
        assert _extract_power_number("1,234,567") == 1234567

    def test_text_mixed_in(self):
        assert _extract_power_number("Power: 5,000") == 5000

    def test_no_number(self):
        assert _extract_power_number("no numbers here") == 0

    def test_empty_string(self):
        assert _extract_power_number("") == 0

    def test_none_returns_zero(self):
        assert _extract_power_number(None) == 0

    def test_single_digit(self):
        assert _extract_power_number("5") == 5

    def test_leading_trailing_spaces(self):
        assert _extract_power_number("  3,456  ") == 3456

    def test_millions_suffix(self):
        assert _extract_power_number("24.68M") == 24680000

    def test_millions_suffix_lowercase(self):
        assert _extract_power_number("14.28m") == 14280000

    def test_thousands_suffix(self):
        assert _extract_power_number("39.29K") == 39290

    def test_whole_millions(self):
        assert _extract_power_number("3M") == 3000000

    def test_suffix_with_surrounding_text(self):
        assert _extract_power_number("Power 24.68M total") == 24680000

    def test_billions_suffix(self):
        assert _extract_power_number("1.5B") == 1500000000

    def test_trillions_suffix(self):
        assert _extract_power_number("2.3T") == 2300000000000


class TestDetectTextColor:
    """_detect_text_color(image, bbox) extracts color from bbox region."""

    def _make_solid_color(self, bgr, size=(50, 100)):
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = bgr
        return img

    def _full_bbox(self, h, w):
        """Bbox covering the entire image."""
        return [[0, 0], [w, 0], [w, h], [0, h]]

    def test_blue_image(self):
        img = self._make_solid_color((255, 100, 0))
        bbox = self._full_bbox(50, 100)
        assert _detect_text_color(img, bbox) == "blue"

    def test_red_image(self):
        img = self._make_solid_color((0, 0, 255))
        bbox = self._full_bbox(50, 100)
        assert _detect_text_color(img, bbox) == "red"

    def test_unknown_gray(self):
        img = self._make_solid_color((128, 128, 128))
        bbox = self._full_bbox(50, 100)
        assert _detect_text_color(img, bbox) == "unknown"

    def test_empty_image(self):
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert _detect_text_color(img, bbox) == "unknown"


class TestIsPowerText:
    """Power text requires a magnitude suffix (K/M/B/T)."""

    def test_millions(self):
        assert _is_power_text("24.68M") is True

    def test_thousands(self):
        assert _is_power_text("14.28K") is True

    def test_billions(self):
        assert _is_power_text("1.5B") is True

    def test_trillions(self):
        assert _is_power_text("2.3T") is True

    def test_whole_suffix(self):
        assert _is_power_text("3M") is True

    def test_plain_number_no_suffix(self):
        """Plain numbers without K/M/B/T suffix are NOT power text."""
        assert _is_power_text("12345") is False
        assert _is_power_text("12,345") is False

    def test_name_not_power(self):
        assert _is_power_text("Primalpat") is False

    def test_mixed_text(self):
        assert _is_power_text("Defense Info") is False

    def test_level_monument_not_power(self):
        """'Monument' M is not a suffix — letters follow it."""
        assert _is_power_text("evel 11 Monument") is False
        assert _is_power_text("Level 11 Monument") is False

    def test_single_digit_not_power(self):
        assert _is_power_text("9") is False
        assert _is_power_text("2") is False

    def test_earnings_not_power(self):
        """Earnings text like '4500/hour' has no suffix."""
        assert _is_power_text("4500/hour") is False
        assert _is_power_text("450/hour") is False


class TestIsNameText:
    """Player names: 2+ alpha chars, not power/noise/button text."""

    def test_player_name(self):
        assert _is_name_text("Primalpat") is True

    def test_player_name_with_quote(self):
        assert _is_name_text("'Primalpat") is True

    def test_short_name(self):
        assert _is_name_text("Lonz") is True

    def test_single_letter_not_name(self):
        """OCR noise like 'U' or 'X' filtered out."""
        assert _is_name_text("U") is False
        assert _is_name_text("X") is False

    def test_digit_not_name(self):
        assert _is_name_text("2") is False
        assert _is_name_text("123") is False

    def test_noise_words_not_name(self):
        assert _is_name_text("Defense Info") is False
        assert _is_name_text("estimated") is False

    def test_button_words_not_name(self):
        assert _is_name_text("attack") is False
        assert _is_name_text("Exit") is False

    def test_power_text_not_name(self):
        assert _is_name_text("24.68M") is False

    def test_faction_names_not_player_names(self):
        """Faction words should be filtered out as noise."""
        assert _is_name_text("Star Spirit") is False
        assert _is_name_text("Galactic Empire") is False
        assert _is_name_text("Interstellar Federation") is False
        assert _is_name_text("Star Alliance") is False


class TestCleanName:
    def test_strips_leading_quote(self):
        assert _clean_name("'Primalpat") == "Primalpat"

    def test_strips_curly_quotes(self):
        assert _clean_name("\u2018Sionna\u2019") == "Sionna"

    def test_no_artifacts(self):
        assert _clean_name("Lonz") == "Lonz"

    def test_strips_backtick(self):
        assert _clean_name("`Bob`") == "Bob"


class TestCropRegion:
    def test_correct_bounds(self):
        """Crop region with percentage-based coords produces correct shape."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        region = (0.25, 0.1, 0.75, 0.9)  # left 25%-75%, top 10%-90%
        cropped = _crop_region(img, region)
        # Width: 75%-25% = 50% of 200 = 100
        # Height: 90%-10% = 80% of 100 = 80
        assert cropped.shape == (80, 100, 3)

    def test_full_image(self):
        img = np.zeros((50, 80, 3), dtype=np.uint8)
        cropped = _crop_region(img, (0.0, 0.0, 1.0, 1.0))
        assert cropped.shape == img.shape

    def test_zero_region(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = _crop_region(img, (0.5, 0.5, 0.5, 0.5))
        assert cropped.size == 0


class TestOCRDataclasses:
    def test_defender_reading_defaults(self):
        d = OCRDefenderReading(slot=1)
        assert d.slot == 1
        assert d.name == ""
        assert d.power == 0
        assert d.status == "unknown"
        assert d.confidence == 0.0

    def test_monument_reading_defaults(self):
        r = OCRMonumentReading()
        assert r.ownership_text == ""
        assert r.is_friendly is None
        assert r.defenders == []
        assert r.action_button_text == ""
        assert r.overall_confidence == 0.0
        assert r.total_garrison_power == 0

    def test_monument_reading_with_defenders(self):
        defenders = [
            OCRDefenderReading(slot=1, name="Alice", power=5000, status="active"),
            OCRDefenderReading(slot=2, name="Bob", power=3000, status="active"),
        ]
        r = OCRMonumentReading(
            defenders=defenders,
            total_garrison_power=8000,
            is_friendly=False,
        )
        assert len(r.defenders) == 2
        assert r.total_garrison_power == 8000
        assert r.is_friendly is False
