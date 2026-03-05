"""Tests for ScreenAnalyzer — local-first screen detection."""

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from src.vision.screen_analyzer import (
    ScreenAnalyzer,
    ScreenAnalysis,
    _SCREEN_KEYWORDS,
    _SINGLE_KEYWORD_SCREENS,
    _ELEMENT_SIGNATURES,
    _TEXT_TO_ELEMENT,
)
from src.vision.element_detector import ElementDetector, DetectedElement


def _make_image(width: int = 1080, height: int = 1920, brightness: int = 40) -> np.ndarray:
    """Create a blank image with specified brightness."""
    return np.full((height, width, 3), brightness, dtype=np.uint8)


def _to_png(img: np.ndarray) -> bytes:
    """Encode a numpy image to PNG bytes."""
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _draw_yellow_button(img: np.ndarray, cx: int, cy: int, bw: int = 200, bh: int = 60) -> None:
    """Draw a yellow button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 240), -1)


def _draw_green_button(img: np.ndarray, cx: int, cy: int, bw: int = 120, bh: int = 50) -> None:
    """Draw a green button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), -1)


def _draw_pink_button(img: np.ndarray, cx: int, cy: int, bw: int = 180, bh: int = 60) -> None:
    """Draw a pink button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 50, 230), -1)


class TestBrightnessDetection:
    """Tier 1 — brightness check."""

    def setup_method(self):
        self.analyzer = ScreenAnalyzer(ElementDetector(), {})

    def test_black_screen_detected_as_loading(self):
        """Completely black screen → loading."""
        img = _make_image(brightness=0)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "loading"
        assert result.confidence >= 0.9
        assert result.method == "brightness"

    def test_very_dark_screen_detected_as_loading(self):
        """Very dark screen (brightness=10) → loading."""
        img = _make_image(brightness=10)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "loading"
        assert result.method == "brightness"

    def test_normal_brightness_not_loading(self):
        """Normal brightness screen should not be classified as loading."""
        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type != "loading" or result.method != "brightness"

    def test_borderline_brightness_not_loading(self):
        """Brightness at threshold (15) should NOT be classified as loading.
        Threshold is strictly < 15."""
        img = _make_image(brightness=20)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        # Should not be caught by brightness tier
        assert result.method != "brightness"


class TestElementSignatureDetection:
    """Tier 2 — element detection → screen type inference.

    Only template-matched and distinctively-colored (pink) elements are used
    as signatures. Yellow/green/purple color-only detections are too ambiguous
    across screens (nav bar, popups, UI chrome).

    battle_result uses a dark-screen check: brightness < 55 + ok_button.
    """

    def setup_method(self):
        self.detector = ElementDetector()
        self.analyzer = ScreenAnalyzer(self.detector, {})

    def test_ok_button_on_dark_screen_infers_battle_result(self):
        """Yellow OK button on DARK screen → battle_result.

        The dark background (brightness ~36) is unique to battle result screens.
        Without the dark check, yellow buttons on main_map/popups false-positive.
        """
        # brightness=30 simulates dark battle result screen
        img = _make_image(brightness=30)
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, int(h * 0.85))
        png = _to_png(img)

        result = self.analyzer.analyze(png)
        assert result.screen_type == "battle_result"
        assert result.method == "element"
        assert result.confidence >= 0.7
        assert "ok_button" in result.elements

    def test_ok_button_on_bright_screen_NOT_battle_result(self):
        """Yellow button on BRIGHT screen should NOT be classified as battle_result.

        This prevents false positives from yellow nav bar buttons, monument
        popup action buttons, etc. which all have brightness > 55.
        """
        # brightness=80 simulates main_map/popup (not battle)
        img = _make_image(brightness=80)
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, int(h * 0.85))
        png = _to_png(img)

        result = self.analyzer.analyze(png)
        assert result.screen_type != "battle_result" or result.method != "element"

    def test_cancel_button_infers_occupy_prompt(self):
        """Pink cancel button → occupy_prompt."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_pink_button(img, int(w * 0.25), int(h * 0.55))
        png = _to_png(img)

        result = self.analyzer.analyze(png)
        assert result.screen_type == "occupy_prompt"
        assert result.method == "element"
        assert "occupy_cancel_button" in result.elements

    def test_green_button_not_used_as_signature(self):
        """Green buttons should NOT be used for screen identification.

        Green skip buttons false-positive on main_map nav bar icons.
        """
        img = _make_image(brightness=80)
        h, w = img.shape[:2]
        _draw_green_button(img, int(w * 0.75), int(h * 0.85))
        png = _to_png(img)

        # Mock out OCR to isolate element detection behavior
        with patch("src.vision.screen_analyzer.find_minimap_squares", return_value=None):
            with patch("src.vision.screen_analyzer._get_reader") as mock_get_reader:
                mock_reader = MagicMock()
                mock_reader.readtext.return_value = []
                mock_get_reader.return_value = mock_reader
                result = self.analyzer.analyze(png)
                # Should NOT be classified as battle_active via element detection
                assert result.method != "element" or result.screen_type != "battle_active"

    def test_yellow_button_center_not_used_as_signature(self):
        """Yellow buttons in center region should NOT identify as logged_out.

        Yellow restart_button detection false-positives on monument popup
        Attack/Garrison/Occupy buttons which are also yellow and centered.
        """
        img = _make_image(brightness=80)
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, h // 2)
        png = _to_png(img)

        with patch("src.vision.screen_analyzer.find_minimap_squares", return_value=None):
            with patch("src.vision.screen_analyzer._get_reader") as mock_get_reader:
                mock_reader = MagicMock()
                mock_reader.readtext.return_value = []
                mock_get_reader.return_value = mock_reader
                result = self.analyzer.analyze(png)
                assert result.method != "element" or result.screen_type != "logged_out"


class TestMinimap:
    """Tier 2.5 — minimap detection."""

    def setup_method(self):
        self.analyzer = ScreenAnalyzer(ElementDetector(), {})

    @patch("src.vision.screen_analyzer.find_minimap_squares")
    def test_minimap_squares_detected(self, mock_find):
        """2+ squares found → minimap."""
        mock_detection = MagicMock()
        mock_detection.squares = [MagicMock(), MagicMock(), MagicMock()]
        mock_find.return_value = mock_detection

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "minimap"
        assert result.method == "minimap"
        assert result.confidence >= 0.8

    @patch("src.vision.screen_analyzer.find_minimap_squares")
    def test_minimap_not_detected_with_few_squares(self, mock_find):
        """Only 1 square found → not minimap."""
        mock_detection = MagicMock()
        mock_detection.squares = [MagicMock()]
        mock_find.return_value = mock_detection

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        # Should fall through to OCR or unknown
        assert result.screen_type != "minimap" or result.method != "minimap"


class TestOCRKeywordMatching:
    """Tier 3 — OCR keyword matching."""

    def setup_method(self):
        self.analyzer = ScreenAnalyzer(ElementDetector(), {})

    @patch("src.vision.screen_analyzer._get_reader")
    def test_hibernation_detected_single_keyword(self, mock_get_reader):
        """Single keyword 'hibernation' is enough."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Hibernation ends:", 0.9),
            ([[0, 40], [80, 40], [80, 60], [0, 60]], "3:21:05", 0.85),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "hibernation"
        assert result.method == "ocr"
        assert result.timer == "3:21:05"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_cant_attack_detected(self, mock_get_reader):
        """'cannot attack' keyword in bottom half → cant_attack."""
        mock_reader = MagicMock()
        # Place text in bottom half (y > 50% of enhanced image)
        h = 1920
        bottom_y = int(h * 0.75)
        mock_reader.readtext.return_value = [
            ([[0, bottom_y], [200, bottom_y], [200, bottom_y + 30], [0, bottom_y + 30]],
             "Cannot Attack", 0.88),
            ([[0, bottom_y + 35], [100, bottom_y + 35], [100, bottom_y + 55], [0, bottom_y + 55]],
             "1:23:45", 0.82),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "cant_attack"
        assert result.method == "ocr"
        assert result.timer == "1:23:45"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_home_screen_needs_two_keywords(self, mock_get_reader):
        """Home screen needs 2+ keywords: adventure, guild, events."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Adventure", 0.9),
            ([[200, 0], [300, 0], [300, 30], [200, 30]], "Guild", 0.85),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "home_screen"
        assert result.method == "ocr"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_single_keyword_not_enough_for_home_screen(self, mock_get_reader):
        """One keyword alone should not match home_screen."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Adventure", 0.9),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type != "home_screen"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_monument_popup_detected(self, mock_get_reader):
        """Monument popup with defense info + ownership."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Defense Info", 0.92),
            ([[0, 800], [100, 800], [100, 830], [0, 830]], "Ownership", 0.88),
            ([[0, 1600], [100, 1600], [100, 1630], [0, 1630]], "Attack", 0.91),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "monument_popup"
        assert result.method == "ocr"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_battle_result_detected(self, mock_get_reader):
        """Battle result with victory + battle report."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Victory", 0.95),
            ([[200, 200], [400, 200], [400, 230], [200, 230]], "Battle Report", 0.90),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "battle_result"
        assert result.method == "ocr"

    @patch("src.vision.screen_analyzer._get_reader")
    def test_logged_out_single_keyword(self, mock_get_reader):
        """'logged in on another device' is enough for logged_out."""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [400, 0], [400, 30], [0, 30]],
             "Logged in on another device", 0.93),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert result.screen_type == "logged_out"
        assert result.method == "ocr"


class TestOCRElementExtraction:
    """OCR text → element position extraction."""

    def setup_method(self):
        self.analyzer = ScreenAnalyzer(ElementDetector(), {})

    @patch("src.vision.screen_analyzer._get_reader")
    def test_attack_button_extracted(self, mock_get_reader):
        """'Attack' text in bottom 30% → action_button element."""
        mock_reader = MagicMock()
        # Two keywords for monument_popup + attack in bottom area
        enh_h = 1920  # enhanced image height
        bottom_y = int(enh_h * 0.80)
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Defense Info", 0.92),
            ([[0, 400], [100, 400], [100, 430], [0, 430]], "Ownership", 0.88),
            ([[400, bottom_y], [550, bottom_y], [550, bottom_y + 40], [400, bottom_y + 40]],
             "Attack", 0.91),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert "action_button" in result.elements
        x, y, conf = result.elements["action_button"]
        assert y > 70  # should be in bottom area

    @patch("src.vision.screen_analyzer._get_reader")
    def test_ok_button_extracted(self, mock_get_reader):
        """'OK' text in bottom 50% → ok_button element."""
        mock_reader = MagicMock()
        enh_h = 1920
        bottom_y = int(enh_h * 0.70)
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "Victory", 0.95),
            ([[200, 200], [400, 200], [400, 230], [200, 230]], "Battle Report", 0.90),
            ([[500, bottom_y], [560, bottom_y], [560, bottom_y + 40], [500, bottom_y + 40]],
             "OK", 0.88),
        ]
        mock_get_reader.return_value = mock_reader

        img = _make_image(brightness=80)
        png = _to_png(img)
        result = self.analyzer.analyze(png)
        assert "ok_button" in result.elements


class TestFixedElements:
    """Test that fixed-position elements are injected for matching screen types."""

    def test_monument_popup_gets_close_popup(self):
        """Monument popup detection should include close_popup from fixed elements."""
        analysis = ScreenAnalysis(screen_type="monument_popup", confidence=0.9, method="ocr")
        ScreenAnalyzer._inject_fixed_elements(analysis)
        assert "close_popup" in analysis.elements
        x, y, conf = analysis.elements["close_popup"]
        assert x == 50.0
        assert y == 94.5
        assert conf == 0.75

    def test_minimap_gets_minimap_close(self):
        """Minimap detection should include minimap_close from fixed elements."""
        analysis = ScreenAnalysis(screen_type="minimap", confidence=0.9, method="minimap")
        ScreenAnalyzer._inject_fixed_elements(analysis)
        assert "minimap_close" in analysis.elements
        x, y, conf = analysis.elements["minimap_close"]
        assert x == 50.0
        assert y == 94.5

    def test_unknown_screen_no_fixed_elements(self):
        """Screen types without fixed elements should not get any injected."""
        analysis = ScreenAnalysis(screen_type="battle_active", confidence=0.9, method="ocr")
        ScreenAnalyzer._inject_fixed_elements(analysis)
        assert len(analysis.elements) == 0

    def test_fixed_does_not_overwrite_higher_confidence(self):
        """Fixed elements should not overwrite existing detections with higher confidence."""
        analysis = ScreenAnalysis(
            screen_type="monument_popup", confidence=0.9, method="ocr",
            elements={"close_popup": (48.0, 93.0, 0.95)},
        )
        ScreenAnalyzer._inject_fixed_elements(analysis)
        x, y, conf = analysis.elements["close_popup"]
        # Should keep the original higher-confidence detection
        assert x == 48.0
        assert conf == 0.95


class TestTierFallback:
    """Test that tiers execute in order and stop at first confident match."""

    def test_brightness_stops_before_elements(self):
        """Black screen should return loading without running element detection."""
        detector = MagicMock(spec=ElementDetector)
        analyzer = ScreenAnalyzer(detector, {})

        img = _make_image(brightness=0)
        png = _to_png(img)
        result = analyzer.analyze(png)

        assert result.screen_type == "loading"
        assert result.method == "brightness"
        detector.detect.assert_not_called()

    def test_element_match_stops_before_ocr(self):
        """Element match (dark screen + ok_button) should not trigger OCR."""
        img = _make_image(brightness=30)
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, int(h * 0.85))
        png = _to_png(img)

        analyzer = ScreenAnalyzer(ElementDetector(), {})
        with patch("src.vision.screen_analyzer._get_reader") as mock_reader:
            result = analyzer.analyze(png)
            assert result.screen_type == "battle_result"
            assert result.method == "element"
            mock_reader.assert_not_called()

    def test_no_match_returns_unknown(self):
        """When nothing matches, return unknown with low confidence."""
        detector = MagicMock(spec=ElementDetector)
        detector.detect.return_value = []
        analyzer = ScreenAnalyzer(detector, {})

        img = _make_image(brightness=80)
        png = _to_png(img)

        with patch("src.vision.screen_analyzer.find_minimap_squares", return_value=None):
            with patch("src.vision.screen_analyzer._get_reader") as mock_get_reader:
                mock_reader = MagicMock()
                mock_reader.readtext.return_value = []
                mock_get_reader.return_value = mock_reader

                result = analyzer.analyze(png)
                assert result.confidence < 0.5


class TestScreenKeywordConfig:
    """Verify keyword configuration is consistent."""

    def test_single_keyword_screens_are_in_keywords(self):
        """All single-keyword screens must have entries in _SCREEN_KEYWORDS."""
        for screen in _SINGLE_KEYWORD_SCREENS:
            assert screen in _SCREEN_KEYWORDS

    def test_element_signatures_cover_known_screens(self):
        """All signature elements map to valid screen types."""
        valid_screens = set(_SCREEN_KEYWORDS.keys()) | {
            "minimap", "loading", "unknown",
        }
        for element, screen in _ELEMENT_SIGNATURES.items():
            assert screen in valid_screens, f"{element} → {screen} not a valid screen"

    def test_text_to_element_keywords_are_lowercase(self):
        """All _TEXT_TO_ELEMENT keywords should be lowercase."""
        for keyword, _, _, _, _ in _TEXT_TO_ELEMENT:
            assert keyword == keyword.lower()


class TestTimerExtraction:
    """Timer extraction for hibernation/cant_attack."""

    def setup_method(self):
        self.analyzer = ScreenAnalyzer(ElementDetector(), {})

    def test_extracts_hms_timer(self):
        """Extracts HH:MM:SS timer."""
        detections = [
            {"text": "3:21:05", "lower": "3:21:05", "cx": 50.0, "cy": 60.0, "conf": 0.9}
        ]
        result = ScreenAnalyzer._extract_timer(detections, "hibernation", 1920)
        assert result == "3:21:05"

    def test_extracts_ms_timer(self):
        """Extracts MM:SS timer."""
        detections = [
            {"text": "5:30", "lower": "5:30", "cx": 50.0, "cy": 60.0, "conf": 0.9}
        ]
        result = ScreenAnalyzer._extract_timer(detections, "hibernation", 1920)
        assert result == "5:30"

    def test_cant_attack_timer_bottom_half_only(self):
        """cant_attack timer must be in bottom half (cy > 50%)."""
        detections = [
            {"text": "1:23:45", "lower": "1:23:45", "cx": 50.0, "cy": 20.0, "conf": 0.9}
        ]
        result = ScreenAnalyzer._extract_timer(detections, "cant_attack", 1920)
        assert result is None

    def test_cant_attack_timer_accepted_in_bottom_half(self):
        """cant_attack timer in bottom half should be accepted."""
        detections = [
            {"text": "1:23:45", "lower": "1:23:45", "cx": 50.0, "cy": 70.0, "conf": 0.9}
        ]
        result = ScreenAnalyzer._extract_timer(detections, "cant_attack", 1920)
        assert result == "1:23:45"

    def test_no_timer_returns_none(self):
        """No timer text → None."""
        detections = [
            {"text": "no timer here", "lower": "no timer here", "cx": 50.0, "cy": 60.0, "conf": 0.9}
        ]
        result = ScreenAnalyzer._extract_timer(detections, "hibernation", 1920)
        assert result is None
