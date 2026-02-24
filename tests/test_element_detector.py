"""Tests for local UI element detection using OpenCV."""

import numpy as np
import cv2
import pytest

from src.vision.element_detector import ElementDetector, DetectedElement


def _make_image(width: int = 1080, height: int = 1920) -> np.ndarray:
    """Create a blank dark image."""
    return np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)


def _to_png(img: np.ndarray) -> bytes:
    """Encode a numpy image to PNG bytes."""
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _draw_yellow_button(img: np.ndarray, cx: int, cy: int, bw: int = 200, bh: int = 60) -> None:
    """Draw a yellow button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    # Yellow in BGR: B=0, G=220, R=240 → HSV H≈25, S≈255, V≈240
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 240), -1)


def _draw_green_button(img: np.ndarray, cx: int, cy: int, bw: int = 120, bh: int = 50) -> None:
    """Draw a green button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    # Green in BGR: B=0, G=200, R=0 → HSV H≈60, S≈255, V≈200
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), -1)


def _draw_pink_button(img: np.ndarray, cx: int, cy: int, bw: int = 180, bh: int = 60) -> None:
    """Draw a pink button (BGR) at the given center."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    # Pink/magenta in BGR: B=180, G=50, R=230 → HSV H≈168, S≈~190, V≈230
    cv2.rectangle(img, (x1, y1), (x2, y2), (180, 50, 230), -1)


def _draw_purple_square(img: np.ndarray, cx: int, cy: int, size: int = 40) -> None:
    """Draw a small purple square at the given center."""
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    # Purple in BGR: B=160, G=40, R=100 → HSV H≈135, S≈~190, V≈160
    cv2.rectangle(img, (x1, y1), (x2, y2), (160, 40, 100), -1)


class TestColorDetection:
    def setup_method(self):
        self.detector = ElementDetector()

    def test_finds_ok_button(self):
        """Yellow button at bottom-center of battle_result screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, int(h * 0.85))
        png = _to_png(img)

        results = self.detector.detect(png, "battle_result")
        assert len(results) == 1
        det = results[0]
        assert det.name == "ok_button"
        assert det.method == "color"
        assert det.confidence >= 0.7
        # Should be near center-x, bottom area
        assert 40 < det.x_percent < 60
        assert 75 < det.y_percent < 95

    def test_finds_skip_button(self):
        """Green button at bottom-right of battle_active screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_green_button(img, int(w * 0.75), int(h * 0.85))
        png = _to_png(img)

        results = self.detector.detect(png, "battle_active")
        assert len(results) == 1
        det = results[0]
        assert det.name == "skip_battle"
        assert det.method == "color"
        assert det.confidence >= 0.7
        assert 60 < det.x_percent < 90
        assert 75 < det.y_percent < 95

    def test_finds_cancel_button(self):
        """Pink button at left side of occupy_prompt screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_pink_button(img, int(w * 0.25), int(h * 0.55))
        png = _to_png(img)

        results = self.detector.detect(png, "occupy_prompt")
        assert len(results) == 1
        det = results[0]
        assert det.name == "occupy_cancel_button"
        assert det.method == "color"
        assert det.confidence >= 0.7
        assert 10 < det.x_percent < 40
        assert 40 < det.y_percent < 70

    def test_finds_action_button(self):
        """Yellow button at bottom of monument_popup screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, int(h * 0.70))
        png = _to_png(img)

        results = self.detector.detect(png, "monument_popup")
        # Should find action_button (may also find close_popup shape — check at least one)
        action_results = [r for r in results if r.name == "action_button"]
        assert len(action_results) == 1
        det = action_results[0]
        assert det.method == "color"
        assert 35 < det.x_percent < 65
        assert 60 < det.y_percent < 80

    def test_finds_restart_button(self):
        """Yellow button at center of logged_out screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_yellow_button(img, w // 2, h // 2)
        png = _to_png(img)

        results = self.detector.detect(png, "logged_out")
        assert len(results) == 1
        det = results[0]
        assert det.name == "restart_button"
        assert det.method == "color"

    def test_finds_minimap_button(self):
        """Small purple square at top-right of main_map screen."""
        img = _make_image()
        h, w = img.shape[:2]
        _draw_purple_square(img, int(w * 0.85), int(h * 0.12))
        png = _to_png(img)

        results = self.detector.detect(png, "main_map")
        assert len(results) == 1
        det = results[0]
        assert det.name == "minimap_button"
        assert det.method == "color"
        assert 70 < det.x_percent < 95
        assert 5 < det.y_percent < 25

    def test_different_resolutions(self):
        """Detection works at 720p, 1080p, and 1440p."""
        resolutions = [(720, 1280), (1080, 1920), (1440, 2560)]
        for w, h in resolutions:
            img = _make_image(w, h)
            _draw_yellow_button(img, w // 2, int(h * 0.85), bw=int(w * 0.2), bh=int(h * 0.04))
            png = _to_png(img)

            results = self.detector.detect(png, "battle_result")
            assert len(results) == 1, f"Failed at {w}x{h}"
            det = results[0]
            assert det.name == "ok_button"
            assert 40 < det.x_percent < 60

    def test_returns_empty_for_unknown_screen(self):
        """Unknown screen types return empty list."""
        img = _make_image()
        png = _to_png(img)
        results = self.detector.detect(png, "nonexistent_screen")
        assert results == []

    def test_returns_empty_for_invalid_image(self):
        """Invalid PNG data returns empty list."""
        results = self.detector.detect(b"not a png", "battle_result")
        assert results == []

    def test_returns_empty_for_no_matching_color(self):
        """Blank screen with no buttons returns empty list."""
        img = _make_image()
        png = _to_png(img)
        results = self.detector.detect(png, "battle_result")
        assert results == []


class TestTemplateMatching:
    def setup_method(self):
        self.detector = ElementDetector()

    def test_no_template_returns_empty(self):
        """Without a saved template, template elements return nothing."""
        img = _make_image()
        png = _to_png(img)
        results = self.detector.detect(png, "home_screen")
        assert results == []

    def test_has_template_false_initially(self):
        """No templates exist on fresh detector."""
        assert not self.detector.has_template("star_trek_button")
        assert not self.detector.has_template("alien_minefield_button")


class TestSaveTemplate:
    def setup_method(self):
        self.detector = ElementDetector()

    def test_save_and_check(self, tmp_path, monkeypatch):
        """save_template creates a file that has_template can find."""
        import src.vision.element_detector as mod
        monkeypatch.setattr(mod, "_TEMPLATE_DIR", tmp_path)

        img = _make_image(400, 400)
        _draw_yellow_button(img, 200, 200, 80, 40)
        png = _to_png(img)

        ok = self.detector.save_template(png, "test_button", 50.0, 50.0)
        assert ok
        assert (tmp_path / "test_button.png").exists()

    def test_save_template_bad_image(self):
        """Bad image data returns False."""
        ok = self.detector.save_template(b"bad", "test", 50.0, 50.0)
        assert not ok
