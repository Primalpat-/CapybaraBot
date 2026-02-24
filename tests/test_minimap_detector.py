"""Tests for pixel-based minimap square detection."""

import io
import numpy as np
import cv2
import pytest

from src.vision.minimap_detector import find_minimap_squares, MinimapDetection


def _make_minimap_image(
    width: int = 1080,
    height: int = 1920,
    colors: list[str] | None = None,
) -> bytes:
    """Create a synthetic minimap screenshot with colored squares.

    Args:
        width: Image width.
        height: Image height.
        colors: List of 4 colors for the squares (top-left, top-right,
                bottom-left, bottom-right). Each is "red" or "blue".
    """
    if colors is None:
        colors = ["red", "red", "blue", "blue"]

    # Dark background
    img = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)

    # Minimap grid area (centered, covering roughly 60% of width, 30% of height)
    grid_x = int(width * 0.15)
    grid_y = int(height * 0.20)
    grid_w = int(width * 0.70)
    grid_h = int(height * 0.30)

    gap = 10  # gap between squares
    sq_w = (grid_w - gap) // 2
    sq_h = (grid_h - gap) // 2

    positions = [
        (grid_x, grid_y),                          # top-left
        (grid_x + sq_w + gap, grid_y),              # top-right
        (grid_x, grid_y + sq_h + gap),              # bottom-left
        (grid_x + sq_w + gap, grid_y + sq_h + gap), # bottom-right
    ]

    for i, (sx, sy) in enumerate(positions):
        c = colors[i]
        if c == "red":
            bgr = (40, 40, 200)   # Red in BGR
        else:
            bgr = (180, 120, 60)  # Blue in BGR
        cv2.rectangle(img, (sx, sy), (sx + sq_w, sy + sq_h), bgr, -1)

    # Encode to PNG bytes
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class TestFindMinimapSquares:
    def test_finds_four_squares(self):
        png = _make_minimap_image(colors=["red", "red", "blue", "blue"])
        result = find_minimap_squares(png)
        assert result is not None
        assert len(result.squares) == 4

    def test_correct_colors(self):
        png = _make_minimap_image(colors=["red", "blue", "red", "blue"])
        result = find_minimap_squares(png)
        assert result is not None
        assert result.slot_colors[1] == "red"
        assert result.slot_colors[2] == "blue"
        assert result.slot_colors[3] == "red"
        assert result.slot_colors[4] == "blue"

    def test_all_red(self):
        png = _make_minimap_image(colors=["red", "red", "red", "red"])
        result = find_minimap_squares(png)
        assert result is not None
        assert all(c == "red" for c in result.slot_colors.values())

    def test_all_blue(self):
        png = _make_minimap_image(colors=["blue", "blue", "blue", "blue"])
        result = find_minimap_squares(png)
        assert result is not None
        assert all(c == "blue" for c in result.slot_colors.values())

    def test_slot_ordering(self):
        png = _make_minimap_image(colors=["red", "blue", "blue", "red"])
        result = find_minimap_squares(png)
        assert result is not None
        # Slot 1 should be top-left (lowest x, lowest y)
        s1 = result.get_square(1)
        s4 = result.get_square(4)
        assert s1 is not None
        assert s4 is not None
        assert s1.center_x < s4.center_x
        assert s1.center_y < s4.center_y

    def test_returns_none_for_blank_image(self):
        # Blank gray image — no squares
        img = np.full((1920, 1080, 3), (128, 128, 128), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        result = find_minimap_squares(buf.tobytes())
        assert result is None

    def test_image_dimensions_stored(self):
        png = _make_minimap_image(width=720, height=1280)
        result = find_minimap_squares(png)
        assert result is not None
        assert result.image_width == 720
        assert result.image_height == 1280

    def test_get_square(self):
        png = _make_minimap_image()
        result = find_minimap_squares(png)
        assert result is not None
        sq = result.get_square(1)
        assert sq is not None
        assert sq.slot == 1
        assert result.get_square(99) is None
