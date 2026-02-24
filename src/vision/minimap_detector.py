"""Detect minimap squares using pixel color analysis instead of Vision API.

Uses OpenCV HSV thresholding to find the 4 colored squares on the minimap
overlay. Returns their center positions and colors (red/blue) directly —
no API call needed.
"""

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MinimapSquare:
    slot: int           # 1-4 in reading order
    center_x: int       # pixel x in screenshot space
    center_y: int       # pixel y in screenshot space
    color: str          # "red" or "blue"
    area: int           # contour area in pixels


@dataclass
class MinimapDetection:
    squares: list[MinimapSquare]
    image_width: int
    image_height: int

    @property
    def slot_colors(self) -> dict[int, str]:
        """Return {slot: color} dict matching the parser.MinimapColors interface."""
        return {sq.slot: sq.color for sq in self.squares}

    def get_square(self, slot: int) -> MinimapSquare | None:
        for sq in self.squares:
            if sq.slot == slot:
                return sq
        return None


def find_minimap_squares(png_bytes: bytes) -> MinimapDetection | None:
    """Detect the 4 minimap squares using color thresholding.

    Args:
        png_bytes: Raw PNG screenshot bytes from ADB screencap.

    Returns:
        MinimapDetection with square centers and colors, or None if
        the minimap grid couldn't be detected (< 2 squares found).
    """
    # Decode PNG to OpenCV image (BGR)
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not decode image for minimap detection")
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Red mask (for contour detection) ─────────────────────────
    # Red hue wraps around 0/180 in OpenCV HSV (0-180 range).
    # Sampled red squares: H≈175, S≈96, V≈131.
    red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    # ── Wider red mask (for color classification only) ─────────
    # More permissive — used only for counting red-vs-blue
    # inside each contour's bounding box after detection.
    red1_wide = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([20, 255, 255]))
    red2_wide = cv2.inRange(hsv, np.array([155, 25, 25]), np.array([180, 255, 255]))
    red_mask_wide = red1_wide | red2_wide

    # ── Blue mask ───────────────────────────────────────────────
    # Sampled blue squares: H≈126, S≈54, V≈121.
    # Background/border: H≈149, S≈87, V≈100.
    # Upper hue capped at 130 to exclude background (H≈149).
    blue_mask = cv2.inRange(hsv, np.array([90, 25, 40]), np.array([130, 255, 255]))

    # ── Combined + morphological cleanup ────────────────────────
    combined = red_mask | blue_mask

    # Remove small noise (dots, icons) but do NOT close gaps between squares
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # ── Find contours ───────────────────────────────────────────
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Each square should be roughly 2-15% of image area
    min_area = int(w * h * 0.015)
    max_area = int(w * h * 0.15)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)

        # Squares should be roughly square-ish (aspect ratio 0.5-2.0)
        aspect = cw / ch if ch > 0 else 0
        if not (0.4 < aspect < 2.5):
            continue

        cx = x + cw // 2
        cy = y + ch // 2

        # Determine color by counting red vs blue mask pixels in the
        # entire bounding box.  Center-sampling is unreliable because the
        # monument icon (which has blue/purple gems) sits right at the
        # center and skews the hue reading.
        # Use the WIDE red mask here to catch the dark maroon background.
        region_red = red_mask_wide[y:y + ch, x:x + cw]
        region_blue = blue_mask[y:y + ch, x:x + cw]
        red_count = int(np.count_nonzero(region_red))
        blue_count = int(np.count_nonzero(region_blue))
        color = "red" if red_count > blue_count else "blue"
        logger.debug(
            f"  Contour ({cx},{cy}): red_px={red_count} blue_px={blue_count} → {color}"
        )

        candidates.append((cx, cy, color, area))

    if len(candidates) < 2:
        logger.info(f"Minimap detection: found only {len(candidates)} squares (need >= 2)")
        return None

    # Take the 4 largest regions
    candidates.sort(key=lambda c: c[3], reverse=True)
    squares = candidates[:4]

    logger.info(f"Minimap detection: found {len(squares)} squares")

    # ── Sort into grid positions ────────────────────────────────
    # Split by Y into top/bottom rows, then by X into left/right
    avg_y = sum(s[1] for s in squares) / len(squares)
    top_row = sorted([s for s in squares if s[1] <= avg_y], key=lambda s: s[0])
    bottom_row = sorted([s for s in squares if s[1] > avg_y], key=lambda s: s[0])

    # If all are in one row (no vertical split), try splitting differently
    if not top_row:
        top_row = bottom_row[:len(bottom_row) // 2]
        bottom_row = bottom_row[len(bottom_row) // 2:]
    elif not bottom_row:
        bottom_row = top_row[len(top_row) // 2:]
        top_row = top_row[:len(top_row) // 2]

    result_squares = []
    slot = 1
    for row in [top_row, bottom_row]:
        for cx, cy, color, area in row:
            result_squares.append(MinimapSquare(
                slot=slot, center_x=cx, center_y=cy, color=color, area=area,
            ))
            logger.info(
                f"  Slot {slot}: ({cx}, {cy}) = {color}  "
                f"({cx / w * 100:.1f}%, {cy / h * 100:.1f}%)"
            )
            slot += 1

    return MinimapDetection(
        squares=result_squares,
        image_width=w,
        image_height=h,
    )


def save_detection_debug(png_bytes: bytes, detection: MinimapDetection | None, path: str) -> None:
    """Save an annotated image showing detected squares for debugging."""
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return

    if detection:
        for sq in detection.squares:
            color_bgr = (0, 0, 255) if sq.color == "red" else (255, 150, 0)
            # Draw crosshair
            arm = 30
            cv2.line(img, (sq.center_x - arm, sq.center_y), (sq.center_x + arm, sq.center_y), color_bgr, 3)
            cv2.line(img, (sq.center_x, sq.center_y - arm), (sq.center_x, sq.center_y + arm), color_bgr, 3)
            cv2.circle(img, (sq.center_x, sq.center_y), 20, color_bgr, 2)
            # Label
            label = f"Slot {sq.slot} ({sq.color})"
            cv2.putText(img, label, (sq.center_x + 35, sq.center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
    else:
        cv2.putText(img, "NO SQUARES DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imwrite(path, img)
    logger.info(f"Saved minimap detection debug image: {path}")
