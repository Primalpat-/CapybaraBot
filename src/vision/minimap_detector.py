"""Detect minimap squares using pixel color analysis instead of Vision API.

Uses a two-pass approach:
1. **Contour-based** (original): HSV thresholding → contours → grid sorting.
2. **Adaptive grid-based** (fallback): Very wide HSV masks to find the minimap
   region, then split into a 2x2 grid and sample center colors directly.

The grid-based approach is more robust across devices because it doesn't
depend on tight HSV thresholds — it assumes a 2x2 layout and reads colors.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

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

    Tries the contour-based approach first. If it finds fewer than 2 squares,
    falls back to the adaptive grid-based approach.

    Args:
        png_bytes: Raw PNG screenshot bytes from ADB screencap.

    Returns:
        MinimapDetection with square centers and colors, or None if
        the minimap grid couldn't be detected.
    """
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not decode image for minimap detection")
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Try contour-based detection first (works well when thresholds match)
    result = _detect_contour_based(hsv, w, h)
    if result is not None:
        return result

    # Fall back to adaptive grid-based detection
    logger.info("Contour detection failed — trying adaptive grid-based approach")
    return _detect_grid_based(hsv, w, h)


def _detect_contour_based(hsv: np.ndarray, w: int, h: int) -> MinimapDetection | None:
    """Original contour-based detection with fixed HSV ranges."""
    # ── Red mask (for contour detection) ─────────────────────────
    red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    # ── Wider red mask (for color classification only) ─────────
    red1_wide = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([20, 255, 255]))
    red2_wide = cv2.inRange(hsv, np.array([155, 25, 25]), np.array([180, 255, 255]))
    red_mask_wide = red1_wide | red2_wide

    # ── Blue mask ───────────────────────────────────────────────
    blue_mask = cv2.inRange(hsv, np.array([90, 25, 40]), np.array([130, 255, 255]))

    # ── Combined + morphological cleanup ────────────────────────
    combined = red_mask | blue_mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # ── Find contours ───────────────────────────────────────────
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = int(w * h * 0.015)
    max_area = int(w * h * 0.15)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / ch if ch > 0 else 0
        if not (0.4 < aspect < 2.5):
            continue

        cx = x + cw // 2
        cy = y + ch // 2

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
        logger.info(f"Contour detection: found only {len(candidates)} squares (need >= 2)")
        return None

    candidates.sort(key=lambda c: c[3], reverse=True)
    squares = candidates[:4]

    logger.info(f"Contour detection: found {len(squares)} squares")
    return _sort_into_grid(squares, w, h)


def _detect_grid_based(hsv: np.ndarray, w: int, h: int) -> MinimapDetection | None:
    """Adaptive grid-based detection — sample colors from 2x2 quadrants.

    Uses very wide HSV masks to find the minimap region (area with highest
    density of red+blue pixels), splits it into a 2x2 grid, and classifies
    each quadrant by sampling center pixel colors.
    """
    # ── Very wide masks to catch any red/blue across devices ────
    red1_wide = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([25, 255, 255]))
    red2_wide = cv2.inRange(hsv, np.array([150, 20, 20]), np.array([180, 255, 255]))
    red_wide = red1_wide | red2_wide

    blue_wide = cv2.inRange(hsv, np.array([80, 20, 20]), np.array([140, 255, 255]))

    combined = red_wide | blue_wide

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # ── Find the bounding region of all colored pixels ──────────
    points = cv2.findNonZero(combined)
    if points is None or len(points) < 100:
        logger.info("Adaptive grid: not enough colored pixels found")
        return None

    rx, ry, rw, rh = cv2.boundingRect(points)

    # Sanity check: the region should be roughly square-ish and
    # cover a reasonable portion of the image
    region_area = rw * rh
    image_area = w * h
    if region_area < image_area * 0.02 or region_area > image_area * 0.8:
        logger.info(f"Adaptive grid: region area out of range ({region_area}/{image_area})")
        return None

    aspect = rw / rh if rh > 0 else 0
    if not (0.3 < aspect < 3.0):
        logger.info(f"Adaptive grid: bad aspect ratio ({aspect:.2f})")
        return None

    logger.debug(f"Adaptive grid: minimap region at ({rx},{ry}) size {rw}x{rh}")

    # ── Split into 2x2 grid and sample each quadrant ────────────
    mid_x = rx + rw // 2
    mid_y = ry + rh // 2

    # Quadrants: (x_center, y_center) for sampling
    quadrants = [
        (rx + rw // 4, ry + rh // 4),         # top-left (slot 1)
        (rx + 3 * rw // 4, ry + rh // 4),     # top-right (slot 2)
        (rx + rw // 4, ry + 3 * rh // 4),     # bottom-left (slot 3)
        (rx + 3 * rw // 4, ry + 3 * rh // 4), # bottom-right (slot 4)
    ]

    # Sample a small region around each quadrant center
    sample_radius_x = max(5, rw // 8)
    sample_radius_y = max(5, rh // 8)

    squares = []
    for slot_idx, (qx, qy) in enumerate(quadrants, 1):
        sx1 = max(0, qx - sample_radius_x)
        sy1 = max(0, qy - sample_radius_y)
        sx2 = min(w, qx + sample_radius_x)
        sy2 = min(h, qy + sample_radius_y)

        sample = hsv[sy1:sy2, sx1:sx2]
        if sample.size == 0:
            continue

        # Get median hue from pixels that have some saturation (skip gray)
        sat_mask = sample[:, :, 1] > 15  # minimal saturation filter
        if not np.any(sat_mask):
            continue

        median_h = float(np.median(sample[:, :, 0][sat_mask]))

        # Classify: red hue wraps around 0/180
        if median_h < 25 or median_h > 150:
            color = "red"
        elif 80 < median_h < 140:
            color = "blue"
        else:
            # Ambiguous — skip this quadrant
            logger.debug(f"  Quadrant {slot_idx}: ambiguous hue {median_h:.0f}")
            continue

        # Estimate area from quadrant size
        quad_area = (rw // 2) * (rh // 2)

        squares.append((qx, qy, color, quad_area))
        logger.debug(f"  Quadrant {slot_idx}: center=({qx},{qy}) hue={median_h:.0f} → {color}")

    if len(squares) < 2:
        logger.info(f"Adaptive grid: classified only {len(squares)} quadrants (need >= 2)")
        return None

    logger.info(f"Adaptive grid detection: found {len(squares)} squares")

    # Build result directly — quadrants are already in slot order
    result_squares = []
    for slot_idx, (cx, cy, color, area) in enumerate(squares, 1):
        result_squares.append(MinimapSquare(
            slot=slot_idx, center_x=cx, center_y=cy, color=color, area=area,
        ))
        logger.info(
            f"  Slot {slot_idx}: ({cx}, {cy}) = {color}  "
            f"({cx / w * 100:.1f}%, {cy / h * 100:.1f}%)"
        )

    return MinimapDetection(squares=result_squares, image_width=w, image_height=h)


def _sort_into_grid(
    candidates: list[tuple[int, int, str, int]], w: int, h: int,
) -> MinimapDetection:
    """Sort candidate squares into a 2x2 grid by position."""
    avg_y = sum(s[1] for s in candidates) / len(candidates)
    top_row = sorted([s for s in candidates if s[1] <= avg_y], key=lambda s: s[0])
    bottom_row = sorted([s for s in candidates if s[1] > avg_y], key=lambda s: s[0])

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

    return MinimapDetection(squares=result_squares, image_width=w, image_height=h)


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
