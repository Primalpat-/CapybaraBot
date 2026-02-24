"""Detect minimap squares using pixel color analysis instead of Vision API.

Uses a two-pass approach:
1. **Contour-based** (original): HSV thresholding -> contours -> grid sorting.
2. **Frame-based** (fallback): Detects the dark purple grid frame, finds the
   4 square regions as "holes" inside the filled frame contour, then samples
   center pixel colors to classify red vs blue.

The frame-based approach is more robust across devices because it finds the
grid structure from the frame shape (not the square colors), then adaptively
reads whatever colors are actually in each quadrant.
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
    falls back to the frame-based approach.

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

    # Fall back to frame-based detection
    logger.info("Contour detection failed -- trying frame-based approach")
    return _detect_frame_based(hsv, w, h)


def _detect_contour_based(hsv: np.ndarray, w: int, h: int) -> MinimapDetection | None:
    """Original contour-based detection with fixed HSV ranges."""
    red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 40, 40]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    red1_wide = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([20, 255, 255]))
    red2_wide = cv2.inRange(hsv, np.array([155, 25, 25]), np.array([180, 255, 255]))
    red_mask_wide = red1_wide | red2_wide

    blue_mask = cv2.inRange(hsv, np.array([90, 25, 40]), np.array([130, 255, 255]))

    combined = red_mask | blue_mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

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
            f"  Contour ({cx},{cy}): red_px={red_count} blue_px={blue_count} -> {color}"
        )

        candidates.append((cx, cy, color, area))

    if len(candidates) < 2:
        logger.info(f"Contour detection: found only {len(candidates)} squares (need >= 2)")
        return None

    candidates.sort(key=lambda c: c[3], reverse=True)
    squares = candidates[:4]

    logger.info(f"Contour detection: found {len(squares)} squares")
    return _sort_into_grid(squares, w, h)


def _detect_frame_based(hsv: np.ndarray, w: int, h: int) -> MinimapDetection | None:
    """Frame-based detection -- find the dark purple grid frame, then locate
    the 4 square regions as "holes" inside the filled frame contour.

    The grid frame and dividers have a distinctive dark purple color
    (H~165-170, S~85-105, V~70-80). By finding the largest frame contour
    and inverting the mask inside it, the 4 square content regions appear
    as separate blobs that can be independently located and color-sampled.
    """
    # -- Mask the dark purple frame & dividers -------------------
    frame_mask = cv2.inRange(hsv, np.array([155, 40, 30]), np.array([178, 160, 110]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel)
    frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel)

    # -- Find the largest contour = the grid frame ---------------
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.info("Frame detection: no frame contours found")
        return None

    largest = max(contours, key=cv2.contourArea)
    frame_area = cv2.contourArea(largest)
    image_area = w * h

    if frame_area < image_area * 0.02:
        logger.info(f"Frame detection: largest contour too small ({frame_area:.0f})")
        return None

    fx, fy, fw, fh = cv2.boundingRect(largest)
    logger.debug(f"Frame detection: frame contour at ({fx},{fy}) {fw}x{fh}")

    # -- Invert: find square content as holes inside the frame ---
    # Fill the frame contour solid, then subtract frame-colored pixels.
    # The remaining regions = the square content areas.
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 255, -1)

    content_mask = filled & (~frame_mask)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, k2)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, k2)

    # -- Find the 4 square regions ------------------------------
    sq_contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_sq_area = frame_area * 0.03
    max_sq_area = frame_area * 0.40

    candidates = []
    for c in sq_contours:
        area = cv2.contourArea(c)
        if area < min_sq_area or area > max_sq_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh > 0 else 0
        if not (0.4 < aspect < 2.5):
            continue

        cx = bx + bw // 2
        cy = by + bh // 2

        # -- Adaptive color sampling at center -------------------
        sr = max(5, min(bw, bh) // 4)
        sx1 = max(0, cx - sr)
        sy1 = max(0, cy - sr)
        sx2 = min(w, cx + sr)
        sy2 = min(h, cy + sr)

        sample = hsv[sy1:sy2, sx1:sx2]
        if sample.size == 0:
            continue

        sat_mask = sample[:, :, 1] > 15
        if not np.any(sat_mask):
            median_v = float(np.median(sample[:, :, 2]))
            color = "blue" if median_v > 100 else "red"
        else:
            median_h = float(np.median(sample[:, :, 0][sat_mask]))
            if median_h < 25 or median_h > 150:
                color = "red"
            elif 80 < median_h < 145:
                color = "blue"
            else:
                median_s = float(np.median(sample[:, :, 1][sat_mask]))
                color = "red" if median_s > 100 else "blue"

        logger.debug(f"  Frame content ({cx},{cy}): area={area:.0f} -> {color}")
        candidates.append((cx, cy, color, area))

    if len(candidates) < 2:
        logger.info(f"Frame detection: found only {len(candidates)} content regions")
        return None

    # Take the 4 largest
    candidates.sort(key=lambda c: c[3], reverse=True)
    squares = candidates[:4]

    logger.info(f"Frame detection: found {len(squares)} squares")
    return _sort_into_grid(squares, w, h)


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
