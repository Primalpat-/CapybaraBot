"""Detect UI elements locally using OpenCV — color, shape, and template matching.

Replaces Vision API calibration for most elements. Each element uses a
constrained screen region and detection strategy (color thresholding,
shape analysis, or template matching) to find button centers.

Returns positions as percentages (0-100) matching the calibration system.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "config" / "templates"
_DEBUG_DIR = Path(__file__).resolve().parents[2] / "screenshots" / "detection"


@dataclass
class DetectedElement:
    name: str
    x_percent: float    # 0-100
    y_percent: float    # 0-100
    confidence: float   # 0-1
    method: str         # "color", "shape", "template"


# ── Region helpers ────────────────────────────────────────────────
# Regions are (x_start, y_start, x_end, y_end) as fractions of image size (0-1).

_BOTTOM_CENTER = (0.15, 0.65, 0.85, 1.0)
_BOTTOM_CENTER_LOW = (0.15, 0.82, 0.85, 1.0)   # OK button only (below popup buttons)
_BOTTOM_RIGHT = (0.50, 0.70, 1.0, 1.0)
_CENTER = (0.10, 0.25, 0.90, 0.75)
_LEFT_HALF = (0.0, 0.30, 0.55, 0.80)
_TOP_RIGHT = (0.50, 0.0, 1.0, 0.25)
_RIGHT_HALF = (0.40, 0.0, 1.0, 0.65)
_BOTTOM_POPUP = (0.15, 0.50, 0.85, 0.85)
_BOTTOM_CENTER_STRIP = (0.20, 0.60, 0.80, 0.95)

# ── HSV ranges (OpenCV H: 0-180, S: 0-255, V: 0-255) ────────────

_YELLOW_LOWER = np.array([15, 80, 180])
_YELLOW_UPPER = np.array([30, 255, 255])

_GREEN_LOWER = np.array([40, 80, 120])
_GREEN_UPPER = np.array([80, 255, 255])

# The skip button can appear as a dark teal/blue depending on the game's
# rendering state (animations, overlays).  Measured from real screenshots:
# H≈120-135, S≈70-140, V≈80-160.
_DARK_TEAL_LOWER = np.array([118, 65, 75])
_DARK_TEAL_UPPER = np.array([138, 150, 165])

_PINK_LOWER = np.array([155, 60, 150])
_PINK_UPPER = np.array([175, 255, 255])

_PURPLE_LOWER = np.array([120, 30, 40])
_PURPLE_UPPER = np.array([155, 255, 255])

# ── Dispatch table ────────────────────────────────────────────────
# Maps (screen_type, element_name) → detection config.

_COLOR_ELEMENTS: dict[tuple[str, str], dict] = {
    ("battle_result", "ok_button"): {
        "region": _BOTTOM_CENTER_LOW,
        "hsv_lower": _YELLOW_LOWER,
        "hsv_upper": _YELLOW_UPPER,
        "min_area_frac": 0.005,
    },
    ("logged_out", "restart_button"): {
        "region": _CENTER,
        "hsv_lower": _YELLOW_LOWER,
        "hsv_upper": _YELLOW_UPPER,
        "min_area_frac": 0.005,
    },
    ("battle_active", "skip_battle"): {
        "region": _BOTTOM_RIGHT,
        "hsv_ranges": [
            (_GREEN_LOWER, _GREEN_UPPER),          # bright green (normal)
            (_DARK_TEAL_LOWER, _DARK_TEAL_UPPER),  # dark teal (dimmed/overlay)
        ],
        "min_area_frac": 0.001,
    },
    ("monument_popup", "action_button"): {
        "region": _BOTTOM_POPUP,
        "hsv_lower": _YELLOW_LOWER,
        "hsv_upper": _YELLOW_UPPER,
        "min_area_frac": 0.003,
    },
    ("monument_popup", "close_popup"): {
        "strategy": "shape",
        "region": _BOTTOM_CENTER_STRIP,
    },
    ("minimap", "minimap_close"): {
        "strategy": "shape",
        "region": _BOTTOM_CENTER_STRIP,
    },
    ("occupy_prompt", "occupy_cancel_button"): {
        "region": _LEFT_HALF,
        "hsv_lower": _PINK_LOWER,
        "hsv_upper": _PINK_UPPER,
        "min_area_frac": 0.003,
    },
    ("main_map", "minimap_button"): {
        "region": _TOP_RIGHT,
        "hsv_lower": _PURPLE_LOWER,
        "hsv_upper": _PURPLE_UPPER,
        "min_area_frac": 0.0003,
    },
}

_TEMPLATE_ELEMENTS: dict[tuple[str, str], dict] = {
    ("home_screen", "star_trek_button"): {
        "region": _RIGHT_HALF,
    },
    ("mode_select", "alien_minefield_button"): {
        "region": _CENTER,
    },
}

# Build a quick lookup: screen_type → list of element names we can detect
_SCREEN_DETECTABLE: dict[str, list[str]] = {}
for (screen, name) in list(_COLOR_ELEMENTS.keys()) + list(_TEMPLATE_ELEMENTS.keys()):
    _SCREEN_DETECTABLE.setdefault(screen, []).append(name)


class ElementDetector:
    """Detect UI elements locally using OpenCV."""

    def __init__(self) -> None:
        _TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    def detect(self, png_bytes: bytes, screen_type: str) -> list[DetectedElement]:
        """Detect all known elements for a given screen type.

        Args:
            png_bytes: Raw PNG screenshot bytes.
            screen_type: The identified screen type (e.g. "battle_result").

        Returns:
            List of detected elements with positions as percentages.
        """
        if screen_type not in _SCREEN_DETECTABLE:
            return []

        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("ElementDetector: could not decode image")
            return []

        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = None  # lazy — only compute if needed

        results: list[DetectedElement] = []

        for element_name in _SCREEN_DETECTABLE[screen_type]:
            key = (screen_type, element_name)

            if key in _COLOR_ELEMENTS:
                cfg = _COLOR_ELEMENTS[key]
                strategy = cfg.get("strategy", "color")

                if strategy == "shape":
                    if gray is None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    det = self._detect_circle(gray, w, h, element_name, cfg["region"])
                else:
                    det = self._detect_color(
                        hsv, w, h, element_name,
                        cfg["region"],
                        cfg.get("hsv_lower"), cfg.get("hsv_upper"),
                        cfg.get("min_area_frac", 0.001),
                        hsv_ranges=cfg.get("hsv_ranges"),
                    )
                if det:
                    results.append(det)

            elif key in _TEMPLATE_ELEMENTS:
                cfg = _TEMPLATE_ELEMENTS[key]
                det = self._detect_template(img, w, h, element_name, cfg["region"])
                if det:
                    results.append(det)

        if results:
            self._save_debug_image(img, results, screen_type)

        return results

    def save_template(self, png_bytes: bytes, name: str, x_pct: float, y_pct: float) -> bool:
        """Capture a template from a screenshot at the given percentage position.

        Crops a region around the position and saves it for future template matching.

        Args:
            png_bytes: Raw PNG screenshot bytes.
            name: Element name (used as filename).
            x_pct: X position as percentage (0-100).
            y_pct: Y position as percentage (0-100).

        Returns:
            True if template was saved successfully.
        """
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]
        cx = int(x_pct / 100 * w)
        cy = int(y_pct / 100 * h)

        # Crop a region around the element — 8% of screen width/height
        crop_w = int(w * 0.08)
        crop_h = int(h * 0.08)
        x1 = max(0, cx - crop_w)
        y1 = max(0, cy - crop_h)
        x2 = min(w, cx + crop_w)
        y2 = min(h, cy + crop_h)

        template = img[y1:y2, x1:x2]
        if template.size == 0:
            return False

        path = _TEMPLATE_DIR / f"{name}.png"
        cv2.imwrite(str(path), template)
        logger.info(f"Saved template for {name}: {path} ({x2 - x1}x{y2 - y1}px)")
        return True

    def has_template(self, name: str) -> bool:
        """Check if a saved template exists for the given element name."""
        return (_TEMPLATE_DIR / f"{name}.png").exists()

    # ── Detection methods ─────────────────────────────────────────

    @staticmethod
    def _detect_color(
        hsv: np.ndarray, w: int, h: int,
        name: str, region: tuple,
        hsv_lower: np.ndarray | None, hsv_upper: np.ndarray | None,
        min_area_frac: float,
        hsv_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> DetectedElement | None:
        """Find largest color blob in a constrained region."""
        rx1 = int(region[0] * w)
        ry1 = int(region[1] * h)
        rx2 = int(region[2] * w)
        ry2 = int(region[3] * h)

        roi = hsv[ry1:ry2, rx1:rx2]

        # Smooth to reduce noise from text, gradients, and anti-aliasing
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # Build mask — OR-combine multiple HSV ranges if provided
        ranges = hsv_ranges if hsv_ranges else [(hsv_lower, hsv_upper)]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(roi, lo, hi)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug(f"  {name}: no contours found in region")
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = w * h * min_area_frac
        if area < min_area:
            logger.debug(f"  {name}: largest contour too small ({area:.0f} < {min_area:.0f})")
            return None

        # Get center
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + rx1
        cy = int(M["m01"] / M["m00"]) + ry1

        # Confidence: the button was found and passes min_area — base 0.8,
        # boosted toward 1.0 as area grows relative to the minimum threshold.
        confidence = min(1.0, 0.8 + 0.2 * (area / (min_area * 10))) if min_area > 0 else 0.8

        x_pct = cx / w * 100
        y_pct = cy / h * 100
        logger.info(f"  Local detection: {name} at ({x_pct:.1f}%, {y_pct:.1f}%) [color] conf={confidence:.2f}")
        return DetectedElement(name=name, x_percent=x_pct, y_percent=y_pct, confidence=confidence, method="color")

    @staticmethod
    def _detect_circle(
        gray: np.ndarray, w: int, h: int,
        name: str, region: tuple,
    ) -> DetectedElement | None:
        """Find a circular dark shape (close/X button) in a constrained region."""
        rx1 = int(region[0] * w)
        ry1 = int(region[1] * h)
        rx2 = int(region[2] * w)
        ry2 = int(region[3] * h)

        roi = gray[ry1:ry2, rx1:rx2]

        # The X button is a dark circle on a lighter background
        _, thresh = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the most circular contour with reasonable size
        best = None
        best_circ = 0.0

        min_area = w * h * 0.0005
        max_area = w * h * 0.01

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6 and circularity > best_circ:
                best_circ = circularity
                best = c

        if best is None:
            logger.debug(f"  {name}: no circular contour found")
            return None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + rx1
        cy = int(M["m01"] / M["m00"]) + ry1

        x_pct = cx / w * 100
        y_pct = cy / h * 100
        confidence = min(1.0, best_circ)
        logger.info(f"  Local detection: {name} at ({x_pct:.1f}%, {y_pct:.1f}%) [shape] circ={best_circ:.2f}")
        return DetectedElement(name=name, x_percent=x_pct, y_percent=y_pct, confidence=confidence, method="shape")

    def _detect_template(
        self, img: np.ndarray, w: int, h: int,
        name: str, region: tuple,
    ) -> DetectedElement | None:
        """Multi-scale template matching within a constrained region."""
        template_path = _TEMPLATE_DIR / f"{name}.png"
        if not template_path.exists():
            logger.debug(f"  {name}: no template saved yet — skipping")
            return None

        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            return None

        rx1 = int(region[0] * w)
        ry1 = int(region[1] * h)
        rx2 = int(region[2] * w)
        ry2 = int(region[3] * h)

        roi = img[ry1:ry2, rx1:rx2]
        roi_h, roi_w = roi.shape[:2]

        best_val = -1.0
        best_loc = None
        best_scale = 1.0
        templ_h, templ_w = template.shape[:2]

        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            sw = int(templ_w * scale)
            sh = int(templ_h * scale)
            if sw >= roi_w or sh >= roi_h or sw < 10 or sh < 10:
                continue

            scaled = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(roi, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        if best_val < 0.6 or best_loc is None:
            logger.debug(f"  {name}: template match too low ({best_val:.2f})")
            return None

        # Center of matched region in full image coords
        matched_w = int(templ_w * best_scale)
        matched_h = int(templ_h * best_scale)
        cx = best_loc[0] + matched_w // 2 + rx1
        cy = best_loc[1] + matched_h // 2 + ry1

        x_pct = cx / w * 100
        y_pct = cy / h * 100
        logger.info(
            f"  Local detection: {name} at ({x_pct:.1f}%, {y_pct:.1f}%) "
            f"[template] scale={best_scale:.1f} conf={best_val:.2f}"
        )
        return DetectedElement(
            name=name, x_percent=x_pct, y_percent=y_pct,
            confidence=best_val, method="template",
        )

    # ── Debug output ──────────────────────────────────────────────

    @staticmethod
    def _save_debug_image(img: np.ndarray, detections: list[DetectedElement], screen_type: str) -> None:
        """Save annotated image showing detected elements."""
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        debug_img = img.copy()
        h, w = debug_img.shape[:2]

        for det in detections:
            cx = int(det.x_percent / 100 * w)
            cy = int(det.y_percent / 100 * h)

            # Color by method
            color_bgr = {
                "color": (0, 255, 0),
                "shape": (255, 255, 0),
                "template": (0, 165, 255),
            }.get(det.method, (255, 255, 255))

            # Crosshair
            arm = 25
            cv2.line(debug_img, (cx - arm, cy), (cx + arm, cy), color_bgr, 2)
            cv2.line(debug_img, (cx, cy - arm), (cx, cy + arm), color_bgr, 2)
            cv2.circle(debug_img, (cx, cy), 15, color_bgr, 2)

            # Label
            label = f"{det.name} [{det.method}] {det.confidence:.2f}"
            cv2.putText(debug_img, label, (cx + 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

        path = _DEBUG_DIR / f"{screen_type}_detection.png"
        cv2.imwrite(str(path), debug_img)
        logger.debug(f"Saved detection debug image: {path}")
