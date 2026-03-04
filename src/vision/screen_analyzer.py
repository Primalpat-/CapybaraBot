"""Local-first screen detection using brightness, element detection, and OCR.

Replaces most Vision API identify_screen calls with a tiered local pipeline:
  Tier 1 — Brightness (<1ms): black/transition → "loading"
  Tier 2 — Element signatures (~5-20ms): known buttons → infer screen type
  Tier 3 — OCR keyword matching (~200-500ms): text patterns → screen type
  Tier 4 — Vision API fallback (3-5s, $$$): only when tiers 1-3 fail

Also extracts element positions from OCR text (e.g. "attack" button bounding
box → action_button coordinate), reducing the need for separate calibration.
"""

import io
import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageStat

from src.vision.element_detector import ElementDetector
from src.vision.minimap_detector import find_minimap_squares
from src.vision.ocr_reader import _get_reader, _enhance_for_ocr

logger = logging.getLogger(__name__)


@dataclass
class ScreenAnalysis:
    """Result of local screen analysis."""
    screen_type: str = "unknown"        # Same enum as identify_screen
    confidence: float = 0.0             # 0.0-1.0
    method: str = "unknown"             # "brightness", "element", "ocr", "minimap", "vision"
    elements: dict[str, tuple[float, float, float]] = field(default_factory=dict)
        # name → (x_pct, y_pct, confidence)
    texts: list[tuple[str, float, float]] = field(default_factory=list)
        # (text, x_pct, y_pct)
    timer: str | None = None            # For hibernation/cant_attack


# ── Element signature mapping ─────────────────────────────────────
# If a signature element is found with high confidence → infer screen type.
# Maps element_name → screen_type.
#
# IMPORTANT: Only include elements that are visually UNIQUE to their screen.
# Color-only detections (yellow buttons, green buttons) are too ambiguous —
# yellow appears in nav bars, monument popups, battle results, and logged_out.
# Green appears in nav bar icons, skip buttons, and minimap checkmarks.
# Purple appears in various UI chrome across most screens.
#
# Reliable signatures:
#   - Template matches (star_trek_button, alien_minefield_button): pixel patterns
#   - Pink cancel button: pink is rare in game UI
#   - Dark screen + ok_button: battle_result has uniquely dark background (~36 brightness)

_ELEMENT_SIGNATURES: dict[str, str] = {
    "star_trek_button": "home_screen",           # template match — very specific
    "alien_minefield_button": "mode_select",      # template match — very specific
    "occupy_cancel_button": "occupy_prompt",      # pink button — distinctive color
}

# Which screen types to probe for element detection (template + pink only).
_SIGNATURE_SCREEN_TYPES = [
    "home_screen",
    "mode_select",
    "occupy_prompt",
]

# Battle result screens have very dark backgrounds (brightness ~36).
# Other screens with yellow buttons are much brighter (main_map ~76,
# monument_popup ~113). This threshold gates ok_button detection to
# avoid false positives from yellow nav bar / popup buttons.
_DARK_SCREEN_THRESHOLD = 55

# ── OCR keyword patterns ──────────────────────────────────────────
# Each screen type has keywords. Match threshold varies per screen.

_SCREEN_KEYWORDS: dict[str, set[str]] = {
    "monument_popup":  {"defense info", "ownership", "not garrisoned", "garrisoned",
                        "mecha armament"},
    "hibernation":     {"hibernation"},
    "cant_attack":     {"cannot attack"},
    "logged_out":      {"logged in on another device"},
    "home_screen":     {"adventure", "guild", "events"},
    "mode_select":     {"star trek", "alien minefield"},
    "occupy_prompt":   {"continue to occupy", "abandon"},
    "daily_popup":     {"do not show again"},
    "battle_result":   {"victory", "defeat", "battle report"},
    "battle_active":   {"skip"},
    "main_map":        {"dormant"},
}

# Screens where 1 keyword is enough (they're unique enough)
_SINGLE_KEYWORD_SCREENS = {"hibernation", "cant_attack", "logged_out",
                           "battle_active", "main_map"}

# ── OCR text → element position mapping ───────────────────────────
# When OCR finds specific text, its bounding box center becomes an element.
# Format: (keyword, element_name, min_y_pct, max_y_pct, allowed_screens)
# allowed_screens: if set, only extract the element when the detected screen
# matches. None = always extract.

_TEXT_TO_ELEMENT: list[tuple[str, str, float, float, set[str] | None]] = [
    ("attack", "action_button", 70.0, 100.0, {"monument_popup"}),
    ("occupy", "action_button", 70.0, 100.0, {"monument_popup"}),
    ("garrison", "action_button", 70.0, 100.0, {"monument_popup"}),
    ("visit", "action_button", 70.0, 100.0, {"monument_popup"}),
    ("mining", "action_button", 70.0, 100.0, {"monument_popup"}),
    ("skip", "skip_battle", 0.0, 100.0, {"battle_active"}),
    ("ok", "ok_button", 50.0, 100.0, {"battle_result"}),
    ("restart", "restart_button", 0.0, 100.0, {"logged_out"}),
    ("cancel", "occupy_cancel_button", 50.0, 100.0, {"occupy_prompt"}),
    ("star trek", "star_trek_button", 0.0, 100.0, {"home_screen", "mode_select"}),
    ("alien minefield", "alien_minefield_button", 0.0, 100.0, {"mode_select"}),
]


class ScreenAnalyzer:
    """Local-first screen detection with tiered fallback to Vision API."""

    def __init__(self, element_detector: ElementDetector, config: dict):
        self._element_detector = element_detector
        self._config = config

    def analyze(self, png_bytes: bytes) -> ScreenAnalysis:
        """Single-pass analysis: screen type + element positions + text.

        Runs detection tiers in order, stopping when confident:
          1. Brightness check (<1ms)
          2. Element signatures (~5-20ms)
          3. Minimap detection (~10-30ms)
          4. OCR keyword matching (~200-500ms)

        Vision API (tier 4) is NOT called here — the caller should fall back
        to Vision if analysis.confidence < 0.6.
        """
        # ── Tier 1: Brightness check ─────────────────────────────
        result = self._check_brightness(png_bytes)
        if result is not None:
            return result

        # ── Tier 2: Element signatures ────────────────────────────
        result = self._check_element_signatures(png_bytes)
        if result is not None:
            return result

        # ── Tier 2.5: Minimap detection ───────────────────────────
        result = self._check_minimap(png_bytes)
        if result is not None:
            return result

        # ── Tier 3: OCR keyword matching ──────────────────────────
        result = self._check_ocr_keywords(png_bytes)
        if result is not None:
            return result

        # No tier matched with sufficient confidence
        logger.info("ScreenAnalyzer: no tier matched — Vision API fallback needed")
        return ScreenAnalysis(screen_type="unknown", confidence=0.0, method="none")

    # ── Tier 1: Brightness ────────────────────────────────────────

    @staticmethod
    def _check_brightness(png_bytes: bytes) -> ScreenAnalysis | None:
        """Dark screen → loading/transition."""
        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("L")
            mean_brightness = ImageStat.Stat(image).mean[0]
            if mean_brightness < 15:
                logger.info(f"ScreenAnalyzer: brightness={mean_brightness:.1f} → loading")
                return ScreenAnalysis(
                    screen_type="loading",
                    confidence=0.95,
                    method="brightness",
                )
        except Exception:
            pass
        return None

    # ── Tier 2: Element signatures ────────────────────────────────

    def _check_element_signatures(self, png_bytes: bytes) -> ScreenAnalysis | None:
        """Detect known UI elements → infer screen type.

        Only checks template-matched elements (star_trek, alien_minefield) and
        distinctively-colored elements (pink cancel button). Color-only yellow/
        green/purple detections are NOT used here because they false-positive
        across many screens (nav bar, popups, UI chrome).

        Additionally checks for battle_result via dark background + ok_button:
        battle_result screens have uniquely low brightness (~36) compared to
        all other screens with yellow buttons (main_map ~76, popup ~113).
        """
        analysis = ScreenAnalysis()
        best_screen = None
        best_conf = 0.0

        # Check template + pink signatures
        for screen_type in _SIGNATURE_SCREEN_TYPES:
            detections = self._element_detector.detect(png_bytes, screen_type)
            for det in detections:
                analysis.elements[det.name] = (
                    det.x_percent, det.y_percent, det.confidence
                )
                if det.name in _ELEMENT_SIGNATURES and det.confidence >= 0.7:
                    sig_screen = _ELEMENT_SIGNATURES[det.name]
                    if det.confidence > best_conf:
                        best_screen = sig_screen
                        best_conf = det.confidence

        if best_screen is not None:
            analysis.screen_type = best_screen
            analysis.confidence = min(1.0, best_conf)
            analysis.method = "element"
            logger.info(
                f"ScreenAnalyzer: element signature → {best_screen} "
                f"(conf={analysis.confidence:.2f})"
            )
            return analysis

        # Dark screen + ok_button → battle_result
        # Only check when screen is dark enough to rule out main_map/popup yellow buttons
        brightness = self._get_brightness(png_bytes)
        if brightness is not None and brightness < _DARK_SCREEN_THRESHOLD:
            detections = self._element_detector.detect(png_bytes, "battle_result")
            for det in detections:
                if det.name == "ok_button" and det.confidence >= 0.7:
                    logger.info(
                        f"ScreenAnalyzer: dark screen (brightness={brightness:.0f}) "
                        f"+ ok_button → battle_result (conf={det.confidence:.2f})"
                    )
                    return ScreenAnalysis(
                        screen_type="battle_result",
                        confidence=det.confidence,
                        method="element",
                        elements={det.name: (det.x_percent, det.y_percent, det.confidence)},
                    )

        return None

    @staticmethod
    def _get_brightness(png_bytes: bytes) -> float | None:
        """Get mean pixel brightness (0-255) of an image."""
        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("L")
            return ImageStat.Stat(image).mean[0]
        except Exception:
            return None

    # ── Tier 2.5: Minimap detection ───────────────────────────────

    @staticmethod
    def _check_minimap(png_bytes: bytes) -> ScreenAnalysis | None:
        """Detect minimap by colored squares."""
        detection = find_minimap_squares(png_bytes)
        if detection is not None and len(detection.squares) >= 2:
            logger.info(
                f"ScreenAnalyzer: minimap detected ({len(detection.squares)} squares)"
            )
            return ScreenAnalysis(
                screen_type="minimap",
                confidence=0.9,
                method="minimap",
            )
        return None

    # ── Tier 3: OCR keyword matching ──────────────────────────────

    def _check_ocr_keywords(self, png_bytes: bytes) -> ScreenAnalysis | None:
        """Run OCR, match keywords to screen types, extract element positions."""
        try:
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return None

            img_h, img_w = image.shape[:2]

            # Enhance for OCR
            enhanced = _enhance_for_ocr(image)
            enh_h, enh_w = enhanced.shape[:2]

            reader = _get_reader()
            results = reader.readtext(enhanced, detail=1)
            if not results:
                return None

            # Build detection list with positions in original image percentages
            detections: list[dict] = []
            for bbox, text, conf in results:
                text = text.strip()
                if not text:
                    continue
                pts = np.array(bbox)
                cx = float(pts[:, 0].mean()) / enh_w * 100  # as % of width
                cy = float(pts[:, 1].mean()) / enh_h * 100  # as % of height
                detections.append({
                    "text": text,
                    "lower": text.lower(),
                    "cx": cx,
                    "cy": cy,
                    "conf": conf,
                })

            if not detections:
                return None

            # Score each screen type by keyword matches
            best_screen = None
            best_score = 0
            best_conf = 0.0

            for screen_type, keywords in _SCREEN_KEYWORDS.items():
                hits = 0
                max_conf = 0.0
                for d in detections:
                    for kw in keywords:
                        if kw in d["lower"]:
                            hits += 1
                            max_conf = max(max_conf, d["conf"])
                            break

                # Threshold: 1 keyword for unique screens, 2 for others
                min_hits = 1 if screen_type in _SINGLE_KEYWORD_SCREENS else 2
                if hits >= min_hits and hits > best_score:
                    best_screen = screen_type
                    best_score = hits
                    best_conf = max_conf

            if best_screen is None:
                return None

            # Build result
            analysis = ScreenAnalysis(
                screen_type=best_screen,
                confidence=min(1.0, best_conf * 0.9),  # slight discount for OCR
                method="ocr",
                texts=[(d["text"], d["cx"], d["cy"]) for d in detections],
            )

            # Extract element positions from OCR text
            for d in detections:
                for keyword, element_name, min_y, max_y, allowed in _TEXT_TO_ELEMENT:
                    if keyword in d["lower"] and min_y <= d["cy"] <= max_y:
                        # Only extract if screen type matches (when filter is set)
                        if allowed is not None and best_screen not in allowed:
                            break
                        # Don't overwrite higher-confidence detections
                        if element_name not in analysis.elements or d["conf"] > analysis.elements[element_name][2]:
                            analysis.elements[element_name] = (
                                d["cx"], d["cy"], d["conf"] * 0.8
                            )
                        break

            # Extract timer for hibernation/cant_attack
            if best_screen in ("hibernation", "cant_attack"):
                analysis.timer = self._extract_timer(
                    detections, best_screen, img_h
                )

            logger.info(
                f"ScreenAnalyzer: OCR → {best_screen} "
                f"(hits={best_score}, conf={analysis.confidence:.2f}, "
                f"elements={list(analysis.elements.keys())})"
            )
            return analysis

        except Exception as e:
            logger.warning(f"ScreenAnalyzer OCR failed: {e}")
            return None

    @staticmethod
    def _extract_timer(
        detections: list[dict], screen_type: str, img_h: int
    ) -> str | None:
        """Extract timer text (HH:MM:SS or MM:SS) from OCR detections."""
        for d in detections:
            timer_match = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", d["text"])
            if timer_match:
                # For cant_attack, timer must be in bottom half
                if screen_type == "cant_attack" and d["cy"] < 50.0:
                    continue
                return timer_match.group(0)
        return None
