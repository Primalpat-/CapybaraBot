"""Local OCR reader for monument popup text using EasyOCR.

Provides fast (~200ms), free alternative to Vision API for reading
monument popup status, defender names, power levels, and action buttons.
Falls back gracefully when confidence is low.
"""

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded EasyOCR singleton (model loads ~2-3s on first call)
# ---------------------------------------------------------------------------

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Loading EasyOCR model (first call)...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR model loaded")
    return _reader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRDefenderReading:
    slot: int
    name: str = ""
    power: int = 0
    status: str = "unknown"  # "active", "empty", "defeated"
    confidence: float = 0.0


@dataclass
class OCRMonumentReading:
    ownership_text: str = ""
    is_friendly: bool | None = None
    defenders: list[OCRDefenderReading] = field(default_factory=list)
    action_button_text: str = ""
    overall_confidence: float = 0.0
    total_garrison_power: int = 0


# ---------------------------------------------------------------------------
# Popup region definitions (percentages of full screenshot)
#
# These define where each section of the monument popup appears.
# Tuned for 1080x1920 BlueStacks; percentage-based for resolution independence.
# ---------------------------------------------------------------------------

# The monument popup occupies roughly the center of the screen
POPUP_REGION = (0.10, 0.18, 0.90, 0.88)  # (left%, top%, right%, bottom%)

# Regions within the popup (relative to POPUP_REGION)
# Ownership text at the bottom of the popup area
OWNERSHIP_REGION = (0.15, 0.72, 0.85, 0.82)

# Three defender slots — each has name + power
DEFENDER_REGIONS = [
    # slot 1 (top)
    {"name": (0.25, 0.22, 0.75, 0.30), "power": (0.25, 0.30, 0.75, 0.37)},
    # slot 2 (middle)
    {"name": (0.25, 0.38, 0.75, 0.46), "power": (0.25, 0.46, 0.75, 0.53)},
    # slot 3 (bottom)
    {"name": (0.25, 0.54, 0.75, 0.62), "power": (0.25, 0.62, 0.75, 0.69)},
]

# Action button at the very bottom of popup
ACTION_BUTTON_REGION = (0.20, 0.84, 0.80, 0.94)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _crop_region(image: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """Crop image using percentage-based region (left%, top%, right%, bottom%)."""
    h, w = image.shape[:2]
    left = int(region[0] * w)
    top = int(region[1] * h)
    right = int(region[2] * w)
    bottom = int(region[3] * h)
    return image[top:bottom, left:right]


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR accuracy: upscale, grayscale, threshold."""
    # Upscale 2x for better character recognition
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled

    # Adaptive threshold for variable lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    return thresh


def _detect_text_color(image: np.ndarray) -> str:
    """Detect dominant text color in a region: 'blue', 'red', or 'unknown'.

    Uses HSV color space to distinguish blue (friendly) from red (enemy) text.
    """
    if image.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Blue range in HSV
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_pixels = cv2.countNonZero(blue_mask)

    # Red range in HSV (wraps around 0)
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    red_pixels = cv2.countNonZero(red_mask)

    total_pixels = image.shape[0] * image.shape[1]
    if total_pixels == 0:
        return "unknown"

    # Need at least 1% of pixels to be colored to make a call
    min_threshold = total_pixels * 0.01
    if blue_pixels > min_threshold and blue_pixels > red_pixels * 1.5:
        return "blue"
    elif red_pixels > min_threshold and red_pixels > blue_pixels * 1.5:
        return "red"
    return "unknown"


def _extract_power_number(text: str) -> int:
    """Extract a numeric power value from OCR text.

    Handles formats like: "12,345", "12345", "Power: 12,345", "12.3K"
    Returns 0 if no number found.
    """
    if not text:
        return 0

    # Remove common OCR artifacts and normalize
    cleaned = text.strip()

    # Try to find numbers with commas (e.g., "12,345")
    match = re.search(r"[\d,]+\d", cleaned)
    if match:
        num_str = match.group(0).replace(",", "")
        try:
            return int(num_str)
        except ValueError:
            pass

    # Try plain numbers
    match = re.search(r"\d+", cleaned)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            pass

    return 0


def _ocr_region(image: np.ndarray, region: tuple[float, float, float, float],
                preprocess: bool = True) -> list[tuple[str, float]]:
    """Run OCR on a cropped region, return list of (text, confidence) pairs."""
    cropped = _crop_region(image, region)
    if cropped.size == 0:
        return []

    if preprocess:
        processed = _preprocess_for_ocr(cropped)
    else:
        processed = cropped

    reader = _get_reader()
    try:
        results = reader.readtext(processed, detail=1)
        return [(text.strip(), conf) for (_, text, conf) in results if text.strip()]
    except Exception as e:
        logger.debug(f"OCR error: {e}")
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def read_monument_popup(png_bytes: bytes) -> OCRMonumentReading:
    """Read monument popup from a screenshot using local OCR.

    Args:
        png_bytes: PNG screenshot as bytes

    Returns:
        OCRMonumentReading with extracted text, ownership, defenders, and confidence.
    """
    # Decode PNG to OpenCV image
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("OCR: failed to decode image")
        return OCRMonumentReading()

    # Crop to popup area
    popup = _crop_region(image, POPUP_REGION)
    if popup.size == 0:
        return OCRMonumentReading()

    confidences = []
    reading = OCRMonumentReading()

    # --- Ownership text ---
    ownership_crop = _crop_region(popup, OWNERSHIP_REGION)
    if ownership_crop.size > 0:
        color = _detect_text_color(ownership_crop)
        ownership_results = _ocr_region(popup, OWNERSHIP_REGION)
        if ownership_results:
            reading.ownership_text = " ".join(t for t, _ in ownership_results)
            avg_conf = sum(c for _, c in ownership_results) / len(ownership_results)
            confidences.append(avg_conf)

            # Determine friendly/enemy from color + text
            text_lower = reading.ownership_text.lower()
            if color == "blue" and "star spirit" in text_lower:
                reading.is_friendly = True
            elif color == "red":
                reading.is_friendly = False
            elif "star spirit" in text_lower:
                reading.is_friendly = True
            else:
                reading.is_friendly = False
        else:
            reading.is_friendly = None

    # --- Defender slots ---
    for i, slot_regions in enumerate(DEFENDER_REGIONS):
        slot_num = i + 1
        defender = OCRDefenderReading(slot=slot_num)

        # Read name
        name_results = _ocr_region(popup, slot_regions["name"])
        if name_results:
            name_text = " ".join(t for t, _ in name_results)
            name_conf = sum(c for _, c in name_results) / len(name_results)
            confidences.append(name_conf)

            if "not garrisoned" in name_text.lower() or "empty" in name_text.lower():
                defender.status = "empty"
                defender.name = ""
            else:
                defender.status = "active"
                defender.name = name_text
        else:
            defender.status = "empty"

        # Read power
        power_results = _ocr_region(popup, slot_regions["power"])
        if power_results and defender.status == "active":
            power_text = " ".join(t for t, _ in power_results)
            defender.power = _extract_power_number(power_text)
            power_conf = sum(c for _, c in power_results) / len(power_results)
            confidences.append(power_conf)
            defender.confidence = power_conf

        reading.defenders.append(defender)

    # --- Action button ---
    button_results = _ocr_region(popup, ACTION_BUTTON_REGION)
    if button_results:
        reading.action_button_text = " ".join(t for t, _ in button_results)
        btn_conf = sum(c for _, c in button_results) / len(button_results)
        confidences.append(btn_conf)

    # Cross-check: "Attack" button overrides ownership to enemy
    btn_lower = reading.action_button_text.lower()
    if "attack" in btn_lower:
        reading.is_friendly = False
    elif "visit" in btn_lower:
        reading.is_friendly = True

    # Compute overall confidence and total power
    reading.overall_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    reading.total_garrison_power = sum(
        d.power for d in reading.defenders if d.status == "active"
    )

    logger.info(
        f"OCR reading: friendly={reading.is_friendly}, "
        f"ownership='{reading.ownership_text}', "
        f"button='{reading.action_button_text}', "
        f"power={reading.total_garrison_power}, "
        f"confidence={reading.overall_confidence:.2f}"
    )

    return reading
