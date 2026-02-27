"""Local OCR reader for monument popup text using EasyOCR.

Provides fast (~200ms), free alternative to Vision API for reading
monument popup status, defender names, power levels, and action buttons.
Falls back gracefully when confidence is low.

Approach: OCR the entire popup region at once, then interpret results by
pattern-matching text content and grouping by spatial position. No
hardcoded per-element crop regions needed.
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
# Popup crop — we only need ONE region: the popup itself
# ---------------------------------------------------------------------------

POPUP_REGION = (0.08, 0.06, 0.92, 0.95)


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


def _detect_text_color(image: np.ndarray, bbox) -> str:
    """Detect text color at a specific bounding box: 'blue', 'red', or 'unknown'.

    bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] from EasyOCR.
    """
    if image.size == 0:
        return "unknown"

    # Extract bounding rect from the 4-point bbox
    pts = np.array(bbox, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    # Clamp to image bounds
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return "unknown"

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

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

    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return "unknown"

    min_threshold = total_pixels * 0.01
    if blue_pixels > min_threshold and blue_pixels > red_pixels * 1.5:
        return "blue"
    elif red_pixels > min_threshold and red_pixels > blue_pixels * 1.5:
        return "red"
    return "unknown"


def _extract_power_number(text: str) -> int:
    """Extract a numeric power value from OCR text.

    Handles formats like: "24.68M", "14.28K", "12,345", "12345"
    M = millions, K = thousands. Returns 0 if no number found.
    """
    if not text:
        return 0

    cleaned = text.strip()

    # Try number with M/K suffix first (e.g., "24.68M", "14.28K", "3M")
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*([MmKk])", cleaned)
    if match:
        num_str = match.group(1).replace(",", ".")
        suffix = match.group(2).upper()
        try:
            value = float(num_str)
            if suffix == "M":
                return int(value * 1_000_000)
            elif suffix == "K":
                return int(value * 1_000)
        except ValueError:
            pass

    # Try numbers with commas (e.g., "12,345")
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


def _is_power_text(text: str) -> bool:
    """Check if text looks like a power value (e.g., '24.68M', '5,000')."""
    return bool(re.search(r"\d+[.,]?\d*\s*[MmKk]", text)) or bool(
        re.search(r"^[\d,]+$", text.strip())
    )


_BUTTON_WORDS = {"attack", "exit", "visit", "claim", "quick", "mining"}
_NOISE_WORDS = {
    "defense", "info", "estimated", "earnings", "hour", "level", "monument",
    "subject", "actual", "output", "ownership", "guard", "guarding",
    "not", "garrisoned", "empty",
}


def _center_y(bbox) -> float:
    """Get vertical center of an EasyOCR bounding box as a pixel coordinate."""
    pts = np.array(bbox)
    return float(pts[:, 1].mean())


def _center_x(bbox) -> float:
    """Get horizontal center of an EasyOCR bounding box as a pixel coordinate."""
    pts = np.array(bbox)
    return float(pts[:, 0].mean())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def read_monument_popup(png_bytes: bytes) -> OCRMonumentReading:
    """Read monument popup from a screenshot using local OCR.

    Scans the entire popup region, then interprets all detected text by
    pattern matching — no hardcoded per-element crop positions needed.
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

    popup_h, popup_w = popup.shape[:2]

    # Run OCR on the entire popup at once
    reader = _get_reader()
    try:
        raw_results = reader.readtext(popup, detail=1)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return OCRMonumentReading()

    if not raw_results:
        return OCRMonumentReading()

    # Build list of (bbox, text, confidence, y_center_pct, x_center_pct)
    detections = []
    for bbox, text, conf in raw_results:
        text = text.strip()
        if not text:
            continue
        cy = _center_y(bbox) / popup_h  # as fraction of popup height
        cx = _center_x(bbox) / popup_w  # as fraction of popup width
        detections.append({
            "bbox": bbox,
            "text": text,
            "conf": conf,
            "cy": cy,
            "cx": cx,
        })

    logger.debug(f"OCR found {len(detections)} text regions")
    for d in detections:
        logger.debug(f"  [{d['cy']:.2f}, {d['cx']:.2f}] conf={d['conf']:.2f} '{d['text']}'")

    reading = OCRMonumentReading()
    confidences = []

    # --- Find ownership text ---
    # Look for "Monument Ownership:" followed by a name on the same line
    ownership_name = ""
    ownership_color = "unknown"
    for d in detections:
        lower = d["text"].lower()
        if "ownership" in lower and "monument" in lower:
            # The ownership label itself — look for the name nearby (similar y)
            for d2 in detections:
                if d2 is d:
                    continue
                if abs(d2["cy"] - d["cy"]) < 0.02 and d2["cx"] > d["cx"]:
                    ownership_name = d2["text"]
                    ownership_color = _detect_text_color(popup, d2["bbox"])
                    confidences.append(d2["conf"])
                    break
            break

    if ownership_name:
        reading.ownership_text = ownership_name
        name_lower = ownership_name.lower()
        if ownership_color == "blue" and "star spirit" in name_lower:
            reading.is_friendly = True
        elif ownership_color == "red":
            reading.is_friendly = False
        elif "star spirit" in name_lower:
            reading.is_friendly = True
        else:
            reading.is_friendly = False
    else:
        # Fallback: look for "Star Spirit" anywhere
        for d in detections:
            if "star spirit" in d["text"].lower():
                reading.ownership_text = d["text"]
                color = _detect_text_color(popup, d["bbox"])
                reading.is_friendly = (color == "blue")
                confidences.append(d["conf"])
                break

    # --- Find power values and associate with defender names ---
    # Power values match patterns like "24.68M", "14.28K", "5,000"
    # They appear to the left/center of the popup, names to the right
    power_entries = []
    for d in detections:
        if _is_power_text(d["text"]):
            power = _extract_power_number(d["text"])
            if power > 0:
                power_entries.append({
                    "power": power,
                    "cy": d["cy"],
                    "cx": d["cx"],
                    "conf": d["conf"],
                    "text": d["text"],
                })

    # Sort power entries by vertical position (top to bottom = slot 1, 2, 3)
    power_entries.sort(key=lambda p: p["cy"])

    # For each power entry, find the closest name-like text at a similar y position
    # Names are typically to the right of center and DON'T look like power/noise
    used_names = set()
    defenders = []
    for slot_idx, pe in enumerate(power_entries[:3]):  # max 3 defenders
        slot_num = slot_idx + 1
        best_name = ""
        best_conf = 0.0

        for d in detections:
            if id(d) in used_names:
                continue
            # Must be at similar y position (within ~4% of popup height)
            if abs(d["cy"] - pe["cy"]) > 0.04:
                continue
            # Skip power text, noise, and button words
            lower = d["text"].lower()
            if _is_power_text(d["text"]):
                continue
            words = set(lower.split())
            if words & _NOISE_WORDS:
                continue
            if words & _BUTTON_WORDS:
                continue
            # Skip single digits (slot numbers like "1", "2", "3")
            if re.match(r"^\d$", d["text"].strip()):
                continue
            # Prefer text that's further right (names are right of character art)
            if d["cx"] > 0.3:
                best_name = d["text"]
                best_conf = d["conf"]
                used_names.add(id(d))
                break

        defender = OCRDefenderReading(
            slot=slot_num,
            name=best_name,
            power=pe["power"],
            status="active" if best_name else "active",  # has power, so active
            confidence=pe["conf"],
        )
        confidences.append(pe["conf"])
        if best_conf > 0:
            confidences.append(best_conf)
        defenders.append(defender)

    # Fill remaining slots as empty (up to 3)
    existing_slots = {d.slot for d in defenders}
    for s in range(1, 4):
        if s not in existing_slots:
            # Check if there's a "Not Garrisoned" text for this slot
            defenders.append(OCRDefenderReading(slot=s, status="empty"))

    defenders.sort(key=lambda d: d.slot)
    reading.defenders = defenders

    # --- Find action button text ---
    # Buttons are in the bottom ~25% of the popup, look for known button words
    for d in detections:
        if d["cy"] < 0.70:
            continue
        lower = d["text"].lower()
        for word in ("attack", "exit", "visit", "claim"):
            if word in lower:
                reading.action_button_text = d["text"]
                confidences.append(d["conf"])
                break
        if reading.action_button_text:
            break

    # Cross-check: button overrides ownership
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
        f"defenders={[(d.name, d.power) for d in reading.defenders if d.status == 'active']}, "
        f"confidence={reading.overall_confidence:.2f}"
    )

    return reading
