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

    # Try number with magnitude suffix (e.g., "24.68M", "14.28K", "3B")
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*([KkMmBbTt])", cleaned)
    if match:
        num_str = match.group(1).replace(",", ".")
        suffix = match.group(2).upper()
        try:
            value = float(num_str)
            multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
            return int(value * multipliers.get(suffix, 1))
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
    """Power values always have a magnitude suffix (K/M/B/T).

    Number directly followed by suffix letter, suffix not followed by more
    letters (avoids "11 Monument" matching on "M").
    """
    return bool(re.search(r"\d+[.,]?\d*[KkMmBbTt](?![a-zA-Z])", text))


_BUTTON_WORDS = {"attack", "exit", "visit", "claim", "quick", "mining"}
_NOISE_WORDS = {
    "defense", "info", "estimated", "earnings", "hour", "level", "monument",
    "subject", "actual", "output", "ownership", "guard", "guarding",
    "not", "garrisoned", "empty",
}


def _is_name_text(text: str) -> bool:
    """Player names are the only alphabetic text in the defender section.

    Must have 2+ alpha characters and not be a known header/noise/button word.
    """
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False
    if _is_power_text(text):
        return False
    words = set(text.lower().split())
    if words & _NOISE_WORDS:
        return False
    if words & _BUTTON_WORDS:
        return False
    return True


def _clean_name(text: str) -> str:
    """Strip common OCR artifacts (quotes, backticks) from player names."""
    return text.strip().strip("'\"'\u2018\u2019\u201c\u201d`")


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

    # --- Find section boundaries ---
    # "Defense Info" and "Ownership Info" headers divide the popup into sections
    defense_info_y = 0.0   # default: don't filter lower bound
    ownership_info_y = 1.0  # default: don't filter upper bound
    for d in detections:
        lower = d["text"].lower()
        if "defense" in lower and "info" in lower:
            defense_info_y = d["cy"]
        if "ownership" in lower and "info" in lower:
            ownership_info_y = d["cy"]

    # --- Find defenders (between Defense Info and Ownership Info) ---
    # Powers always have K/M/B/T suffix; names are the only alpha text
    power_entries = []
    name_entries = []
    for d in detections:
        if d["cy"] <= defense_info_y or d["cy"] >= ownership_info_y:
            continue
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
        elif _is_name_text(d["text"]):
            name_entries.append({
                "name": _clean_name(d["text"]),
                "cy": d["cy"],
                "conf": d["conf"],
            })

    # Sort power entries by vertical position (top to bottom = slot 1, 2, 3)
    power_entries.sort(key=lambda p: p["cy"])

    # Pair each power entry with the closest name by y-position
    used_names = set()
    defenders = []
    for slot_idx, pe in enumerate(power_entries[:3]):  # max 3 defenders
        slot_num = slot_idx + 1
        best_name = ""
        best_conf = 0.0
        best_dist = 999.0
        best_idx = -1

        for i, ne in enumerate(name_entries):
            if i in used_names:
                continue
            dy = abs(ne["cy"] - pe["cy"])
            if dy < 0.08 and dy < best_dist:
                best_name = ne["name"]
                best_conf = ne["conf"]
                best_dist = dy
                best_idx = i

        if best_idx >= 0:
            used_names.add(best_idx)

        defender = OCRDefenderReading(
            slot=slot_num,
            name=best_name,
            power=pe["power"],
            status="active",
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
            defenders.append(OCRDefenderReading(slot=s, status="empty"))

    defenders.sort(key=lambda d: d.slot)
    reading.defenders = defenders

    # --- Find ownership (below Ownership Info header) ---
    # Faction name appears after "Monument Ownership:" text
    ownership_name = ""
    ownership_color = "unknown"
    for d in detections:
        # Only search below Ownership Info header (when found)
        if ownership_info_y < 1.0 and d["cy"] < ownership_info_y:
            continue
        lower = d["text"].lower()
        if "ownership" in lower and "monument" in lower:
            # Try to extract name from same text (e.g., "Monument Ownership: Star Spirit")
            name_match = re.search(r"[Oo]wnership\s*:\s*(.+)", d["text"])
            if name_match:
                ownership_name = name_match.group(1).strip()
                ownership_color = _detect_text_color(popup, d["bbox"])
                confidences.append(d["conf"])
            else:
                # Name is a separate detection to the right at similar y
                for d2 in detections:
                    if d2 is d:
                        continue
                    if ownership_info_y < 1.0 and d2["cy"] < ownership_info_y:
                        continue
                    if abs(d2["cy"] - d["cy"]) < 0.03 and d2["cx"] > d["cx"]:
                        ownership_name = d2["text"]
                        ownership_color = _detect_text_color(popup, d2["bbox"])
                        confidences.append(d2["conf"])
                        break
            break

    if not ownership_name:
        # Fallback: look for faction names below Ownership Info header
        for d in detections:
            if ownership_info_y < 1.0 and d["cy"] < ownership_info_y:
                continue
            if "star spirit" in d["text"].lower():
                ownership_name = d["text"]
                ownership_color = _detect_text_color(popup, d["bbox"])
                confidences.append(d["conf"])
                break

    if ownership_name:
        reading.ownership_text = ownership_name
        name_lower = ownership_name.lower()
        # Name match takes priority over color — the bbox for the whole
        # ownership line often picks up background reds from UI/crystals
        if "star spirit" in name_lower:
            reading.is_friendly = True
        elif ownership_color == "red":
            reading.is_friendly = False
        elif ownership_color == "blue":
            reading.is_friendly = True
        else:
            reading.is_friendly = False

    # --- Find action button text ---
    # Scan all button text in bottom ~30%, pick the most informative one.
    # Priority: attack (enemy) > mining/visit (friendly) > exit/claim (neutral)
    button_candidates = []
    for d in detections:
        if d["cy"] < 0.70:
            continue
        lower = d["text"].lower()
        for word in ("attack", "exit", "visit", "claim", "quick", "mining"):
            if word in lower:
                button_candidates.append({"text": d["text"], "lower": lower, "conf": d["conf"]})
                break

    # Pick most informative button
    for bc in button_candidates:
        if "attack" in bc["lower"]:
            reading.action_button_text = bc["text"]
            confidences.append(bc["conf"])
            break
    if not reading.action_button_text:
        for bc in button_candidates:
            if "mining" in bc["lower"] or "visit" in bc["lower"]:
                reading.action_button_text = bc["text"]
                confidences.append(bc["conf"])
                break
    if not reading.action_button_text and button_candidates:
        reading.action_button_text = button_candidates[0]["text"]
        confidences.append(button_candidates[0]["conf"])

    # Cross-check: button overrides ownership
    btn_lower = reading.action_button_text.lower()
    if "attack" in btn_lower:
        reading.is_friendly = False
    elif any(w in btn_lower for w in ("visit", "mining", "exit")):
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
