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
    wrong_screen: str = ""  # non-empty if OCR detected a wrong screen (e.g. "shop")


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


def _enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Upscale and sharpen popup image for better OCR accuracy.

    EasyOCR works better on larger text.  We upscale to at least 1200px
    wide and apply a mild sharpen to crispen text edges on colored badge
    backgrounds.  Keeps the original color (no grayscale conversion).
    """
    h, w = image.shape[:2]
    min_width = 1200
    if w < min_width:
        scale = min_width / w
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # Mild sharpen kernel — improves text edges without adding noise
    sharpen_kernel = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5],
                               [0, -0.5, 0]], dtype=np.float32)
    image = cv2.filter2D(image, -1, sharpen_kernel)
    return image


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

# All possible faction names in the game
FACTION_NAMES = [
    "star spirit",
    "galactic empire",
    "interstellar federation",
    "star alliance",
]

_NOISE_WORDS = {
    "defense", "info", "estimated", "earnings", "hour", "level", "monument",
    "subject", "actual", "output", "ownership", "guard", "guarding",
    "not", "garrisoned", "empty",
    # Game-specific UI text that appears in the defender section
    "win", "streak", "cannot", "garrison", "cooldown", "debuff",
    "buff", "timer", "expires", "remaining", "locked", "occupied",
    # Faction names — these appear in the ownership section, not defender section
    "star", "spirit", "galactic", "empire", "interstellar", "federation", "alliance",
}

# Regex for timer patterns — these are never player names
_TIMER_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")

# Regex for "X Win Streak" pattern
_WIN_STREAK_RE = re.compile(r"\d+\s*win\s*streak", re.IGNORECASE)


def _is_noise_text(text: str) -> bool:
    """Check if text is game UI noise (not a player name).

    Filters out: known noise/button words, timer patterns, win streak text,
    power values, and pure numbers.
    """
    stripped = text.strip()
    if not stripped:
        return True
    # Timer patterns (e.g. "01:23:45", "5:30")
    if _TIMER_RE.match(stripped):
        return True
    # "X Win Streak" pattern
    if _WIN_STREAK_RE.search(stripped):
        return True
    # Power text (has K/M/B/T suffix)
    if _is_power_text(stripped):
        return True
    # Pure digits / digit-only with punctuation (e.g. "123", "1,234")
    if not any(c.isalpha() for c in stripped):
        return True
    # Known noise or button words
    words = set(stripped.lower().split())
    if words & _NOISE_WORDS:
        return True
    if words & _BUTTON_WORDS:
        return True
    return False


def _is_name_text(text: str) -> bool:
    """Player names are the only alphabetic text in the defender section.

    Must have 2+ alpha characters and not be a known header/noise/button word.
    """
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False
    if _is_noise_text(text):
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

def read_monument_popup(png_bytes: bytes, friendly_faction: str = "star spirit") -> OCRMonumentReading:
    """Read monument popup from a screenshot using local OCR.

    Scans the entire popup region, then interprets all detected text by
    pattern matching — no hardcoded per-element crop positions needed.

    Args:
        png_bytes: Screenshot PNG data.
        friendly_faction: Our faction name (lowercase). Used to determine
            friendly vs enemy ownership.
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

    # Upscale + sharpen for better OCR accuracy.
    # Keep the original popup for color detection (ownership text color).
    popup_enhanced = _enhance_for_ocr(popup)
    enh_h, enh_w = popup_enhanced.shape[:2]
    scale_x = popup_w / enh_w  # to map enhanced coords back to original
    scale_y = popup_h / enh_h

    # Run OCR on the enhanced image
    reader = _get_reader()
    try:
        raw_results = reader.readtext(popup_enhanced, detail=1)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return OCRMonumentReading()

    if not raw_results:
        return OCRMonumentReading()

    # Build list of (bbox, text, confidence, y_center_pct, x_center_pct)
    # Bbox coords are in enhanced image space; convert to fractions using
    # enhanced dimensions, and scale bbox back for color detection on original.
    detections = []
    for bbox, text, conf in raw_results:
        text = text.strip()
        if not text:
            continue
        cy = _center_y(bbox) / enh_h  # as fraction of popup height
        cx = _center_x(bbox) / enh_w  # as fraction of popup width
        # Scale bbox back to original popup coordinates for color detection
        orig_bbox = [
            [int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in bbox
        ]
        detections.append({
            "bbox": orig_bbox,
            "text": text,
            "conf": conf,
            "cy": cy,
            "cx": cx,
        })

    logger.info(f"OCR found {len(detections)} text regions in popup")
    for d in detections:
        logger.info(f"  OCR [{d['cy']:.2f}, {d['cx']:.2f}] conf={d['conf']:.2f} '{d['text']}'")

    reading = OCRMonumentReading()

    # --- Wrong-screen detection ---
    # Check if we accidentally opened the shop instead of the monument popup.
    # Reuses the OCR we just ran — no extra call needed.
    shop_hits = 0
    for d in detections:
        lower = d["text"].lower()
        for kw in _SHOP_KEYWORDS:
            if kw in lower:
                shop_hits += 1
                break
    if shop_hits >= 2:
        logger.info(f"OCR detected shop screen ({shop_hits} shop keywords) — not a monument popup")
        reading.wrong_screen = "shop"
        return reading

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
    # Power numbers (K/M/B/T suffix) are the most reliable anchor for each
    # defender slot.  For each power entry we look in a tight spatial window
    # ABOVE it for the player name, which filters out noise text like
    # "Win Streak", "Cannot garrison", debuff timers, etc.
    defender_detections = [
        d for d in detections
        if d["cy"] > defense_info_y and d["cy"] < ownership_info_y
    ]

    power_entries = []
    for d in defender_detections:
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

    # For each power entry, search a tight window ABOVE it for the name.
    # Player names appear directly above their power number within ~6% of
    # popup height.  Accept any non-noise text as a potential name — players
    # can have unicode symbols, special chars, etc.
    used_detections = set()  # indices into defender_detections
    defenders = []
    for slot_idx, pe in enumerate(power_entries[:3]):  # max 3 defenders
        slot_num = slot_idx + 1
        best_name = ""
        best_conf = 0.0
        best_dist = 999.0
        best_didx = -1

        for di, d in enumerate(defender_detections):
            if di in used_detections:
                continue
            # Name must be ABOVE the power entry (lower cy) and close
            dy = pe["cy"] - d["cy"]  # positive = d is above pe
            if dy < 0.005 or dy > 0.06:
                continue
            if _is_noise_text(d["text"]):
                continue
            if dy < best_dist:
                best_name = _clean_name(d["text"])
                best_conf = d["conf"]
                best_dist = dy
                best_didx = di

        if best_didx >= 0:
            used_detections.add(best_didx)

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
        # Fallback: look for any faction name below Ownership Info header
        for d in detections:
            if ownership_info_y < 1.0 and d["cy"] < ownership_info_y:
                continue
            d_lower = d["text"].lower()
            for faction in FACTION_NAMES:
                if faction in d_lower:
                    ownership_name = d["text"]
                    ownership_color = _detect_text_color(popup, d["bbox"])
                    confidences.append(d["conf"])
                    break
            if ownership_name:
                break

    if ownership_name:
        reading.ownership_text = ownership_name
        name_lower = ownership_name.lower()
        # Name match takes priority over color — the bbox for the whole
        # ownership line often picks up background reds from UI/crystals.
        # Check if ownership matches our faction name.
        friendly_lower = friendly_faction.lower()
        if friendly_lower in name_lower:
            reading.is_friendly = True
        elif any(f in name_lower for f in FACTION_NAMES if f != friendly_lower):
            # Recognized enemy faction name
            reading.is_friendly = False
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


# ---------------------------------------------------------------------------
# Quick OCR checks for wrong-screen detection
# ---------------------------------------------------------------------------

# Keywords that indicate we accidentally opened the shop
_SHOP_KEYWORDS = {
    "season shop", "sergeant medal", "armour chip", "armament cone",
    "purchase limit", "exchange", "shop",
}


def check_if_shop(png_bytes: bytes) -> bool:
    """Quick OCR check: did we accidentally open the shop?

    Looks for shop-specific keywords in the screenshot.  Returns True if
    2+ shop keywords are found (avoids false positives from a single word
    like 'Shop' in the bottom nav bar).
    """
    try:
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return False

        reader = _get_reader()
        results = reader.readtext(image, detail=0)  # text only, faster

        hits = 0
        for text in results:
            lower = text.lower().strip()
            for kw in _SHOP_KEYWORDS:
                if kw in lower:
                    hits += 1
                    break
            if hits >= 2:
                logger.info(f"OCR detected shop screen ({hits} keywords found)")
                return True
        return False
    except Exception as e:
        logger.warning(f"OCR shop check failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Hibernation / Can't-attack screen detection via OCR
# ---------------------------------------------------------------------------

@dataclass
class OCRScreenReading:
    """Result of OCR-based screen check for hibernation/cant_attack overlays."""
    screen_type: str = ""       # "hibernation", "cant_attack", or ""
    timer: str = ""             # raw timer text (e.g. "3:21:05")
    confidence: float = 0.0


def check_screen_ocr(png_bytes: bytes) -> OCRScreenReading:
    """Quick OCR scan of the full screen for hibernation / can't-attack text.

    Looks for keywords like "hibernation" or "cannot attack" and a nearby
    timer pattern (HH:MM:SS or MM:SS).  Much faster and cheaper than a
    Vision API call.

    Can't-attack detection requires "cannot attack" text (NOT "dormant" alone,
    because the sidebar always shows a "Dormant Period" icon — that refers to
    hibernation).  The debuff banner appears at the bottom-middle of the screen,
    so both the keyword and timer must be in the bottom half (y > 50%).
    """
    reading = OCRScreenReading()
    try:
        img_array = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return reading

        # Upscale small images for better OCR accuracy
        h, w = image.shape[:2]
        if max(h, w) < 1024:
            scale = 1024 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
            h, w = image.shape[:2]

        reader = _get_reader()
        results = reader.readtext(image)

        detected_type = ""
        timer_text = ""
        best_confidence = 0.0

        for bbox, text, conf in results:
            lower = text.lower().strip()
            cy = _center_y(bbox)
            in_bottom_half = cy > h * 0.5

            # Detect screen type keywords
            if "hibernat" in lower:
                detected_type = "hibernation"
                best_confidence = max(best_confidence, conf)
            elif "cannot attack" in lower and in_bottom_half:
                # Only match the actual debuff banner at the bottom, not the
                # "Dormant Period" sidebar icon (which refers to hibernation).
                detected_type = "cant_attack"
                best_confidence = max(best_confidence, conf)

            # Look for timer pattern (H:MM:SS or MM:SS)
            timer_match = re.search(r"\d{1,2}:\d{2}:\d{2}", text)
            if not timer_match:
                timer_match = re.search(r"\d{1,2}:\d{2}", text)
            if timer_match:
                # For cant_attack, only accept timers in bottom half of screen.
                # The skull timer in the top-left is NOT the debuff timer.
                if detected_type != "cant_attack" or in_bottom_half:
                    timer_text = timer_match.group(0)

        if detected_type:
            reading.screen_type = detected_type
            reading.timer = timer_text
            reading.confidence = best_confidence
            logger.info(
                f"OCR screen check: {detected_type}, timer='{timer_text}', "
                f"confidence={best_confidence:.2f}"
            )

    except Exception as e:
        logger.warning(f"OCR screen check failed: {e}")

    return reading
