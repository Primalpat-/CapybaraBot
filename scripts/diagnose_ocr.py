"""Diagnostic tool: capture screenshot, run OCR on monument popup, save annotated image.

Run with:  python -m scripts.diagnose_ocr

This will:
1. Connect to ADB and capture a screenshot
2. Draw crop regions for each popup section
3. Run OCR on each region and print results
4. Save annotated output to screenshots/ocr/ for visual tuning
"""

import asyncio
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adb.connection import ADBConnection
from src.adb.capture import ScreenCapture
from src.vision.ocr_reader import (
    POPUP_REGION,
    OWNERSHIP_REGION,
    DEFENDER_REGIONS,
    ACTION_BUTTON_REGION,
    _crop_region,
    read_monument_popup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "screenshots" / "ocr"


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abs_region(popup_region, sub_region, img_w, img_h):
    """Convert a sub-region (relative to popup) to absolute pixel coords."""
    pl = int(popup_region[0] * img_w)
    pt = int(popup_region[1] * img_h)
    pw = int((popup_region[2] - popup_region[0]) * img_w)
    ph = int((popup_region[3] - popup_region[1]) * img_h)

    x1 = pl + int(sub_region[0] * pw)
    y1 = pt + int(sub_region[1] * ph)
    x2 = pl + int(sub_region[2] * pw)
    y2 = pt + int(sub_region[3] * ph)
    return x1, y1, x2, y2


def draw_regions(image: np.ndarray) -> np.ndarray:
    """Draw all OCR crop regions on the image with labels."""
    annotated = image.copy()
    h, w = image.shape[:2]

    # Draw popup region
    pl, pt = int(POPUP_REGION[0] * w), int(POPUP_REGION[1] * h)
    pr, pb = int(POPUP_REGION[2] * w), int(POPUP_REGION[3] * h)
    cv2.rectangle(annotated, (pl, pt), (pr, pb), (0, 255, 0), 2)
    cv2.putText(annotated, "POPUP", (pl + 5, pt + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw ownership region
    x1, y1, x2, y2 = _abs_region(POPUP_REGION, OWNERSHIP_REGION, w, h)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(annotated, "OWNERSHIP", (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw defender regions
    colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for i, regions in enumerate(DEFENDER_REGIONS):
        color = colors[i % len(colors)]
        for label_key in ("name", "power"):
            x1, y1, x2, y2 = _abs_region(POPUP_REGION, regions[label_key], w, h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"D{i+1}_{label_key}", (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw action button region
    x1, y1, x2, y2 = _abs_region(POPUP_REGION, ACTION_BUTTON_REGION, w, h)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(annotated, "BUTTON", (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return annotated


async def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to ADB
    adb_cfg = config.get("adb", {})
    connection = ADBConnection(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=adb_cfg.get("port", 5555),
        adb_path=adb_cfg.get("adb_path", "adb"),
    )

    logger.info("Connecting to ADB...")
    if not await connection.connect_with_retry():
        logger.error("Failed to connect to ADB")
        return

    capture = ScreenCapture(connection)

    # Capture screenshot
    logger.info("Capturing screenshot...")
    png_bytes = await capture.capture()
    logger.info(f"Screenshot captured: {len(png_bytes)} bytes")

    # Save raw screenshot
    raw_path = OUTPUT_DIR / "raw_screenshot.png"
    raw_path.write_bytes(png_bytes)
    logger.info(f"Saved raw screenshot: {raw_path}")

    # Decode and draw regions
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    annotated = draw_regions(image)

    annotated_path = OUTPUT_DIR / "annotated_regions.png"
    cv2.imwrite(str(annotated_path), annotated)
    logger.info(f"Saved annotated image: {annotated_path}")

    # Run OCR
    logger.info("Running OCR...")
    reading = read_monument_popup(png_bytes)

    # Print results
    print("\n" + "=" * 60)
    print("OCR RESULTS")
    print("=" * 60)
    print(f"  Ownership text:  '{reading.ownership_text}'")
    print(f"  Is friendly:     {reading.is_friendly}")
    print(f"  Action button:   '{reading.action_button_text}'")
    print(f"  Overall conf:    {reading.overall_confidence:.2f}")
    print(f"  Total power:     {reading.total_garrison_power}")
    print()
    for d in reading.defenders:
        print(f"  Defender {d.slot}: name='{d.name}', power={d.power}, "
              f"status={d.status}, conf={d.confidence:.2f}")
    print("=" * 60)

    await connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
