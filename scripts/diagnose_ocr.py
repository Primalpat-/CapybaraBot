"""Diagnostic tool: capture screenshot, run OCR on monument popup, save annotated image.

Run with:  python -m scripts.diagnose_ocr

This will:
1. Connect to ADB and capture a screenshot
2. Run full-popup OCR and draw all detected text bounding boxes
3. Print parsed results
4. Save annotated output to screenshots/ocr/ for visual review
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
    _crop_region,
    _get_reader,
    _is_power_text,
    read_monument_popup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "screenshots" / "ocr"


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_ocr_results(image: np.ndarray) -> np.ndarray:
    """Run raw OCR on popup and draw every detected bounding box with text."""
    annotated = image.copy()
    h, w = image.shape[:2]

    # Draw popup region outline
    pl, pt = int(POPUP_REGION[0] * w), int(POPUP_REGION[1] * h)
    pr, pb = int(POPUP_REGION[2] * w), int(POPUP_REGION[3] * h)
    cv2.rectangle(annotated, (pl, pt), (pr, pb), (0, 255, 0), 2)
    cv2.putText(annotated, "POPUP", (pl + 5, pt + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Crop popup and run raw OCR
    popup = _crop_region(image, POPUP_REGION)
    reader = _get_reader()
    results = reader.readtext(popup, detail=1)

    print(f"\nRaw OCR detections ({len(results)} items):")
    print("-" * 70)

    for bbox, text, conf in results:
        text = text.strip()
        if not text:
            continue

        # Convert popup-relative bbox to full image coords
        pts = np.array(bbox, dtype=np.int32)
        pts[:, 0] = pts[:, 0] + pl
        pts[:, 1] = pts[:, 1] + pt

        # Color code: green=power, blue=button keywords, yellow=other
        lower = text.lower()
        if _is_power_text(text):
            color = (0, 255, 0)
            label = "PWR"
        elif any(w in lower for w in ("attack", "exit", "visit", "claim", "quick", "mining")):
            color = (255, 100, 100)
            label = "BTN"
        elif "ownership" in lower or "star spirit" in lower:
            color = (255, 0, 255)
            label = "OWN"
        else:
            color = (0, 255, 255)
            label = ""

        cv2.polylines(annotated, [pts], True, color, 2)

        # Label with text and confidence
        x_min = pts[:, 0].min()
        y_min = pts[:, 1].min() - 5
        display = f"{label} '{text}' {conf:.2f}" if label else f"'{text}' {conf:.2f}"
        cv2.putText(annotated, display, (x_min, max(y_min, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Also print to console
        cy_pct = (np.mean(pts[:, 1]) - pt) / (pb - pt) * 100
        cx_pct = (np.mean(pts[:, 0]) - pl) / (pr - pl) * 100
        print(f"  [{cy_pct:5.1f}%, {cx_pct:5.1f}%] conf={conf:.2f}  {label:3s}  '{text}'")

    print("-" * 70)
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

    # Decode and draw OCR results
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    annotated = draw_ocr_results(image)

    annotated_path = OUTPUT_DIR / "annotated_regions.png"
    cv2.imwrite(str(annotated_path), annotated)
    logger.info(f"Saved annotated image: {annotated_path}")

    # Run parsed OCR
    logger.info("Running parsed OCR...")
    reading = read_monument_popup(png_bytes)

    # Print results
    print("\n" + "=" * 60)
    print("PARSED RESULTS")
    print("=" * 60)
    print(f"  Ownership text:  '{reading.ownership_text}'")
    print(f"  Is friendly:     {reading.is_friendly}")
    print(f"  Action button:   '{reading.action_button_text}'")
    print(f"  Overall conf:    {reading.overall_confidence:.2f}")
    print(f"  Total power:     {reading.total_garrison_power:,}")
    print()
    for d in reading.defenders:
        print(f"  Defender {d.slot}: name='{d.name}', power={d.power:,}, "
              f"status={d.status}, conf={d.confidence:.2f}")
    print("=" * 60)

    await connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
