"""Diagnostic tool: capture screenshot, run Vision calibration, save annotated image.

Run with:  python -m scripts.diagnose_calibration

This will:
1. Connect to ADB and capture a screenshot
2. Log the screenshot dimensions vs wm size
3. Send the screenshot to Vision for element calibration
4. Save an annotated screenshot with markers where Vision found elements
5. Print all coordinates so you can verify visually
"""

import asyncio
import base64
import io
import json
import logging
import re
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adb.connection import ADBConnection
from src.adb.capture import ScreenCapture
from src.vision.client import VisionClient
from src.vision.parser import parse_calibration_result
from src.vision.prompts import get_prompt
from src.bot.calibration import ELEMENT_DESCRIPTIONS, SCREEN_ELEMENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "screenshots" / "diagnostics"


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def get_wm_size(connection: ADBConnection) -> tuple[int, int] | None:
    try:
        stdout, _, rc = await connection.run_adb("shell", "wm", "size")
        if rc != 0:
            return None
        for line in reversed(stdout.strip().splitlines()):
            match = re.search(r"(\d+)x(\d+)", line)
            if match:
                return int(match.group(1)), int(match.group(2))
    except Exception as e:
        logger.error(f"wm size failed: {e}")
    return None


def draw_markers(image: Image.Image, elements: list, wm_w: int, wm_h: int) -> Image.Image:
    """Draw crosshair markers on the image at the reported percentage positions."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "blue"]

    for i, el in enumerate(elements):
        color = colors[i % len(colors)]

        # Position on the actual image (percentages of image dimensions)
        ix = int(el.x_percent / 100 * img_w)
        iy = int(el.y_percent / 100 * img_h)

        # Position for input tap (percentages of wm size)
        tap_x = int(el.x_percent / 100 * wm_w)
        tap_y = int(el.y_percent / 100 * wm_h)

        # Draw crosshair
        arm = 30
        draw.line([(ix - arm, iy), (ix + arm, iy)], fill=color, width=3)
        draw.line([(ix, iy - arm), (ix, iy + arm)], fill=color, width=3)

        # Draw circle
        r = 15
        draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], outline=color, width=2)

        # Label
        label = f"{el.name}\n({el.x_percent:.1f}%, {el.y_percent:.1f}%)\ntap=({tap_x}, {tap_y})\nconf={el.confidence:.2f}"
        draw.text((ix + arm + 5, iy - 20), label, fill=color)

    return img


async def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adb_cfg = config.get("adb", {})
    connection = ADBConnection(
        host=adb_cfg.get("host", "127.0.0.1"),
        port=adb_cfg.get("port", 5555),
        adb_path=adb_cfg.get("adb_path", "adb"),
        connect_timeout=adb_cfg.get("connect_timeout", 10),
        command_timeout=adb_cfg.get("command_timeout", 30),
    )

    logger.info("Connecting to ADB...")
    if not await connection.connect_with_retry():
        logger.error("Failed to connect to ADB")
        return

    # Get wm size
    wm_dims = await get_wm_size(connection)
    logger.info(f"wm size: {wm_dims}")

    # Capture screenshot
    capture = ScreenCapture(connection)
    png_bytes = await capture.capture()

    # Get actual image dimensions
    image = Image.open(io.BytesIO(png_bytes))
    img_w, img_h = image.size
    logger.info(f"Screenshot dimensions: {img_w}x{img_h}")

    if wm_dims:
        wm_w, wm_h = wm_dims
        if (img_w, img_h) != (wm_w, wm_h):
            logger.warning(f"MISMATCH: screenshot={img_w}x{img_h}  wm_size={wm_w}x{wm_h}")
            logger.warning(f"  Scale factor: x={img_w/wm_w:.4f}  y={img_h/wm_h:.4f}")
        else:
            logger.info("Screenshot dims match wm size — OK")
    else:
        wm_w, wm_h = img_w, img_h
        logger.warning("Could not get wm size, using screenshot dims")

    # Save raw screenshot
    raw_path = OUTPUT_DIR / "raw_screenshot.png"
    raw_path.write_bytes(png_bytes)
    logger.info(f"Saved raw screenshot: {raw_path}")

    # Detect screen type first
    vision_cfg = config.get("vision", {})
    vision = VisionClient(
        model=vision_cfg.get("model", "claude-sonnet-4-6"),
        max_tokens=vision_cfg.get("max_tokens", 1024),
        max_image_dimension=0,  # NO resize for diagnostics
        temperature=0.0,
    )

    logger.info("Identifying screen type...")
    from src.vision.parser import parse_screen_identification
    system_id, prompt_id = get_prompt("identify_screen")
    resp = vision.analyze_screenshot(png_bytes, prompt_id, system_id)
    screen = parse_screen_identification(resp.text)
    logger.info(f"Screen type: {screen.screen_type} (confidence={screen.confidence:.2f})")
    logger.info(f"Details: {screen.details}")

    # Run calibration for detected screen type
    screen_type = screen.screen_type
    if screen_type not in SCREEN_ELEMENTS:
        screen_type = "main_map"
    elements_to_find = SCREEN_ELEMENTS.get(screen_type, [])

    if not elements_to_find:
        logger.info(f"No calibratable elements for screen type '{screen_type}'")
        await connection.disconnect()
        return

    logger.info(f"Calibrating elements for '{screen_type}': {elements_to_find}")

    descriptions = []
    for i, name in enumerate(elements_to_find, 1):
        desc = ELEMENT_DESCRIPTIONS.get(name, name)
        descriptions.append(f"{i}. **{name}**: {desc}")
    elements_description = "\n".join(descriptions)

    system, prompt_template = get_prompt("calibrate_elements")
    prompt = prompt_template.replace("{elements_description}", elements_description)

    logger.info("Sending to Vision API (full resolution, no resize)...")
    response = vision.analyze_screenshot(png_bytes, prompt, system)

    logger.info(f"Raw Vision response:\n{response.text}")

    result = parse_calibration_result(response.text)

    logger.info("\n=== CALIBRATION RESULTS ===")
    for el in result.elements:
        # Pixel position on the image
        img_x = int(el.x_percent / 100 * img_w)
        img_y = int(el.y_percent / 100 * img_h)
        # Pixel position for input tap (using wm size)
        tap_x = int(el.x_percent / 100 * wm_w)
        tap_y = int(el.y_percent / 100 * wm_h)
        logger.info(
            f"  {el.name}:\n"
            f"    Vision: ({el.x_percent:.1f}%, {el.y_percent:.1f}%)  conf={el.confidence:.2f}\n"
            f"    Image pixel: ({img_x}, {img_y}) / ({img_w}x{img_h})\n"
            f"    Tap pixel (wm): ({tap_x}, {tap_y}) / ({wm_w}x{wm_h})"
        )

    # Draw markers and save
    annotated = draw_markers(image, result.elements, wm_w, wm_h)
    annotated_path = OUTPUT_DIR / f"annotated_{screen_type}.png"
    annotated.save(annotated_path)
    logger.info(f"\nSaved annotated screenshot: {annotated_path}")
    logger.info("Open this image to verify if the markers are on the correct elements!")

    # Also try with RESIZED image (what the bot normally sends)
    logger.info("\n=== TESTING WITH RESIZED IMAGE (what the bot normally uses) ===")
    from src.utils.image_utils import resize_for_api, image_to_base64, png_bytes_to_pil
    resized = resize_for_api(png_bytes_to_pil(png_bytes), 1024)
    rw, rh = resized.size
    logger.info(f"Resized image: {rw}x{rh}")

    vision_resized = VisionClient(
        model=vision_cfg.get("model", "claude-sonnet-4-6"),
        max_tokens=vision_cfg.get("max_tokens", 1024),
        max_image_dimension=1024,
        temperature=0.0,
    )

    from src.utils.image_utils import pil_to_png_bytes
    resized_bytes = pil_to_png_bytes(resized)
    response2 = vision_resized.analyze_screenshot(resized_bytes, prompt, system)
    logger.info(f"Raw Vision response (resized):\n{response2.text}")

    result2 = parse_calibration_result(response2.text)

    for el in result2.elements:
        tap_x = int(el.x_percent / 100 * wm_w)
        tap_y = int(el.y_percent / 100 * wm_h)
        logger.info(
            f"  {el.name} (resized): ({el.x_percent:.1f}%, {el.y_percent:.1f}%)  "
            f"tap=({tap_x}, {tap_y})  conf={el.confidence:.2f}"
        )

    # Draw markers on resized image too
    annotated2 = draw_markers(resized, result2.elements, wm_w, wm_h)
    annotated2_path = OUTPUT_DIR / f"annotated_{screen_type}_resized.png"
    annotated2.save(annotated2_path)
    logger.info(f"Saved annotated (resized): {annotated2_path}")

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"  wm size:        {wm_w}x{wm_h}")
    logger.info(f"  Screenshot:     {img_w}x{img_h}")
    logger.info(f"  Resized for API: {rw}x{rh}")
    logger.info(f"  Screen type:    {screen.screen_type}")
    logger.info(f"  Files saved to: {OUTPUT_DIR}")
    logger.info("\n  Check the annotated images!")
    logger.info("  If crosshairs are ON the correct elements → coordinate mapping bug")
    logger.info("  If crosshairs are WRONG → Vision is misidentifying positions")

    await connection.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
