"""Image utilities: resize, crop, base64 encoding for Vision API."""

import base64
import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def resize_for_api(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """Resize image so its longest side is at most max_dimension pixels.

    Pass max_dimension=0 to skip resizing entirely.
    """
    if max_dimension <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_dimension:
        return image

    scale = max_dimension / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"Resized {w}x{h} → {new_w}x{new_h}")
    return resized


def crop_region(image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Crop a region from the image."""
    return image.crop((x, y, x + w, y + h))


def crop_percent(image: Image.Image, left: float, top: float,
                 right: float, bottom: float) -> Image.Image:
    """Crop using percentage coordinates (0-100)."""
    iw, ih = image.size
    return image.crop((
        int(iw * left / 100),
        int(ih * top / 100),
        int(iw * right / 100),
        int(ih * bottom / 100),
    ))


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def bytes_to_base64(png_bytes: bytes) -> str:
    """Encode raw PNG bytes to base64 string."""
    return base64.standard_b64encode(png_bytes).decode("utf-8")


def png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    """Convert raw PNG bytes to PIL Image."""
    return Image.open(io.BytesIO(png_bytes))


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
