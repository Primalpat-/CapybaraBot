"""Image-hash cache to skip duplicate Vision API calls."""

import hashlib
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    response_text: str
    timestamp: float
    prompt_name: str


class VisionCache:
    """Caches Vision API responses keyed by perceptual image hash + prompt.

    Uses a simple average-hash approach: resize to 8x8 grayscale, threshold
    at mean brightness. Two images with the same hash are visually similar
    enough to reuse the previous API result.
    """

    def __init__(self, ttl: float = 30.0, max_entries: int = 100):
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def _image_hash(self, png_bytes: bytes) -> str:
        """Compute a perceptual hash of the image.

        Falls back to SHA-256 if PIL is unavailable or the image is too small.
        """
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((8, 8))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return hex(int(bits, 2))
        except Exception:
            return hashlib.sha256(png_bytes).hexdigest()[:16]

    def _cache_key(self, png_bytes: bytes, prompt_name: str) -> str:
        return f"{self._image_hash(png_bytes)}:{prompt_name}"

    def get(self, png_bytes: bytes, prompt_name: str) -> str | None:
        """Look up a cached response. Returns None on miss."""
        key = self._cache_key(png_bytes, prompt_name)
        entry = self._cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        if time.time() - entry.timestamp > self.ttl:
            del self._cache[key]
            self.misses += 1
            return None

        self.hits += 1
        logger.debug(f"Cache hit for {prompt_name} (key={key[:20]}...)")
        return entry.response_text

    def put(self, png_bytes: bytes, prompt_name: str, response_text: str) -> None:
        """Store a response in the cache."""
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        key = self._cache_key(png_bytes, prompt_name)
        self._cache[key] = CacheEntry(
            response_text=response_text,
            timestamp=time.time(),
            prompt_name=prompt_name,
        )

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses),
        }
