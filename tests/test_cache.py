"""Tests for vision cache."""

import time
import pytest
from src.vision.cache import VisionCache


def _make_png_stub(content: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) -> bytes:
    return content


class TestVisionCache:
    def test_miss_then_hit(self):
        cache = VisionCache(ttl=10.0)
        png = _make_png_stub()

        assert cache.get(png, "test_prompt") is None
        assert cache.misses == 1

        cache.put(png, "test_prompt", '{"result": "ok"}')
        result = cache.get(png, "test_prompt")
        assert result == '{"result": "ok"}'
        assert cache.hits == 1

    def test_different_prompts_different_keys(self):
        cache = VisionCache(ttl=10.0)
        png = _make_png_stub()

        cache.put(png, "prompt_a", "response_a")
        cache.put(png, "prompt_b", "response_b")

        assert cache.get(png, "prompt_a") == "response_a"
        assert cache.get(png, "prompt_b") == "response_b"

    def test_ttl_expiry(self):
        cache = VisionCache(ttl=0.1)  # 100ms TTL
        png = _make_png_stub()

        cache.put(png, "test", "value")
        assert cache.get(png, "test") == "value"

        time.sleep(0.15)
        assert cache.get(png, "test") is None

    def test_max_entries_eviction(self):
        cache = VisionCache(ttl=60.0, max_entries=2)
        png1 = _make_png_stub(b"\x89PNG" + b"\x01" * 100)
        png2 = _make_png_stub(b"\x89PNG" + b"\x02" * 100)
        png3 = _make_png_stub(b"\x89PNG" + b"\x03" * 100)

        cache.put(png1, "p", "v1")
        cache.put(png2, "p", "v2")
        cache.put(png3, "p", "v3")

        # One of the first two should have been evicted
        assert len(cache._cache) <= 2

    def test_stats(self):
        cache = VisionCache(ttl=10.0)
        png = _make_png_stub()

        cache.get(png, "a")  # miss
        cache.put(png, "a", "val")
        cache.get(png, "a")  # hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_clear(self):
        cache = VisionCache()
        png = _make_png_stub()
        cache.put(png, "test", "value")
        cache.clear()
        assert cache.get(png, "test") is None
        assert cache.stats()["entries"] == 0
