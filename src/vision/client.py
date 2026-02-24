"""Claude Vision API wrapper with token tracking."""

import logging
import time
from dataclasses import dataclass, field

import anthropic

from src.utils.image_utils import (
    png_bytes_to_pil,
    resize_for_api,
    image_to_base64,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    total_cost: float = 0.0


@dataclass
class VisionResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float


class VisionClient:
    """Sends screenshots to Claude Vision API and tracks usage."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        max_image_dimension: int = 1024,
        temperature: float = 0.0,
        input_token_cost_per_million: float = 3.0,
        output_token_cost_per_million: float = 15.0,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_image_dimension = max_image_dimension
        self.temperature = temperature
        self.input_cost_rate = input_token_cost_per_million / 1_000_000
        self.output_cost_rate = output_token_cost_per_million / 1_000_000
        self.usage = TokenUsage()

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_cost_rate) + (output_tokens * self.output_cost_rate)

    def analyze_screenshot(
        self, png_bytes: bytes, prompt: str, system: str = ""
    ) -> VisionResponse:
        """Send a screenshot to Claude Vision and get a text response.

        Args:
            png_bytes: Raw PNG screenshot bytes.
            prompt: The text prompt to send alongside the image.
            system: Optional system prompt.

        Returns:
            VisionResponse with the text, token counts, and cost.
        """
        # Resize image for cost efficiency
        image = png_bytes_to_pil(png_bytes)
        image = resize_for_api(image, self.max_image_dimension)
        b64_data = image_to_base64(image, format="PNG")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        start = time.time()

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        latency = time.time() - start

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        # Update cumulative usage
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.total_calls += 1
        self.usage.total_cost += cost

        text = response.content[0].text

        logger.info(
            f"Vision API call: {input_tokens} in / {output_tokens} out, "
            f"${cost:.4f}, {latency:.2f}s"
        )

        return VisionResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
        )

    def get_usage_summary(self) -> dict:
        """Return cumulative usage stats."""
        return {
            "total_calls": self.usage.total_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_cost": round(self.usage.total_cost, 4),
        }
