"""Prompt templates loaded from config/prompts.yaml."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_prompts: dict | None = None


def _load_prompts() -> dict:
    global _prompts
    if _prompts is None:
        path = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"
        with open(path, "r", encoding="utf-8") as f:
            _prompts = yaml.safe_load(f)
        logger.debug(f"Loaded {len(_prompts)} prompt templates from {path}")
    return _prompts


def get_prompt(name: str) -> tuple[str, str]:
    """Get (system, prompt) tuple for a given prompt template name.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    prompts = _load_prompts()
    if name not in prompts:
        raise KeyError(f"Unknown prompt template: {name}")
    entry = prompts[name]
    return entry.get("system", ""), entry["prompt"]


def get_all_prompt_names() -> list[str]:
    """List all available prompt template names."""
    return list(_load_prompts().keys())
