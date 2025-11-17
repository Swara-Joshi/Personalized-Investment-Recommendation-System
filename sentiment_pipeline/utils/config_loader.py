"""Helpers for loading and accessing YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@lru_cache(maxsize=1)
def load_config(path: str | Path = "sentiment_pipeline/config.yaml") -> Dict[str, Any]:
    """Load configuration YAML and memoize it."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping at the root.")
    return data

