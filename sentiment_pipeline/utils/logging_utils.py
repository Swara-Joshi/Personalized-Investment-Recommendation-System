"""Centralized logging configuration."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logger(name: Optional[str] = None) -> logging.Logger:
    """Configure a logger with timestamped formatting."""
    logger = logging.getLogger(name if name else "sentiment_pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

