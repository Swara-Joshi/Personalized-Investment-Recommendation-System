"""Unit tests for key helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sentiment_pipeline.scripts.combine_signals import compute_combined_signal
from sentiment_pipeline.scripts.prepare_dataset import format_prompt


def test_format_prompt():
    headline = "Apple beats earnings expectations"
    sentiment = "positive"
    formatted = format_prompt(headline, sentiment)
    assert headline in formatted
    assert formatted.endswith("positive")


def test_compute_combined_signal_agreement():
    assert compute_combined_signal(price_direction=5.0, sentiment_score=1) == "strong_buy"
    assert compute_combined_signal(price_direction=-2.0, sentiment_score=-1) == "strong_sell"


def test_compute_combined_signal_watch():
    assert compute_combined_signal(price_direction=-1.0, sentiment_score=1) == "watch"

