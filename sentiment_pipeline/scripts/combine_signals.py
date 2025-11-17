"""Combine LSTM price forecasts with sentiment model outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def load_lstm_predictions(path: Path) -> pd.DataFrame:
    """Load LSTM price forecasts."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"ticker", "date", "predicted_price", "current_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LSTM predictions missing columns: {missing}")
    df["price_direction"] = df["predicted_price"] - df["current_price"]
    return df


def load_sentiment_predictions(path: Path) -> pd.DataFrame:
    """Load sentiment labels (latest per ticker/date)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["ticker", "date"], keep="last")
    return df


def compute_combined_signal(price_direction: float, sentiment_score: int) -> str:
    """Determine portfolio action."""
    if price_direction > 0 and sentiment_score > 0:
        return "strong_buy"
    if price_direction < 0 and sentiment_score < 0:
        return "strong_sell"
    if sentiment_score == 0 or abs(price_direction) < 1e-3:
        return "hold"
    return "watch"


def sentiment_to_score(sentiment: str) -> int:
    return {"positive": 1, "negative": -1}.get(sentiment, 0)


def merge_signals(lstm_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> List[Dict]:
    """Merge on ticker/date and compute combined outputs."""
    merged = pd.merge(lstm_df, sentiment_df, on=["ticker", "date"], how="inner", suffixes=("_lstm", "_sent"))
    results: List[Dict] = []

    for _, row in merged.iterrows():
        sentiment_score = sentiment_to_score(row["sentiment"])
        combined_signal = compute_combined_signal(row["price_direction"], sentiment_score)
        agreement = np.sign(row["price_direction"]) == np.sign(sentiment_score) if sentiment_score != 0 else False
        confidence = float(min(1.0, row.get("confidence", 0.5) + (0.3 if agreement else -0.1)))

        results.append(
            {
                "ticker": row["ticker"],
                "date": row["date"].strftime("%Y-%m-%d"),
                "price_prediction": row["predicted_price"],
                "sentiment": row["sentiment"],
                "combined_signal": combined_signal,
                "confidence": round(max(0.0, confidence), 3),
            }
        )

    return results


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
        lstm_path = Path(config["paths"]["lstm_predictions"])
        sentiment_path = Path(config["paths"]["labeled_news_csv"])
        output_path = Path(config["paths"]["combined_predictions"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lstm_df = load_lstm_predictions(lstm_path)
        sentiment_df = load_sentiment_predictions(sentiment_path)
        results = merge_signals(lstm_df, sentiment_df)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        logger.info("Combined %s signals → %s", len(results), output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to combine signals: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

