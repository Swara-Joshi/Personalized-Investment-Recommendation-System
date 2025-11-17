"""Auto-label Yahoo Finance news with FinBERT and price confirmation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.finance import PriceChange, get_price_change
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def load_finbert(model_name: str) -> TextClassificationPipeline:
    """Load FinBERT sentiment pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, return_all_scores=False)


def sentiment_matches_price(sentiment: str, price_change: PriceChange, threshold: float) -> bool:
    """Validate FinBERT sentiment against realized price movement."""
    pct = price_change.pct_change
    if sentiment == "positive":
        return pct >= threshold
    if sentiment == "negative":
        return pct <= -threshold
    return abs(pct) < threshold


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
        raw_path = Path(config["paths"]["raw_news_csv"])
        labeled_path = Path(config["paths"]["labeled_news_csv"])
        labeled_path.parent.mkdir(parents=True, exist_ok=True)

        finbert_model = config["labeling"]["finbert_model"]
        min_conf = config["labeling"]["min_confidence"]
        price_threshold = config["labeling"]["price_confirm_threshold"]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw news not found: {raw_path}")

        df = pd.read_csv(raw_path)
        if df.empty:
            raise ValueError("Raw dataset is empty.")

        clf = load_finbert(finbert_model)
        records: List[dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling news"):
            headline = row["headline"]
            ticker = row["ticker"]
            source = row.get("source", "Yahoo Finance")
            date_str = row["date"]

            try:
                parsed_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                parsed_date = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")

            finbert_result = clf(headline)[0]
            sentiment = finbert_result["label"].lower()
            confidence = float(finbert_result["score"])
            if confidence < min_conf:
                continue

            price_change = get_price_change(ticker, parsed_date)
            if not price_change or not sentiment_matches_price(sentiment, price_change, price_threshold):
                continue

            records.append(
                {
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "headline": headline,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "price_change": round(price_change.pct_change * 100, 4),
                    "source": source,
                }
            )

        if not records:
            raise RuntimeError("No headlines met hybrid labeling criteria.")

        labeled_df = pd.DataFrame(records)
        labeled_df.to_csv(labeled_path, index=False)
        logger.info("Saved %s labeled headlines → %s", len(labeled_df), labeled_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Labeling failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

