"""Daily automation for refreshing sentiment data and retraining."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import schedule

from sentiment_pipeline.scripts.data_collection import fetch_news_for_ticker
from sentiment_pipeline.scripts.sentiment_inference import batch_predict, load_sentiment_model
from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.finance import get_price_change
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def collect_recent_news(tickers: List[str], lookback_days: int) -> List[dict]:
    """Collect news for each ticker limited by lookback."""
    rows: List[dict] = []
    for ticker in tickers:
        rows.extend(fetch_news_for_ticker(ticker, lookback_days))
    return rows


def label_rows(rows: List[dict], config: dict) -> pd.DataFrame:
    """Label rows using the fine-tuned sentiment model."""
    if not rows:
        return pd.DataFrame()

    model, tokenizer = load_sentiment_model(config)
    predictions = batch_predict([row["headline"] for row in rows], model, tokenizer)

    enriched = []
    for row, pred in zip(rows, predictions, strict=False):
        try:
            parsed_date = datetime.fromisoformat(row["date"])
        except ValueError:
            parsed_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")

        price_change = get_price_change(row["ticker"], parsed_date)
        enriched.append(
            {
                "date": parsed_date.strftime("%Y-%m-%d"),
                "ticker": row["ticker"],
                "headline": row["headline"],
                "sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
                "price_change": price_change.pct_change * 100 if price_change else None,
                "source": row.get("source", "Yahoo Finance"),
            }
        )

    return pd.DataFrame(enriched)


def append_dataset(df: pd.DataFrame, labeled_path: Path) -> None:
    """Append new labeled rows to dataset."""
    if df.empty:
        logger.info("No new rows to append.")
        return

    if labeled_path.exists():
        existing = pd.read_csv(labeled_path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.drop_duplicates(subset=["date", "ticker", "headline"], keep="last", inplace=True)
    combined.to_csv(labeled_path, index=False)
    logger.info("Appended %s rows → %s", len(df), labeled_path)


def maybe_retrain(config: dict) -> None:
    """Trigger dataset prep (and optionally fine-tune) on scheduled weekday."""
    retrain_weekday = config["daily_update"]["retrain_weekday"]
    today = datetime.now(timezone.utc).strftime("%A")
    if today != retrain_weekday:
        return

    logger.info("Weekly retrain triggered; refreshing dataset.")
    prepare_script = Path("sentiment_pipeline/scripts/prepare_dataset.py")
    if prepare_script.exists():
        import subprocess

        subprocess.run([sys.executable, prepare_script.as_posix()], check=False)
        logger.info("Dataset refreshed; re-run notebook to fine-tune with new data.")
    else:
        logger.warning("Prepare script not found; skipping retrain trigger.")


def job() -> None:
    """Scheduled job logic."""
    try:
        config = load_config()
        lookback = config["daily_update"]["lookback_days"]
        tickers = config["tickers"]
        labeled_path = Path(config["paths"]["labeled_news_csv"])

        rows = collect_recent_news(tickers, lookback)
        logger.info("Collected %s new headlines", len(rows))
        labeled_df = label_rows(rows, config)
        append_dataset(labeled_df, labeled_path)
        maybe_retrain(config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Daily update failed: %s", exc)


def main() -> None:
    """Schedule the job based on config."""
    try:
        config = load_config()
        run_time = config["daily_update"]["run_time_est"]
        schedule.every().day.at(run_time).do(job)
        logger.info("Daily update scheduled at %s EST.", run_time)

        job()  # Run once immediately
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Shutting down daily update scheduler.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scheduler exited: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

