"""Collect Yahoo Finance news headlines for multiple tickers."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import feedparser

from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def fetch_news_for_ticker(ticker: str, lookback_days: int) -> List[dict]:
    """Fetch Yahoo Finance news entries for a single ticker."""
    rows: List[dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Try yfinance first, fall back to RSS if it fails
    news_items = []
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance failed for %s, using RSS fallback: %s", ticker, exc)
        news_items = []

    if not news_items:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            published_tuple = entry.get("published_parsed")
            if not published_tuple:
                continue
            published = datetime(*published_tuple[:6], tzinfo=timezone.utc)
            if published < cutoff:
                continue

            rows.append(
                {
                    "date": published.strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "headline": entry.get("title", "").strip(),
                    "source": entry.get("source", "Yahoo Finance"),
                    "url": entry.get("link", ""),
                }
            )
        return rows

    for item in news_items:
        content = item.get("content", {})
        title = (item.get("title") or content.get("title") or "").strip()
        if not title:
            continue

        pub_date_str = content.get("pubDate")
        if pub_date_str:
            published = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        else:
            published = datetime.fromtimestamp(item.get("providerPublishTime", 0), tz=timezone.utc)

        if published < cutoff:
            continue

        provider = (
            item.get("publisher")
            or (item.get("provider") or {}).get("displayName")
            or (content.get("provider") or {}).get("displayName")
            or "Yahoo Finance"
        )
        url = (
            (item.get("link") or "")
            or ((content.get("canonicalUrl") or {}).get("url") or "")
            or ((content.get("clickThroughUrl") or {}).get("url") or "")
        )

        rows.append(
            {
                "date": published.strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "headline": title,
                "source": provider,
                "url": url,
            }
        )

    return rows


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
        tickers = config["tickers"]
        lookback_days = config["collection"]["lookback_days"]
        max_headlines = config["collection"]["max_headlines_per_ticker"]
        output_path = Path(config["paths"]["raw_news_csv"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_rows: List[dict] = []

        for ticker in tqdm(tickers, desc="Fetching Yahoo Finance news"):
            logger.info("Fetching headlines for %s", ticker)
            try:
                rows = fetch_news_for_ticker(ticker, lookback_days)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to fetch %s: %s", ticker, exc)
                continue

            if max_headlines:
                rows = rows[:max_headlines]
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No headlines collected; nothing to save.")
            return

        df = pd.DataFrame(all_rows)
        if "headline" not in df.columns:
            logger.warning("Headline column missing; raw rows: %s", len(all_rows))
            return

        df = df.dropna(subset=["headline"])
        df = df[df["headline"].str.strip().ne("")]

        df.to_csv(output_path, index=False)
        logger.info("Saved %s headlines → %s", len(df), output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data collection failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

