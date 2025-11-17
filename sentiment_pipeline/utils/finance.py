"""Financial helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


_PRICE_CACHE: Dict[str, pd.DataFrame] = {}


@dataclass
class PriceChange:
    """Represents price movement statistics."""

    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def pct_change(self) -> float:
        if self.before == 0:
            return 0.0
        return self.delta / self.before


def get_price_change(ticker: str, date: datetime, lookahead_days: int = 1) -> Optional[PriceChange]:
    """Fetch closing prices around a date and compute change."""
    history = _ensure_history(ticker, date, lookahead_days)
    if history.empty:
        return None

    before = history.iloc[0]["Close"]
    after = history.iloc[-1]["Close"]
    if pd.isna(before) or pd.isna(after):
        return None

    return PriceChange(before=float(before), after=float(after))


def _ensure_history(ticker: str, date: datetime, lookahead_days: int) -> pd.DataFrame:
    start = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    cache_key = f"{ticker}:{start}:{end}"

    if cache_key not in _PRICE_CACHE:
        ticker_obj = yf.Ticker(ticker)
        history = ticker_obj.history(start=start, end=end, auto_adjust=True)
        _PRICE_CACHE[cache_key] = history

    return _PRICE_CACHE[cache_key]

