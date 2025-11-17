"""Utilities for loading and cleaning textual sentiment datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas import DataFrame


def load_financial_news(csv_path: str | Path) -> DataFrame:
    """Load and clean traditional financial news data.

    Args:
        csv_path: Path to a CSV containing ``date``, ``source``, and ``headline`` columns.

    Returns:
        A cleaned ``DataFrame`` with parsed dates and non-empty headlines only.

    Raises:
        ValueError: If any required columns are missing.
    """

    df = _read_csv(csv_path, required_columns=("date", "source", "headline"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _finalize(df)


def load_reddit_data(csv_path: str | Path) -> DataFrame:
    """Load and clean Reddit-sourced financial discussions.

    Args:
        csv_path: Path to a CSV containing ``date``, ``source``, and ``title`` columns.

    Returns:
        A cleaned ``DataFrame`` with parsed dates and non-empty headlines only.

    Raises:
        ValueError: If any required columns are missing.
    """

    df = _read_csv(csv_path, required_columns=("date", "source", "title"))
    df = df.rename(columns={"title": "headline"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _finalize(df)


def _read_csv(csv_path: str | Path, required_columns: Sequence[str]) -> DataFrame:
    """Read a CSV file and ensure required columns exist."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    _ensure_columns(df, required_columns)
    return df


def _ensure_columns(df: DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that a dataframe contains each required column."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _finalize(df: DataFrame) -> DataFrame:
    """Drop empty headlines and return a cleaned view."""
    cleaned = df.dropna(subset=["headline"]).copy()
    cleaned["headline"] = cleaned["headline"].astype(str).str.strip()
    cleaned = cleaned[cleaned["headline"].ne("")]
    cleaned = cleaned.dropna(subset=["date"])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned

