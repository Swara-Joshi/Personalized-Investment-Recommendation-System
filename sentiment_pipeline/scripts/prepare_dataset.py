"""Prepare labeled sentiment data for LLM fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def format_prompt(headline: str, sentiment: str) -> str:
    """Format text prompt for supervised fine-tuning."""
    return f"Headline: {headline}\nSentiment: {sentiment}"


def save_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save sentiment distribution."""
    counts = df["sentiment"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#2ecc71", "#e74c3c", "#95a5a6"])
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Entry point."""
    try:
        config = load_config()
        labeled_path = Path(config["paths"]["labeled_news_csv"])
        dataset_dir = Path(config["paths"]["hf_dataset_dir"])
        distribution_plot = Path("sentiment_pipeline/results/sentiment_distribution.png")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        confidence_cutoff = config["labeling"]["high_quality_confidence"]

        if not labeled_path.exists():
            raise FileNotFoundError(f"Labeled dataset not found: {labeled_path}")

        df = pd.read_csv(labeled_path)
        df = df[df["confidence"] >= confidence_cutoff].reset_index(drop=True)
        if df.empty:
            raise ValueError("No rows met high-quality confidence threshold.")

        df["text"] = df.apply(lambda row: format_prompt(row["headline"], row["sentiment"]), axis=1)
        save_distribution_plot(df, distribution_plot)

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])

        train_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
        val_ds = Dataset.from_pandas(val_df[["text"]], preserve_index=False)

        train_path = dataset_dir / "train"
        val_path = dataset_dir / "validation"
        train_ds.save_to_disk(train_path.as_posix())
        val_ds.save_to_disk(val_path.as_posix())

        logger.info("Prepared dataset: %s train rows, %s validation rows", len(train_ds), len(val_ds))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dataset preparation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

