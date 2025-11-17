"""Text preprocessing utilities for sentiment tasks."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]+")


def preprocess_text(text: str) -> str:
    """Normalize raw text for downstream sentiment or NLP tasks.

    Steps:
        1. Lowercase the text.
        2. Remove URLs.
        3. Remove special characters (keep alphanumerics and whitespace).
        4. Collapse multiple spaces into a single space.

    Args:
        text: Input string.

    Returns:
        Cleaned text with normalized whitespace.
    """

    lowered = text.lower()
    without_urls = URL_PATTERN.sub(" ", lowered)
    alphanumeric_only = NON_ALPHANUMERIC_PATTERN.sub(" ", without_urls)
    cleaned = re.sub(r"\s+", " ", alphanumeric_only).strip()
    return cleaned


if __name__ == "__main__":
    test_cases = {
        "lowercase": ("Hello WORLD!", "hello world"),
        "url removal": ("Check this https://example.com now", "check this now"),
        "special chars": ("Profit & Loss (P/L)!!!", "profit loss p l"),
        "extra spaces": ("  Multiple    spaced\twords\nhere ", "multiple spaced words here"),
    }

    for name, (raw, expected) in test_cases.items():
        result = preprocess_text(raw)
        assert result == expected, f"{name} failed: {result!r} != {expected!r}"
        print(f"{name}: ok")

