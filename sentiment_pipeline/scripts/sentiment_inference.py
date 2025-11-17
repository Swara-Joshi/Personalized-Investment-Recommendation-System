"""Inference utilities for the fine-tuned sentiment model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from sentiment_pipeline.utils.config_loader import load_config
from sentiment_pipeline.utils.logging_utils import configure_logger

logger = configure_logger(__name__)


def load_sentiment_model(config: dict) -> tuple[PeftModel, AutoTokenizer]:
    """Load tokenizer + LoRA-adapted model in 4-bit."""
    training_cfg = config["training"]
    adapter_dir = Path(config["paths"]["sentiment_model_dir"])
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Sentiment adapter not found: {adapter_dir}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(training_cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        training_cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def predict_sentiment(headline: str, model: PeftModel, tokenizer: AutoTokenizer) -> Dict[str, float | str]:
    """Predict sentiment label and confidence from a headline."""
    prompt = f"Headline: {headline}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    sentiment_tokens = {
        "positive": tokenizer.encode(" positive", add_special_tokens=False)[0],
        "negative": tokenizer.encode(" negative", add_special_tokens=False)[0],
        "neutral": tokenizer.encode(" neutral", add_special_tokens=False)[0],
    }

    sentiment_scores = {label: float(probs[0, token_id].item()) for label, token_id in sentiment_tokens.items()}
    sentiment = max(sentiment_scores, key=sentiment_scores.get)

    return {"headline": headline, "sentiment": sentiment, "confidence": sentiment_scores[sentiment]}


def batch_predict(headlines: List[str], model: PeftModel, tokenizer: AutoTokenizer) -> List[Dict[str, float | str]]:
    """Run predictions over multiple headlines."""
    return [predict_sentiment(headline, model, tokenizer) for headline in headlines]


def main() -> None:
    """Example CLI usage."""
    try:
        config = load_config()
        model, tokenizer = load_sentiment_model(config)
        examples = [
            "Tesla shares jump after record deliveries",
            "Apple under investigation for antitrust violations",
            "S&P 500 flat as investors await Fed minutes",
        ]
        results = batch_predict(examples, model, tokenizer)
        print(json.dumps(results, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

