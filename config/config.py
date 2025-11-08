"""
Configuration file for the Personalized Investment Recommendation System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (load from environment variables)
API_KEYS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
    "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
    "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
    "reddit_user_agent": os.getenv("REDDIT_USER_AGENT", "InvestmentBot/1.0"),
}

# Stock symbols to collect data for
STOCK_SYMBOLS = [
    # Major stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
    # ETFs
    "SPY", "QQQ", "VTI", "VOO", "IWM", "DIA", "GLD", "TLT",
    # Crypto (as stocks via Yahoo Finance)
    "BTC-USD", "ETH-USD", "BNB-USD"
]

# Data collection parameters
DATA_COLLECTION_CONFIG = {
    "historical_prices": {
        "period": "5y",  # 5 years of historical data
        "interval": "1d",  # Daily interval
        "sources": ["yahoo_finance", "alpha_vantage"],
    },
    "sentiment": {
        "reddit_subreddits": ["wallstreetbets", "investing", "stocks", "SecurityAnalysis"],
        "reddit_limit": 100,
        "news_sources": [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.bloomberg.com/feed/topics",
        ],
    },
    "user_profiles": {
        "num_profiles": 1000,
        "age_range": (25, 65),
        "income_range": (30000, 200000),
        "investment_horizon_range": (1, 30),  # years
        "risk_preferences": ["conservative", "moderate", "aggressive"],
    },
}

