"""
Example usage of data collection and preprocessing modules

This script demonstrates how to use individual modules for data collection and preprocessing
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.historical_prices import HistoricalPriceCollector
from src.data_collection.market_sentiment import SentimentCollector
from src.data_collection.user_profiles import UserProfileGenerator
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, STOCK_SYMBOLS, API_KEYS


def example_collect_prices():
    """Example: Collect historical prices for a few symbols"""
    print("Example: Collecting historical prices...")
    
    collector = HistoricalPriceCollector(RAW_DATA_DIR)
    
    # Collect data for a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = collector.collect_and_save(
        symbols=symbols,
        period="1y",  # 1 year of data
        interval="1d",  # Daily interval
        filename_prefix="example_prices"
    )
    
    print(f"Collected data for {len(data)} symbols")
    return data


def example_collect_sentiment():
    """Example: Collect market sentiment"""
    print("Example: Collecting market sentiment...")
    
    collector = SentimentCollector(
        output_dir=RAW_DATA_DIR,
        reddit_client_id=API_KEYS.get("reddit_client_id"),
        reddit_client_secret=API_KEYS.get("reddit_client_secret"),
        reddit_user_agent=API_KEYS.get("reddit_user_agent")
    )
    
    # Collect from a few subreddits
    sentiment_df = collector.collect_and_save(
        subreddits=["investing", "stocks"],
        rss_feeds=["https://feeds.finance.yahoo.com/rss/2.0/headline"],
        reddit_limit=10,  # Small limit for example
        filename="example_sentiment"
    )
    
    print(f"Collected {len(sentiment_df)} sentiment records")
    return sentiment_df


def example_generate_profiles():
    """Example: Generate a few user profiles"""
    print("Example: Generating user profiles...")
    
    generator = UserProfileGenerator(RAW_DATA_DIR)
    
    # Generate a small number of profiles
    profiles_df = generator.generate_profiles(
        num_profiles=10,
        age_range=(25, 65),
        income_range=(30000, 200000),
        horizon_range=(1, 30),
        risk_preferences=["conservative", "moderate", "aggressive"]
    )
    
    generator.save_profiles(profiles_df, filename="example_profiles")
    
    print(f"Generated {len(profiles_df)} user profiles")
    print("\nSample profile:")
    print(profiles_df.head(1).to_string())
    return profiles_df


if __name__ == "__main__":
    print("=" * 60)
    print("Data Collection Examples")
    print("=" * 60)
    
    # Run examples
    # Uncomment the examples you want to run
    
    # example_collect_prices()
    # example_collect_sentiment()
    # example_generate_profiles()
    
    print("\nTo run examples, uncomment the function calls in the main block")

