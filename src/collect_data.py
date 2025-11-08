"""
Main Data Collection Script

This script orchestrates the collection of all data for Phase 1:
1. Historical stock/ETF/crypto prices
2. Market sentiment from news & social media
3. Simulated user profiles
4. Data preprocessing for each agent
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.historical_prices import HistoricalPriceCollector
from src.data_collection.market_sentiment import SentimentCollector
from src.data_collection.user_profiles import UserProfileGenerator
from src.data_preprocessing.market_agent_preprocessing import MarketAgentPreprocessor
from src.data_preprocessing.risk_agent_preprocessing import RiskAgentPreprocessor
from src.data_preprocessing.recommendation_agent_preprocessing import RecommendationAgentPreprocessor
from config.config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    STOCK_SYMBOLS, DATA_COLLECTION_CONFIG, API_KEYS
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_historical_prices():
    """Collect historical stock/ETF/crypto prices"""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting Historical Prices")
    logger.info("=" * 60)
    
    collector = HistoricalPriceCollector(RAW_DATA_DIR)
    
    config = DATA_COLLECTION_CONFIG["historical_prices"]
    symbols = STOCK_SYMBOLS
    
    data = collector.collect_and_save(
        symbols=symbols,
        period=config["period"],
        interval=config["interval"],
        filename_prefix="historical_prices"
    )
    
    logger.info(f"Collected historical prices for {len(data)} symbols")
    return data


def collect_market_sentiment():
    """Collect market sentiment from news and social media"""
    logger.info("=" * 60)
    logger.info("STEP 2: Collecting Market Sentiment")
    logger.info("=" * 60)
    
    config = DATA_COLLECTION_CONFIG["sentiment"]
    
    collector = SentimentCollector(
        output_dir=RAW_DATA_DIR,
        reddit_client_id=API_KEYS.get("reddit_client_id"),
        reddit_client_secret=API_KEYS.get("reddit_client_secret"),
        reddit_user_agent=API_KEYS.get("reddit_user_agent")
    )
    
    # Collect sentiment data
    sentiment_df = collector.collect_and_save(
        subreddits=config["reddit_subreddits"],
        rss_feeds=config["news_sources"],
        reddit_limit=config["reddit_limit"],
        filename="market_sentiment"
    )
    
    logger.info(f"Collected {len(sentiment_df)} sentiment records")
    return sentiment_df


def generate_user_profiles():
    """Generate simulated user profiles"""
    logger.info("=" * 60)
    logger.info("STEP 3: Generating User Profiles")
    logger.info("=" * 60)
    
    generator = UserProfileGenerator(RAW_DATA_DIR)
    
    config = DATA_COLLECTION_CONFIG["user_profiles"]
    
    profiles_df = generator.generate_profiles(
        num_profiles=config["num_profiles"],
        age_range=config["age_range"],
        income_range=config["income_range"],
        horizon_range=config["investment_horizon_range"],
        risk_preferences=config["risk_preferences"]
    )
    
    generator.save_profiles(profiles_df, filename="user_profiles")
    
    logger.info(f"Generated {len(profiles_df)} user profiles")
    return profiles_df


def preprocess_market_agent_data():
    """Preprocess data for Market Agent"""
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocessing Market Agent Data")
    logger.info("=" * 60)
    
    preprocessor = MarketAgentPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    processed_df = preprocessor.process_all(
        price_filename="historical_prices_combined.csv",
        sentiment_filename="market_sentiment.csv",
        output_filename="market_agent_processed.csv"
    )
    
    logger.info(f"Processed {len(processed_df)} records for Market Agent")
    return processed_df


def preprocess_risk_agent_data():
    """Preprocess data for Risk Agent"""
    logger.info("=" * 60)
    logger.info("STEP 5: Preprocessing Risk Agent Data")
    logger.info("=" * 60)
    
    preprocessor = RiskAgentPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    processed_df = preprocessor.process_all(
        input_filename="user_profiles.csv",
        output_filename="risk_agent_processed.csv"
    )
    
    logger.info(f"Processed {len(processed_df)} records for Risk Agent")
    return processed_df


def preprocess_recommendation_agent_data():
    """Preprocess data for Recommendation Agent"""
    logger.info("=" * 60)
    logger.info("STEP 6: Preprocessing Recommendation Agent Data")
    logger.info("=" * 60)
    
    preprocessor = RecommendationAgentPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    processed_df = preprocessor.process_all(
        user_profiles_filename="user_profiles.csv",
        price_filename="historical_prices_combined.csv",
        output_filename="recommendation_agent_processed.csv"
    )
    
    logger.info(f"Processed {len(processed_df)} records for Recommendation Agent")
    return processed_df


def main():
    """Main function to run all data collection and preprocessing steps"""
    logger.info("Starting Phase 1: Data Collection & Preprocessing")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")
    
    try:
        # Step 1: Collect historical prices
        historical_prices = collect_historical_prices()
        
        # Step 2: Collect market sentiment
        market_sentiment = collect_market_sentiment()
        
        # Step 3: Generate user profiles
        user_profiles = generate_user_profiles()
        
        # Step 4: Preprocess Market Agent data
        market_agent_data = preprocess_market_agent_data()
        
        # Step 5: Preprocess Risk Agent data
        risk_agent_data = preprocess_risk_agent_data()
        
        # Step 6: Preprocess Recommendation Agent data
        recommendation_agent_data = preprocess_recommendation_agent_data()
        
        logger.info("=" * 60)
        logger.info("Phase 1 Complete!")
        logger.info("=" * 60)
        logger.info(f"Historical prices: {len(historical_prices)} symbols")
        logger.info(f"Market sentiment: {len(market_sentiment)} records")
        logger.info(f"User profiles: {len(user_profiles)} profiles")
        logger.info(f"Market Agent data: {len(market_agent_data)} records")
        logger.info(f"Risk Agent data: {len(risk_agent_data)} records")
        logger.info(f"Recommendation Agent data: {len(recommendation_agent_data)} records")
        
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

