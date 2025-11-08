"""
Historical Price Data Collection Module

This module collects historical stock/ETF/crypto prices from:
- Yahoo Finance (primary)
- Alpha Vantage (backup)
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalPriceCollector:
    """
    Collects historical price data for stocks, ETFs, and cryptocurrencies
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the Historical Price Collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_yahoo_finance_data(
        self,
        symbols: List[str],
        period: str = "5y",
        interval: str = "1d",
        delay: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical price data from Yahoo Finance
        
        Args:
            symbols: List of stock/ETF/crypto symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            delay: Delay between API calls to avoid rate limiting
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        collected_data = {}
        failed_symbols = []
        
        logger.info(f"Collecting historical price data for {len(symbols)} symbols...")
        
        for symbol in tqdm(symbols, desc="Downloading price data"):
            try:
                # Download historical data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Add symbol column
                df['Symbol'] = symbol
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                
                # Rename columns for consistency
                df.columns = [col.replace(' ', '_') for col in df.columns]
                
                # Store collected data
                collected_data[symbol] = df
                
                logger.info(f"Successfully collected {len(df)} records for {symbol}")
                
                # Rate limiting
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to collect data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        return collected_data
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], filename_prefix: str = "historical_prices"):
        """
        Save raw historical price data to CSV files
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            filename_prefix: Prefix for output filenames
        """
        logger.info(f"Saving raw data for {len(data)} symbols...")
        
        # Save individual symbol files
        for symbol, df in data.items():
            filename = f"{filename_prefix}_{symbol}.csv"
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {symbol} data to {filepath}")
        
        # Save combined file
        if data:
            combined_df = pd.concat(data.values(), ignore_index=True)
            combined_filename = f"{filename_prefix}_combined.csv"
            combined_filepath = self.output_dir / combined_filename
            combined_df.to_csv(combined_filepath, index=False)
            logger.info(f"Saved combined data to {combined_filepath}")
    
    def collect_and_save(
        self,
        symbols: List[str],
        period: str = "5y",
        interval: str = "1d",
        filename_prefix: str = "historical_prices"
    ):
        """
        Collect and save historical price data in one step
        
        Args:
            symbols: List of stock/ETF/crypto symbols
            period: Time period for historical data
            interval: Data interval
            filename_prefix: Prefix for output filenames
        """
        data = self.collect_yahoo_finance_data(symbols, period, interval)
        self.save_raw_data(data, filename_prefix)
        return data

