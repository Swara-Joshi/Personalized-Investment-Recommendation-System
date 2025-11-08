"""
Load Historical Price Data

This module loads historical price data from CSV files for model training.
It provides functionality to load individual stock data or combined datasets.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalPriceDataLoader:
    """
    Loads historical price data from CSV files for model training.
    
    Supports loading:
    - Individual stock/ETF/crypto price files
    - Combined price data
    - Processed data from Phase 1
    """
    
    def __init__(self, data_folder: Optional[Union[str, Path]] = None):
        """
        Initialize the Historical Price Data Loader
        
        Args:
            data_folder: Path to folder containing CSV files. 
                        Defaults to 'data/raw' for raw data or 'data/processed' for processed data.
        """
        if data_folder is None:
            # Default to project root's data/raw folder
            project_root = Path(__file__).parent.parent
            data_folder = project_root / "data" / "raw"
        
        self.data_folder = Path(data_folder)
        
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {self.data_folder}")
        
        logger.info(f"Initialized HistoricalPriceDataLoader with folder: {self.data_folder}")
    
    def list_available_files(self) -> List[str]:
        """
        List all CSV files in the data folder
        
        Returns:
            List of CSV filenames
        """
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_folder}")
        return csv_files
    
    def load_single_stock(self, ticker: str, filename_prefix: str = "historical_prices") -> pd.DataFrame:
        """
        Load historical price data for a single stock/ETF/crypto
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'BTC-USD')
            filename_prefix: Prefix of the CSV filename (default: 'historical_prices')
            
        Returns:
            DataFrame with historical price data
        """
        filename = f"{filename_prefix}_{ticker}.csv"
        filepath = self.data_folder / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        logger.info(f"Loaded {len(df)} records for {ticker}")
        return df
    
    def load_multiple_stocks(
        self,
        tickers: List[str],
        filename_prefix: str = "historical_prices"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical price data for multiple stocks/ETFs/crypto
        
        Args:
            tickers: List of stock ticker symbols
            filename_prefix: Prefix of the CSV filename (default: 'historical_prices')
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        price_data = {}
        
        for ticker in tickers:
            try:
                df = self.load_single_stock(ticker, filename_prefix)
                price_data[ticker] = df
            except FileNotFoundError as e:
                logger.warning(f"Skipping {ticker}: {str(e)}")
                continue
        
        logger.info(f"Loaded data for {len(price_data)} tickers")
        return price_data
    
    def load_all_stocks(self, filename_prefix: str = "historical_prices") -> Dict[str, pd.DataFrame]:
        """
        Load all historical price CSV files from the data folder
        
        Args:
            filename_prefix: Prefix to filter CSV files (default: 'historical_prices')
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        csv_files = [
            f for f in os.listdir(self.data_folder) 
            if f.endswith('.csv') and f.startswith(filename_prefix) and not f.endswith('_combined.csv')
        ]
        
        price_data = {}
        
        for file in csv_files:
            # Extract ticker name from filename
            # Format: historical_prices_TICKER.csv
            ticker = file.replace(f"{filename_prefix}_", "").replace(".csv", "")
            
            try:
                filepath = self.data_folder / file
                df = pd.read_csv(filepath)
                
                # Standardize columns
                df = self._standardize_columns(df)
                
                price_data[ticker] = df
                logger.info(f"Loaded {len(df)} records for {ticker}")
            except Exception as e:
                logger.warning(f"Error loading {file}: {str(e)}")
                continue
        
        logger.info(f"Loaded data for {len(price_data)} tickers")
        return price_data
    
    def load_combined_data(self, filename: str = "historical_prices_combined.csv") -> pd.DataFrame:
        """
        Load combined historical price data from a single CSV file
        
        Args:
            filename: Name of the combined CSV file
            
        Returns:
            DataFrame with combined historical price data
        """
        filepath = self.data_folder / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading combined data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        logger.info(f"Loaded {len(df)} records from combined file")
        return df
    
    def create_combined_dataframe(
        self,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Combine multiple ticker DataFrames into one DataFrame with MultiIndex
        
        Args:
            price_data: Dictionary of ticker -> DataFrame (if None, will load all stocks)
            tickers: List of tickers to load (if price_data is None)
            
        Returns:
            Combined DataFrame with MultiIndex (Ticker, Row)
        """
        if price_data is None:
            if tickers is not None:
                price_data = self.load_multiple_stocks(tickers)
            else:
                price_data = self.load_all_stocks()
        
        if not price_data:
            raise ValueError("No price data available to combine")
        
        # Combine DataFrames with MultiIndex
        combined_df = pd.concat(
            price_data.values(),
            keys=price_data.keys(),
            names=['Ticker', 'Row']
        )
        
        logger.info(f"Created combined DataFrame with {len(combined_df)} records from {len(price_data)} tickers")
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and keep only necessary columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        # Expected columns
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for Date column (might be named differently)
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df = df.rename(columns={date_cols[0]: 'Date'})
        
        # Ensure Date column exists
        if 'Date' not in df.columns:
            raise ValueError(f"'Date' column missing. Available columns: {list(df.columns)}")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Keep only necessary columns that exist
        available_cols = [col for col in expected_cols if col in df.columns]
        
        # Also keep Symbol column if it exists (for combined data)
        if 'Symbol' in df.columns:
            available_cols.append('Symbol')
        
        df = df[available_cols]
        
        # Sort by Date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def get_data_summary(self, df: Optional[pd.DataFrame] = None, ticker: Optional[str] = None) -> Dict:
        """
        Get summary statistics for the price data
        
        Args:
            df: DataFrame to summarize (if None, will load for ticker)
            ticker: Ticker symbol to load if df is None
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            if ticker is None:
                raise ValueError("Either df or ticker must be provided")
            df = self.load_single_stock(ticker)
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'columns': list(df.columns),
            'price_stats': {}
        }
        
        # Calculate price statistics if Close column exists
        if 'Close' in df.columns:
            summary['price_stats'] = {
                'min': df['Close'].min(),
                'max': df['Close'].max(),
                'mean': df['Close'].mean(),
                'std': df['Close'].std()
            }
        
        return summary

