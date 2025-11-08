"""
Auto-loading Historical Price Data

This module provides an auto-loading version of HistoricalPriceDataLoader
that automatically loads all CSV files on initialization.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalPriceDataLoader:
    """
    Auto-loading version that loads all CSV files on initialization.
    Provides quick access to individual tickers or combined DataFrame.
    """
    
    def __init__(self, folder_path: Optional[str] = None):
        """
        Initialize the loader and automatically load all CSV files.
        
        Args:
            folder_path: Path to folder containing CSV files. 
                        Defaults to project's data/raw folder.
        """
        if folder_path is None:
            # Default to project root's data/raw folder
            project_root = Path(__file__).parent.parent
            folder_path = project_root / "data" / "raw"
        
        self.folder_path = Path(folder_path)
        
        if not self.folder_path.exists():
            raise ValueError(f"Folder path does not exist: {self.folder_path}")
        
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.combined_df: Optional[pd.DataFrame] = None
        
        # Automatically load all CSVs on initialization
        self._load_all_csvs()
    
    def _load_all_csvs(self):
        """
        Load all CSV files from the folder path automatically.
        """
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in {self.folder_path}")
        
        for file in csv_files:
            # Skip combined file and non-price files (like user_profiles.csv)
            if file == 'historical_prices_combined.csv' or not file.startswith('historical_prices_'):
                continue
            
            # Extract ticker from filename
            # Format: historical_prices_TICKER.csv
            ticker = file.replace('historical_prices_', '').replace('.csv', '')
            
            file_path = self.folder_path / file
            
            try:
                df = pd.read_csv(file_path)
                
                # Check if this is a price data file (must have Date and at least one price column)
                if 'Date' not in df.columns:
                    # Try to find date-like column
                    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if not date_cols:
                        logger.debug(f"Skipping {file}: No Date column found")
                        continue
                    # Rename the date column to 'Date'
                    df = df.rename(columns={date_cols[0]: 'Date'})
                
                # Check if this has price data columns
                price_cols = ['Open', 'High', 'Low', 'Close']
                if not any(col in df.columns for col in price_cols):
                    logger.debug(f"Skipping {file}: No price columns found (Open, High, Low, Close)")
                    continue
                
                # Standardize columns - keep only expected columns that exist
                expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in expected_cols if col in df.columns]
                
                df = df[available_cols]
                
                # Add Symbol/Ticker column if missing
                if 'Symbol' not in df.columns:
                    df['Symbol'] = ticker
                else:
                    # Ensure Symbol column is included in the dataframe
                    if 'Symbol' not in available_cols:
                        available_cols.append('Symbol')
                
                # Convert Date to datetime (handle UTC timezone)
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                
                # Sort by date
                df.sort_values('Date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                # Handle missing values (using forward fill, then backward fill)
                df = df.ffill().bfill()
                
                self.dataframes[ticker] = df
                logger.info(f"Loaded {len(df)} records for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                continue
        
        # Combine all tickers into one DataFrame with MultiIndex
        if self.dataframes:
            self.combined_df = pd.concat(
                self.dataframes.values(),
                keys=self.dataframes.keys(),
                names=['Ticker', 'Row']
            )
            logger.info(
                f"Created combined DataFrame with {len(self.combined_df)} records "
                f"from {len(self.dataframes)} tickers"
            )
        else:
            logger.warning("No dataframes loaded!")
    
    def get_dataframe(self, ticker: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame for a specific ticker or the combined DataFrame.
        
        Args:
            ticker: Ticker symbol to get data for. If None, returns combined DataFrame.
            
        Returns:
            DataFrame for the specified ticker, or combined DataFrame if ticker is None.
        """
        if ticker:
            return self.dataframes.get(ticker)
        return self.combined_df
    
    def get_tickers(self) -> list:
        """
        Get list of available ticker symbols.
        
        Returns:
            List of ticker symbols that were loaded.
        """
        return list(self.dataframes.keys())
    
    def get_summary(self) -> Dict:
        """
        Get summary of loaded data.
        
        Returns:
            Dictionary with summary information.
        """
        summary = {
            'num_tickers': len(self.dataframes),
            'tickers': list(self.dataframes.keys()),
            'total_records': len(self.combined_df) if self.combined_df is not None else 0,
            'ticker_details': {}
        }
        
        for ticker, df in self.dataframes.items():
            summary['ticker_details'][ticker] = {
                'records': len(df),
                'date_range': {
                    'start': str(df['Date'].min()),
                    'end': str(df['Date'].max())
                }
            }
            if 'Close' in df.columns:
                summary['ticker_details'][ticker]['price_range'] = {
                    'min': float(df['Close'].min()),
                    'max': float(df['Close'].max()),
                    'mean': float(df['Close'].mean())
                }
        
        return summary


def main():
    """Main function to demonstrate usage."""
    # Default folder path (project's data/raw)
    project_root = Path(__file__).parent.parent
    folder_path = project_root / "data" / "raw"
    
    print("=" * 60)
    print("Auto-loading Historical Price Data")
    print("=" * 60)
    
    # Initialize loader (automatically loads all CSVs)
    loader = HistoricalPriceDataLoader(folder_path)
    
    # Get summary
    print("\n1. Data Summary:")
    print("-" * 60)
    summary = loader.get_summary()
    print(f"   Total tickers: {summary['num_tickers']}")
    print(f"   Total records: {summary['total_records']}")
    print(f"   Tickers: {', '.join(summary['tickers'][:10])}{'...' if len(summary['tickers']) > 10 else ''}")
    
    # Get combined DataFrame
    print("\n2. Combined DataFrame:")
    print("-" * 60)
    combined_df = loader.get_dataframe()
    if combined_df is not None:
        print(combined_df.head(10))
        print(f"\n   Shape: {combined_df.shape}")
        print(f"   Index levels: {combined_df.index.names}")
    
    # Get specific ticker
    print("\n3. Individual Ticker (AAPL):")
    print("-" * 60)
    aapl_df = loader.get_dataframe('AAPL')
    if aapl_df is not None:
        print(aapl_df.head())
        print(f"\n   Records: {len(aapl_df)}")
        if 'Close' in aapl_df.columns:
            print(f"   Price range: ${aapl_df['Close'].min():.2f} - ${aapl_df['Close'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("Auto-loading complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

