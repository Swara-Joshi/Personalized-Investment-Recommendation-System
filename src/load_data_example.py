"""
Example script for loading historical price data

This script demonstrates how to use the HistoricalPriceDataLoader
to load data for model training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import HistoricalPriceDataLoader
import pandas as pd


def main():
    """Main function demonstrating data loading"""
    
    # Initialize the data loader
    # Defaults to data/raw folder
    loader = HistoricalPriceDataLoader()
    
    print("=" * 60)
    print("Load Historical Price Data")
    print("=" * 60)
    
    # List available files
    print("\n1. Available CSV files:")
    csv_files = loader.list_available_files()
    print(f"   Found {len(csv_files)} CSV files")
    print(f"   Sample files: {csv_files[:5]}")
    
    # Example 1: Load single stock (Apple)
    print("\n2. Loading Apple (AAPL) stock data:")
    print("-" * 60)
    try:
        aapl_data = loader.load_single_stock('AAPL')
        print(aapl_data.head())
        print(f"\n   Total records: {len(aapl_data)}")
        print(f"   Date range: {aapl_data['Date'].min()} to {aapl_data['Date'].max()}")
        print(f"   Columns: {list(aapl_data.columns)}")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
    
    # Example 2: Load multiple stocks
    print("\n3. Loading multiple stocks (AAPL, MSFT, GOOGL):")
    print("-" * 60)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    price_data = loader.load_multiple_stocks(tickers)
    
    for ticker, df in price_data.items():
        print(f"   {ticker}: {len(df)} records, "
              f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Example 3: Create combined DataFrame
    print("\n4. Creating combined DataFrame:")
    print("-" * 60)
    try:
        combined_df = loader.create_combined_dataframe(tickers=tickers)
        print(f"   Combined DataFrame shape: {combined_df.shape}")
        print(f"   Index levels: {combined_df.index.names}")
        print("\n   First few rows:")
        print(combined_df.head(10))
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 4: Load combined data file (if available)
    print("\n5. Loading combined data file:")
    print("-" * 60)
    try:
        combined_data = loader.load_combined_data()
        print(f"   Loaded {len(combined_data)} records")
        if 'Symbol' in combined_data.columns:
            symbols = combined_data['Symbol'].unique()
            print(f"   Symbols: {len(symbols)} unique symbols")
            print(f"   Sample symbols: {list(symbols[:5])}")
    except FileNotFoundError:
        print("   Combined data file not found (this is okay)")
    
    # Example 5: Get data summary
    print("\n6. Data summary for AAPL:")
    print("-" * 60)
    try:
        summary = loader.get_data_summary(ticker='AAPL')
        print(f"   Total records: {summary['total_records']}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        if summary['price_stats']:
            print(f"   Price stats:")
            print(f"     Min: ${summary['price_stats']['min']:.2f}")
            print(f"     Max: ${summary['price_stats']['max']:.2f}")
            print(f"     Mean: ${summary['price_stats']['mean']:.2f}")
            print(f"     Std: ${summary['price_stats']['std']:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Data loading examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

