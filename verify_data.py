"""Quick script to verify collected and processed data"""
import pandas as pd
from pathlib import Path

print("=" * 60)
print("Data Verification Report")
print("=" * 60)

# Check raw data
print("\n1. Raw Data Files:")
raw_dir = Path("data/raw")
if raw_dir.exists():
    price_files = list(raw_dir.glob("historical_prices_*.csv"))
    print(f"   - Historical price files: {len(price_files)}")
    
    if (raw_dir / "user_profiles.csv").exists():
        user_df = pd.read_csv(raw_dir / "user_profiles.csv")
        print(f"   - User profiles: {len(user_df)} records")
    
    if (raw_dir / "historical_prices_combined.csv").exists():
        price_df = pd.read_csv(raw_dir / "historical_prices_combined.csv")
        print(f"   - Combined price data: {len(price_df)} records")
        print(f"   - Symbols: {price_df['Symbol'].nunique()}")

# Check processed data
print("\n2. Processed Data Files:")
processed_dir = Path("data/processed")
if processed_dir.exists():
    files = list(processed_dir.glob("*.csv"))
    print(f"   - Processed files: {len(files)}")
    
    for file in files:
        df = pd.read_csv(file)
        print(f"   - {file.name}: {len(df)} records, {len(df.columns)} columns")

# Check Market Agent data details
print("\n3. Market Agent Data Details:")
market_file = processed_dir / "market_agent_processed.csv"
if market_file.exists():
    market_df = pd.read_csv(market_file)
    print(f"   - Total records: {len(market_df)}")
    print(f"   - Symbols: {market_df['Symbol'].nunique()}")
    print(f"   - Date range: {market_df['Date'].min()} to {market_df['Date'].max()}")
    
    # Check for technical indicators
    tech_indicators = [c for c in market_df.columns if any(x in c for x in ['MA_', 'RSI', 'MACD', 'BB_', 'Price_Change'])]
    print(f"   - Technical indicators: {len(tech_indicators)}")
    print(f"   - Sample indicators: {tech_indicators[:5]}")

# Check Risk Agent data details
print("\n4. Risk Agent Data Details:")
risk_file = processed_dir / "risk_agent_processed.csv"
if risk_file.exists():
    risk_df = pd.read_csv(risk_file)
    print(f"   - Total records: {len(risk_df)}")
    print(f"   - Risk level distribution:")
    if 'risk_level' in risk_df.columns:
        print(risk_df['risk_level'].value_counts().to_string())

# Check Recommendation Agent data details
print("\n5. Recommendation Agent Data Details:")
rec_file = processed_dir / "recommendation_agent_processed.csv"
if rec_file.exists():
    rec_df = pd.read_csv(rec_file)
    print(f"   - Total records: {len(rec_df)}")
    print(f"   - Unique users: {rec_df['user_id'].nunique()}")
    print(f"   - Unique stocks: {rec_df['symbol'].nunique() if 'symbol' in rec_df.columns else 'N/A'}")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)

