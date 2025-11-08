# Personalized Investment Recommendation System

A comprehensive system for personalized investment recommendations using multi-agent AI architecture.

## Project Structure

```
Personalized-Investment-Recommendation-System/
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── historical_prices.py      # Historical price data collection
│   │   ├── market_sentiment.py       # Market sentiment collection
│   │   └── user_profiles.py          # User profile generation
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   ├── market_agent_preprocessing.py      # Market Agent data preprocessing
│   │   ├── risk_agent_preprocessing.py        # Risk Agent data preprocessing
│   │   └── recommendation_agent_preprocessing.py  # Recommendation Agent data preprocessing
│   ├── data_loader.py                # Historical price data loader
│   ├── load_data_auto.py             # Auto-loading data loader
│   ├── load_data_example.py          # Example usage of data loader
│   └── collect_data.py               # Main data collection script
├── config/
│   └── config.py                     # Configuration file
├── data/
│   ├── raw/                          # Raw collected data
│   └── processed/                    # Processed data for ML models
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Phase 1: Data Collection & Preprocessing

### Overview

Phase 1 focuses on collecting and preprocessing data for three main agents:

1. **Risk Tolerance Agent**: User profiling & risk scoring
2. **Market Agent**: Market trend prediction (time-series + sentiment)
3. **Recommendation Agent**: Portfolio advice & optimization

### Features

#### 1. Historical Price Data Collection
- Collects historical stock/ETF/crypto prices from Yahoo Finance
- Supports multiple symbols and time periods
- Saves data in CSV format

#### 2. Market Sentiment Collection
- Collects sentiment from Reddit (r/wallstreetbets, r/investing, etc.)
- Collects financial news from RSS feeds
- Calculates sentiment scores using VADER sentiment analyzer

#### 3. User Profile Generation
- Generates realistic user profiles with:
  - Age (25-65)
  - Income ($30,000 - $200,000)
  - Investment horizon (1-30 years)
  - Risk preference (conservative, moderate, aggressive)
  - Portfolio value and savings rate

#### 4. Data Preprocessing
- **Market Agent**: 
  - Technical indicators (MA, RSI, MACD, Bollinger Bands)
  - Time-series sequence creation for LSTM
  - Sentiment data integration
  - Feature normalization

- **Risk Agent**:
  - Categorical variable encoding
  - Derived feature creation
  - Risk score calculation
  - Feature normalization

- **Recommendation Agent**:
  - Stock feature calculation (returns, volatility, Sharpe ratio)
  - User-stock matching
  - Recommended allocation calculation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

Create a `.env` file in the project root:

```env
# Reddit API (optional, for sentiment collection)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=InvestmentBot/1.0

# Alpha Vantage API (optional, for alternative price data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

**Note**: The system works without API keys, but some features (Reddit sentiment, Alpha Vantage) will be disabled.

### 3. Run Data Collection

```bash
python src/collect_data.py
```

This will:
1. Collect historical prices for stocks/ETFs/crypto
2. Collect market sentiment from news and Reddit (if API keys provided)
3. Generate simulated user profiles
4. Preprocess data for each agent
5. Save processed data to `data/processed/`

## Data Sources

### Historical Prices
- **Yahoo Finance**: Primary source for stock/ETF/crypto prices
- **Alpha Vantage**: Backup source (requires API key)

### Market Sentiment
- **Reddit**: r/wallstreetbets, r/investing, r/stocks, r/SecurityAnalysis
- **News RSS Feeds**: Yahoo Finance, Bloomberg

### User Profiles
- Simulated profiles with realistic financial characteristics
- Configurable parameters in `config/config.py`

## Output Files

### Raw Data (`data/raw/`)
- `historical_prices_*.csv`: Historical price data per symbol
- `historical_prices_combined.csv`: Combined price data
- `market_sentiment.csv`: Sentiment data from news and social media
- `user_profiles.csv`: Generated user profiles

### Processed Data (`data/processed/`)
- `market_agent_processed.csv`: Processed data for Market Agent
- `risk_agent_processed.csv`: Processed data for Risk Agent
- `recommendation_agent_processed.csv`: Processed data for Recommendation Agent

## Configuration

Edit `config/config.py` to customize:
- Stock symbols to collect
- Data collection parameters
- User profile generation parameters
- API keys and data sources

## Data Loading for Model Training

### Overview

Comprehensive historical price data loaders for model training and analysis.

### Features

#### 1. Manual Data Loader (`data_loader.py`)
- Load single or multiple stock/ETF/crypto price data on demand
- Load combined price data from CSV files
- Standardize column names and data formats
- Create combined DataFrames with MultiIndex
- Get data summaries and statistics

#### 2. Auto-Loading Data Loader (`load_data_auto.py`)
- Automatically loads all CSV files on initialization
- Quick access to individual tickers or combined DataFrame
- Simplified API for rapid data access
- Automatic filtering of non-price files

### Usage

#### Manual Loader (On-Demand Loading)

```python
from src.data_loader import HistoricalPriceDataLoader

# Initialize loader
loader = HistoricalPriceDataLoader()

# Load single stock
aapl_data = loader.load_single_stock('AAPL')

# Load multiple stocks
price_data = loader.load_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])

# Create combined DataFrame
combined_df = loader.create_combined_dataframe(tickers=['AAPL', 'MSFT', 'GOOGL'])

# Get data summary
summary = loader.get_data_summary(ticker='AAPL')
```

#### Auto-Loading (Automatic Loading)

```python
from src.load_data_auto import HistoricalPriceDataLoader

# Initialize loader (automatically loads all CSVs)
loader = HistoricalPriceDataLoader()

# Get combined DataFrame
combined_df = loader.get_dataframe()

# Get specific ticker
aapl_df = loader.get_dataframe('AAPL')

# Get list of available tickers
tickers = loader.get_tickers()

# Get summary
summary = loader.get_summary()
```

### Run Examples

```bash
# Manual loader example
python src/load_data_example.py

# Auto-loading example
python src/load_data_auto.py
```

## Next Steps

The system is ready for:
- Feature engineering for ML models
- LSTM model implementation for time-series prediction
- GPT fine-tuning setup for recommendation generation
- Reinforcement Learning implementation for portfolio optimization
- Agent Implementation
- Recommendation System Integration

## License

This project is for educational purposes.
