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

## Next Steps

After completing Phase 1, the system is ready for:
- Phase 2: Model Training (LSTM, GPT fine-tuning, RL)
- Phase 3: Agent Implementation
- Phase 4: Recommendation System Integration

## License

This project is for educational purposes.
