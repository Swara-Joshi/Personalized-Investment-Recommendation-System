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

## Phase 2: Market Agent – ML Prototyping

### Overview

The Market Agent's primary job is to **predict future stock prices**. These predictions will later feed into the **Recommendation Agent** to provide investment advice. We approach this using **time-series modeling** with **historical price data** and **LSTM neural networks**.

The workflow consists of:
1. **Data Loading (2.1)**: Gather and combine historical price data
2. **Data Preprocessing (2.2)**: Scale, clean, and split data
3. **Sequence Creation (2.3)**: Convert data into sliding windows for LSTM
4. **LSTM Model Training (2.4)**: Train models to predict next-day prices
5. **Model Evaluation (2.5)**: Evaluate prediction accuracy

---

### 2.1 – Load Historical Price Data

#### What We Do

- Collect all historical price CSVs for different tickers (e.g., AAPL, MSFT, GOOGL, BTC-USD)
- Automatically combine them into a single dataset or keep per-ticker datasets
- Standardize column names and data formats
- Handle multiple data sources (stocks, ETFs, crypto)

#### Why

- ML models require structured numerical data
- Combining or looping over tickers avoids manual work for every stock
- Standardization ensures consistent data format across all symbols
- Enables batch processing and scalable model training

#### Implementation

See the [Data Loading](#data-loading-for-model-training) section below for detailed usage.

---

### 2.2 – Data Preprocessing

#### What We Do

- **Feature Selection**: Select relevant numerical features (High, Low, Close, Volume, Open)
- **Data Scaling**: Standardize data by scaling features using MinMaxScaler (range 0–1)
- **Missing Values**: Handle missing values by forward filling and backward filling
- **Train/Validation Split**: Split data into training (first 80%) and validation (last 20%) sets
- **Technical Indicators**: Calculate additional features (MA, RSI, MACD, Bollinger Bands)

#### Why

- **LSTMs are sensitive to feature scale**: Normalization ensures faster learning and stable gradients
- **Handling missing values**: Prevents errors in sequence creation and model training
- **Train/validation split**: Ensures we can evaluate model generalization and prevent overfitting
- **Technical indicators**: Provide additional signals that help the model learn price patterns

#### Key Considerations

- Normalization is critical for LSTM performance
- Validation set should be chronologically after training set (no data leakage)
- Feature engineering can significantly improve model accuracy

---

### 2.3 – Sequence Creation

#### What We Do

- Transform flat historical data into **sliding windows** of past prices
- Example: Use last 60 days' prices to predict the **next day's Close price**
- Create `X_train` (input sequences) and `y_train` (target prices) for the LSTM
- Each sequence represents a window of historical data points

#### Why

- **LSTM models need sequences as input**: They are designed to learn **temporal dependencies**
- **Sliding windows**: Allow the model to see "recent history" for predicting the future
- **Sequence length**: Balances model memory (longer = more context) vs. training speed (shorter = faster)
- **Time-series nature**: Stock prices are sequential data, requiring sequence-based modeling

#### Sequence Structure

```
Input (X):  [Day 1-60 prices] → Target (y): [Day 61 Close price]
Input (X):  [Day 2-61 prices] → Target (y): [Day 62 Close price]
...
```

#### Key Parameters

- **Sequence Length**: Number of past days to use (typically 30-60 days)
- **Prediction Horizon**: How far ahead to predict (typically 1 day for next-day prediction)
- **Step Size**: How many days to slide the window (typically 1 day)

---

### 2.4 – LSTM Model Implementation

#### What We Do

- Build a **baseline LSTM model** with:
  - **Input Layer**: Sequences of shape (sequence_length, num_features)
  - **LSTM Layer(s)**: Learn temporal patterns and dependencies
  - **Dense Layer**: Output next-day price prediction
- Train separately for each ticker (or use multi-ticker training)
- Use appropriate loss function: **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**
- Optimizer: **Adam** (adaptive learning rate)
- Use **EarlyStopping** and **ModelCheckpoint** to prevent overfitting
- Monitor training with validation loss

#### Why

- **LSTM is designed for sequential/temporal data**: Ideal for stock price prediction
- **Dense output**: Predicts numerical next-day price directly
- **Training per ticker**: Ensures the model learns stock-specific patterns and behaviors
- **EarlyStopping**: Prevents overfitting by stopping when validation loss stops improving
- **Adam optimizer**: Efficiently adapts learning rate during training

#### Model Architecture

```
Input (sequences) 
  ↓
LSTM Layer(s) [learns temporal patterns]
  ↓
Dense Layer [outputs price prediction]
  ↓
Output (next-day Close price)
```

#### Training Strategy

- **Per-Ticker Training**: Train separate models for each stock/ETF/crypto
- **Multi-Ticker Training**: Train a single model on multiple tickers (requires careful normalization)
- **Hyperparameter Tuning**: Adjust LSTM units, layers, learning rate, sequence length
- **Regularization**: Use dropout, L2 regularization to prevent overfitting

#### Key Metrics

- **Loss Function**: MSE (Mean Squared Error) for regression
- **Optimizer**: Adam with learning rate scheduling
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

---

### 2.5 – Model Evaluation (Next Step)

#### What We'll Do

- Evaluate model performance using metrics:
  - **RMSE** (Root Mean Squared Error): Measures prediction error magnitude
  - **MAE** (Mean Absolute Error): Average absolute prediction error
  - **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- Plot **predicted vs actual** values to visualize model performance
- Analyze prediction errors across different market conditions
- Compare model performance across different tickers

#### Why

- **RMSE/MAE**: Provide quantitative measures of prediction accuracy
- **Visualization**: Helps identify patterns in prediction errors
- **Market condition analysis**: Understand model performance in different scenarios
- **Ticker comparison**: Identify which stocks are easier/harder to predict

---

### Summary: Phase 2 Workflow

1. **Data Loading (2.1)**: ✅ Gather and combine historical prices
2. **Preprocessing (2.2)**: Scale, clean, and split data into train/val
3. **Sequence Creation (2.3)**: Convert historical prices into sliding windows for LSTM input
4. **LSTM Model (2.4)**: Train a model per ticker to predict **next-day Close price**
5. **Model Evaluation (2.5)**: Evaluate prediction accuracy using RMSE, MAE, and visualizations

---

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

## Implementation Status

### ✅ Completed

- **Phase 1**: Data Collection & Preprocessing
  - Historical price data collection
  - Market sentiment collection
  - User profile generation
  - Data preprocessing for all agents

- **Phase 2.1**: Data Loading
  - Manual data loader
  - Auto-loading data loader
  - Data validation and standardization

### 🚧 In Progress

- **Phase 2.2-2.4**: Market Agent ML Prototyping
  - Data preprocessing pipeline
  - Sequence creation for LSTM
  - LSTM model implementation
  - Model training infrastructure

### 📋 Planned

- **Phase 2.5**: Model Evaluation
  - Performance metrics (RMSE, MAE, MAPE)
  - Prediction visualization
  - Model comparison and analysis

- **Phase 3**: Agent Implementation
  - Risk Tolerance Agent
  - Market Agent integration
  - Recommendation Agent

- **Phase 4**: System Integration
  - Multi-agent coordination
  - Recommendation system
  - User interface

## Next Steps

The system is ready for:
1. **Complete Phase 2.2-2.4**: Implement data preprocessing, sequence creation, and LSTM model training
2. **Phase 2.5**: Model evaluation and performance analysis
3. **Phase 3**: Implement remaining agents (Risk Tolerance, Recommendation)
4. **Phase 4**: System integration and deployment

## License

This project is for educational purposes.
