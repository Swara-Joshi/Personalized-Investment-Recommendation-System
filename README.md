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
├── sentiment_pipeline/
│   ├── config.yaml                   # Sentiment pipeline configuration
│   ├── requirements.txt              # Focused dependencies (HF, LoRA, etc.)
│   ├── data/                         # Yahoo Finance raw & labeled news, HF datasets
│   ├── scripts/                      # Data collection, labeling, prep, inference, automation
│   ├── notebooks/
│   │   └── sentiment_finetuning.ipynb# Colab-ready LoRA fine-tuning notebook
│   ├── models/                       # LoRA adapters (saved after fine-tuning)
│   └── results/                      # Sentiment plots & combined signals
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Data Collection & Preprocessing

### Overview

This layer focuses on collecting and preprocessing data for three main agents:

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

## Market Agent – LSTM Prototyping

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

### Summary: Market Agent Workflow

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

## Sentiment Intelligence & Signal Integration

### Overview

`sentiment_pipeline/` extends the advisor with real Yahoo Finance news, hybrid FinBERT + price-action labeling, LoRA fine-tuning on Mistral-7B, and signal fusion with the LSTM forecasts.

### Setup

```bash
pip install -r sentiment_pipeline/requirements.txt
```

Create/update `sentiment_pipeline/config.yaml` to control ticker universe, confidence thresholds, LoRA hyperparameters, output paths, and scheduling windows.

### 1. Collect Yahoo Finance Headlines

```bash
python sentiment_pipeline/scripts/data_collection.py
```

- Uses `yfinance` + RSS fallback for `AAPL, TSLA, NVDA, MSFT, AMZN, META, GOOGL, JPM, SPY`.
- Captures headline, timestamp, publisher, URL for the last 90 days (configurable).
- Writes `sentiment_pipeline/data/sentiment/yahoo_news_raw.csv`.

### 2. Auto-Label with FinBERT + Price Confirmation

```bash
$env:KMP_DUPLICATE_LIB_OK='TRUE'  # Windows OpenMP workaround
python sentiment_pipeline/scripts/label_news.py
```

- Runs `ProsusAI/FinBERT` and keeps rows where confidence ≥ 0.75.
- Verifies sentiment with ±2% next-day price moves via `yfinance`.
- Saves curated rows (date, ticker, headline, sentiment, confidence, price_change, source) to `sentiment_pipeline/data/sentiment/yahoo_news_labeled.csv`.

### 3. Build HF Datasets & Visuals

```bash
python sentiment_pipeline/scripts/prepare_dataset.py
```

- Filters for confidence ≥ 0.8 and formats prompts as `Headline: … Sentiment: …`.
- Produces Hugging Face datasets (80/20 split) in `sentiment_pipeline/data/hf_datasets/{train,validation}`.
- Exports `sentiment_pipeline/results/sentiment_distribution.png`.

### 4. Fine-Tune Mistral-7B with LoRA (Colab-ready)

Notebook: `sentiment_pipeline/notebooks/sentiment_finetuning.ipynb`

- Mount Google Drive, set `PROJECT_ROOT`, and run dependency install cell.
- Loads datasets via `datasets.load_from_disk`, config via `sentiment_pipeline/utils/config_loader`.
- Uses 4-bit quantization (`bitsandbytes`) and LoRA (r=16, α=32, dropout=0.05 on q/k/v/o projections).
- Training config: lr=2e-4, epochs=3, batch_size=4, grad_accum=4, warmup=100, fp16 enabled, W&B logging optional.
- Saves adapters + tokenizer to `sentiment_pipeline/models/sentiment_model`.

### 5. Inference & Signal Fusion

```bash
python sentiment_pipeline/scripts/sentiment_inference.py   # requires trained adapters
python sentiment_pipeline/scripts/combine_signals.py       # needs data/lstm_predictions.csv
```

- `sentiment_inference.py` loads the LoRA adapter, exposes `predict_sentiment` + `batch_predict`, and prints example outputs.
- `combine_signals.py` merges LSTM price moves with sentiment, scores agreement, and writes `sentiment_pipeline/data/combined_predictions.json`.

### 6. Automated Daily Updates

```bash
python sentiment_pipeline/scripts/daily_update.py
```

- Schedules a 6 PM EST job (configurable) to pull 7-day headlines, label with the current model, append to the dataset, and optionally trigger weekly dataset refreshes.

### 7. Tests

```bash
pytest sentiment_pipeline/tests
```

Ensures prompt formatting + signal combination utilities stay stable as the pipeline evolves.

## Implementation Status

### ✅ Completed

- Data collection & preprocessing: historical prices, news sentiment captures, synthetic user profiles, preprocessing pipelines per agent.
- Market agent data loaders: manual + auto loaders with validation, combined DataFrame builders, example scripts.
- Sentiment intelligence foundation: Yahoo Finance scraping, FinBERT hybrid labeling, HF dataset export, LoRA training notebook scaffolding, inference/integration scripts, and unit tests.

### 🚧 In Progress

- Market agent feature engineering and LSTM training loops (sequence creation + model orchestration).
- LoRA adapter training/export (awaiting Colab or local GPU run for `sentiment_finetuning.ipynb`).

### 📋 Planned

- Market agent evaluation (RMSE/MAE/MAPE, prediction visualizations, ticker comparisons).
- Recommendation/risk agents + portfolio orchestration that consume both LSTM forecasts and sentiment scores.
- User-facing surfaces and automation (dashboards, alerts, scheduled retraining).

## Next Steps

1. Run `sentiment_pipeline/notebooks/sentiment_finetuning.ipynb` (or equivalent script) to produce LoRA adapters in `sentiment_pipeline/models/sentiment_model`, then rerun inference/integration scripts.
2. Export the existing LSTM price predictions to `data/lstm_predictions.csv` so that `sentiment_pipeline/scripts/combine_signals.py` can emit final trading signals.
3. Finish the Market Agent training/evaluation loop (sequence creation, LSTM model training, RMSE/MAE dashboards).
4. Integrate sentiment + market outputs into the recommendation and risk agents, then layer on a UI or API for investment advice delivery.

## License

This project is for educational purposes.
