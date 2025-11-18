# Complete System Overview: Theory & Architecture

## 🎯 **The Big Picture: What Are We Building?**

You're building an **AI-Powered Investment Recommendation System** that combines:
1. **Price Prediction** (LSTM models) - "Where will the stock price go?"
2. **Sentiment Analysis** (Fine-tuned LLM) - "What does the market feel about this stock?"
3. **Combined Signals** - "Should I buy, sell, or hold?"

---

## 📊 **Phase 1: Data Collection & Preprocessing**

### **What We Did:**
- Collected historical stock prices (AAPL, MSFT, TSLA, etc.)
- Gathered market sentiment from news and Reddit
- Generated synthetic user profiles
- Preprocessed data for three different ML agents

### **Why We Did It:**
**Theory:** Machine learning models need **structured, clean data** to learn patterns. Raw data from APIs is messy and needs transformation.

**Key Concepts:**

1. **Historical Price Data**
   - **What:** Stock prices (Open, High, Low, Close, Volume) over time
   - **Why:** LSTMs need historical sequences to predict future prices
   - **How:** Time-series data captures trends, volatility, and patterns

2. **Market Sentiment**
   - **What:** News headlines, Reddit posts, social media buzz
   - **Why:** Market psychology affects prices. Good news → price up, bad news → price down
   - **How:** Sentiment scores (positive/negative/neutral) quantify market emotions

3. **User Profiles**
   - **What:** Age, income, risk tolerance, investment horizon
   - **Why:** Different investors need different recommendations
   - **How:** Risk-averse users get conservative stocks; aggressive users get volatile stocks

4. **Data Preprocessing**
   - **Normalization:** Scale all features to 0-1 range (LSTMs work better with normalized data)
   - **Feature Engineering:** Create technical indicators (RSI, MACD, Moving Averages)
   - **Train/Test Split:** Use 80% for training, 20% for validation (prevent overfitting)

**Output:** Clean, structured datasets ready for ML models

---

## 🧠 **Phase 2: LSTM Price Prediction**

### **What We Did:**
- Built LSTM (Long Short-Term Memory) neural networks
- Trained separate models for each stock ticker
- Predicted next-day stock prices
- Evaluated model accuracy with RMSE, MAE metrics

### **Why We Did It:**
**Theory:** Stock prices follow **temporal patterns**. Past prices influence future prices. LSTMs are designed to learn these sequential dependencies.

**Key Concepts:**

1. **What is an LSTM?**
   - **Definition:** A type of Recurrent Neural Network (RNN) that can remember long-term dependencies
   - **Why LSTMs?** Regular neural networks can't remember past information. LSTMs have "memory cells" that remember important patterns
   - **Analogy:** Like reading a book - you remember what happened earlier to understand what's happening now

2. **Sequence Creation**
   - **What:** Convert flat price data into sliding windows
   - **Example:** Use last 60 days → predict day 61
   - **Why:** LSTMs need sequences as input, not single data points
   ```
   Input: [Day 1-60 prices] → Output: [Day 61 price]
   Input: [Day 2-61 prices] → Output: [Day 62 price]
   ```

3. **Model Architecture**
   ```
   Input Layer (60 days × 5 features)
        ↓
   LSTM Layer(s) - Learns temporal patterns
        ↓
   Dense Layer - Outputs single price prediction
        ↓
   Predicted Price
   ```

4. **Training Process**
   - **Loss Function:** Mean Squared Error (MSE) - measures prediction error
   - **Optimizer:** Adam - adapts learning rate during training
   - **Early Stopping:** Prevents overfitting (stops when validation loss stops improving)
   - **Per-Ticker Training:** Each stock gets its own model (different stocks have different patterns)

5. **Why Separate Models Per Ticker?**
   - AAPL behaves differently than TSLA
   - Each stock has unique volatility, trends, and patterns
   - One-size-fits-all model would be less accurate

**Output:** Trained LSTM models that predict next-day stock prices with ~2-5% error

**Limitation:** LSTMs only look at price data. They don't understand news, events, or market sentiment.

---

## 💬 **Phase 3: Sentiment Analysis Pipeline**

### **What We Did:**
- Collected real-time Yahoo Finance news headlines
- Auto-labeled headlines with FinBERT (sentiment classifier)
- Validated labels using actual price movements
- Fine-tuned Mistral-7B LLM with LoRA adapters
- Combined sentiment + price predictions for final recommendations

### **Why We Did It:**
**Theory:** Stock prices don't just follow technical patterns - they react to **news, events, and market psychology**. A company announcing great earnings → stock goes up. A scandal → stock goes down.

**Key Concepts:**

1. **Data Collection (`data_collection.py`)**
   - **What:** Scrape Yahoo Finance for news headlines (last 90 days)
   - **Why:** Need real, current news to analyze sentiment
   - **How:** yfinance library fetches news for each ticker
   - **Output:** `yahoo_news_raw.csv` with headlines, dates, sources

2. **Auto-Labeling (`label_news.py`)**
   - **What:** Use FinBERT to classify sentiment (positive/negative/neutral)
   - **Why:** Need labeled data to train our LLM, but manual labeling is expensive
   - **How:**
     - FinBERT (pre-trained on financial text) predicts sentiment
     - Check if actual price moved in same direction (validation)
     - Only keep labels where FinBERT confidence > 75% AND price confirms sentiment
   - **Hybrid Labeling:** Combines AI prediction + real-world validation
   - **Output:** `yahoo_news_labeled.csv` with high-quality labels

3. **Dataset Preparation (`prepare_dataset.py`)**
   - **What:** Format data for LLM fine-tuning
   - **Why:** LLMs need specific prompt formats to learn
   - **How:**
     - Format: `"Headline: {headline}\nSentiment: {sentiment}"`
     - Filter high-confidence labels (>80%)
     - Split 80/20 train/validation
     - Convert to Hugging Face Dataset format
   - **Output:** Train/validation datasets ready for fine-tuning

4. **Fine-Tuning Mistral-7B (`sentiment_finetuning.ipynb`)**
   - **What:** Train a large language model to predict sentiment from headlines
   - **Why:** Generic LLMs aren't good at financial sentiment. We need domain-specific knowledge.
   - **How:**
     - **Base Model:** Mistral-7B-Instruct (7 billion parameters, instruction-tuned)
     - **LoRA (Low-Rank Adaptation):** Instead of training all 7B parameters, train small adapter layers
     - **4-bit Quantization:** Reduce memory usage (model fits in GPU)
     - **Supervised Fine-Tuning:** Train on our labeled headlines
   - **LoRA Explained:**
     ```
     Original: Train all 7B parameters (expensive, slow)
     LoRA: Train only 0.1% of parameters (cheap, fast)
     Result: Model learns financial sentiment without full retraining
     ```
   - **Training Config:**
     - Learning rate: 2e-4 (small steps to avoid breaking pre-trained knowledge)
     - Epochs: 3 (enough to learn, not overfit)
     - Batch size: 4 (fits in GPU memory)
   - **Output:** Fine-tuned adapter weights saved to `models/sentiment_model/`

5. **Inference (`sentiment_inference.py`)**
   - **What:** Use fine-tuned model to predict sentiment on new headlines
   - **Why:** Real-time sentiment analysis for trading decisions
   - **How:**
     - Load base model + LoRA adapters
     - Input: `"Headline: Apple beats earnings\nSentiment:"`
     - Output: `"positive"` with confidence score
   - **Output:** Sentiment predictions for any headline

6. **Signal Combination (`combine_signals.py`)**
   - **What:** Merge LSTM price predictions + sentiment analysis
   - **Why:** Use both technical (price patterns) and fundamental (news) signals
   - **How:**
     ```
     LSTM says: Price will go UP (+5%)
     Sentiment says: POSITIVE (good news)
     → Combined Signal: STRONG BUY (both agree)
     
     LSTM says: Price will go UP (+3%)
     Sentiment says: NEGATIVE (bad news)
     → Combined Signal: WATCH (conflicting signals)
     ```
   - **Confidence Calculation:**
     - High confidence when LSTM and sentiment agree
     - Low confidence when they disagree
   - **Output:** `combined_predictions.json` with final trading signals

7. **Daily Updates (`daily_update.py`)**
   - **What:** Automated script to refresh data daily
   - **Why:** Markets change daily. Need fresh data for accurate predictions.
   - **How:**
     - Collect last 7 days of news
     - Label with current model
     - Append to dataset
     - Optionally retrain weekly
   - **Output:** Continuously updated dataset

---

## 🔄 **How Everything Connects**

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA LAYER                       │
│  Historical Prices + News + User Profiles → Clean Data      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼──────────┐
│  PHASE 2:      │          │  PHASE 3:          │
│  LSTM Models   │          │  Sentiment LLM     │
│                │          │                    │
│  Price Data →  │          │  News Headlines →  │
│  LSTM →        │          │  FinBERT →         │
│  Price Pred    │          │  Fine-tune →       │
│                │          │  Sentiment Pred    │
└───────┬────────┘          └─────────┬──────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
              ┌────────▼─────────┐
              │  COMBINE SIGNALS │
              │                  │
              │  Price Pred +    │
              │  Sentiment →     │
              │  Final Signal    │
              └──────────────────┘
                       │
              ┌────────▼─────────┐
              │  RECOMMENDATION  │
              │                  │
              │  BUY/SELL/HOLD   │
              │  + Confidence    │
              └──────────────────┘
```

---

## 🎓 **Key Theoretical Concepts**

### **1. Time-Series Forecasting (Phase 2)**
- **Theory:** Future values depend on past values
- **Application:** Stock prices follow trends and patterns
- **Model:** LSTM captures long-term dependencies

### **2. Transfer Learning (Phase 3)**
- **Theory:** Pre-trained models can be adapted to new tasks
- **Application:** Mistral-7B (general LLM) → Financial sentiment (specific task)
- **Method:** LoRA fine-tuning (efficient adaptation)

### **3. Ensemble Methods (Combining Signals)**
- **Theory:** Multiple models together > single model
- **Application:** LSTM (technical) + Sentiment (fundamental) = Better predictions
- **Benefit:** Reduces risk, increases confidence

### **4. Hybrid Labeling (Phase 3)**
- **Theory:** Combine AI predictions with real-world validation
- **Application:** FinBERT predicts → Price movement confirms
- **Benefit:** Higher quality training data

### **5. Quantization (Phase 3)**
- **Theory:** Reduce model precision to save memory
- **Application:** 4-bit quantization allows 7B model on consumer GPU
- **Trade-off:** Slight accuracy loss for massive memory savings

---

## 🎯 **Why This Architecture Works**

1. **Complementary Signals:**
   - LSTM: Technical analysis (price patterns)
   - Sentiment: Fundamental analysis (news, events)
   - Together: More robust predictions

2. **Real-World Validation:**
   - Labels confirmed by actual price movements
   - Reduces false positives/negatives

3. **Efficient Training:**
   - LoRA: Fast fine-tuning (hours, not days)
   - Quantization: Fits on free Colab GPUs

4. **Scalable:**
   - Per-ticker models: Easy to add new stocks
   - Daily updates: Stays current with market

5. **Production-Ready:**
   - Automated data collection
   - Batch inference
   - Confidence scoring

---

## 📈 **Expected Performance**

- **LSTM Price Prediction:** 2-5% RMSE (predicts within 2-5% of actual price)
- **Sentiment Accuracy:** 85-90% (on validated labels)
- **Combined Signal:** Higher confidence when both agree, lower when they disagree

---

## 🚀 **Next Steps (Future Phases)**

1. **Risk Agent:** Calculate portfolio risk based on user profile
2. **Recommendation Agent:** Generate personalized portfolio allocations
3. **Backtesting:** Test strategy on historical data
4. **Live Trading:** Connect to broker API (paper trading first!)

---

## 💡 **Key Takeaways**

1. **Data is King:** Clean, validated data → Better models
2. **Domain Adaptation:** Generic models need fine-tuning for finance
3. **Ensemble Wins:** Multiple signals > Single signal
4. **Efficiency Matters:** LoRA + Quantization = Accessible AI
5. **Validation is Critical:** Always verify with real-world outcomes

This system combines **technical analysis** (LSTM) with **fundamental analysis** (sentiment) to create a comprehensive investment recommendation engine! 🎯

