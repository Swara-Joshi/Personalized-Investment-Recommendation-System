"""
LSTM Model Evaluation Script

This script evaluates trained LSTM models for stock price prediction.
It calculates RMSE, MAE, and MAPE metrics and generates visualizations.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_PATH = PROJECT_ROOT
PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed"
MODELS_PATH = BASE_PATH / "models" / "lstm"
RAW_DATA_PATH = BASE_PATH / "data" / "raw"
SCALERS_PATH = BASE_PATH / "models" / "scalers"
PLOTS_PATH = BASE_PATH / "results" / "plots"

# Create directories if they don't exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)
SCALERS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# --- Parameters ---
SEQUENCE_LENGTH = 60  # Same as during training
# Model was trained with 4 features: ['High', 'Low', 'Close', 'Volume']
# Note: Model input shape is (60, 4), so we must use exactly these 4 features
FEATURES = ['High', 'Low', 'Close', 'Volume']  # Must match training features

# Tickers to evaluate
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
    'JPM', 'SPY', 'QQQ', 'VTI', 'VOO', 'IWM', 'DIA', 'GLD', 
    'TLT', 'BTC-USD', 'ETH-USD', 'BNB-USD'
]


def create_sequences(data, seq_len=60):
    """
    Create sliding window sequences for LSTM input.
    
    Args:
        data: Scaled feature data
        seq_len: Sequence length (number of time steps)
        
    Returns:
        X: Input sequences (samples, timesteps, features)
        y: Target values (next Close price)
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        # 'Close' is at index 2 in FEATURES ['High', 'Low', 'Close', 'Volume']
        close_index = FEATURES.index('Close') if 'Close' in FEATURES else 2
        y.append(data[i, close_index])
    return np.array(X), np.array(y)


def load_and_prepare_data(ticker, raw_data_path, scaler_path=None):
    """
    Load and prepare data for evaluation.
    
    Args:
        ticker: Stock ticker symbol
        raw_data_path: Path to raw data directory
        scaler_path: Path to scaler file (if exists)
        
    Returns:
        df: DataFrame with prepared data
        scaler: MinMaxScaler object
        scaled_data: Scaled feature data
    """
    # Load raw CSV (our files are named historical_prices_{ticker}.csv)
    csv_file = raw_data_path / f"historical_prices_{ticker}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Data file not found: {csv_file}")
    
    logger.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert Date to datetime and sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Select available features
    available_features = [f for f in FEATURES if f in df.columns]
    if 'Close' not in available_features:
        raise ValueError(f"'Close' column not found in {ticker} data")
    
    df_features = df[available_features].values
    
    # Load or create scaler
    if scaler_path and Path(scaler_path).exists():
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        logger.warning(f"Scaler not found for {ticker}, creating new one")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_features)
    
    scaled_data = scaler.transform(df_features)
    
    return df, scaler, scaled_data, available_features


def evaluate_model(ticker, model_path, scaler_path, raw_data_path):
    """
    Evaluate a trained LSTM model for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to trained model file
        scaler_path: Path to scaler file
        raw_data_path: Path to raw data directory
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    try:
        # Load and prepare data
        df, scaler, scaled_data, available_features = load_and_prepare_data(
            ticker, raw_data_path, scaler_path
        )
        
        # Create sequences
        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
        
        if len(X) == 0:
            raise ValueError(f"Insufficient data for {ticker} (need at least {SEQUENCE_LENGTH + 1} samples)")
        
        # Train/val split (same as training: first 80% train, last 20% val)
        train_size = int(len(X) * 0.8)
        X_val = X[train_size:]
        y_val = y[train_size:]
        
        if len(X_val) == 0:
            raise ValueError(f"Insufficient validation data for {ticker}")
        
        # Load trained model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        try:
            # Try loading with custom_objects to handle version differences
            # This is a common issue when models are trained with different TensorFlow/Keras versions
            try:
                # Method 1: Try with TensorFlow 2.x style
                import tensorflow as tf
                # Suppress warnings about version mismatches
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    model = tf.keras.models.load_model(
                        str(model_path),
                        compile=False  # Don't compile to avoid optimizer issues
                    )
            except Exception as e1:
                try:
                    # Method 2: Try with standalone Keras
                    import keras
                    model = keras.models.load_model(str(model_path), compile=False)
                except Exception as e2:
                    logger.error(f"Error loading model with TensorFlow: {e1}")
                    logger.error(f"Error loading model with Keras: {e2}")
                    logger.error("This may be due to TensorFlow/Keras version mismatch.")
                    logger.error("Try retraining the models or matching TensorFlow versions.")
                    raise e1
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Model file exists but cannot be loaded. This is likely a version compatibility issue.")
            raise
        
        # Predict
        logger.info(f"Making predictions for {ticker}...")
        y_pred = model.predict(X_val, verbose=0)
        y_pred = y_pred.flatten()
        
        # Inverse scale predictions and actual values
        close_index = available_features.index('Close')
        
        # Inverse transform actual values
        y_val_scaled = np.zeros((len(y_val), len(available_features)))
        y_val_scaled[:, close_index] = y_val
        y_val_actual = scaler.inverse_transform(y_val_scaled)[:, close_index]
        
        # Inverse transform predictions
        y_pred_scaled = np.zeros((len(y_pred), len(available_features)))
        y_pred_scaled[:, close_index] = y_pred
        y_pred_actual = scaler.inverse_transform(y_pred_scaled)[:, close_index]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        
        # Calculate MAPE (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_val_actual - y_pred_actual) / y_val_actual)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate R-squared
        ss_res = np.sum((y_val_actual - y_pred_actual) ** 2)
        ss_tot = np.sum((y_val_actual - np.mean(y_val_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'y_val_actual': y_val_actual,
            'y_pred_actual': y_pred_actual,
            'num_samples': len(y_val_actual)
        }
        
        logger.info(
            f"✅ {ticker} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, "
            f"MAPE: {mape:.2f}%, R²: {r2:.4f}"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error evaluating {ticker}: {e}")
        return None


def plot_predictions(ticker, y_val_actual, y_pred_actual, save_path):
    """
    Plot predicted vs actual values.
    
    Args:
        ticker: Stock ticker symbol
        y_val_actual: Actual values
        y_pred_actual: Predicted values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_actual, label='Actual', linewidth=2, alpha=0.7)
    plt.plot(y_pred_actual, label='Predicted', linewidth=2, alpha=0.7)
    plt.title(f"{ticker} - Predicted vs Actual Close Price", fontsize=14, fontweight='bold')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Close Price ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = save_path / f"{ticker}_predictions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_file}")
    plt.close()


def main():
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("LSTM Model Evaluation")
    logger.info("=" * 60)
    
    results = {}
    successful_evaluations = 0
    failed_evaluations = 0
    
    for ticker in TICKERS:
        logger.info(f"\nEvaluating {ticker}...")
        
        # Paths
        model_path = MODELS_PATH / f"lstm_model_{ticker}.h5"
        scaler_path = SCALERS_PATH / f"scaler_{ticker}.pkl"
        
        # Check if model exists
        if not model_path.exists():
            logger.warning(f"⚠️  Model not found for {ticker}: {model_path}")
            logger.warning(f"   Skipping {ticker}. Train the model first.")
            failed_evaluations += 1
            continue
        
        # Evaluate model
        result = evaluate_model(ticker, model_path, scaler_path, RAW_DATA_PATH)
        
        if result:
            results[ticker] = result
            successful_evaluations += 1
            
            # Plot predictions
            plot_predictions(
                ticker,
                result['y_val_actual'],
                result['y_pred_actual'],
                PLOTS_PATH
            )
        else:
            failed_evaluations += 1
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Successfully evaluated: {successful_evaluations} tickers")
    logger.info(f"Failed/Skipped: {failed_evaluations} tickers")
    
    if results:
        # Create summary DataFrame
        summary_data = {
            ticker: {
                'RMSE': result['RMSE'],
                'MAE': result['MAE'],
                'MAPE': result['MAPE'],
                'R²': result['R2'],
                'Samples': result['num_samples']
            }
            for ticker, result in results.items()
        }
        
        results_df = pd.DataFrame(summary_data).T
        results_df = results_df.sort_values('RMSE')
        
        print("\n" + "=" * 60)
        print("Evaluation Metrics Summary")
        print("=" * 60)
        print(results_df.to_string())
        
        # Save results to CSV
        results_csv = BASE_PATH / "results" / "evaluation_results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv)
        logger.info(f"\n✅ Saved results to {results_csv}")
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Average RMSE: {results_df['RMSE'].mean():.2f}")
        print(f"Average MAE: {results_df['MAE'].mean():.2f}")
        print(f"Average MAPE: {results_df['MAPE'].mean():.2f}%")
        print(f"Average R²: {results_df['R²'].mean():.4f}")
        print(f"\nBest Model (Lowest RMSE): {results_df.index[0]}")
        print(f"Worst Model (Highest RMSE): {results_df.index[-1]}")
    else:
        logger.warning("\n⚠️  No models were evaluated. Please train models first.")
        logger.info("To train models, run the LSTM training script.")


if __name__ == "__main__":
    main()

