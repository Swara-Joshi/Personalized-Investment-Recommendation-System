"""
Market Agent Data Preprocessing

This module preprocesses historical price data and sentiment data for the Market Agent
- Handles missing values
- Normalizes numerical columns for LSTM input
- Creates time-series sequences
- Merges price and sentiment data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketAgentPreprocessor:
    """
    Preprocesses data for Market Agent (time-series prediction)
    """
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        """
        Initialize the Market Agent Preprocessor
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = StandardScaler()
    
    def load_price_data(self, filename: str = "historical_prices_combined.csv") -> pd.DataFrame:
        """
        Load historical price data from CSV
        
        Args:
            filename: Name of the price data file
            
        Returns:
            DataFrame with price data
        """
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Price data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded price data: {len(df)} records")
        return df
    
    def load_sentiment_data(self, filename: str = "market_sentiment.csv") -> pd.DataFrame:
        """
        Load sentiment data from CSV
        
        Args:
            filename: Name of the sentiment data file
            
        Returns:
            DataFrame with sentiment data
        """
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            logger.warning(f"Sentiment data file not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded sentiment data: {len(df)} records")
        return df
    
    def preprocess_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical price data
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing price data...")
        
        # Convert Date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'], utc=True)
        else:
            raise ValueError("Date or Datetime column not found in price data")
        
        # Sort by Date
        df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        # Handle missing values
        # Forward fill for missing prices within each symbol
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df.groupby('Symbol')[col].ffill()
                df[col] = df.groupby('Symbol')[col].bfill()
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Remove rows with any remaining NaN values
        df = df.dropna()
        
        logger.info(f"Preprocessed price data: {len(df)} records")
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for each symbol
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info("Calculating technical indicators...")
        
        df = df.copy()
        
        # Group by symbol and calculate indicators
        for symbol in df['Symbol'].unique():
            symbol_mask = df['Symbol'] == symbol
            symbol_df = df[symbol_mask].copy()
            
            if len(symbol_df) < 20:  # Need enough data for indicators
                continue
            
            # Moving averages
            symbol_df['MA_5'] = symbol_df['Close'].rolling(window=5).mean()
            symbol_df['MA_10'] = symbol_df['Close'].rolling(window=10).mean()
            symbol_df['MA_20'] = symbol_df['Close'].rolling(window=20).mean()
            symbol_df['MA_50'] = symbol_df['Close'].rolling(window=50).mean()
            
            # RSI (Relative Strength Index)
            symbol_df['RSI'] = self._calculate_rsi(symbol_df['Close'], period=14)
            
            # MACD
            symbol_df['MACD'], symbol_df['MACD_signal'] = self._calculate_macd(symbol_df['Close'])
            
            # Bollinger Bands
            symbol_df['BB_upper'], symbol_df['BB_middle'], symbol_df['BB_lower'] = \
                self._calculate_bollinger_bands(symbol_df['Close'])
            
            # Price change features
            symbol_df['Price_Change'] = symbol_df['Close'].pct_change()
            symbol_df['Price_Change_5d'] = symbol_df['Close'].pct_change(5)
            symbol_df['Price_Change_20d'] = symbol_df['Close'].pct_change(20)
            
            # Volume features
            symbol_df['Volume_MA'] = symbol_df['Volume'].rolling(window=20).mean()
            symbol_df['Volume_Ratio'] = symbol_df['Volume'] / symbol_df['Volume_MA']
            
            # Update dataframe
            df.loc[symbol_mask, symbol_df.columns] = symbol_df
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def preprocess_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess sentiment data and aggregate by date
        
        Args:
            df: Raw sentiment DataFrame
            
        Returns:
            Preprocessed DataFrame with daily sentiment aggregates
        """
        if df.empty:
            logger.warning("Sentiment data is empty")
            return pd.DataFrame()
        
        logger.info("Preprocessing sentiment data...")
        
        # Convert date columns to datetime
        date_columns = ['created_utc', 'published']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df['Date'] = df[col].dt.date
                break
        
        if 'Date' not in df.columns:
            logger.warning("No date column found in sentiment data")
            return pd.DataFrame()
        
        # Aggregate sentiment by date
        sentiment_cols = ['sentiment_compound', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
        available_cols = [col for col in sentiment_cols if col in df.columns]
        
        if available_cols:
            daily_sentiment = df.groupby('Date')[available_cols].agg(['mean', 'std', 'count']).reset_index()
            daily_sentiment.columns = ['Date', 'sentiment_mean', 'sentiment_std', 'sentiment_count',
                                     'positive_mean', 'positive_std', 'positive_count',
                                     'neutral_mean', 'neutral_std', 'neutral_count',
                                     'negative_mean', 'negative_std', 'negative_count'][:len(daily_sentiment.columns)]
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
            logger.info(f"Aggregated sentiment data: {len(daily_sentiment)} days")
            return daily_sentiment
        else:
            return pd.DataFrame()
    
    def merge_price_sentiment_data(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge price and sentiment data by date
        
        Args:
            price_df: Preprocessed price DataFrame
            sentiment_df: Preprocessed sentiment DataFrame
            
        Returns:
            Merged DataFrame
        """
        if sentiment_df.empty:
            logger.warning("No sentiment data to merge")
            return price_df
        
        logger.info("Merging price and sentiment data...")
        
        # Ensure Date is datetime
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        
        # Merge on Date
        merged_df = price_df.merge(
            sentiment_df,
            on='Date',
            how='left'
        )
        
        # Forward fill sentiment for missing dates
        sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col.lower() or 'positive' in col.lower() or 'negative' in col.lower() or 'neutral' in col.lower()]
        for col in sentiment_cols:
            merged_df[col] = merged_df.groupby('Symbol')[col].ffill()
            merged_df[col] = merged_df.groupby('Symbol')[col].fillna(0)  # Fill remaining NaN with 0
        
        logger.info(f"Merged data: {len(merged_df)} records")
        return merged_df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_column: str = 'Close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            df: Preprocessed DataFrame
            sequence_length: Length of input sequence
            prediction_horizon: Number of days ahead to predict
            target_column: Column to predict
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        logger.info(f"Creating sequences (length={sequence_length}, horizon={prediction_horizon})...")
        
        # Select features for LSTM
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_middle', 'BB_lower',
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d',
            'Volume_Ratio'
        ]
        
        # Add sentiment columns if available
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'positive' in col.lower() or 'negative' in col.lower()]
        feature_columns.extend([col for col in sentiment_cols if col in df.columns])
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        sequences_X = []
        sequences_y = []
        
        # Create sequences for each symbol
        for symbol in df['Symbol'].unique():
            symbol_df = df[df['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
            
            if len(symbol_df) < sequence_length + prediction_horizon:
                continue
            
            # Extract features
            features = symbol_df[available_features].values
            
            # Normalize features
            features_normalized = self.price_scaler.fit_transform(features)
            
            # Create sequences
            for i in range(len(features_normalized) - sequence_length - prediction_horizon + 1):
                X = features_normalized[i:i + sequence_length]
                y = symbol_df[target_column].iloc[i + sequence_length + prediction_horizon - 1]
                
                sequences_X.append(X)
                sequences_y.append(y)
        
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        
        logger.info(f"Created {len(X)} sequences")
        return X, y
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "market_agent_processed.csv"
    ):
        """
        Save processed data to CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        filepath = self.processed_data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_all(
        self,
        price_filename: str = "historical_prices_combined.csv",
        sentiment_filename: str = "market_sentiment.csv",
        output_filename: str = "market_agent_processed.csv"
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for Market Agent
        
        Args:
            price_filename: Name of price data file
            sentiment_filename: Name of sentiment data file
            output_filename: Output filename
            
        Returns:
            Processed DataFrame
        """
        # Load data
        price_df = self.load_price_data(price_filename)
        sentiment_df = self.load_sentiment_data(sentiment_filename)
        
        # Preprocess
        price_df = self.preprocess_price_data(price_df)
        sentiment_df = self.preprocess_sentiment_data(sentiment_df)
        
        # Merge
        merged_df = self.merge_price_sentiment_data(price_df, sentiment_df)
        
        # Save
        self.save_processed_data(merged_df, output_filename)
        
        return merged_df

