"""
Recommendation Agent Data Preprocessing

This module preprocesses data for the Recommendation Agent
- Combines user profiles with market data
- Creates portfolio recommendation features
- Prepares data for portfolio optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationAgentPreprocessor:
    """
    Preprocesses data for Recommendation Agent (portfolio advice & optimization)
    """
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        """
        Initialize the Recommendation Agent Preprocessor
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_user_profiles(self, filename: str = "user_profiles.csv") -> pd.DataFrame:
        """Load user profile data"""
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"User profile file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def load_price_data(self, filename: str = "historical_prices_combined.csv") -> pd.DataFrame:
        """Load historical price data"""
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Price data file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def load_market_data(self, filename: str = "market_agent_processed.csv") -> pd.DataFrame:
        """Load processed market data"""
        filepath = self.processed_data_dir / filename
        if not filepath.exists():
            logger.warning(f"Market data file not found: {filepath}")
            return pd.DataFrame()
        return pd.read_csv(filepath)
    
    def calculate_stock_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stock features for portfolio recommendation
        
        Args:
            price_df: Historical price DataFrame
            
        Returns:
            DataFrame with stock features
        """
        logger.info("Calculating stock features...")
        
        # Convert Date to datetime
        if 'Date' in price_df.columns:
            price_df['Date'] = pd.to_datetime(price_df['Date'], utc=True)
        elif 'Datetime' in price_df.columns:
            price_df['Date'] = pd.to_datetime(price_df['Datetime'], utc=True)
        
        stock_features = []
        
        for symbol in price_df['Symbol'].unique():
            symbol_df = price_df[price_df['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
            
            if len(symbol_df) < 30:
                continue
            
            # Calculate returns
            symbol_df['Returns'] = symbol_df['Close'].pct_change()
            symbol_df['Returns_5d'] = symbol_df['Close'].pct_change(5)
            symbol_df['Returns_30d'] = symbol_df['Close'].pct_change(30)
            symbol_df['Returns_90d'] = symbol_df['Close'].pct_change(90)
            
            # Calculate volatility
            symbol_df['Volatility_30d'] = symbol_df['Returns'].rolling(window=30).std()
            symbol_df['Volatility_90d'] = symbol_df['Returns'].rolling(window=90).std()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            symbol_df['Sharpe_Ratio_30d'] = (
                (symbol_df['Returns'].rolling(window=30).mean() - risk_free_rate) /
                symbol_df['Volatility_30d']
            )
            
            # Calculate maximum drawdown
            symbol_df['Cumulative_Returns'] = (1 + symbol_df['Returns']).cumprod()
            symbol_df['Running_Max'] = symbol_df['Cumulative_Returns'].expanding().max()
            symbol_df['Drawdown'] = (symbol_df['Cumulative_Returns'] - symbol_df['Running_Max']) / symbol_df['Running_Max']
            symbol_df['Max_Drawdown'] = symbol_df['Drawdown'].rolling(window=90).min()
            
            # Get latest values
            latest = symbol_df.iloc[-1]
            
            features = {
                'Symbol': symbol,
                'Current_Price': latest['Close'],
                'Return_30d': latest['Returns_30d'] if not pd.isna(latest['Returns_30d']) else 0,
                'Return_90d': latest['Returns_90d'] if not pd.isna(latest['Returns_90d']) else 0,
                'Volatility_30d': latest['Volatility_30d'] if not pd.isna(latest['Volatility_30d']) else 0,
                'Volatility_90d': latest['Volatility_90d'] if not pd.isna(latest['Volatility_90d']) else 0,
                'Sharpe_Ratio_30d': latest['Sharpe_Ratio_30d'] if not pd.isna(latest['Sharpe_Ratio_30d']) else 0,
                'Max_Drawdown_90d': latest['Max_Drawdown'] if not pd.isna(latest['Max_Drawdown']) else 0,
                'Market_Cap_Category': self._categorize_market_cap(symbol),
                'Sector': self._get_sector(symbol),
            }
            
            stock_features.append(features)
        
        return pd.DataFrame(stock_features)
    
    def _categorize_market_cap(self, symbol: str) -> str:
        """Categorize stock by market cap (simplified)"""
        # This is a simplified categorization
        # In production, you would fetch actual market cap data
        large_cap = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ', 'SPY', 'QQQ', 'VTI', 'VOO', 'DIA']
        if symbol in large_cap:
            return 'Large Cap'
        elif 'ETF' in symbol or symbol in ['IWM', 'GLD', 'TLT']:
            return 'ETF'
        elif 'USD' in symbol:
            return 'Crypto'
        else:
            return 'Mid Cap'
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for stock (simplified)"""
        # Simplified sector mapping
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
            'META': 'Technology', 'NVDA': 'Technology',
            'JPM': 'Financials', 'V': 'Financials',
            'JNJ': 'Healthcare',
            'SPY': 'ETF', 'QQQ': 'ETF', 'VTI': 'ETF', 'VOO': 'ETF',
            'IWM': 'ETF', 'DIA': 'ETF', 'GLD': 'ETF', 'TLT': 'ETF',
            'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'BNB-USD': 'Crypto',
        }
        return sectors.get(symbol, 'Unknown')
    
    def create_recommendation_dataset(
        self,
        user_profiles_df: pd.DataFrame,
        stock_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create dataset for recommendation agent by combining user profiles with stock features
        
        Args:
            user_profiles_df: User profile DataFrame
            stock_features_df: Stock features DataFrame
            
        Returns:
            Combined DataFrame for recommendations
        """
        logger.info("Creating recommendation dataset...")
        
        recommendation_data = []
        
        for _, user in user_profiles_df.iterrows():
            user_id = user['user_id']
            risk_level = user.get('risk_preference', 'moderate')
            
            # Filter stocks based on risk level
            suitable_stocks = self._filter_stocks_by_risk(stock_features_df, risk_level)
            
            for _, stock in suitable_stocks.iterrows():
                recommendation = {
                    'user_id': user_id,
                    'symbol': stock['Symbol'],
                    'user_age': user['age'],
                    'user_income': user['income'],
                    'user_investment_horizon': user['investment_horizon'],
                    'user_risk_preference': risk_level,
                    'user_portfolio_value': user.get('existing_portfolio_value', 0),
                    'user_savings_rate': user.get('savings_rate', 0.15),
                    'stock_return_30d': stock['Return_30d'],
                    'stock_return_90d': stock['Return_90d'],
                    'stock_volatility_30d': stock['Volatility_30d'],
                    'stock_volatility_90d': stock['Volatility_90d'],
                    'stock_sharpe_ratio': stock['Sharpe_Ratio_30d'],
                    'stock_max_drawdown': stock['Max_Drawdown_90d'],
                    'stock_market_cap': stock['Market_Cap_Category'],
                    'stock_sector': stock['Sector'],
                    'recommended_allocation': self._calculate_recommended_allocation(user, stock),
                }
                recommendation_data.append(recommendation)
        
        df = pd.DataFrame(recommendation_data)
        logger.info(f"Created recommendation dataset: {len(df)} records")
        return df
    
    def _filter_stocks_by_risk(self, stock_df: pd.DataFrame, risk_level: str) -> pd.DataFrame:
        """Filter stocks based on user risk level"""
        risk_level = risk_level.lower()
        
        if risk_level == 'conservative':
            # Low volatility, stable returns
            filtered = stock_df[
                (stock_df['Volatility_30d'] < 0.02) &
                (stock_df['Max_Drawdown_90d'] > -0.15)
            ]
        elif risk_level == 'moderate':
            # Medium volatility
            filtered = stock_df[
                (stock_df['Volatility_30d'] < 0.04) &
                (stock_df['Max_Drawdown_90d'] > -0.25)
            ]
        else:  # aggressive
            # Higher risk tolerance
            filtered = stock_df
        
        # If filtering too strict, return top stocks by Sharpe ratio
        if len(filtered) == 0:
            filtered = stock_df.nlargest(20, 'Sharpe_Ratio_30d')
        
        return filtered
    
    def _calculate_recommended_allocation(self, user: pd.Series, stock: pd.Series) -> float:
        """Calculate recommended allocation percentage for a stock"""
        base_allocation = 0.10  # 10% base allocation
        
        # Adjust based on risk level
        risk_level = user.get('risk_preference', 'moderate').lower()
        if risk_level == 'conservative':
            base_allocation *= 0.5
        elif risk_level == 'aggressive':
            base_allocation *= 1.5
        
        # Adjust based on Sharpe ratio
        sharpe_ratio = stock.get('Sharpe_Ratio_30d', 0)
        if sharpe_ratio > 1.0:
            base_allocation *= 1.2
        elif sharpe_ratio < 0:
            base_allocation *= 0.8
        
        return np.clip(base_allocation, 0.01, 0.20)  # Cap between 1% and 20%
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "recommendation_agent_processed.csv"
    ):
        """Save processed data to CSV"""
        filepath = self.processed_data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_all(
        self,
        user_profiles_filename: str = "user_profiles.csv",
        price_filename: str = "historical_prices_combined.csv",
        output_filename: str = "recommendation_agent_processed.csv"
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for Recommendation Agent
        
        Args:
            user_profiles_filename: Name of user profile file
            price_filename: Name of price data file
            output_filename: Output filename
            
        Returns:
            Processed DataFrame
        """
        # Load data
        user_profiles_df = self.load_user_profiles(user_profiles_filename)
        price_df = self.load_price_data(price_filename)
        
        # Calculate stock features
        stock_features_df = self.calculate_stock_features(price_df)
        
        # Create recommendation dataset
        recommendation_df = self.create_recommendation_dataset(user_profiles_df, stock_features_df)
        
        # Save
        self.save_processed_data(recommendation_df, output_filename)
        
        return recommendation_df

