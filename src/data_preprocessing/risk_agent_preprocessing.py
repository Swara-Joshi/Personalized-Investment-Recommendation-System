"""
Risk Agent Data Preprocessing

This module preprocesses user profile data for the Risk Agent
- Handles missing values
- Encodes categorical variables
- Normalizes numerical features
- Creates risk score labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAgentPreprocessor:
    """
    Preprocesses data for Risk Agent (user profiling & risk scoring)
    """
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        """
        Initialize the Risk Agent Preprocessor
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
    
    def load_user_profiles(self, filename: str = "user_profiles.csv") -> pd.DataFrame:
        """
        Load user profile data from CSV
        
        Args:
            filename: Name of the user profile file
            
        Returns:
            DataFrame with user profiles
        """
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"User profile file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded user profiles: {len(df)} records")
        return df
    
    def preprocess_user_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess user profile data
        
        Args:
            df: Raw user profile DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing user profiles...")
        
        df = df.copy()
        
        # Handle missing values
        # Fill missing numerical values with median
        numerical_cols = ['age', 'income', 'investment_horizon', 'existing_portfolio_value', 'savings_rate']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = ['risk_preference', 'financial_goals']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Calculate risk score
        df = self._calculate_risk_score(df)
        
        # Normalize numerical features
        df = self._normalize_features(df)
        
        logger.info(f"Preprocessed user profiles: {len(df)} records")
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: DataFrame with categorical variables
            
        Returns:
            DataFrame with encoded variables
        """
        logger.info("Encoding categorical variables...")
        
        df = df.copy()
        
        # Encode risk_preference
        if 'risk_preference' in df.columns:
            risk_mapping = {'conservative': 0, 'moderate': 1, 'aggressive': 2}
            df['risk_preference_encoded'] = df['risk_preference'].map(risk_mapping).fillna(1)
        
        # Encode financial_goals using label encoding
        if 'financial_goals' in df.columns:
            df['financial_goals_encoded'] = self.label_encoder.fit_transform(df['financial_goals'])
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from user profile data
        
        Args:
            df: DataFrame with user profiles
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Creating derived features...")
        
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 30, 40, 50, 60, 100],
            labels=['20s', '30s', '40s', '50s', '60+']
        )
        df['age_group_encoded'] = pd.Categorical(df['age_group']).codes
        
        # Income groups
        df['income_group'] = pd.cut(
            df['income'],
            bins=[0, 50000, 100000, 150000, 200000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High', 'Ultra High']
        )
        df['income_group_encoded'] = pd.Categorical(df['income_group']).codes
        
        # Investment horizon groups
        df['horizon_group'] = pd.cut(
            df['investment_horizon'],
            bins=[0, 3, 5, 10, 20, 100],
            labels=['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long']
        )
        df['horizon_group_encoded'] = pd.Categorical(df['horizon_group']).codes
        
        # Portfolio value to income ratio
        df['portfolio_income_ratio'] = df['existing_portfolio_value'] / df['income']
        df['portfolio_income_ratio'] = df['portfolio_income_ratio'].replace([np.inf, -np.inf], 0)
        df['portfolio_income_ratio'] = df['portfolio_income_ratio'].fillna(0)
        
        # Savings rate categories
        df['savings_rate_category'] = pd.cut(
            df['savings_rate'],
            bins=[0, 0.10, 0.15, 0.20, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        df['savings_rate_category_encoded'] = pd.Categorical(df['savings_rate_category']).codes
        
        return df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk score for each user
        
        Args:
            df: DataFrame with user profiles
            
        Returns:
            DataFrame with risk scores
        """
        logger.info("Calculating risk scores...")
        
        df = df.copy()
        
        # Normalize individual factors to 0-1 scale
        age_score = 1 - (df['age'] - 25) / 40  # Younger = higher risk tolerance
        age_score = np.clip(age_score, 0, 1)
        
        income_score = (df['income'] - 30000) / 170000  # Higher income = higher risk tolerance
        income_score = np.clip(income_score, 0, 1)
        
        horizon_score = df['investment_horizon'] / 30  # Longer horizon = higher risk tolerance
        horizon_score = np.clip(horizon_score, 0, 1)
        
        preference_score = df['risk_preference_encoded'] / 2  # 0=conservative, 1=moderate, 2=aggressive
        
        # Weighted combination
        weights = {
            'age': 0.25,
            'income': 0.20,
            'horizon': 0.30,
            'preference': 0.25
        }
        
        df['risk_score'] = (
            age_score * weights['age'] +
            income_score * weights['income'] +
            horizon_score * weights['horizon'] +
            preference_score * weights['preference']
        )
        
        # Categorize risk level
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['conservative', 'moderate', 'aggressive']
        )
        df['risk_level_encoded'] = pd.Categorical(df['risk_level']).codes
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features for ML models
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        df = df.copy()
        
        # Features to normalize
        features_to_normalize = [
            'age', 'income', 'investment_horizon',
            'existing_portfolio_value', 'savings_rate',
            'portfolio_income_ratio'
        ]
        
        available_features = [col for col in features_to_normalize if col in df.columns]
        
        if available_features:
            df[available_features] = self.scaler.fit_transform(df[available_features])
        
        return df
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "risk_agent_processed.csv"
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
        input_filename: str = "user_profiles.csv",
        output_filename: str = "risk_agent_processed.csv"
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for Risk Agent
        
        Args:
            input_filename: Name of input user profile file
            output_filename: Output filename
            
        Returns:
            Processed DataFrame
        """
        # Load data
        df = self.load_user_profiles(input_filename)
        
        # Preprocess
        df = self.preprocess_user_profiles(df)
        
        # Save
        self.save_processed_data(df, output_filename)
        
        return df

