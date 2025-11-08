"""
User Profile Generation Module

This module generates simulated user profiles with realistic financial data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: int
    age: int
    income: float
    investment_horizon: int  # in years
    risk_preference: str  # conservative, moderate, aggressive
    existing_portfolio_value: float
    savings_rate: float  # percentage of income
    financial_goals: str


class UserProfileGenerator:
    """
    Generates simulated user profiles with realistic financial data
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the User Profile Generator
        
        Args:
            output_dir: Directory to save generated profiles
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Financial goals templates
        self.financial_goals = [
            "Retirement Planning",
            "House Down Payment",
            "Children's Education",
            "Emergency Fund",
            "Wealth Accumulation",
            "Early Retirement",
            "Debt Payoff",
            "Vacation Fund",
        ]
    
    def generate_age(self, age_range: Tuple[int, int] = (25, 65)) -> int:
        """
        Generate age following a realistic distribution
        More users in middle age ranges
        
        Args:
            age_range: Tuple of (min_age, max_age)
            
        Returns:
            Generated age
        """
        # Beta distribution skewed towards middle ages
        age = int(np.random.beta(2, 2) * (age_range[1] - age_range[0]) + age_range[0])
        return np.clip(age, age_range[0], age_range[1])
    
    def generate_income(self, age: int, income_range: Tuple[float, float] = (30000, 200000)) -> float:
        """
        Generate income correlated with age
        Income typically increases with age up to a point
        
        Args:
            age: User's age
            income_range: Tuple of (min_income, max_income)
            
        Returns:
            Generated annual income
        """
        # Income correlates with age (peaks around 45-50)
        age_factor = np.sin((age - 25) * np.pi / 40) if 25 <= age <= 65 else 0.5
        base_income = income_range[0] + (income_range[1] - income_range[0]) * (0.3 + 0.7 * age_factor)
        
        # Add some randomness
        noise = np.random.normal(0, (income_range[1] - income_range[0]) * 0.1)
        income = base_income + noise
        
        return np.clip(income, income_range[0], income_range[1])
    
    def generate_investment_horizon(
        self,
        age: int,
        horizon_range: Tuple[int, int] = (1, 30)
    ) -> int:
        """
        Generate investment horizon based on age
        Younger users typically have longer horizons
        
        Args:
            age: User's age
            horizon_range: Tuple of (min_horizon, max_horizon) in years
            
        Returns:
            Generated investment horizon in years
        """
        # Younger users have longer horizons (retirement age ~65)
        retirement_age = 65
        years_to_retirement = max(1, retirement_age - age)
        
        # Add some variability
        horizon = int(years_to_retirement * (0.8 + np.random.random() * 0.4))
        
        return np.clip(horizon, horizon_range[0], min(horizon_range[1], years_to_retirement))
    
    def generate_risk_preference(
        self,
        age: int,
        income: float,
        investment_horizon: int,
        risk_preferences: List[str] = ["conservative", "moderate", "aggressive"]
    ) -> str:
        """
        Generate risk preference based on age, income, and investment horizon
        
        Args:
            age: User's age
            income: User's income
            investment_horizon: Investment horizon in years
            risk_preferences: List of possible risk preferences
            
        Returns:
            Generated risk preference
        """
        # Factors that influence risk tolerance
        age_factor = 1.0 - (age - 25) / 40  # Decreases with age
        income_factor = (income - 30000) / 170000  # Increases with income
        horizon_factor = investment_horizon / 30  # Increases with horizon
        
        # Combined risk score
        risk_score = (age_factor * 0.3 + income_factor * 0.3 + horizon_factor * 0.4)
        
        # Map to risk preference
        if risk_score < 0.33:
            return risk_preferences[0]  # conservative
        elif risk_score < 0.67:
            return risk_preferences[1]  # moderate
        else:
            return risk_preferences[2]  # aggressive
    
    def generate_portfolio_value(self, income: float, age: int) -> float:
        """
        Generate existing portfolio value based on income and age
        
        Args:
            income: User's annual income
            age: User's age
            
        Returns:
            Generated portfolio value
        """
        # Assume users save 10-20% of income annually
        annual_savings_rate = np.random.uniform(0.10, 0.20)
        years_investing = max(1, age - 25)
        
        # Simple compound growth assumption (7% annual return)
        annual_return = 0.07
        portfolio_value = 0
        
        for year in range(years_investing):
            portfolio_value = portfolio_value * (1 + annual_return) + income * annual_savings_rate
        
        return max(0, portfolio_value)
    
    def generate_savings_rate(self) -> float:
        """
        Generate savings rate (percentage of income)
        
        Returns:
            Savings rate as decimal (e.g., 0.15 for 15%)
        """
        return np.random.uniform(0.05, 0.25)
    
    def generate_financial_goals(self, age: int) -> str:
        """
        Generate financial goals based on age
        
        Args:
            age: User's age
            
        Returns:
            Financial goal string
        """
        # Age-appropriate goals
        if age < 30:
            goals = ["House Down Payment", "Emergency Fund", "Debt Payoff", "Vacation Fund"]
        elif age < 45:
            goals = ["Children's Education", "Retirement Planning", "Wealth Accumulation", "House Down Payment"]
        else:
            goals = ["Retirement Planning", "Early Retirement", "Wealth Accumulation"]
        
        return np.random.choice(goals)
    
    def generate_user_profile(
        self,
        user_id: int,
        age_range: Tuple[int, int] = (25, 65),
        income_range: Tuple[float, float] = (30000, 200000),
        horizon_range: Tuple[int, int] = (1, 30),
        risk_preferences: List[str] = ["conservative", "moderate", "aggressive"]
    ) -> UserProfile:
        """
        Generate a single user profile
        
        Args:
            user_id: Unique user ID
            age_range: Age range tuple
            income_range: Income range tuple
            horizon_range: Investment horizon range tuple
            risk_preferences: List of risk preferences
            
        Returns:
            UserProfile object
        """
        age = self.generate_age(age_range)
        income = self.generate_income(age, income_range)
        investment_horizon = self.generate_investment_horizon(age, horizon_range)
        risk_preference = self.generate_risk_preference(age, income, investment_horizon, risk_preferences)
        existing_portfolio_value = self.generate_portfolio_value(income, age)
        savings_rate = self.generate_savings_rate()
        financial_goals = self.generate_financial_goals(age)
        
        return UserProfile(
            user_id=user_id,
            age=age,
            income=income,
            investment_horizon=investment_horizon,
            risk_preference=risk_preference,
            existing_portfolio_value=existing_portfolio_value,
            savings_rate=savings_rate,
            financial_goals=financial_goals
        )
    
    def generate_profiles(
        self,
        num_profiles: int = 1000,
        age_range: Tuple[int, int] = (25, 65),
        income_range: Tuple[float, float] = (30000, 200000),
        horizon_range: Tuple[int, int] = (1, 30),
        risk_preferences: List[str] = ["conservative", "moderate", "aggressive"]
    ) -> pd.DataFrame:
        """
        Generate multiple user profiles
        
        Args:
            num_profiles: Number of profiles to generate
            age_range: Age range tuple
            income_range: Income range tuple
            horizon_range: Investment horizon range tuple
            risk_preferences: List of risk preferences
            
        Returns:
            DataFrame with user profiles
        """
        logger.info(f"Generating {num_profiles} user profiles...")
        
        profiles = []
        for user_id in range(1, num_profiles + 1):
            profile = self.generate_user_profile(
                user_id=user_id,
                age_range=age_range,
                income_range=income_range,
                horizon_range=horizon_range,
                risk_preferences=risk_preferences
            )
            profiles.append(profile)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(p) for p in profiles])
        
        logger.info(f"Generated {len(df)} user profiles successfully")
        return df
    
    def save_profiles(self, df: pd.DataFrame, filename: str = "user_profiles"):
        """
        Save user profiles to CSV file
        
        Args:
            df: DataFrame with user profiles
            filename: Output filename (without extension)
        """
        filepath = self.output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved user profiles to {filepath}")

