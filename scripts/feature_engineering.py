"""
Feature engineering module for DigiCow Farmer Training Adoption Challenge.
Optimized for performance with vectorized operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class with optimized methods.
    """
    
    def __init__(self, topic_categories: Dict[str, List[str]] = None):
        """
        Initialize feature engineer.
        
        Args:
            topic_categories: Dictionary mapping category names to keywords
        """
        self.topic_categories = topic_categories or {}
        self.topic_encodings = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from training dates and timing.
        Vectorized for performance.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")
        
        # Extract date components
        df['first_training_day_of_week'] = df['first_training_date'].dt.dayofweek.astype('int8')
        df['first_training_month'] = df['first_training_date'].dt.month.astype('int8')
        df['first_training_day'] = df['first_training_date'].dt.day.astype('int8')
        df['first_training_week_of_month'] = ((df['first_training_date'].dt.day - 1) // 7 + 1).astype('int8')
        
        # Days to second training features
        df['attended_second_within_7days'] = (
            (df['days_to_second_training'] <= 7) & (df['days_to_second_training'].notna())
        ).astype('int8')
        
        df['attended_second_same_day'] = (
            (df['days_to_second_training'] == 0) & (df['days_to_second_training'].notna())
        ).astype('int8')
        
        # Bin days to second training
        df['days_to_second_binned'] = pd.cut(
            df['days_to_second_training'],
            bins=[-1, 0, 7, 14, 30, np.inf],
            labels=['same_day', '1-7days', '8-14days', '15-30days', '30+days']
        ).astype('category')
        
        # Fill NaN with 'no_second'
        df['days_to_second_binned'] = df['days_to_second_binned'].cat.add_categories(['no_second'])
        df['days_to_second_binned'] = df['days_to_second_binned'].fillna('no_second')
        
        logger.info(f"Created {5} temporal features")
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement intensity features.
        Vectorized operations for speed.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engagement features
        """
        logger.info("Creating engagement features...")
        
        # Training frequency
        df['training_frequency_30d'] = (df['num_trainings_30d'] / 30).astype('float32')
        df['training_frequency_60d'] = (df['num_trainings_60d'] / 60).astype('float32')
        
        # Training acceleration (avoid division by zero)
        df['training_acceleration'] = np.where(
            df['num_trainings_60d'] > 0,
            df['num_trainings_30d'] / df['num_trainings_60d'],
            0
        ).astype('float32')
        
        # Engagement score (weighted recent activity)
        df['engagement_score'] = (
            df['num_trainings_30d'] * 2 + df['num_trainings_60d']
        ).astype('float32')
        
        # Repeat rate
        df['repeat_rate'] = np.where(
            df['num_total_trainings'] > 0,
            df['num_repeat_trainings'] / df['num_total_trainings'],
            0
        ).astype('float32')
        
        # Training consistency
        df['training_consistency'] = np.where(
            df['num_trainings_60d'] > 0,
            df['num_trainings_30d'] / df['num_trainings_60d'],
            0
        ).astype('float32')
        
        # Engagement level flags
        df['is_highly_engaged'] = (df['num_total_trainings'] > 10).astype('int8')
        df['is_super_engaged'] = (df['num_total_trainings'] > 50).astype('int8')
        df['is_low_engaged'] = (df['num_total_trainings'] <= 3).astype('int8')
        
        # Training gap indicators
        df['has_training_gap'] = (df['days_to_second_training'] > 7).fillna(False).astype('int8')
        df['immediate_follower'] = (df['days_to_second_training'] == 0).fillna(False).astype('int8')
        
        # Early engagement ratio
        df['early_engagement_ratio'] = np.where(
            df['num_total_trainings'] > 0,
            df['num_trainings_30d'] / df['num_total_trainings'],
            0
        ).astype('float32')
        
        logger.info(f"Created {12} engagement features")
        return df
    
    def create_topic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from topics_parsed column.
        Optimized with vectorized operations where possible.
        
        Args:
            df: Input DataFrame with 'topics_parsed' column
        
        Returns:
            DataFrame with topic features
        """
        logger.info("Creating topic features...")
        
        # Number of unique topics
        df['num_unique_topics'] = df['topics_parsed'].apply(
            lambda x: len(set(x)) if isinstance(x, list) else 0
        ).astype('int16')
        
        # Topic diversity ratio
        df['topic_diversity_ratio'] = np.where(
            df['num_total_trainings'] > 0,
            df['num_unique_topics'] / df['num_total_trainings'],
            0
        ).astype('float32')
        
        # Topic repetition rate
        df['topic_repetition_rate'] = (1 - df['topic_diversity_ratio']).astype('float32')
        
        # Trainings per unique topic
        df['trainings_per_unique_topic'] = np.where(
            df['num_unique_topics'] > 0,
            df['num_total_trainings'] / df['num_unique_topics'],
            0
        ).astype('float32')
        
        # Topic category features
        for category, keywords in self.topic_categories.items():
            col_name = f'has_{category}_topics'
            df[col_name] = df['topics_parsed'].apply(
                lambda topics: self._has_category_topics(topics, keywords)
            ).astype('int8')
        
        # Count of topic categories
        category_cols = [f'has_{cat}_topics' for cat in self.topic_categories.keys()]
        if category_cols:
            df['topic_category_count'] = df[category_cols].sum(axis=1).astype('int8')
        
        # Topic focus score (most common topic frequency)
        df['topic_focus_score'] = df['topics_parsed'].apply(
            self._calculate_topic_focus
        ).astype('float32')
        
        logger.info(f"Created {7 + len(self.topic_categories)} topic features")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features with binary encoding.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with demographic features
        """
        logger.info("Creating demographic features...")
        
        # Binary encodings
        df['is_female'] = (df['gender'] == 'Female').astype('int8')
        df['is_above_35'] = (df['age'] == 'Above 35').astype('int8')
        df['registration_is_manual'] = (df['registration'] == 'Manual').astype('int8')
        
        # Age-gender interaction
        df['age_gender_combo'] = (
            df['age'].astype(str) + '_' + df['gender'].astype(str)
        ).astype('category')
        
        logger.info(f"Created {4} demographic features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Cooperative x engagement
        df['cooperative_x_engagement'] = (
            df['belong_to_cooperative'] * df['num_trainings_30d']
        ).astype('float32')
        
        # Gender x cooperative
        df['female_x_cooperative'] = (
            df['is_female'] * df['belong_to_cooperative']
        ).astype('int8')
        
        # Age x trainings
        df['above35_x_trainings'] = (
            df['is_above_35'] * df['num_total_trainings']
        ).astype('float32')
        
        # Registration x second training
        df['manual_x_second_training'] = (
            df['registration_is_manual'] * df['has_second_training']
        ).astype('int8')
        
        # Topics x engagement
        df['topics_x_engagement'] = (
            df['num_unique_topics'] * df['num_trainings_30d']
        ).astype('float32')
        
        logger.info(f"Created {5} interaction features")
        return df
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and derived features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with ratio features
        """
        logger.info("Creating ratio features...")
        
        # Sustained engagement ratio
        df['sustained_engagement_ratio'] = np.where(
            df['num_total_trainings'] > 0,
            df['num_trainings_60d'] / df['num_total_trainings'],
            0
        ).astype('float32')
        
        # Training intensity delta
        df['training_intensity_delta'] = (
            df['num_trainings_30d'] - (df['num_trainings_60d'] - df['num_trainings_30d'])
        ).astype('float32')
        
        logger.info(f"Created {2} ratio features")
        return df
    
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create indicators for missing values.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with missing indicators
        """
        logger.info("Creating missing value indicators...")
        
        df['missing_days_to_second'] = df['days_to_second_training'].isna().astype('int8')
        
        logger.info(f"Created {1} missing indicator feature")
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in optimized order.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create features in order
        df = self.create_temporal_features(df)
        df = self.create_engagement_features(df)
        df = self.create_topic_features(df)
        df = self.create_demographic_features(df)
        df = self.create_interaction_features(df)
        df = self.create_ratio_features(df)
        df = self.create_missing_indicators(df)
        
        logger.info("Feature engineering completed!")
        return df
    
    # Helper methods
    @staticmethod
    def _has_category_topics(topics: list, keywords: List[str]) -> int:
        """Check if any topic contains category keywords."""
        if not isinstance(topics, list):
            return 0
        
        topics_lower = ' '.join([str(t).lower() for t in topics])
        return int(any(keyword in topics_lower for keyword in keywords))
    
    @staticmethod
    def _calculate_topic_focus(topics: list) -> float:
        """Calculate focus score based on most common topic."""
        if not isinstance(topics, list) or len(topics) == 0:
            return 0.0
        
        # Count topic frequencies
        topic_counts = Counter(topics)
        most_common_count = topic_counts.most_common(1)[0][1]
        
        return most_common_count / len(topics)
