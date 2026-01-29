"""
Unit tests for feature engineering.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import unittest
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test data."""
        self.fe = FeatureEngineer(topic_categories={
            'dairy': ['dairy', 'cow', 'milk'],
            'poultry': ['poultry', 'chicken']
        })
        
        # Create sample DataFrame
        self.df = pd.DataFrame({
            'first_training_date': pd.to_datetime(['2024-02-19', '2024-02-20', '2024-02-21']),
            'days_to_second_training': [0.0, 7.0, np.nan],
            'num_trainings_30d': [5.0, 10.0, 2.0],
            'num_trainings_60d': [10.0, 15.0, 3.0],
            'num_total_trainings': [20, 30, 5],
            'num_repeat_trainings': [19, 29, 4],
            'has_second_training': [1, 1, 0],
            'topics_parsed': [
                ['Dairy Cow Feeding', 'Poultry Feeding'],
                ['Dairy Cow Feeding', 'Dairy Cow Feeding', 'Ndume App'],
                ['Poultry Feeding']
            ],
            'gender': pd.Categorical(['Female', 'Male', 'Female']),
            'age': pd.Categorical(['Above 35', 'Below 35', 'Above 35']),
            'registration': pd.Categorical(['Manual', 'Ussd', 'Manual']),
            'belong_to_cooperative': [1, 0, 1]
        })
    
    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        result = self.fe.create_temporal_features(self.df.copy())
        
        # Check new columns exist
        self.assertIn('first_training_day_of_week', result.columns)
        self.assertIn('first_training_month', result.columns)
        self.assertIn('attended_second_within_7days', result.columns)
        self.assertIn('days_to_second_binned', result.columns)
        
        # Check values
        self.assertEqual(result['attended_second_within_7days'].iloc[0], 1)
        self.assertEqual(result['attended_second_within_7days'].iloc[1], 1)
        self.assertEqual(result['attended_second_within_7days'].iloc[2], 0)
    
    def test_create_engagement_features(self):
        """Test engagement feature creation."""
        result = self.fe.create_engagement_features(self.df.copy())
        
        # Check new columns exist
        self.assertIn('training_frequency_30d', result.columns)
        self.assertIn('engagement_score', result.columns)
        self.assertIn('is_highly_engaged', result.columns)
        
        # Check calculations
        self.assertGreater(result['engagement_score'].iloc[0], 0)
        self.assertEqual(result['is_highly_engaged'].iloc[0], 1)  # 20 trainings > 10
        self.assertEqual(result['is_highly_engaged'].iloc[2], 0)  # 5 trainings <= 10
    
    def test_create_topic_features(self):
        """Test topic feature creation."""
        result = self.fe.create_topic_features(self.df.copy())
        
        # Check new columns exist
        self.assertIn('num_unique_topics', result.columns)
        self.assertIn('topic_diversity_ratio', result.columns)
        self.assertIn('has_dairy_topics', result.columns)
        self.assertIn('has_poultry_topics', result.columns)
        
        # Check values
        self.assertEqual(result['num_unique_topics'].iloc[0], 2)
        self.assertEqual(result['has_dairy_topics'].iloc[0], 1)
        self.assertEqual(result['has_poultry_topics'].iloc[0], 1)
        self.assertEqual(result['has_dairy_topics'].iloc[2], 0)
    
    def test_create_demographic_features(self):
        """Test demographic feature creation."""
        result = self.fe.create_demographic_features(self.df.copy())
        
        # Check new columns exist
        self.assertIn('is_female', result.columns)
        self.assertIn('is_above_35', result.columns)
        self.assertIn('registration_is_manual', result.columns)
        
        # Check values
        self.assertEqual(result['is_female'].iloc[0], 1)
        self.assertEqual(result['is_female'].iloc[1], 0)
        self.assertEqual(result['is_above_35'].iloc[0], 1)
        self.assertEqual(result['is_above_35'].iloc[1], 0)
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        # Add required columns
        df = self.df.copy()
        df['is_female'] = (df['gender'] == 'Female').astype(int)
        df['is_above_35'] = (df['age'] == 'Above 35').astype(int)
        df['registration_is_manual'] = (df['registration'] == 'Manual').astype(int)
        df['num_unique_topics'] = [2, 2, 1]
        
        result = self.fe.create_interaction_features(df)
        
        # Check new columns exist
        self.assertIn('cooperative_x_engagement', result.columns)
        self.assertIn('female_x_cooperative', result.columns)
        
        # Check calculations
        expected = df['belong_to_cooperative'].iloc[0] * df['num_trainings_30d'].iloc[0]
        self.assertEqual(result['cooperative_x_engagement'].iloc[0], expected)
    
    def test_create_all_features(self):
        """Test complete feature engineering pipeline."""
        result = self.fe.create_all_features(self.df.copy())
        
        # Check that DataFrame has more columns than original
        self.assertGreater(len(result.columns), len(self.df.columns))
        
        # Check no NaN in critical features (except where expected)
        self.assertFalse(result['is_female'].isna().any())
        self.assertFalse(result['engagement_score'].isna().any())
    
    def test_helper_has_category_topics(self):
        """Test topic category detection helper."""
        topics = ['Dairy Cow Feeding', 'Poultry Management']
        keywords = ['dairy', 'cow']
        
        result = self.fe._has_category_topics(topics, keywords)
        self.assertEqual(result, 1)
        
        # Test no match
        keywords = ['sheep', 'goat']
        result = self.fe._has_category_topics(topics, keywords)
        self.assertEqual(result, 0)
    
    def test_helper_calculate_topic_focus(self):
        """Test topic focus calculation helper."""
        topics = ['Topic A', 'Topic A', 'Topic A', 'Topic B']
        result = self.fe._calculate_topic_focus(topics)
        self.assertEqual(result, 0.75)  # 3 out of 4
        
        # Test empty list
        result = self.fe._calculate_topic_focus([])
        self.assertEqual(result, 0.0)


class TestFeatureEngineeringEdgeCases(unittest.TestCase):
    """Test edge cases in feature engineering."""
    
    def setUp(self):
        """Set up test data with edge cases."""
        self.fe = FeatureEngineer()
    
    def test_zero_trainings(self):
        """Test handling of zero trainings."""
        df = pd.DataFrame({
            'num_trainings_30d': [0.0],
            'num_trainings_60d': [0.0],
            'num_total_trainings': [0],
            'num_repeat_trainings': [0],
            'days_to_second_training': [np.nan],
            'has_second_training': [0],
            'topics_parsed': [[]]
        })
        
        result = self.fe.create_engagement_features(df)
        
        # Should not raise errors
        self.assertEqual(result['training_acceleration'].iloc[0], 0)
        self.assertEqual(result['repeat_rate'].iloc[0], 0)
    
    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'days_to_second_training': [np.nan, np.nan],
            'num_trainings_30d': [5.0, 10.0],
            'num_trainings_60d': [10.0, 15.0],
            'num_total_trainings': [20, 30],
            'num_repeat_trainings': [19, 29],
            'has_second_training': [0, 0],
            'first_training_date': pd.to_datetime(['2024-02-19', '2024-02-20'])
        })
        
        result = self.fe.create_temporal_features(df)
        result = self.fe.create_missing_indicators(result)
        
        # Check missing indicator
        self.assertEqual(result['missing_days_to_second'].iloc[0], 1)
        self.assertEqual(result['attended_second_within_7days'].iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
