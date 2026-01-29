"""
Unit tests for data utilities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import unittest
import pandas as pd
import numpy as np
from utils.data_utils import (
    parse_topics_list, clean_topic_name, reduce_memory_usage
)


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""
    
    def test_parse_topics_list_valid(self):
        """Test parsing valid topics list."""
        topics_str = "['Dairy Cow Feeding', 'Poultry Feeding', 'Ndume App']"
        result = parse_topics_list(topics_str)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn('Dairy Cow Feeding', result)
    
    def test_parse_topics_list_empty(self):
        """Test parsing empty or NaN topics."""
        result = parse_topics_list(np.nan)
        self.assertEqual(result, [])
        
        result = parse_topics_list("")
        self.assertIsInstance(result, list)
    
    def test_parse_topics_list_malformed(self):
        """Test parsing malformed topics string."""
        topics_str = "Dairy, Poultry, Ndume"
        result = parse_topics_list(topics_str)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
    
    def test_clean_topic_name(self):
        """Test topic name cleaning."""
        # Test lowercase conversion
        result = clean_topic_name("DAIRY COW FEEDING")
        self.assertEqual(result, "dairy cow feeding")
        
        # Test special character removal
        result = clean_topic_name("Dairy (Cow) Feeding!")
        self.assertEqual(result, "dairy cow feeding")
        
        # Test whitespace normalization
        result = clean_topic_name("  Dairy   Cow  ")
        self.assertEqual(result, "dairy cow")
    
    def test_reduce_memory_usage(self):
        """Test memory reduction function."""
        # Create test DataFrame
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
        })
        
        original_mem = df.memory_usage(deep=True).sum()
        df_reduced = reduce_memory_usage(df, verbose=False)
        reduced_mem = df_reduced.memory_usage(deep=True).sum()
        
        # Memory should be reduced
        self.assertLess(reduced_mem, original_mem)


class TestTopicParsing(unittest.TestCase):
    """Test cases for topic parsing edge cases."""
    
    def test_nested_lists(self):
        """Test parsing nested list structures."""
        topics_str = "[['Topic1', 'Topic2'], 'Topic3']"
        result = parse_topics_list(topics_str)
        self.assertIsInstance(result, list)
    
    def test_single_topic(self):
        """Test parsing single topic."""
        topics_str = "['Single Topic']"
        result = parse_topics_list(topics_str)
        self.assertEqual(len(result), 1)
    
    def test_topics_with_quotes(self):
        """Test parsing topics with various quote styles."""
        topics_str = '["Topic1", "Topic2", "Topic3"]'
        result = parse_topics_list(topics_str)
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
