"""
Utility functions for data loading and preprocessing.
Optimized for performance and memory efficiency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV data with optimized dtypes.
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to load (for testing)
    
    Returns:
        DataFrame with optimized dtypes
    """
    logger.info(f"Loading data from {file_path}")
    
    # Define dtypes for memory optimization
    dtype_dict = {
        'ID': 'string',
        'gender': 'category',
        'age': 'category',
        'registration': 'category',
        'belong_to_cooperative': 'int8',
        'county': 'category',
        'subcounty': 'category',
        'ward': 'category',
        'trainer': 'category',
        'topics_list': 'string',
        'num_trainings_30d': 'float32',
        'num_trainings_60d': 'float32',
        'num_total_trainings': 'int16',
        'num_repeat_trainings': 'int16',
        'days_to_second_training': 'float32',
        'num_unique_trainers': 'int8',
        'has_second_training': 'int8'
    }
    
    # Load data
    df = pd.read_csv(
        file_path,
        dtype=dtype_dict,
        parse_dates=['first_training_date'],
        nrows=nrows
    )
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def parse_topics_list(topics_str: str) -> list:
    """
    Parse the topics_list string into a Python list.
    Handles various formats and errors gracefully.
    
    Args:
        topics_str: String representation of topics list
    
    Returns:
        List of topics
    """
    if pd.isna(topics_str):
        return []
    
    try:
        # Try to evaluate as Python literal
        topics = ast.literal_eval(topics_str)
        if isinstance(topics, list):
            return topics
        return [str(topics)]
    except (ValueError, SyntaxError):
        # Fallback: split by comma
        return [t.strip().strip("'\"") for t in topics_str.split(',')]


def extract_topics_vectorized(df: pd.DataFrame, topics_col: str = 'topics_list') -> pd.DataFrame:
    """
    Vectorized extraction of topics from topics_list column.
    
    Args:
        df: DataFrame with topics_list column
        topics_col: Name of the topics column
    
    Returns:
        DataFrame with parsed topics
    """
    logger.info("Parsing topics list (vectorized)...")
    
    # Parse topics efficiently
    df['topics_parsed'] = df[topics_col].apply(parse_topics_list)
    
    return df


def clean_topic_name(topic: str) -> str:
    """
    Clean and normalize topic names.
    
    Args:
        topic: Raw topic name
    
    Returns:
        Cleaned topic name
    """
    if pd.isna(topic):
        return ""
    
    # Convert to lowercase and strip whitespace
    topic = str(topic).lower().strip()
    
    # Remove special characters but keep spaces and hyphens
    topic = ''.join(c if c.isalnum() or c in [' ', '-', '_'] else ' ' for c in topic)
    
    # Remove extra spaces
    topic = ' '.join(topic.split())
    
    return topic


def get_memory_usage(df: pd.DataFrame) -> dict:
    """
    Get detailed memory usage statistics.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with memory statistics
    """
    mem_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': mem_usage.sum() / 1024**2,
        'per_column_mb': (mem_usage / 1024**2).to_dict(),
        'shape': df.shape
    }


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print memory reduction info
    
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Skip non-numeric types (object, category, string, datetime)
        if col_type == object or col_type.name == 'category' or col_type.name == 'string':
            continue
            
        # Only process numeric types
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Skip if min/max are not numeric (shouldn't happen, but safety check)
            if not isinstance(c_min, (int, float, np.number)) or not isinstance(c_max, (int, float, np.number)):
                continue
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def save_processed_data(df: pd.DataFrame, file_path: Path, compress: bool = False):
    """
    Save processed data with optional compression.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        compress: Whether to use compression
    """
    logger.info(f"Saving processed data to {file_path}")
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        df.to_csv(file_path.with_suffix('.csv.gz'), index=False, compression='gzip')
        logger.info(f"Saved compressed file: {file_path.with_suffix('.csv.gz')}")
    else:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved file: {file_path}")
