"""
Main script to run the complete feature engineering pipeline.
Optimized for performance and memory efficiency.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import (
    TRAIN_FILE, TEST_FILE, TRAIN_FEATURES_FILE, TEST_FEATURES_FILE,
    TOPIC_CATEGORIES, RANDOM_SEED, PROCESSED_DATA_DIR
)
from utils.data_utils import (
    load_data, extract_topics_vectorized, reduce_memory_usage, save_processed_data
)
from feature_engineering import FeatureEngineer
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_dataset(
    df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Process a dataset through the feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        feature_engineer: FeatureEngineer instance
        is_train: Whether this is training data
    
    Returns:
        Processed DataFrame with engineered features
    """
    logger.info(f"Processing {'train' if is_train else 'test'} dataset...")
    
    # Extract topics
    df = extract_topics_vectorized(df)
    
    # Create all features
    df = feature_engineer.create_all_features(df)
    
    # Reduce memory usage
    df = reduce_memory_usage(df, verbose=True)
    
    return df


def main():
    """
    Main execution function.
    """
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("DigiCow Feature Engineering Pipeline")
    logger.info("="*60)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(topic_categories=TOPIC_CATEGORIES)
    
    # Load training data
    logger.info("\n" + "="*60)
    logger.info("Loading Training Data")
    logger.info("="*60)
    train_df = load_data(TRAIN_FILE)
    logger.info(f"Train shape: {train_df.shape}")
    
    # Process training data
    logger.info("\n" + "="*60)
    logger.info("Processing Training Data")
    logger.info("="*60)
    train_processed = process_dataset(train_df, feature_engineer, is_train=True)
    
    # Save processed training data
    save_processed_data(train_processed, TRAIN_FEATURES_FILE)
    
    # Load test data
    logger.info("\n" + "="*60)
    logger.info("Loading Test Data")
    logger.info("="*60)
    test_df = load_data(TEST_FILE)
    logger.info(f"Test shape: {test_df.shape}")
    
    # Process test data
    logger.info("\n" + "="*60)
    logger.info("Processing Test Data")
    logger.info("="*60)
    test_processed = process_dataset(test_df, feature_engineer, is_train=False)
    
    # Save processed test data
    save_processed_data(test_processed, TEST_FEATURES_FILE)
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("Feature Engineering Complete!")
    logger.info("="*60)
    logger.info(f"Train features shape: {train_processed.shape}")
    logger.info(f"Test features shape: {test_processed.shape}")
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info(f"Train features saved to: {TRAIN_FEATURES_FILE}")
    logger.info(f"Test features saved to: {TEST_FEATURES_FILE}")
    
    # Feature summary
    logger.info("\n" + "="*60)
    logger.info("Feature Summary")
    logger.info("="*60)
    
    # Count new features
    original_cols = set(train_df.columns)
    new_cols = set(train_processed.columns) - original_cols
    logger.info(f"Original features: {len(original_cols)}")
    logger.info(f"New features created: {len(new_cols)}")
    logger.info(f"Total features: {len(train_processed.columns)}")
    
    # Display sample of new features
    logger.info("\nSample of new features:")
    for i, col in enumerate(sorted(new_cols)[:10], 1):
        logger.info(f"  {i}. {col}")
    if len(new_cols) > 10:
        logger.info(f"  ... and {len(new_cols) - 10} more features")
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline execution completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
