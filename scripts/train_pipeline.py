"""
Complete training pipeline for DigiCow Farmer Training Adoption Challenge.
Runs feature engineering, model training, and ensemble creation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime

# Import project modules
from config import (
    TRAIN_FILE, TEST_FILE, TRAIN_FEATURES_FILE, TEST_FEATURES_FILE,
    TARGET_COL, RANDOM_SEED, PROCESSED_DATA_DIR, TOPIC_CATEGORIES
)
from utils.data_utils import load_data, extract_topics_vectorized, reduce_memory_usage
from feature_engineering import FeatureEngineer
from model_config import BASE_MODELS, ADVANCED_MODELS, MODEL_PRESETS
from ensemble_pipeline import EnsembleModelPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_feature_engineering(use_cache: bool = True):
    """
    Run feature engineering pipeline.
    
    Args:
        use_cache: If True, load cached features if available
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Check if cached features exist
    if use_cache and TRAIN_FEATURES_FILE.exists() and TEST_FEATURES_FILE.exists():
        logger.info("Loading cached features...")
        train_df = pd.read_csv(TRAIN_FEATURES_FILE)
        test_df = pd.read_csv(TEST_FEATURES_FILE)
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        return train_df, test_df
    
    # Load raw data
    logger.info("Loading raw data...")
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
    
    # Parse topics
    logger.info("Parsing topics...")
    train_df = extract_topics_vectorized(train_df)
    test_df = extract_topics_vectorized(test_df)
    
    # Create features
    logger.info("Creating features...")
    fe = FeatureEngineer(topic_categories=TOPIC_CATEGORIES)
    train_df = fe.create_all_features(train_df)
    test_df = fe.create_all_features(test_df)
    
    # Optimize memory
    train_df = reduce_memory_usage(train_df)
    test_df = reduce_memory_usage(test_df)
    
    # Save processed features
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FEATURES_FILE, index=False)
    test_df.to_csv(TEST_FEATURES_FILE, index=False)
    logger.info(f"Saved features to {PROCESSED_DATA_DIR}")
    
    return train_df, test_df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepare data for modeling.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
    
    Returns:
        Tuple of (X_train, y_train, X_test, feature_names)
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DATA PREPARATION")
    logger.info("="*80)
    
    # Separate features and target
    y_train = train_df[TARGET_COL].values
    
    # Drop non-feature columns
    drop_cols = [
        'ID', TARGET_COL, 'topics_list', 'topics_parsed',
        'first_training_date'  # Already extracted temporal features
    ]
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in drop_cols]
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Handle categorical features
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Categorical features: {len(categorical_cols)}")
    logger.info(f"Numerical features: {len(feature_cols) - len(categorical_cols)}")
    
    # One-hot encode categorical features
    if len(categorical_cols) > 0:
        logger.info("One-hot encoding categorical features...")
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        # Align columns
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # Convert to numpy arrays
    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)
    y_train = y_train.astype(np.int8)
    
    # Handle any remaining NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Target distribution: {np.bincount(y_train)}")
    logger.info(f"Class balance: {y_train.mean():.2%} positive class")
    
    return X_train, y_train, X_test, feature_cols


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_preset: str = 'balanced'
):
    """
    Train base models and create ensemble.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_preset: Model preset to use
    
    Returns:
        Trained EnsembleModelPipeline
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*80)
    
    # Combine all available models
    all_models = {**BASE_MODELS, **ADVANCED_MODELS}
    
    # Initialize pipeline
    pipeline = EnsembleModelPipeline(
        models_config=all_models,
        cv_folds=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    # Get model names from preset
    model_names = MODEL_PRESETS.get(model_preset, MODEL_PRESETS['balanced'])
    logger.info(f"Using preset: '{model_preset}'")
    logger.info(f"Models to train: {model_names}")
    
    # Build base models
    pipeline.build_base_models(model_names)
    
    # Evaluate base models
    logger.info("\nEvaluating base models with 5-fold CV...")
    results_df = pipeline.evaluate_base_models(X_train, y_train, scoring='roc_auc')
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*80)
    print("\n" + results_df.to_string(index=False))
    
    return pipeline, results_df


def create_ensembles(
    pipeline: EnsembleModelPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_n: int = 5
):
    """
    Create multiple ensemble models.
    
    Args:
        pipeline: Trained pipeline
        X_train: Training features
        y_train: Training target
        top_n: Number of top models to use
    
    Returns:
        Dictionary of ensemble models
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: ENSEMBLE CREATION")
    logger.info("="*80)
    
    ensembles = {}
    
    # 1. Voting Ensemble (Soft)
    logger.info(f"\n1. Creating Voting Ensemble (top {top_n} models)...")
    voting_ensemble = pipeline.build_voting_ensemble(
        top_n=top_n,
        voting='soft',
        weights='auto'
    )
    voting_ensemble.fit(X_train, y_train)
    ensembles['voting_soft'] = voting_ensemble
    logger.info("✓ Voting ensemble trained")
    
    # 2. Stacking Ensemble
    logger.info(f"\n2. Creating Stacking Ensemble (top {top_n} models)...")
    from sklearn.linear_model import LogisticRegression
    stacking_ensemble = pipeline.build_stacking_ensemble(
        top_n=top_n,
        final_estimator=LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        cv=5
    )
    stacking_ensemble.fit(X_train, y_train)
    ensembles['stacking_lr'] = stacking_ensemble
    logger.info("✓ Stacking ensemble trained")
    
    # 3. Weighted Average Ensemble
    logger.info(f"\n3. Creating Weighted Average Ensemble (top {top_n+2} models)...")
    weighted_ensemble = pipeline.build_weighted_average_ensemble(
        X_train, y_train,
        top_n=top_n + 2
    )
    # Train individual models
    for model in weighted_ensemble['models']:
        model.fit(X_train, y_train)
    ensembles['weighted_average'] = weighted_ensemble
    logger.info("✓ Weighted average ensemble ready")
    
    return ensembles


def generate_predictions(
    ensembles: dict,
    pipeline: EnsembleModelPipeline,
    X_test: np.ndarray,
    test_ids: pd.Series,
    output_dir: Path
):
    """
    Generate predictions from all ensembles.
    
    Args:
        ensembles: Dictionary of ensemble models
        pipeline: Pipeline instance
        X_test: Test features
        test_ids: Test IDs
        output_dir: Output directory
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 5: GENERATING PREDICTIONS")
    logger.info("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_predictions = {}
    
    # Generate predictions from each ensemble
    for name, ensemble in ensembles.items():
        logger.info(f"\nGenerating predictions: {name}")
        
        if name == 'weighted_average':
            predictions = pipeline.predict_weighted_average(X_test, ensemble)
        else:
            predictions = ensemble.predict_proba(X_test)[:, 1]
        
        all_predictions[name] = predictions
        
        # Save individual submission
        submission = pd.DataFrame({
            'ID': test_ids,
            TARGET_COL: predictions
        })
        filepath = output_dir / f"submission_{name}.csv"
        submission.to_csv(filepath, index=False)
        logger.info(f"  Saved: {filepath}")
    
    # Create meta-ensemble (average of all ensembles)
    logger.info("\nCreating meta-ensemble (average of all methods)...")
    meta_predictions = np.mean(list(all_predictions.values()), axis=0)
    
    submission = pd.DataFrame({
        'ID': test_ids,
        TARGET_COL: meta_predictions
    })
    filepath = output_dir / "submission_meta_ensemble.csv"
    submission.to_csv(filepath, index=False)
    logger.info(f"  Saved: {filepath}")
    
    logger.info(f"\n✓ All predictions saved to {output_dir}")
    
    return all_predictions


def main(
    model_preset: str = 'balanced',
    top_n_models: int = 5,
    use_cache: bool = True
):
    """
    Main training pipeline.
    
    Args:
        model_preset: Model preset to use ('fast', 'balanced', 'powerful', etc.)
        top_n_models: Number of top models for ensemble
        use_cache: Whether to use cached features
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("\n" + "="*80)
    logger.info("DIGICOW FARMER TRAINING ADOPTION - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Model Preset: {model_preset}")
    logger.info(f"Top N Models: {top_n_models}")
    logger.info(f"Random Seed: {RANDOM_SEED}")
    
    try:
        # Step 1: Feature Engineering
        train_df, test_df = run_feature_engineering(use_cache=use_cache)
        
        # Step 2: Data Preparation
        X_train, y_train, X_test, feature_names = prepare_data(train_df, test_df)
        
        # Step 3: Model Training
        pipeline, results_df = train_models(X_train, y_train, model_preset)
        
        # Step 4: Ensemble Creation
        ensembles = create_ensembles(pipeline, X_train, y_train, top_n_models)
        
        # Step 5: Generate Predictions
        test_ids = test_df['ID']
        output_dir = PROCESSED_DATA_DIR / f"submissions_{timestamp}"
        predictions = generate_predictions(ensembles, pipeline, X_test, test_ids, output_dir)
        
        # Save models and results
        models_dir = PROCESSED_DATA_DIR / f"models_{timestamp}"
        pipeline.save_models(str(models_dir))
        
        # Final summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Total execution time: {elapsed_time/60:.2f} minutes")
        logger.info(f"Models saved to: {models_dir}")
        logger.info(f"Submissions saved to: {output_dir}")
        logger.info("\nSubmission files created:")
        logger.info(f"  1. submission_voting_soft.csv")
        logger.info(f"  2. submission_stacking_lr.csv")
        logger.info(f"  3. submission_weighted_average.csv")
        logger.info(f"  4. submission_meta_ensemble.csv (RECOMMENDED)")
        logger.info("\n" + "="*80)
        
        return pipeline, ensembles, predictions
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(f"{'='*80}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DigiCow Training Pipeline')
    parser.add_argument(
        '--preset',
        type=str,
        default='balanced',
        choices=['fast', 'balanced', 'powerful', 'diverse', 'gradient_boosting_only'],
        help='Model preset to use'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top models for ensemble'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable feature caching'
    )
    
    args = parser.parse_args()
    
    main(
        model_preset=args.preset,
        top_n_models=args.top_n,
        use_cache=not args.no_cache
    )
