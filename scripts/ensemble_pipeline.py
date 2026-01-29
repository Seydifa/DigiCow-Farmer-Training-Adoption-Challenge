"""
Ensemble modeling pipeline for DigiCow Farmer Training Adoption Challenge.
Implements various ensemble strategies with optimized training and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
from scripts.utils.model_utils import get_calibrated_classifier
from scripts.utils.printing_utils import SimpleProgressBar, print_header
from collections import defaultdict
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModelPipeline:
    """
    Comprehensive ensemble modeling pipeline with multiple strategies.
    """
    
    def __init__(
        self,
        models_config: Dict,
        ensemble_strategy: str = 'voting_soft',
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize ensemble pipeline.
        
        Args:
            models_config: Dictionary of model configurations
            ensemble_strategy: Type of ensemble ('voting_soft', 'stacking_lr', etc.)
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.models_config = models_config
        self.ensemble_strategy = ensemble_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.base_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.cv_results = {}
        self.feature_importance = {}
        
    def build_base_models(self, model_names: List[str]) -> Dict:
        """
        Build base models from configuration.
        
        Args:
            model_names: List of model names to build
        
        Returns:
            Dictionary of instantiated models
        """
        logger.info(f"Building {len(model_names)} base models...")
        
        models = {}
        for name in model_names:
            if name not in self.models_config:
                logger.warning(f"Model '{name}' not found in config, skipping...")
                continue
            
            config = self.models_config[name]
            model_class = config['model']
            params = config['params']
            
            try:
                models[name] = model_class(**params)
                logger.info(f"  ✓ {name}: {config['description']}")
            except Exception as e:
                logger.error(f"  ✗ {name}: Failed to build - {e}")
        
        self.base_models = models
        return models
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_spaces: Dict,
        n_iter: int = 10,
        cv: int = 3
    ):
        """
        Optimize hyperparameters for base models using RandomizedSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_spaces: Dictionary of hyperparameter search spaces
            n_iter: Number of iterations for random search
            cv: Number of CV folds for optimization
        """
        print_header(f"Hyperparameter Optimization ({n_iter} iterations)")
        
        models_to_optimize = [name for name in self.base_models if name in param_spaces]
        if not models_to_optimize:
            logger.info("No models to optimize.")
            return self.base_models
            
        progress = SimpleProgressBar(len(models_to_optimize), desc="Optimizing Models")
        optimized_models = {}
        
        for name, model in self.base_models.items():
            if name not in param_spaces:
                optimized_models[name] = model
                continue
            
            # Use negative log loss for scoring (since we want to minimize log loss)
            # Check if model supports predicted probabilities
            if hasattr(model, 'predict_proba'):
                scoring = 'neg_log_loss'
            else:
                scoring = 'roc_auc' # Fallback for non-probabilistic models
            
            try:
                progress.update(0, context=f"Tuning {name}...")
                
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_spaces[name],
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0 # Suppress internal printing
                )
                
                search.fit(X, y)
                
                # Update the model with best estimator
                optimized_models[name] = search.best_estimator_
                progress.update(1, context=f"Finished {name}")
                
            except Exception as e:
                logger.error(f"  Optimization failed for {name}: {e}")
                optimized_models[name] = model
                progress.update(1, context=f"Failed {name}")
        
        progress.close()
        self.base_models = optimized_models
        logger.info("Optimization completed.")
        return optimized_models

    def calibrate_base_models(self, cv: int = 3, method: str = 'isotonic'):
        """
        Wrap base models in CalibratedClassifierCV to improve probability estimates.
        Useful when using 'balanced' class weights.
        
        Args:
            cv: Number of folds for calibration
            method: 'isotonic' or 'sigmoid'
        """
        print_header(f"Probability Calibration ({method}, cv={cv})")
        
        progress = SimpleProgressBar(len(self.base_models), desc="Calibrating")
        calibrated_models = {}
        
        for name, model in self.base_models.items():
            progress.update(0, context=f"Wrapping {name}...")
            # We wrap the UN-FITTED model. The calibration happens during .fit() later.
            calibrated_models[name] = get_calibrated_classifier(model, cv=cv, method=method)
            progress.update(1)
            
        progress.close()
        self.base_models = calibrated_models
        logger.info("Calibration wrappers applied.")
        return calibrated_models

    def evaluate_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = 'roc_auc'
    ) -> pd.DataFrame:
        """
        Evaluate all base models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            scoring: Scoring metric
        
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"\nEvaluating {len(self.base_models)} base models...")
        logger.info(f"Using {self.cv_folds}-fold cross-validation")
        logger.info("="*80)
        
        results = []
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        for name, model in self.base_models.items():
            logger.info(f"\nEvaluating: {name}")
            start_time = time.time()
            
            # Define scoring metrics
            # 70% Log Loss (minimize), 30% ROC-AUC (maximize)
            scoring_dict = {
                'roc_auc': 'roc_auc',
                'accuracy': 'accuracy',
                'f1': 'f1'
            }
            
            # Add log_loss only if model supports probabilities
            supports_proba = hasattr(model, 'predict_proba')
            if supports_proba:
                scoring_dict['neg_log_loss'] = 'neg_log_loss'
            
            try:
                # Cross-validation with multiple metrics
                cv_scores = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=scoring_dict,
                    n_jobs=self.n_jobs,
                    return_train_score=True
                )
                
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                roc_auc = cv_scores['test_roc_auc'].mean()
                
                if supports_proba:
                    log_loss = -cv_scores['test_neg_log_loss'].mean()
                else:
                    # Approximation or NaN for non-probabilistic models
                    log_loss = np.nan
                
                # Calculate Competition Score: 0.3 * AUC - 0.7 * LogLoss
                # Note: We maximize this score. (Higher AUC good, Lower LogLoss good)
                if not np.isnan(log_loss):
                    comp_score = (0.3 * roc_auc) - (0.7 * log_loss)
                else:
                    comp_score = -np.inf # Penalize models without probability estimates
                
                # Store results
                result = {
                    'model': name,
                    'comp_score': comp_score,
                    'roc_auc_mean': roc_auc,
                    'roc_auc_std': cv_scores['test_roc_auc'].std(),
                    'log_loss_mean': log_loss,
                    'accuracy_mean': cv_scores['test_accuracy'].mean(),
                    'f1_mean': cv_scores['test_f1'].mean(),
                    'train_roc_auc': cv_scores['train_roc_auc'].mean(),
                    'fit_time': elapsed_time
                }
                
                results.append(result)
                
                logger.info(f"  Comp Score: {comp_score:.4f} (0.3*AUC - 0.7*LogLoss)")
                logger.info(f"  ROC-AUC:    {result['roc_auc_mean']:.4f} ± {result['roc_auc_std']:.4f}")
                if supports_proba:
                    logger.info(f"  Log Loss:   {result['log_loss_mean']:.4f}")
                else:
                    logger.info(f"  Log Loss:   N/A (no predict_proba)")
                logger.info(f"  Time:       {elapsed_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                results.append({
                    'model': name,
                    'comp_score': -np.inf,
                    'roc_auc_mean': 0,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        # Sort by Competition Score
        results_df = results_df.sort_values('comp_score', ascending=False)
        
        self.cv_results = results_df
        
        logger.info("\n" + "="*80)
        logger.info("Base Model Rankings (by Competition Score):")
        logger.info("="*80)
        print(f"{'Model':<25} | {'Comp Score':<12} | {'ROC-AUC':<10} | {'Log Loss':<10}")
        print("-" * 65)
        for idx, row in results_df.head(10).iterrows():
            log_loss_str = f"{row['log_loss_mean']:.4f}" if pd.notnull(row['log_loss_mean']) else "N/A"
            print(f"{row['model']:<25} | {row['comp_score']:<12.4f} | {row['roc_auc_mean']:<10.4f} | {log_loss_str:<10}")
        
        return results_df
    
    def build_voting_ensemble(
        self,
        top_n: int = 5,
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """
        Build voting ensemble from top performing models.
        
        Args:
            top_n: Number of top models to include
            voting: 'soft' or 'hard' voting
            weights: Optional weights for each model
        
        Returns:
            VotingClassifier instance
        """
        logger.info(f"\nBuilding Voting Ensemble (top {top_n} models)...")
        
        # Get top models
        top_models = self.cv_results.head(top_n)['model'].tolist()
        
        # Filter models based on voting type
        if voting == 'soft':
            # Only include models with predict_proba for soft voting
            valid_models = []
            for name in top_models:
                model = self.base_models[name]
                if hasattr(model, 'predict_proba'):
                    valid_models.append(name)
                else:
                    logger.warning(f"  Skipping {name} - no predict_proba method for soft voting")
            top_models = valid_models
        
        # Build estimators list
        estimators = [(name, self.base_models[name]) for name in top_models]
        
        # Auto-weight based on CV scores if requested
        if weights == 'auto':
            # Get scores only for the valid models
            scores = self.cv_results[self.cv_results['model'].isin(top_models)]['roc_auc_mean'].values
            weights = scores / scores.sum()
            logger.info(f"Auto-weights: {weights}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=self.n_jobs
        )
        
        self.ensemble_model = ensemble
        logger.info(f"Voting ensemble created with {len(estimators)} models")
        
        return ensemble
    
    def build_stacking_ensemble(
        self,
        top_n: int = 5,
        final_estimator: Any = None,
        cv: int = 5
    ) -> StackingClassifier:
        """
        Build stacking ensemble from top performing models.
        
        Args:
            top_n: Number of top models to include
            final_estimator: Meta-learner model
            cv: Cross-validation folds for stacking
        
        Returns:
            StackingClassifier instance
        """
        logger.info(f"\nBuilding Stacking Ensemble (top {top_n} models)...")
        
        # Get top models
        top_models = self.cv_results.head(top_n)['model'].tolist()
        
        # Filter models - only include those with predict_proba
        valid_models = []
        for name in top_models:
            model = self.base_models[name]
            if hasattr(model, 'predict_proba'):
                valid_models.append(name)
            else:
                logger.warning(f"  Skipping {name} - no predict_proba method for stacking")
        top_models = valid_models
        
        # Build estimators list
        estimators = [(name, self.base_models[name]) for name in top_models]
        
        # Default final estimator
        if final_estimator is None:
            from sklearn.linear_model import LogisticRegression
            final_estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=self.n_jobs
        )
        
        self.ensemble_model = ensemble
        logger.info(f"Stacking ensemble created with {len(estimators)} base models")
        
        return ensemble
    
    def build_weighted_average_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        top_n: int = 5
    ) -> Dict:
        """
        Build weighted average ensemble using CV scores as weights.
        
        Args:
            X: Feature matrix
            y: Target vector
            top_n: Number of top models to include
        
        Returns:
            Dictionary with models and weights
        """
        logger.info(f"\nBuilding Weighted Average Ensemble (top {top_n} models)...")
        
        # Get top models and their scores
        top_results = self.cv_results.head(top_n)
        top_models = top_results['model'].tolist()
        scores = top_results['roc_auc_mean'].values
        
        # Normalize scores to weights
        weights = scores / scores.sum()
        
        ensemble_info = {
            'models': [self.base_models[name] for name in top_models],
            'model_names': top_models,
            'weights': weights,
            'type': 'weighted_average'
        }
        
        logger.info("Model weights:")
        for name, weight in zip(top_models, weights):
            logger.info(f"  {name:25s}: {weight:.4f}")
        
        self.ensemble_model = ensemble_info
        return ensemble_info
    
    def predict_weighted_average(
        self,
        X: np.ndarray,
        ensemble_info: Dict
    ) -> np.ndarray:
        """
        Make predictions using weighted average ensemble.
        
        Args:
            X: Feature matrix
            ensemble_info: Ensemble configuration
        
        Returns:
            Predicted probabilities (clipped to [0, 1])
        """
        predictions = []
        
        for model in ensemble_info['models']:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                # Use decision_function and convert to probabilities using sigmoid
                decision = model.decision_function(X)
                # Sigmoid transformation: 1 / (1 + exp(-x))
                pred = 1 / (1 + np.exp(-decision))
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = ensemble_info['weights'].reshape(-1, 1)
        final_pred = (predictions * weights).sum(axis=0)
        
        # Ensure predictions are in valid probability range [0, 1]
        final_pred = np.clip(final_pred, 0.0, 1.0)
        
        return final_pred
    
    def extract_feature_importance(
        self,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract and aggregate feature importance from tree-based models.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame with aggregated feature importance
        """
        logger.info("\nExtracting feature importance...")
        
        importance_dict = defaultdict(list)
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                for feat, imp in zip(feature_names, importance):
                    importance_dict[feat].append(imp)
        
        # Aggregate importance
        aggregated = {
            'feature': [],
            'importance_mean': [],
            'importance_std': [],
            'importance_max': []
        }
        
        for feat, imps in importance_dict.items():
            aggregated['feature'].append(feat)
            aggregated['importance_mean'].append(np.mean(imps))
            aggregated['importance_std'].append(np.std(imps))
            aggregated['importance_max'].append(np.max(imps))
        
        importance_df = pd.DataFrame(aggregated)
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        self.feature_importance = importance_df
        
        logger.info(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance_mean']:.4f}")
        
        return importance_df
    
    def save_models(self, output_dir: str):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\nSaving models to {output_dir}...")
        
        # Save base models
        for name, model in self.base_models.items():
            filepath = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(model, filepath)
            logger.info(f"  Saved: {name}")
        
        # Save ensemble model
        if self.ensemble_model is not None:
            filepath = os.path.join(output_dir, "ensemble_model.pkl")
            joblib.dump(self.ensemble_model, filepath)
            logger.info(f"  Saved: ensemble_model")
        
        # Save CV results
        if hasattr(self, 'cv_results'):
            filepath = os.path.join(output_dir, "cv_results.csv")
            self.cv_results.to_csv(filepath, index=False)
            logger.info(f"  Saved: cv_results.csv")
        
        # Save feature importance
        if hasattr(self, 'feature_importance') and isinstance(self.feature_importance, pd.DataFrame):
            filepath = os.path.join(output_dir, "feature_importance.csv")
            self.feature_importance.to_csv(filepath, index=False)
            logger.info(f"  Saved: feature_importance.csv")
        
        logger.info("All models saved successfully!")
    
    def load_models(self, input_dir: str):
        """
        Load trained models from disk.
        
        Args:
            input_dir: Directory containing saved models
        """
        import os
        
        logger.info(f"\nLoading models from {input_dir}...")
        
        # Load ensemble model
        ensemble_path = os.path.join(input_dir, "ensemble_model.pkl")
        if os.path.exists(ensemble_path):
            self.ensemble_model = joblib.load(ensemble_path)
            logger.info(f"  Loaded: ensemble_model")
        
        logger.info("Models loaded successfully!")
