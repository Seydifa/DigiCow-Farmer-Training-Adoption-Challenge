import optuna
import numpy as np
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from scripts.utils.model_utils import SklearnCompatibleCatBoostClassifier
from scripts.utils.printing_utils import print_header
from sklearn.metrics import make_scorer, roc_auc_score, log_loss

logger = logging.getLogger(__name__)

from sklearn.metrics import make_scorer, roc_auc_score, log_loss

def competition_metric(y_true, y_pred, **kwargs):
    """
    Custom metric: 0.3 * AUC - 0.7 * LogLoss
    """
    # Handle probability array shapes
    if y_pred.ndim > 1:
        y_prob = y_pred[:, 1]
    else:
        y_prob = y_pred
        
    try:
        auc = roc_auc_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        return (0.3 * auc) - (0.7 * ll)
    except Exception:
        return -100.0 # Interaction penalty

# Create scorer
comp_scorer = make_scorer(competition_metric, needs_proba=True)

class OptunaTuner:
    """
    Hyperparameter tuning using Optuna.
    Focuses on maximizing Competition Score (0.3*AUC - 0.7*LogLoss).
    """
    
    def __init__(self, n_trials=200, cv=3, random_state=42, n_jobs=-1):
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def tune(self, model_name, X, y):
        """
        Tune a specific model.
        """
        print_header(f"Optuna Tuning: {model_name}")
        
        def objective(trial):
            params = self._suggest_params(trial, model_name)
            model = self._get_model(model_name, params)
            
            # Use StratifiedKFold for imbalance handling
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            
            # Scoring: maximize negative log loss (so minimize log loss)
            try:
                if hasattr(model, 'predict_proba'):
                    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=self.n_jobs)
                else:
                    # Fallback to AUC for non-probabilistic models
                    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=self.n_jobs)
                    
                return scores.mean()
            except Exception as e:
                # Prune failed trials
                raise optuna.exceptions.TrialPruned(f"Error: {e}")

        # Create study
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction="maximize")
        
        logger.info(f"Starting Optuna optimization for {model_name} (Metric: Log Loss)...")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"  Best Params: {study.best_params}")
        logger.info(f"  Best Score: {study.best_value:.4f}")
        
        return self._get_model(model_name, study.best_params)

    def _get_model(self, name, params):
        """Instantiate model with parameters."""
        common_args = {'random_state': self.random_state}
        if name in ['random_forest', 'extra_trees', 'logistic_regression', 'xgboost', 'lightgbm']:
             common_args['n_jobs'] = self.n_jobs
             
        # Merge params with defaults (handling base instances logic if needed)
        # Here we reconstruct from scratch for purity
        if name == 'xgboost':
            return XGBClassifier(**params, **common_args, eval_metric='logloss', use_label_encoder=False)
        elif name == 'lightgbm':
            return LGBMClassifier(**params, **common_args, verbose=-1, metric='binary_logloss')
        elif name == 'catboost':
            # Use our wrapper
            return SklearnCompatibleCatBoostClassifier(**params, random_seed=self.random_state, verbose=False, eval_metric='Logloss', allow_writing_files=False)
        elif name == 'random_forest':
            return RandomForestClassifier(**params, **common_args)
        elif name == 'extra_trees':
            return ExtraTreesClassifier(**params, **common_args)
        elif name == 'gradient_boosting':
            return GradientBoostingClassifier(**params, random_state=self.random_state)
        elif name == 'hist_gradient_boosting':
            return HistGradientBoostingClassifier(**params, random_state=self.random_state)
        elif name == 'logistic_regression':
            return LogisticRegression(**params, random_state=self.random_state, n_jobs=self.n_jobs, max_iter=1000)
        elif name == 'ridge_classifier':
            return RidgeClassifier(**params, random_state=self.random_state)
        
        raise ValueError(f"Model {name} not supported for Optuna tuning")

    def _suggest_params(self, trial, name):
        """Define search spaces with complex regularization."""
        
        if name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12), # Deep trees
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0), # Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0), # L1
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0), # L2
                'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, 8.0])
            }
            
        elif name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150), # Complex leaves
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, 8.0])
            }
            
        elif name == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 200, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, 8.0])
            }
            
        elif name == 'random_forest' or name == 'extra_trees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'criterion': 'log_loss'
            }
            
        elif name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'solver': 'lbfgs',
                'penalty': 'l2',
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            
        elif name == 'hist_gradient_boosting':
             return {
                'max_iter': trial.suggest_int('max_iter', 200, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
             }
             
        elif name == 'ridge_classifier':
             return {
                'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
             }

        # Return empty dict if not configured, or basic defaults
        return {}
