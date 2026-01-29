"""
Model configurations for ensemble learning.
Defines all models and their hyperparameters for the DigiCow challenge.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Try to import advanced models (optional dependencies)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    # Use wrapper for sklearn 1.6+ compatibility
    from scripts.utils.model_utils import SklearnCompatibleCatBoostClassifier
    CatBoostClassifier = SklearnCompatibleCatBoostClassifier
except ImportError:
    CATBOOST_AVAILABLE = False


# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# BASE MODELS CONFIGURATION
# ============================================================================

BASE_MODELS = {
    # -------------------------------------------------------------------------
    # TREE-BASED MODELS (Strong for tabular data)
    # -------------------------------------------------------------------------
    'random_forest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'criterion': 'log_loss',
            'class_weight': 'balanced' # Handle imbalance
        },
        'description': 'Robust ensemble of decision trees (Balanced)'
    },
    
    'extra_trees': {
        'model': ExtraTreesClassifier,
        'params': {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'criterion': 'log_loss',
            'class_weight': 'balanced' # Handle imbalance
        },
        'description': 'Extra randomized trees (Balanced)'
    },
    
    # -------------------------------------------------------------------------
    # GRADIENT BOOSTING MODELS (High performance)
    # -------------------------------------------------------------------------
    'gradient_boosting': {
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'random_state': RANDOM_SEED,
            'loss': 'log_loss'
            # GradientBoostingClassifier doesn't support class_weight/scale_pos_weight unfortunately
            # We rely on calibration for this one
        },
        'description': 'Classic gradient boosting'
    },
    
    'hist_gradient_boosting': {
        'model': HistGradientBoostingClassifier,
        'params': {
            'max_iter': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'min_samples_leaf': 20,
            'l2_regularization': 1.0,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced' # Handle imbalance
        },
        'description': 'Fast histogram-based gradient boosting (Balanced)'
    },
    
    # -------------------------------------------------------------------------
    # LINEAR MODELS (Fast and interpretable)
    # -------------------------------------------------------------------------
    'logistic_regression': {
        'model': LogisticRegression,
        'params': {
            'C': 0.1,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced', # Handle imbalance
            'n_jobs': -1
        },
        'description': 'Linear model with L2 regularization (Balanced)'
    },
    
    'ridge_classifier': {
        'model': RidgeClassifier,
        'params': {
            'alpha': 1.0,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced' # Handle imbalance
        },
        'description': 'Ridge regression (Balanced)'
    },
    
    # -------------------------------------------------------------------------
    # BOOSTING VARIANTS
    # -------------------------------------------------------------------------
    'adaboost': {
        'model': AdaBoostClassifier,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': RANDOM_SEED,
            'algorithm': 'SAMME'
        },
        'description': 'Adaptive boosting'
    },
    
    # -------------------------------------------------------------------------
    # INSTANCE-BASED MODELS
    # -------------------------------------------------------------------------
    'knn': {
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': 30,
            'weights': 'distance',
            'metric': 'minkowski',
            'n_jobs': -1
        },
        'description': 'K-nearest neighbors'
    },
    
    # -------------------------------------------------------------------------
    # PROBABILISTIC MODELS
    # -------------------------------------------------------------------------
    'naive_bayes': {
        'model': GaussianNB,
        'params': {},
        'description': 'Gaussian Naive Bayes'
    },
    
    # -------------------------------------------------------------------------
    # NEURAL NETWORKS
    # -------------------------------------------------------------------------
    'mlp': {
        'model': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': RANDOM_SEED,
            'early_stopping': True
        },
        'description': 'Multi-layer perceptron'
    }
}

# Import device utils
from scripts.utils.device_utils import (
    get_xgboost_device_params,
    get_lightgbm_device_params,
    get_catboost_device_params
)

# ============================================================================
# ADVANCED GRADIENT BOOSTING MODELS (Optional)
# ============================================================================

ADVANCED_MODELS = {}

if XGBOOST_AVAILABLE:
    # Merge base params with device params
    xgb_params = {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'scale_pos_weight': 8.0 # Handle imbalance (~11% positive)
    }
    xgb_params.update(get_xgboost_device_params())
    
    ADVANCED_MODELS['xgboost'] = {
        'model': XGBClassifier,
        'params': xgb_params,
        'description': 'XGBoost (Balanced)'
    }

if LIGHTGBM_AVAILABLE:
    lgbm_params = {
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'metric': 'binary_logloss',
        'verbose': -1,
        'scale_pos_weight': 8.0 # Handle imbalance
    }
    lgbm_params.update(get_lightgbm_device_params())

    ADVANCED_MODELS['lightgbm'] = {
        'model': LGBMClassifier,
        'params': lgbm_params,
        'description': 'LightGBM (Balanced)'
    }

if CATBOOST_AVAILABLE:
    cat_params = {
        'iterations': 500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5,
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'eval_metric': 'Logloss',
        'loss_function': 'Logloss',
        'thread_count': -1,
        'scale_pos_weight': 8.0 # Handle imbalance
    }
    cat_params.update(get_catboost_device_params())

    ADVANCED_MODELS['catboost'] = {
        'model': CatBoostClassifier,
        'params': cat_params,
        'description': 'CatBoost (Balanced)'
    }

# ============================================================================
# ENSEMBLE STRATEGIES
# ============================================================================

ENSEMBLE_STRATEGIES = {
    # -------------------------------------------------------------------------
    # VOTING ENSEMBLES
    # -------------------------------------------------------------------------
    'voting_soft': {
        'type': 'VotingClassifier',
        'voting': 'soft',
        'weights': None,  # Equal weights
        'description': 'Soft voting (average probabilities)'
    },
    
    'voting_hard': {
        'type': 'VotingClassifier',
        'voting': 'hard',
        'weights': None,
        'description': 'Hard voting (majority vote)'
    },
    
    'voting_weighted': {
        'type': 'VotingClassifier',
        'voting': 'soft',
        'weights': 'auto',  # Will be determined by CV performance
        'description': 'Weighted soft voting based on CV scores'
    },
    
    # -------------------------------------------------------------------------
    # STACKING ENSEMBLES
    # -------------------------------------------------------------------------
    'stacking_lr': {
        'type': 'StackingClassifier',
        'final_estimator': LogisticRegression(
            C=1.0,
            random_state=RANDOM_SEED,
            max_iter=1000
        ),
        'cv': 5,
        'description': 'Stacking with Logistic Regression meta-learner'
    },
    
    'stacking_rf': {
        'type': 'StackingClassifier',
        'final_estimator': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'cv': 5,
        'description': 'Stacking with Random Forest meta-learner'
    },
    
    'stacking_xgb': {
        'type': 'StackingClassifier',
        'final_estimator': XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ) if XGBOOST_AVAILABLE else None,
        'cv': 5,
        'description': 'Stacking with XGBoost meta-learner'
    },
    
    # -------------------------------------------------------------------------
    # BLENDING
    # -------------------------------------------------------------------------
    'blending': {
        'type': 'Blending',
        'holdout_size': 0.2,
        'description': 'Blending with holdout set'
    },
    
    # -------------------------------------------------------------------------
    # AVERAGING
    # -------------------------------------------------------------------------
    'simple_average': {
        'type': 'SimpleAverage',
        'description': 'Simple average of predictions'
    },
    
    'weighted_average': {
        'type': 'WeightedAverage',
        'weights': 'auto',
        'description': 'Weighted average based on CV performance'
    },
    
    'rank_average': {
        'type': 'RankAverage',
        'description': 'Average of rank-transformed predictions'
    }
}

# ============================================================================
# MODEL SELECTION PRESETS
# ============================================================================

MODEL_PRESETS = {
    'fast': [
        'logistic_regression',
        'ridge_classifier',
        'naive_bayes',
        'hist_gradient_boosting'
    ],
    
    'balanced': [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'hist_gradient_boosting',
        'xgboost' if XGBOOST_AVAILABLE else 'extra_trees'
    ],
    
    'powerful': [
        'random_forest',
        'extra_trees',
        'gradient_boosting',
        'hist_gradient_boosting',
        'xgboost' if XGBOOST_AVAILABLE else 'adaboost',
        'lightgbm' if LIGHTGBM_AVAILABLE else 'mlp',
        'catboost' if CATBOOST_AVAILABLE else 'knn'
    ],
    
    'diverse': [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'knn',
        'naive_bayes',
        'mlp'
    ],
    
    'gradient_boosting_only': [
        'gradient_boosting',
        'hist_gradient_boosting',
        'xgboost' if XGBOOST_AVAILABLE else 'adaboost',
        'lightgbm' if LIGHTGBM_AVAILABLE else 'extra_trees',
        'catboost' if CATBOOST_AVAILABLE else 'random_forest'
    ],
    
    'all': list(BASE_MODELS.keys()) + list(ADVANCED_MODELS.keys())
}

# ============================================================================
# HYPERPARAMETER SEARCH SPACES (for optimization)
# ============================================================================

HYPERPARAMETER_SPACES = {
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 12, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    
    'extra_trees': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 12, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [4, 10, 20],
        'subsample': [0.7, 0.8, 0.9]
    },
    
    'hist_gradient_boosting': {
        'max_iter': [100, 200, 300],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [5, 8, 10, None],
        'min_samples_leaf': [20, 30, 50],
        'l2_regularization': [0.0, 1.0, 10.0]
    },
    
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'newton-cg'],
        'penalty': ['l2'],
        'class_weight': [None, 'balanced']
    },
    
    'ridge_classifier': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'class_weight': [None, 'balanced']
    },
    
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1.0, 3.0, 5.0, 8.0] # Handle imbalance (11% positive)
    },
    
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.7, 0.8, 0.9],
        'scale_pos_weight': [1.0, 3.0, 5.0, 8.0]
    },
    
    'catboost': {
        'iterations': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'scale_pos_weight': [1.0, 3.0, 5.0, 8.0]
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_METRICS = {
    'primary': 'roc_auc',  # Primary metric for model selection
    'secondary': ['accuracy', 'precision', 'recall', 'f1'],
    'threshold_metrics': ['balanced_accuracy', 'matthews_corrcoef']
}

# ============================================================================
# CROSS-VALIDATION STRATEGY
# ============================================================================

CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_SEED,
    'stratified': True
}
