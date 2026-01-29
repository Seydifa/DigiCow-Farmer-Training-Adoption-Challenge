"""
Model utilities and wrappers.
Includes compatibility fixes for CatBoost and sklearn 1.6+.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)

# Try to import CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    # Define a proper dummy class for inheritance safety
    class CatBoostClassifier(BaseEstimator, ClassifierMixin):
        def fit(self, X, y=None, **kwargs): raise ImportError("CatBoost not installed")
        def predict(self, X, **kwargs): raise ImportError("CatBoost not installed")
        def predict_proba(self, X, **kwargs): raise ImportError("CatBoost not installed")

class SklearnCompatibleCatBoostClassifier(CatBoostClassifier):
    """
    Wrapper for CatBoostClassifier to fix compatibility issues with scikit-learn 1.6+.
    Specifically addresses the missing '__sklearn_tags__' attribute.
    """
    def __init__(self, **kwargs):
        if CATBOOST_AVAILABLE:
            super().__init__(**kwargs)
        else:
            logger.warning("CatBoost not available, wrapper initialized as dummy.")
            
    def __sklearn_tags__(self):
        """
        Implementation of sklearn tags for compatibility.
        """
        # Return default tags generator from BaseEstimator
        # or construct minimal tags expected by sklearn 1.6 check
        
        # If BaseEstimator implements it (it should in 1.6), delegates
        if hasattr(BaseEstimator, '__sklearn_tags__'):
            return BaseEstimator.__sklearn_tags__(self)
        return {}

    def fit(self, X, y=None, **kwargs):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed.")
        return super().fit(X, y, **kwargs)
    
    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)

def get_calibrated_classifier(estimator, cv=3, method='isotonic'):
    """
    Wrap an estimator with CalibratedClassifierCV.
    
    Args:
        estimator: The base estimator
        cv: Number of cross-validation folds for calibration
        method: Calibration method ('sigmoid' or 'isotonic')
        
    Returns:
        CalibratedClassifierCV instance
    """
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(estimator=estimator, cv=cv, method=method)
