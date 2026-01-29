"""
Device detection utilities for GPU acceleration.
Checks for NVIDIA GPU availability and provides configuration for ML libraries.
"""

import shutil
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_gpu_available() -> bool:
    """
    Check if NVIDIA GPU is available via nvidia-smi.
    
    Returns:
        bool: True if GPU detected, False otherwise.
    """
    if shutil.which('nvidia-smi') is None:
        return False
    
    try:
        # Run nvidia-smi to confirm it works
        subprocess.check_call(
            ['nvidia-smi'], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def get_xgboost_device_params() -> Dict[str, Any]:
    """
    Get XGBoost parameters for available device.
    
    Returns:
        Dict: parameters for 'tree_method' and 'device'
    """
    if is_gpu_available():
        logger.info("XGBoost: GPU detected, using 'gpu_hist'")
        return {
            'tree_method': 'gpu_hist',
            'device': 'cuda'
        }
    else:
        logger.info("XGBoost: No GPU detected, using CPU")
        return {
            'tree_method': 'hist',
            'device': 'cpu'
        }

def get_lightgbm_device_params() -> Dict[str, Any]:
    """
    Get LightGBM parameters for available device.
    
    Returns:
        Dict: parameters for 'device'
    """
    if is_gpu_available():
        logger.info("LightGBM: GPU detected, attempting to use 'gpu'")
        return {'device': 'gpu'}
    else:
        logger.info("LightGBM: No GPU detected, using 'cpu'")
        return {'device': 'cpu'}

def get_catboost_device_params() -> Dict[str, Any]:
    """
    Get CatBoost parameters for available device.
    
    Returns:
        Dict: parameters for 'task_type'
    """
    if is_gpu_available():
        logger.info("CatBoost: GPU detected, using 'GPU'")
        return {'task_type': 'GPU'}
    else:
        logger.info("CatBoost: No GPU detected, using 'CPU'")
        return {'task_type': 'CPU'}
