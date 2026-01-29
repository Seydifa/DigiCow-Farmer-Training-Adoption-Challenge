"""
Configuration file for the DigiCow Farmer Training Adoption Challenge.
Contains all paths, constants, and configuration parameters.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data files
TRAIN_FILE = RAW_DATA_DIR / "Train.csv"
TEST_FILE = RAW_DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "SampleSubmission.csv"
DATA_DICT_FILE = RAW_DATA_DIR / "dataset_data_dictionary.csv"

# Processed data files
TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.csv"

# Target variable
TARGET_COL = "adopted_within_07_days"

# Feature categories
CATEGORICAL_FEATURES = [
    "gender",
    "age",
    "registration",
    "belong_to_cooperative",
    "county",
    "subcounty",
    "ward",
    "trainer",
    "has_second_training"
]

NUMERICAL_FEATURES = [
    "num_trainings_30d",
    "num_trainings_60d",
    "num_total_trainings",
    "num_repeat_trainings",
    "days_to_second_training",
    "num_unique_trainers"
]

TEXT_FEATURES = [
    "topics_list"
]

DATE_FEATURES = [
    "first_training_date"
]

# Topic categories for feature engineering
TOPIC_CATEGORIES = {
    "dairy": ["dairy", "cow", "milk", "calf", "lactating", "maziwa"],
    "poultry": ["poultry", "chicken", "kienyeji", "layer", "broiler"],
    "crops": ["maize", "bean", "seed", "fertilizer", "weed", "pest", "harvest"],
    "health": ["health", "disease", "vaccination", "deworming", "biosecurity"],
    "feeding": ["feeding", "feed", "nutrition", "tyari"],
    "technology": ["ndume", "app", "digicow"],
    "breeding": ["breeding", "ai", "infertility", "mating", "crv"],
    "management": ["management", "record", "housing"],
    "biodeal": ["biodeal"],
    "biogas": ["biogas", "sistema"]
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Model parameters
CV_FOLDS = 5
TEST_SIZE = 0.2

# Feature engineering parameters
MIN_TOPIC_FREQUENCY = 5  # Minimum frequency for topic to be included
MAX_TOPICS_TO_EXTRACT = 50  # Maximum number of individual topics to extract

# Logging
LOG_LEVEL = "INFO"
