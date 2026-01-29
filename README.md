# DigiCow Farmer Training Adoption Challenge

## 📁 Project Structure

```
DigiCow Farmer Training Adoption Challenge/
├── data/
│   ├── raw_data/          # Original dataset files
│   └── processed/         # Processed feature files
├── scripts/
│   ├── config.py          # Configuration and constants
│   ├── feature_engineering.py  # Feature engineering class
│   ├── run_feature_engineering.py  # Main pipeline script
│   └── utils/
│       ├── __init__.py
│       └── data_utils.py  # Data loading and preprocessing utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py  # Tests for data utilities
│   └── test_feature_engineering.py  # Tests for feature engineering
├── notebooks/             # Jupyter notebooks for exploration
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Feature Engineering Pipeline

```bash
cd scripts
python run_feature_engineering.py
```

This will:
- Load training and test data
- Parse topics lists
- Create 50+ engineered features
- Optimize memory usage
- Save processed datasets to `data/processed/`

### 3. Run Tests

```bash
cd tests
python -m pytest test_data_utils.py -v
python -m pytest test_feature_engineering.py -v
```

Or run all tests:
```bash
python -m pytest tests/ -v
```

## 📊 Features Created

The pipeline creates **50+ optimized features** across multiple categories:

### 1. **Temporal Features** (8 features)
- Day of week, month, week of month
- Days to second training (binned)
- Attended second within 7 days
- Attended second same day

### 2. **Engagement Features** (12 features)
- Training frequency (30d, 60d)
- Training acceleration
- Engagement score
- Repeat rate
- Training consistency
- Engagement level flags (high, super, low)
- Training gap indicators

### 3. **Topic Features** (15+ features)
- Number of unique topics
- Topic diversity ratio
- Topic repetition rate
- Trainings per unique topic
- Topic category flags (dairy, poultry, crops, health, etc.)
- Topic category count
- Topic focus score

### 4. **Demographic Features** (4 features)
- Binary gender encoding
- Binary age encoding
- Registration method encoding
- Age-gender combination

### 5. **Interaction Features** (5 features)
- Cooperative × engagement
- Gender × cooperative
- Age × trainings
- Registration × second training
- Topics × engagement

### 6. **Ratio Features** (2 features)
- Sustained engagement ratio
- Training intensity delta

### 7. **Missing Indicators** (1 feature)
- Missing days to second training

## 🔧 Configuration

All configuration is centralized in `scripts/config.py`:

- **Paths**: Data directories, file locations
- **Features**: Categorical, numerical, text features
- **Topic Categories**: Keywords for topic classification
- **Model Parameters**: Random seed, CV folds, test size
- **Feature Engineering**: Min topic frequency, max topics

## 🧪 Testing

Comprehensive unit tests cover:

- ✅ Data loading and parsing
- ✅ Topic list extraction
- ✅ Memory optimization
- ✅ All feature engineering methods
- ✅ Edge cases (missing values, zero trainings)
- ✅ Helper functions

## 📈 Performance Optimizations

1. **Vectorized Operations**: All feature engineering uses pandas vectorization
2. **Memory Optimization**: Automatic dtype downcasting
3. **Efficient Parsing**: Optimized topic list parsing
4. **Categorical Encoding**: Uses pandas Categorical dtype
5. **Logging**: Detailed progress tracking

## 💡 Usage Examples

### Load and Process Data

```python
from scripts.config import TRAIN_FILE
from scripts.utils.data_utils import load_data, extract_topics_vectorized
from scripts.feature_engineering import FeatureEngineer

# Load data
df = load_data(TRAIN_FILE)

# Parse topics
df = extract_topics_vectorized(df)

# Create features
fe = FeatureEngineer(topic_categories=TOPIC_CATEGORIES)
df_features = fe.create_all_features(df)
```

### Run Specific Feature Groups

```python
# Create only engagement features
df = fe.create_engagement_features(df)

# Create only topic features
df = fe.create_topic_features(df)
```

## 📝 Notes

- All scripts are optimized for performance
- Memory usage is minimized through dtype optimization
- Comprehensive logging for debugging
- Modular design for easy extension
- Well-tested with unit tests

## 🎯 Target Variable

**`adopted_within_07_days`**: Binary indicator (1/0) showing whether the farmer adopted training practices within 7 days of their first training.

## 📧 Contact

For questions or issues, please refer to the project documentation.
