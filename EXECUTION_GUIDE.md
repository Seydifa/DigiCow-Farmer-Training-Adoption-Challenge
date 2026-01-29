# 🚀 Execution Guide - DigiCow Pipeline

## Quick Start (3 Commands)

### **Option 1: Balanced Preset (Recommended)** ⚖️
```bash
cd "/home/mseydifaye/Documents/Kaggle/DigiCow Farmer Training Adoption Challenge
/scripts"
python train_pipeline.py --preset balanced --top-n 5
```
**Time**: ~10-15 minutes  
**Models**: Logistic Regression, Random Forest, Gradient Boosting, Hist GB, XGBoost

---

### **Option 2: Fast Preset (Quick Test)** ⚡
```bash
python train_pipeline.py --preset fast --top-n 3
```
**Time**: ~2-3 minutes  
**Models**: Logistic Regression, Ridge, Naive Bayes, Hist GB

---

### **Option 3: Powerful Preset (Maximum Performance)** 💪
```bash
python train_pipeline.py --preset powerful --top-n 7
```
**Time**: ~20-30 minutes  
**Models**: All gradient boosting models + tree ensembles

---

## What Happens When You Run

### **Step 1: Feature Engineering** (1-2 min)
```
✓ Load Train.csv and Test.csv
✓ Parse topics lists
✓ Create 50+ engineered features
✓ Optimize memory usage
✓ Save to data/processed/
```

### **Step 2: Data Preparation** (< 1 min)
```
✓ Separate features and target
✓ One-hot encode categorical features
✓ Handle missing values
✓ Convert to numpy arrays
```

### **Step 3: Model Training** (5-20 min depending on preset)
```
✓ Train base models with 5-fold CV
✓ Evaluate with ROC-AUC, Accuracy, F1, etc.
✓ Rank models by performance
✓ Display results table
```

### **Step 4: Ensemble Creation** (2-5 min)
```
✓ Build Voting Ensemble (top N models)
✓ Build Stacking Ensemble (with Logistic Regression)
✓ Build Weighted Average Ensemble
✓ Train all ensembles
```

### **Step 5: Generate Predictions** (< 1 min)
```
✓ Generate predictions from each ensemble
✓ Create meta-ensemble (average of all)
✓ Save 4 submission files
✓ Save trained models
```

---

## Output Files

### **Submissions** (in `data/processed/submissions_TIMESTAMP/`)
1. `submission_voting_soft.csv` - Soft voting ensemble
2. `submission_stacking_lr.csv` - Stacking with LR meta-learner
3. `submission_weighted_average.csv` - Weighted average
4. `submission_meta_ensemble.csv` - **RECOMMENDED** (average of all 3)

### **Models** (in `data/processed/models_TIMESTAMP/`)
- All trained base models (`.pkl` files)
- Ensemble model (`.pkl` file)
- CV results (`.csv` file)
- Feature importance (`.csv` file)

---

## Command Line Options

```bash
python train_pipeline.py [OPTIONS]

Options:
  --preset {fast,balanced,powerful,diverse,gradient_boosting_only}
      Model preset to use (default: balanced)
  
  --top-n N
      Number of top models for ensemble (default: 5)
  
  --no-cache
      Disable feature caching (re-run feature engineering)
```

---

## Examples

### **1. Quick Test Run**
```bash
python train_pipeline.py --preset fast --top-n 3
```

### **2. Production Run**
```bash
python train_pipeline.py --preset powerful --top-n 7
```

### **3. Re-run Feature Engineering**
```bash
python train_pipeline.py --preset balanced --no-cache
```

### **4. Gradient Boosting Only**
```bash
python train_pipeline.py --preset gradient_boosting_only --top-n 5
```

---

## Expected Output

```
================================================================================
DIGICOW FARMER TRAINING ADOPTION - COMPLETE PIPELINE
================================================================================
Timestamp: 20260129_002500
Model Preset: balanced
Top N Models: 5
Random Seed: 42

================================================================================
STEP 1: FEATURE ENGINEERING
================================================================================
Loading raw data...
Train shape: (15000, 17)
Test shape: (5000, 16)
Parsing topics...
Creating features...
✓ Created 8 temporal features
✓ Created 12 engagement features
✓ Created 22 topic features
✓ Created 4 demographic features
✓ Created 5 interaction features
✓ Created 2 ratio features
✓ Created 1 missing indicator feature
Memory usage decreased from 45.23 MB to 18.67 MB (58.7% reduction)

================================================================================
STEP 2: DATA PREPARATION
================================================================================
Total features: 54
Categorical features: 8
Numerical features: 46
One-hot encoding categorical features...
Training set: (15000, 120)
Test set: (5000, 120)
Target distribution: [10500  4500]
Class balance: 30.00% positive class

================================================================================
STEP 3: MODEL TRAINING
================================================================================
Using preset: 'balanced'
Models to train: ['logistic_regression', 'random_forest', 'gradient_boosting', 
                  'hist_gradient_boosting', 'xgboost']

Building 5 base models...
  ✓ logistic_regression: Linear model with L2 regularization
  ✓ random_forest: Robust ensemble of decision trees
  ✓ gradient_boosting: Classic gradient boosting
  ✓ hist_gradient_boosting: Fast histogram-based gradient boosting
  ✓ xgboost: XGBoost - Extreme Gradient Boosting

Evaluating base models with 5-fold CV...

Evaluating: logistic_regression
  ROC-AUC: 0.7523 ± 0.0134
  Accuracy: 0.7245
  F1-Score: 0.6789
  Time: 2.34s

Evaluating: random_forest
  ROC-AUC: 0.7891 ± 0.0156
  Accuracy: 0.7567
  F1-Score: 0.7123
  Time: 45.67s

[... more models ...]

================================================================================
MODEL EVALUATION RESULTS
================================================================================
                    model  roc_auc_mean  roc_auc_std  accuracy_mean  f1_mean
               xgboost         0.8234       0.0142         0.7823    0.7456
       random_forest         0.7891       0.0156         0.7567    0.7123
  gradient_boosting         0.7845       0.0167         0.7489    0.7034
hist_gradient_boosting      0.7798       0.0151         0.7445    0.6989
logistic_regression         0.7523       0.0134         0.7245    0.6789

================================================================================
STEP 4: ENSEMBLE CREATION
================================================================================

1. Creating Voting Ensemble (top 5 models)...
Auto-weights: [0.2456 0.2123 0.1987 0.1789 0.1645]
✓ Voting ensemble trained

2. Creating Stacking Ensemble (top 5 models)...
✓ Stacking ensemble trained

3. Creating Weighted Average Ensemble (top 7 models)...
Model weights:
  xgboost                  : 0.2456
  random_forest            : 0.2123
  gradient_boosting        : 0.1987
  [...]
✓ Weighted average ensemble ready

================================================================================
STEP 5: GENERATING PREDICTIONS
================================================================================

Generating predictions: voting_soft
  Saved: data/processed/submissions_20260129_002500/submission_voting_soft.csv

Generating predictions: stacking_lr
  Saved: data/processed/submissions_20260129_002500/submission_stacking_lr.csv

Generating predictions: weighted_average
  Saved: data/processed/submissions_20260129_002500/submission_weighted_average.csv

Creating meta-ensemble (average of all methods)...
  Saved: data/processed/submissions_20260129_002500/submission_meta_ensemble.csv

✓ All predictions saved to data/processed/submissions_20260129_002500

================================================================================
PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
Total execution time: 12.45 minutes
Models saved to: data/processed/models_20260129_002500
Submissions saved to: data/processed/submissions_20260129_002500

Submission files created:
  1. submission_voting_soft.csv
  2. submission_stacking_lr.csv
  3. submission_weighted_average.csv
  4. submission_meta_ensemble.csv (RECOMMENDED)

================================================================================
```

---

## Troubleshooting

### **Issue: XGBoost/LightGBM/CatBoost not found**
**Solution**: Install optional libraries
```bash
pip install xgboost lightgbm catboost
```

### **Issue: Out of memory**
**Solution**: Use 'fast' preset or reduce top-n
```bash
python train_pipeline.py --preset fast --top-n 3
```

### **Issue: Takes too long**
**Solution**: Use 'fast' preset
```bash
python train_pipeline.py --preset fast
```

---

## Next Steps After Running

1. ✅ Check the CV results to see model performance
2. ✅ Review feature importance (saved in models directory)
3. ✅ Submit `submission_meta_ensemble.csv` to Kaggle
4. ✅ If needed, tune hyperparameters of top models
5. ✅ Experiment with different presets

---

## Performance Tips

1. **Install all optional libraries** for best performance:
   ```bash
   pip install xgboost lightgbm catboost
   ```

2. **Use powerful preset** for competitions:
   ```bash
   python train_pipeline.py --preset powerful --top-n 7
   ```

3. **Monitor CV scores** - if overfitting, reduce model complexity

4. **Try different top-n values** - usually 5-7 is optimal

---

**Ready to run?** Just execute one of the commands above! 🚀
