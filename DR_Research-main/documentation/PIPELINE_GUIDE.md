# APTOS 2019 DR Classification - Complete Pipeline Guide

## Overview

This guide explains the refactored PyTorch training pipeline for the **original Kaggle APTOS 2019 Blindness Detection dataset**. The pipeline uses **Stratified K-Fold Cross-Validation** with comprehensive metrics computation.

---

## Table of Contents

1. [Dataset Structure](#dataset-structure)
2. [Validation Strategy Comparison](#validation-strategy-comparison)
3. [Why Not Use Test Set for Validation](#why-not-use-test-set-for-validation)
4. [Class Imbalance Handling](#class-imbalance-handling)
5. [Metrics Explanation](#metrics-explanation)
6. [Training Process](#training-process)
7. [Results Interpretation](#results-interpretation)
8. [Publication Guidelines](#publication-guidelines)

---

## Dataset Structure

### Original Kaggle Layout

```
APTOS2019/
├── train.csv                 (id_code, diagnosis) - 3662 labeled images
├── test.csv                  (id_code only) - 1923 unlabeled images
├── train_images/             (3662 training images)
└── test_images/              (1923 test images for Kaggle submission)
```

### Key Points

- **Train Set**: 3662 images with labels (diagnosis 0-4)
- **Test Set**: 1923 images WITHOUT labels
- **Classes**: 5 (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Imbalance**: Severe (class 0: 1805 images, class 4: 85 images)

### Class Distribution

```
Class 0 (No DR):        1805 images (49.3%) █████████████████
Class 1 (Mild):         1300 images (35.5%) █████████████
Class 2 (Moderate):      435 images (11.9%) ████
Class 3 (Severe):         99 images ( 2.7%) █
Class 4 (Proliferative):  85 images ( 2.3%) █
```

---

## Validation Strategy Comparison

### ❌ Strategy 1: Single Train/Val Split (NOT RECOMMENDED)

```
Train Set (80%) → Training
Val Set (20%)   → Validation & Early Stopping & Model Selection
Test Set (1923) → Final Evaluation (Kaggle submission only)
```

**Advantages:**
- Simple to implement
- Faster training

**Disadvantages:**
- High variance in performance estimate (depends on ONE random split)
- Final metrics may not reflect true generalization
- One unlucky random split can give biased results
- **NOT publishable in top-tier venues** without additional validation

### ✅ Strategy 2: Stratified K-Fold Cross-Validation (RECOMMENDED)

```
Fold 1: Train on folds 2,3,4,5  (2929 samples) → Validate on fold 1 (733 samples)
Fold 2: Train on folds 1,3,4,5  (2929 samples) → Validate on fold 2 (733 samples)
Fold 3: Train on folds 1,2,4,5  (2929 samples) → Validate on fold 3 (733 samples)
Fold 4: Train on folds 1,2,3,5  (2929 samples) → Validate on fold 4 (733 samples)
Fold 5: Train on folds 1,2,3,4  (2929 samples) → Validate on fold 5 (733 samples)

Final Test Evaluation: Test Set (1923 unlabeled images) - separate from all training
Final Metrics: Average across all 5 folds → Mean ± Std
```

**Advantages:**
- **Lower variance** in performance estimate
- Each sample used for training ~80% of folds, validation ~20%
- Reproducible with fixed random seed
- **Better class distribution preservation**
- **Publishable in top conferences** (IEEE TPAMI, MICCAI, etc.)
- Confidence estimates (±Std) for metrics
- Reduces dependency on one random split

**Why Stratified?**
- Standard K-Fold: Random assignment of folds (class distribution may vary across folds)
- **Stratified K-Fold**: Ensures EACH fold has same class distribution as original
- Critical for imbalanced data like DR (each fold has ~49% class 0, ~35% class 1, etc.)

### Comparison Table

| Feature | Single Split | K-Fold |
|---------|-------------|--------|
| Training data used | ~80% for training | ~80% for training |
| Validation data used | ~20% for validation | ~20% per fold, rotated |
| Generalization estimate | Single value (high variance) | Mean ± Std (robust) |
| Computational cost | Low | 5× higher |
| Publishability | Lower (needs additional validation) | **Higher (standard in papers)** |
| Recommendation | For quick prototyping | **For final results & publication** |

---

## Why Not Use Test Set for Validation?

### ❌ Test Set is NOT a Validation Set

**What is the Kaggle test set for?**
- The 1923 unlabeled test images are for **final leaderboard submission**
- Kaggle will evaluate your predictions and rank submissions
- You cannot use them for validation because:

1. **No Labels Available**
   - test.csv only has `id_code`, NOT `diagnosis`
   - To validate, you'd need labels
   - Leaderboard uses hidden ground truth (only Kaggle has it)

2. **Leaderboard is the Final Evaluation**
   - Submitting to Kaggle is ONE-WAY evaluation
   - Can't iterate on test set during development
   - Not practical for hyperparameter tuning

3. **Test Set Contamination**
   - If you use test set for model selection, you're overfitting to it
   - Final leaderboard performance will be worse than your validation metrics
   - Violates proper train/val/test separation

### ✅ Proper Protocol

```
TRAINING DATA (3662 from train.csv):
├── Train Fold 1 (2929) → Train model
├── Train Fold 2 (2929) → Train model
├── Train Fold 3 (2929) → Train model
├── Train Fold 4 (2929) → Train model
├── Train Fold 5 (2929) → Train model
├── Val Fold 1 (733)    → Evaluate (keep separate!)
├── Val Fold 2 (733)    → Evaluate (keep separate!)
├── Val Fold 3 (733)    → Evaluate (keep separate!)
├── Val Fold 4 (733)    → Evaluate (keep separate!)
└── Val Fold 5 (733)    → Evaluate (keep separate!)
        ↓
    Average results across folds
        ↓
  VALIDATION METRICS (Mean ± Std)

TEST SET (1923 unlabeled images):
        ↓
    Make predictions
        ↓
    Submit to Kaggle
        ↓
    Get leaderboard score
```

### Key Principle

> **Never use the test set for validation, hyperparameter tuning, or model selection. Keep it completely separate until final evaluation.**

---

## Class Imbalance Handling

### Problem: Severe Imbalance

APTOS has extremely imbalanced data:
- Class 0: 1805 images (49.3%)
- Class 4: 85 images (2.3%)
- **Ratio: ~21:1**

**Naive Model Problem**: A model trained with standard loss could learn to predict "Class 0" for everything and still get ~49% accuracy!

### Solution 1: WeightedRandomSampler

**How it works:**
```python
# Calculate class weights inversely proportional to frequency
class_counts = [1805, 1300, 435, 99, 85]
class_weights = 1 / class_counts = [0.00055, 0.00077, 0.00230, 0.0101, 0.0118]

# Each sample gets a weight based on its class
sample_weights = class_weights[sample_class]

# Random sampler oversamples rare classes
WeightedRandomSampler(weights=sample_weights, num_samples=len(data))
```

**Result**: In each epoch, the model sees a **balanced distribution** of classes
- ~20% Class 0 (instead of 49%)
- ~20% Class 1 (instead of 35%)
- ~20% Class 2, 3, 4 (instead of rare)

**Advantage**: Model learns to recognize all classes equally well

### Solution 2: Focal Loss

**Problem with Standard Cross-Entropy:**
- Easy examples (high confidence, correct prediction): Loss ≈ 0, gradient ≈ 0
- Hard examples (low confidence, wrong prediction): Loss > 1, gradient > 0
- Imbalance: Most examples are easy (class 0), swamp the gradient
- Result: Model gets stuck in local minimum with poor minority class performance

**Focal Loss Formula:**
```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

**Where:**
- `p_t` = predicted probability of correct class (0-1)
- `gamma` = focusing parameter (usually 2.0)
  - `gamma=0`: Standard cross-entropy
  - `gamma=2`: Focuses 100× more on hard examples
- `alpha` = balance parameter (0.25 recommended)

**Intuition:**
```
Easy example (p_t=0.99, correct):
  Focal weight = (1 - 0.99)^2 = 0.0001 → Small loss contribution

Hard example (p_t=0.3, wrong):
  Focal weight = (1 - 0.3)^2 = 0.49 → Large loss contribution
```

**Result**: Focus training on hard examples, especially minority classes

### Solution 3: Combined Approach (USED IN PIPELINE)

**Weighted Sampler + Focal Loss**:
- Sampler: Ensures each epoch sees balanced data distribution
- Focal Loss: Ensures model learns hard examples well
- Synergistic: Together they're stronger than separately

**Why both?**
1. Weighted Sampler solves "unbalanced gradient problem" during epoch
2. Focal Loss solves "easy examples dominate" problem within batch
3. Double protection against class imbalance

---

## Metrics Explanation

### 1. **Accuracy**
```
Accuracy = (# Correct Predictions) / (# Total Samples)
```

**Interpretation**: Percentage of samples classified correctly

**Problem for imbalanced data**: 
- If class 0 is 50%, predicting "class 0" always gives 50% accuracy
- Misleading for minority classes

**Use case**: General performance overview

---

### 2. **Precision (Macro & Weighted)**

```
Precision = (True Positives) / (True Positives + False Positives)
          = "Of samples we predicted as class i, how many were correct?"
```

**Macro Precision**: Average precision across all classes
- Gives equal weight to each class
- Class 0 and Class 4 contribute equally (even though Class 0 is 21× more common)

**Weighted Precision**: Average precision weighted by class frequency
- Classes with more samples contribute more to average
- Reflects real-world distribution

**Use case**: How often is our positive prediction correct? (Important for rare diseases)

---

### 3. **Recall (Macro & Weighted)**

```
Recall = (True Positives) / (True Positives + False Negatives)
       = "Of actual class i samples, how many did we find?"
```

**Macro Recall**: Average recall across all classes
**Weighted Recall**: Weighted recall across all classes

**Use case**: How many patients with disease did we diagnose? (Crucial in medical imaging!)

---

### 4. **F1-Score (Macro & Weighted)**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic mean of precision and recall
   = Balanced metric when you care about both precision and recall
```

**Macro F1**: Average F1 across classes (equal weighting)
**Weighted F1**: Average F1 weighted by class frequency

**Use case**: Overall balance between precision and recall

---

### 5. **Quadratic Weighted Kappa (QWK) - CRITICAL FOR APTOS**

```
QWK = Agreement - Expected Agreement (weighted) / (1 - Expected Agreement)
```

**What makes it special for ordinal data:**
- Standard metrics treat all errors equally
- **QWK penalizes errors between distant classes MORE**

**Example:**
- Predicting Class 1 instead of Class 0: Small penalty
- Predicting Class 4 instead of Class 0: LARGE penalty

**Why important for DR:**
- Dr severity is ordinal: No DR → Mild → Moderate → Severe → Proliferative
- Misclassifying No DR as Severe is worse than misclassifying as Mild
- QWK captures this medical domain knowledge

**Range**: -1 to +1
- +1: Perfect agreement
- 0: Random guessing
- -1: Complete disagreement

**QWK Interpretation for APTOS:**
- < 0.4: Poor agreement
- 0.4 - 0.6: Fair agreement
- 0.6 - 0.75: Good agreement
- 0.75 - 0.9: Very good agreement
- > 0.9: Excellent agreement

---

### 6. **ROC-AUC (One-vs-Rest, Macro)**

```
AUC = Area Under the Receiver Operating Characteristic Curve
```

**What it measures:**
- Probability that model ranks a random positive sample higher than a random negative sample
- **One-vs-Rest (OvR)**: For each class, compute AUC vs all other classes
- **Macro**: Average AUC across all classes

**Range**: 0 to 1
- 0.5: Random guessing
- 0.7-0.8: Acceptable
- 0.8-0.9: Excellent  
- > 0.9: Outstanding

**Use case**: How well does model discriminate between classes at all thresholds?

---

### 7. **Per-Class F1 Scores**

```
F1_class_i = F1 score for only class i vs all others
```

**Use case**: 
- Identify which classes the model struggles with
- Minority classes often have lower F1
- Helps understand where to improve

---

## Training Process

### High-Level Flow

```
1. SETUP (Section 1)
   ├─ Set random seeds (reproducibility)
   ├─ Load libraries
   └─ Configure device (GPU/CPU)

2. LOAD DATA (Section 2)
   ├─ Read train.csv (3662 labels)
   ├─ Read test.csv (1923 unlabeled)
   ├─ Display class distribution
   └─ Explain validation strategy

3. CREATE FOLDS (Section 3)
   ├─ Use StratifiedKFold(n_splits=5)
   ├─ Ensure each fold has same class distribution
   ├─ Save fold assignments (reproducible)
   └─ Display fold statistics

4. FOR EACH FOLD (Section 11):
   ├─ DATASET (Section 4)
   │  ├─ Create APTOSDataset class
   │  ├─ Load images (224×224, normalized)
   │  ├─ Resize and pad to handle different input sizes
   │  └─ Support train/val/test modes
   │
   ├─ AUGMENTATION (Section 5)
   │  ├─ Train: Random flips, rotation, affine, color jitter
   │  └─ Val: Only normalization
   │
   ├─ DATALOADER (Section 7)
   │  ├─ WeightedRandomSampler for balanced batches
   │  ├─ Batch size: 32
   │  └─ num_workers: 0 (Windows compatible)
   │
   ├─ MODEL (Section 6)
   │  ├─ Load pretrained backbone (EfficientNet-B4 / ResNet-50)
   │  ├─ Replace head with Linear(backbone_features → 5)
   │  └─ Add dropout(0.3)
   │
   ├─ LOSS FUNCTION (Section 7)
   │  └─ Focal Loss (alpha=0.25, gamma=2.0)
   │
   ├─ OPTIMIZER & SCHEDULER (Sections 8, 10)
   │  ├─ AdamW(lr=1e-3, weight_decay=2e-4)
   │  ├─ CosineAnnealingLR (T_max=num_epochs-warmup)
   │  └─ Warmup: 2 epochs (linear from 1e-6 to 1e-3)
   │
   ├─ TRAINING LOOP (Section 8)
   │  ├─ For epoch in range(80):
   │  │  ├─ train_epoch()
   │  │  │  ├─ Forward pass
   │  │  │  ├─ Backward pass (+ gradient clipping)
   │  │  │  ├─ Optimizer step
   │  │  │  └─ Track loss & accuracy
   │  │  │
   │  │  └─ validate()
   │  │     ├─ No gradient computation
   │  │     ├─ Compute loss & accuracy
   │  │     └─ Compute all metrics (Section 12)
   │  │
   │  ├─ Early Stopping
   │  │  ├─ Monitor: validation loss
   │  │  ├─ Patience: 15 epochs
   │  │  └─ Save best model when val_loss decreases
   │  │
   │  └─ Log every 10 epochs
   │
   └─ SAVE RESULTS
      ├─ fold_N_results.json (validation metrics)
      ├─ best_model_fold_N.pth (best model weights)
      └─ Training history

5. AGGREGATE (Section 12)
   ├─ For each metric:
   │  ├─ Collect values from all 5 folds
   │  └─ Compute Mean ± Std ± Min ± Max
   │
   └─ Save final_summary.json

6. VISUALIZE (Section 13)
   ├─ Training curves (loss & accuracy)
   ├─ Cross-fold metrics comparison
   ├─ Per-class F1 scores
   └─ Save PNG plots

7. SUMMARY (Section 14)
   ├─ Print final results
   ├─ Display experiment configuration
   └─ Save EXPERIMENT_SUMMARY.txt
```

---

## Results Interpretation

### Example Results

```
Accuracy:    0.8234 ± 0.0156
F1 Macro:    0.7821 ± 0.0234
F1 Weighted: 0.8156 ± 0.0198
QWK:         0.7456 ± 0.0312
```

### What This Means

- **Accuracy 0.8234 ± 0.0156**: Model is correct 82.34% of the time, ±1.56% (confidence interval)
- **F1 Macro 0.7821**: On average, class F1 scores are 0.78 (good balance across classes)
- **QWK 0.7456**: Very good ordinal agreement (considers class hierarchy)
- **± Std**: Variation across 5 folds (low std = robust, high std = unstable)

### Per-Class F1 Example

```
Class 0 (No DR):       0.89 ± 0.02  ← High (common class)
Class 1 (Mild):        0.81 ± 0.03  ← Good
Class 2 (Moderate):    0.72 ± 0.05  ← Fair (less common)
Class 3 (Severe):      0.55 ± 0.08  ← Poor (rare)
Class 4 (Proliferative): 0.48 ± 0.10 ← Challenging (rarest)
```

**Interpretation**: Model struggles with rare classes (expected with imbalance)

---

## Publication Guidelines

### For Top-Tier Medical Imaging Conference (MICCAI, IPMI, etc.)

**Required Sections:**

1. **Methodology**
   ```
   "We employed stratified 5-fold cross-validation to estimate 
    generalization performance. Each fold preserved the original 
    class distribution. We report mean ± standard deviation across 
    all folds, with fold-wise results in supplementary material."
   ```

2. **Results Table**
   ```
   | Metric           | Our Method      | Baseline        |
   |------------------|-----------------|-----------------|
   | Accuracy         | 0.823 ± 0.016   | 0.805 ± 0.022   |
   | F1 (Macro)       | 0.782 ± 0.023   | 0.751 ± 0.031   |
   | QWK              | 0.746 ± 0.031   | 0.710 ± 0.042   |
   ```

3. **Per-Class Performance**
   ```
   Include bar plot of F1 scores per class
   ```

4. **Fold-Wise Stability**
   ```
   "Low standard deviations (±1-3%) indicate stable performance 
    across folds, suggesting robust generalization."
   ```

5. **Supplementary Material**
   - Fold assignments (JSON) for reproducibility
   - Per-fold metrics
   - Confusion matrix
   - ROC curves

### Why Avoid Single Split in Papers

❌ **Weak**: "We achieved 82% accuracy on validation set"
- Reviewers will ask: "On which random split? What if you chose differently?"
- No confidence intervals
- High risk of luck/overfitting to split

✅ **Strong**: "We achieved 82.3% ± 1.6% accuracy (5-fold cross-validation)"
- Reproducible with fixed seed
- Confidence intervals provided
- Reviewers confident in generalization
- Standard in MICCAI, IEEE TPAMI, etc.

---

## Code Structure

### File Organization

```
dr_kfold_original_dataset.ipynb
├── Section 1: Setup & Configuration
├── Section 2: Load & Explore Data
├── Section 3: Create Stratified K-Fold Splits
├── Section 4: Custom Dataset Class (APTOSDataset)
├── Section 5: Data Augmentation Pipelines
├── Section 6: Model Architecture (DRClassifier)
├── Section 7: Class Imbalance Handling (WeightedSampler + FocalLoss)
├── Section 8: Training Loop
├── Section 9: Evaluation Loop
├── Section 10: Learning Rate Scheduler
├── Section 11: K-Fold Cross-Validation (MAIN LOOP)
├── Section 12: Compute Metrics
├── Section 13: Visualizations
└── Section 14: Summary & Export
```

### Key Classes

**APTOSDataset**
- Loads images from disk
- Handles train/val/test modes
- Applies transforms
- Supports fold-based indexing

**DRClassifier**
- Flexible backbone (EfficientNet / ResNet)
- Pretrained weights
- Custom head (N → 5 classes)
- Dropout for regularization

**FocalLoss**
- Focal Loss implementation
- Alpha and gamma parameters
- Down-weights easy examples

---

## Troubleshooting

### Q: Why is the model only predicting Class 0?
**A**: Class imbalance is severe. Check:
1. Is WeightedRandomSampler enabled?
2. Is Focal Loss being used?
3. Is class 0 actually overrepresented in your data?

### Q: Why are validation metrics different each run?
**A**: Different random seed. Use fixed `SEED = 42` at top of notebook.

### Q: Can I get the test predictions?
**A**: Yes! Add test loader and predict on test set after training.

### Q: How do I compare with other methods?
**A**: Use same k-fold setup. Report Mean ± Std from same folds.

### Q: Should I use 5-Fold or 10-Fold?
**A**: 5-Fold is standard (good balance). 10-Fold is more robust but 2× slower.

---

## Summary

| Aspect | Decision | Reason |
|--------|----------|--------|
| **Validation Strategy** | Stratified 5-Fold CV | Robust generalization, publishable, handles imbalance |
| **Class Imbalance** | WeightedSampler + Focal Loss | Proven effective for severe imbalance (21:1) |
| **Metrics** | Accuracy, F1 (macro/weighted), QWK, ROC-AUC | Comprehensive evaluation, QWK critical for ord class |
| **Model** | Pretrained EfficientNet-B4 | SOTA accuracy-efficiency tradeoff |
| **Optimizer** | AdamW | Robust, handles pretrained weights well |
| **LR Scheduler** | CosineAnnealing | Smooth decay, helps escape local minima |
| **Early Stopping** | Patience=15 | Prevents overfitting |
| **Test Set** | Never touched during training | Proper protocol, final leaderboard only |

---

## References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
2. **Stratified K-Fold**: Scikit-learn documentation
3. **Quadratic Weighted Kappa**: Cohen (1968), extended for ordinal data
4. **APTOS Dataset**: Kaggle competition, https://www.kaggle.com/c/aptos2019-blindness-detection
5. **Medical Imaging Best Practices**: MICCAI conference guidelines

---

**Last Updated**: February 26, 2026  
**Author**: DR Classification Research Group  
**Status**: Production Ready
