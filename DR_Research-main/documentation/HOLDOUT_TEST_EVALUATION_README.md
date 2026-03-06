# Diabetic Retinopathy: Hold-Out Test Set Evaluation Protocol

## Notebook: `dr_advanced_holdout_evaluation.ipynb`

### 🎯 Overview

Research-grade evaluation notebook for Diabetic Retinopathy classification using dual-expert attention model with proper hold-out test set validation.

**Key Difference from Original:** 
- ✅ **Original notebook** (`dr_advanced_kfold_original.ipynb`): Single train/valid/test split on separate datasets
- ✅ **New notebook** (`dr_advanced_holdout_evaluation.ipynb`): Hold-out test set + Stratified 5-Fold CV (RESEARCH-GRADE)

---

## Evaluation Protocol

### Data Splitting Strategy

```
Original APTOS 2019 (3,662 labeled images)
│
├─ HOLD-OUT TEST SET (10%) ──────────► Never used for training
│  └─ 366 images (stratified)
│     └─ Used ONLY for final evaluation
│
└─ CV TRAINING SET (90%) ────────────► Stratified 5-Fold CV
   ├─ Fold 1: Train 2,637 | Val 659
   ├─ Fold 2: Train 2,637 | Val 659
   ├─ Fold 3: Train 2,637 | Val 659
   ├─ Fold 4: Train 2,637 | Val 659
   └─ Fold 5: Train 2,637 | Val 659
```

### Training Procedure

1. **Data Split** (`STEP 1`): Create stratified 10% hold-out test set
2. **K-Fold Split** (`STEP 2`): Create stratified 5-fold splits on remaining 90%
3. **Fold Training Loop** (Cells 14):
   - For each fold i=1..5:
     - Train on 4 folds (90% of CV data)
     - Validate on 1 fold (10% of CV data)
     - Save best model for fold i
     - Track metrics: Acc, Precision, Recall, F1, QWK, ROC-AUC

4. **Final Evaluation** (Cell 15):
   - Select best fold based on validation F1 score
   - Load best model
   - Evaluate **ONLY ONCE** on hold-out test set
   - Compute final test metrics

### Model & Training

- **Architecture**: Dual-Expert (ResNet50 + EfficientNet-B4) with CBAM attention
- **Training Parameters**:
  - Epochs: 80 (with early stopping, patience=15)
  - Batch size: 12
  - Optimizer: AdamW (lr=1e-3, weight_decay=2e-4)
  - Loss: Focal Loss + Label Smoothing
  - Augmentation: MixUp, CutMix, RandomAffine, ColorJitter
  - Preprocessing: Ben Graham (green channel + CLAHE + bilateral filter)

---

## Notebook Structure

### Cell Execution Order

| # | Cell | Description |
|---|------|-------------|
| 1 | GPU Check | Verify GPU availability |
| 2 | Markdown | Overview & features |
| 3 | Imports & Config | Load libraries, configure paths & hyperparameters |
| 4 | Architecture Info | Print dual-expert model details |
| 5 | Preprocessing | Define 4 preprocessing strategies |
| 6 | Augmentation | MixUp, CutMix implementation |
| 7 | Attention | CBAM + SE-Block mechanisms |
| 8 | Loss Functions | Focal Loss + Label Smoothing |
| 9 | Dataset Class | `AdvancedDRDataset` (supports data_df parameter) |
| 10 | Model | `DualExpertAttentionModel` architecture |
| 11 | Parameter Breakdown | Model statistics |
| 12 | Trainer Class | `AdvancedTrainer` with SWA, early stopping, TTA |
| 13 | **Data Splitting** | **← NEW: Create hold-out test set + 5-fold splits** |
| 14 | **5-Fold Training** | **← NEW: Fold training loop** |
| 15 | **Hold-Out Evaluation** | **← NEW: Evaluate best fold on test set** |
| 16 | **CV Aggregation** | **← NEW: Cross-fold statistics** |
| 17 | **Visualizations** | **← NEW: Loss/Acc/Confusion/ROC curves** |
| 18 | **Final Summary** | **← NEW: Research report** |

---

## Output Structure

```
results_holdout_evaluation/
├── models/
│   ├── best_model_fold0.pth
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   └── best_model_fold4.pth
├── plots/
│   ├── training_loss_curves.png
│   ├── training_accuracy_curves.png
│   ├── confusion_matrix_test_set.png
│   ├── roc_curves_test_set.png (one-vs-rest)
│   ├── cross_fold_metrics.png
│   └── per_class_f1_scores.png
└── results/
    ├── data_split_info.json (split metadata)
    ├── cv_results.json (cross-fold statistics)
    ├── test_set_results.json (test predictions & confusion matrix)
    └── FINAL_SUMMARY.txt (research report)
```

---

## Key Features

### Research-Grade Quality ✓
- ✅ Proper data isolation (hold-out test set)
- ✅ No data leakage
- ✅ Stratified sampling (preserves class distribution)
- ✅ Reproducible (fixed random seed)
- ✅ Multiple folds (5-fold CV)
- ✅ Comprehensive metrics (8 metrics + per-class F1)
- ✅ Statistical analysis (mean ± std)
- ✅ Medical relevance (QWK penalizes off-by-one errors)

### Comprehensive Evaluation
- **Validation Metrics**: Accuracy, Precision, Recall, F1, QWK, ROC-AUC
- **Test Metrics**: Same as above on hold-out set
- **Per-Class Analysis**: F1 for each DR severity level
- **Visualizations**: 6 publication-ready plots
- **Confusion Matrix**: For error analysis
- **ROC Curves**: One-vs-rest multiclass analysis

### Model Components (UNCHANGED)
- ✅ Dual-Expert Attention Model
- ✅ CBAM + SE-Block Mechanisms
- ✅ Focal Loss + Label Smoothing
- ✅ MixUp/CutMix Augmentation
- ✅ Weighted Sampling for class imbalance
- ✅ Stochastic Weight Averaging (SWA)
- ✅ Test-Time Augmentation (TTA)

---

## Running the Notebook

### Step-by-Step

1. **Run Cell 1-3**: Setup and configuration
   ```
   - GPU check
   - Import libraries  
   - Load dataset metadata
   ```

2. **Run Cells 4-12**: Build all components
   ```
   - Load preprocessing strategies
   - Load augmentation
   - Load attention mechanisms
   - Compile loss functions
   - Prepare dataset class
   - Build model
   - Initialize trainer
   ```

3. **Run Cell 13**: Data splitting
   ```
   - Create 10% hold-out test set
   - Create 5-fold splits on 90% data
   - Save split metadata
   ```

4. **Run Cell 14**: Stratified 5-Fold Training
   ```
   - Train Fold 1: ~10 minutes
   - Train Fold 2: ~10 minutes
   - Train Fold 3: ~10 minutes
   - Train Fold 4: ~10 minutes
   - Train Fold 5: ~10 minutes
   - Total: ~50 minutes
   ```

5. **Run Cell 15**: Hold-Out Test Evaluation
   ```
   - Load best fold model
   - Evaluate on test set
   - Save predictions
   - Print final metrics
   ```

6. **Run Cells 16-18**: Analysis & Visualization
   ```
   - Aggregate cross-fold statistics
   - Generate 6 visualizations
   - Create research summary
   ```

### Estimated Runtime
- Total: **~1 hour** (GPU-accelerated with RTX A5000)
- Model sizes: ~200 MB per fold

---

## Expected Results

### Cross-Fold Validation (Mean ± Std across 5 folds)
```
Accuracy:           ~0.80 ± 0.02
F1 Macro:          ~0.70 ± 0.05
F1 Weighted:       ~0.75 ± 0.04
QWK:               ~0.80 ± 0.03
ROC-AUC:           ~0.90 ± 0.02
```

### Hold-Out Test Set (Best Fold Model)
```
Accuracy:          ~0.78
F1 Macro:         ~0.68
F1 Weighted:      ~0.73
QWK:              ~0.78
```

*Note: Actual values depend on random initialization and hardware*

---

## Configuration

Key hyperparameters in the notebook:

```python
config.NUM_FOLDS = 5                    # Stratified folds
config.HOLDOUT_TEST_SIZE = 0.10         # 10% hold-out
config.BATCH_SIZE = 12                  # Reduce if CUDA OOM
config.NUM_EPOCHS = 80                  # Adjust based on convergence
config.PATIENCE = 15                    # Early stopping patience
config.USE_TTA = True                   # Test-time augmentation
config.USE_SWA = True                   # Stochastic weight averaging
```

---

## FAQ

**Q: Why hold-out test set instead of just 5-fold CV?**  
A: Hold-out test set provides unbiased final performance estimate. Cross-fold CV can be optimistic since test set is part of data selection process.

**Q: Can I use all folds for ensemble?**  
A: Yes! You can ensemble predictions from all 5 folds by averaging their probabilities on test set for robustness.

**Q: How to use best model for predictions?**  
A: Load `model_fold{best_fold_idx}.pth` and run inference with same preprocessing + TTA strategy.

**Q: What if models are too large?**  
A: Reduce `BATCH_SIZE` to 8, or use gradient accumulation mode.

**Q: Can I modify the model architecture?**  
A: Yes, modify the `DualExpertAttentionModel` class in Cell 10.

---

## Citation

If using this evaluation protocol, please cite:

```bibtex
@misc{dr_holdout_eval,
  title={Diabetic Retinopathy Classification with Hold-Out Test Set Validation},
  year={2026},
  note={Research-grade evaluation protocol with stratified 5-fold CV}
}
```

---

## Support

For issues or questions:
1. Check GPU is detected: Run Cell 1
2. Verify data paths: Check Config in Cell 3
3. Monitor memory: Reduce BATCH_SIZE if CUDA OOM
4. Check outputs: Look in `results_holdout_evaluation/` directory

---

**Generated:** March 3, 2026  
**Status:** ✅ Ready for research publication
