# APTOS 2019 DR Classification - Configuration & Best Practices

## Quick Configuration Reference

### Recommended Settings for APTOS 2019

```python
# Dataset
BASE_DIR = r'd:\Ece_DR\APTOS2019'
NUM_CLASSES = 5
IMAGE_SIZE = 224

# Model
MODEL_BACKBONE = 'efficientnet_b4'  # or 'resnet50', 'resnet101', 'efficientnet_b5'
PRETRAINED = True
DROPOUT = 0.3

# Training
BATCH_SIZE = 32                 # Adjust based on GPU memory
NUM_EPOCHS = 80                 # With early stopping
NUM_FOLDS = 5                   # or 10 for more robust estimates
PATIENCE = 15                   # Early stopping patience

# Learning Rate
MAX_LR = 1e-3                   # Peak learning rate
MIN_LR = 1e-6                   # Minimum learning rate
WARMUP_EPOCHS = 2               # Linear warmup

# Optimizer
OPTIMIZER = 'AdamW'
WEIGHT_DECAY = 2e-4
GRADIENT_CLIP = 1.0

# Loss Function
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25              # Balance parameter
FOCAL_GAMMA = 2.0               # Focusing parameter
LABEL_SMOOTHING = 0.1           # Optional

# Class Imbalance
USE_WEIGHTED_SAMPLER = True     # Oversample minority classes

# Data Augmentation
AUGMENTATION = True
AUGMENT_PROBABILITY = 0.5       # Apply aug to 50% of samples

# Reproducibility
SEED = 42
```

---

## Backbone Selection Guide

### EfficientNet-B4 (RECOMMENDED for APTOS)
```
Parameters:     ~19M
Inference Time: ~39ms
Accuracy:       Top-1: 82.9% (ImageNet)
✓ Good balance of accuracy and speed
✓ Smaller than ResNet-101
✓ Pretrained weights available in timm
✓ SOTA for medical imaging (2018-2020)
```

### EfficientNet-B5
```
Parameters:     ~30M
Inference Time: ~58ms
Accuracy:       Top-1: 83.6% (ImageNet)
✓ Slightly better accuracy
✗ Slower and larger
- Use only if more GPU memory available
```

### ResNet-50
```
Parameters:     ~26M
Inference Time: ~45ms
Accuracy:       Top-1: 76.1% (ImageNet)
✓ Simple, well-understood
✗ Lower accuracy than EfficientNet
- Use if you need maximum compatibility
```

### ResNet-101
```
Parameters:     ~45M
Inference Time: ~75ms
Accuracy:       Top-1: 77.4% (ImageNet)
✓ Better than ResNet-50
✗ Much slower
- Use only if you have compute resources
```

**RECOMMENDATION**: **EfficientNet-B4** (best accuracy/speed/size tradeoff)

---

## Hyperparameter Tuning Grid

### If Validation Performance is Poor:

#### Model
```
└─ Try deeper backbone
   ├─ EfficientNet-B5 (slightly better)
   └─ ResNet-101 (significantly heavier)

└─ Increase dropout to 0.4-0.5 (reduce overfitting)
```

#### Learning Rate
```
If loss is unstable:         Use warmup_epochs=3-5, MAX_LR=5e-4
If loss is decreasing slowly: Increase MAX_LR to 2e-3
If loss diverges:            Decrease MAX_LR to 5e-4
```

#### Batch Size
```
Too small (< 16):   Noisy gradients, unstable training
Good range (16-64): Most stable
Too large (> 128):  Less robust, needs higher LR
```

#### Class Imbalance
```
If minority classes (3,4) have low F1:
├─ Increase FOCAL_GAMMA to 3.0 (more focus on hard examples)
├─ Adjust FOCAL_ALPHA to 0.35-0.5
└─ Ensure WeightedRandomSampler is enabled
```

#### Augmentation
```
If overfitting (train_loss << val_loss):
├─ Increase augmentation probability
└─ Add more aggressive augmentations

If underfitting (train_loss ≈ val_loss, both high):
├─ Reduce augmentation severity
└─ Implement simpler augmentations
```

---

## Data Augmentation Strategies

### Strategy 1: Conservative (Current Implementation)
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```
**Use when**: Data is limited, model is simple

### Strategy 2: Aggressive (Medical Imaging)
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), 
                           scale=(0.85, 1.15), shear=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                          saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])
```
**Use when**: Model capacity is high, data is sufficient

### Strategy 3: Domain-Specific (Retinal Imaging)
```python
# Preserve optic disc / vessel structure
transforms.Compose([
    transforms.RandomRotation(degrees=360),  # Rotation-invariant
    transforms.RandomHorizontalFlip(p=0.5),  # Preserves structure
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Lighting variations
    # NO extreme affine - retinal structure is important
])
```
**Use when**: You want to preserve medical features

---

## GPU Memory Guidelines

### EfficientNet-B4, Image Size 224×224

| Batch Size | GPU Memory | FP32 Model | Recommended |
|------------|-----------|------------|-------------|
| 8          | ~4GB      | All GPUs   | ✓ Safe |
| 16         | ~6GB      | RTX 2070+  | ✓ Recommended |
| 32         | ~10GB     | RTX 2080+  | ✓ Good |
| 64         | ~18GB     | RTX 3090   | ✓ For large batch |
| 128        | ~35GB     | A100       | Not needed |

**For typical NVIDIA RTX 3080 (10GB)**: Use batch_size=32 safely

---

## Training Time Estimates

### On NVIDIA RTX 3080 (10GB)

| Model | Batch=16 | Batch=32 | Batch=64 |
|-------|----------|----------|----------|
| ResNet-50 | ~20s/epoch | ~12s/epoch | ~8s/epoch |
| EfficientNet-B4 | ~25s/epoch | ~15s/epoch | ~10s/epoch |
| ResNet-101 | ~35s/epoch | ~20s/epoch | ~15s/epoch |

**For 5-Fold, 80 epochs**:
- EfficientNet-B4, batch=32: ~80 epochs × 5 folds × 15s/epoch ≈ 100 min ≈ 1.7 hours
- ResNet-50, batch=32: ~80 epochs × 5 folds × 12s/epoch ≈ 80 min ≈ 1.3 hours

---

## Validation Metric Interpretation

### Classification Metrics

```
Accuracy = (TP + TN) / Total
├─ What: % of correct predictions
├─ Range: 0-1 (higher is better)
└─ For APTOS: 0.80-0.85 is good

Precision = TP / (TP + FP)
├─ What: Of predicted as Class-i, how many are correct?
├─ Range: 0-1 (higher is better)
└─ For APTOS: Focus on minority classes

Recall = TP / (TP + FN)
├─ What: Of actual Class-i, how many did we find?
├─ Range: 0-1 (higher is better)
└─ For APTOS: Recall for disease classes (3,4) is critical

F1 = 2 × (Precision × Recall) / (Precision + Recall)
├─ What: Harmonic mean of precision & recall
├─ Range: 0-1 (higher is better)
└─ For APTOS: 0.75-0.85 is good
```

### Domain-Specific Metrics

```
QWK (Quadratic Weighted Kappa)
├─ What: Agreement accounting for class hierarchy
├─ Range: -1 to +1
├─ Interpretation:
│  ├─ < 0.40: Poor
│  ├─ 0.40-0.60: Fair
│  ├─ 0.60-0.75: Good ✓
│  ├─ 0.75-0.90: Very Good ✓✓
│  └─ > 0.90: Excellent ✓✓✓
└─ For APTOS: 0.70-0.80 is very good

ROC-AUC (One-vs-Rest)
├─ What: Discrimination ability at all thresholds
├─ Range: 0-1 (higher is better)
├─ Interpretation:
│  ├─ 0.50-0.70: Fair
│  ├─ 0.70-0.80: Good ✓
│  ├─ 0.80-0.90: Excellent ✓✓
│  └─ > 0.90: Outstanding ✓✓✓
└─ For APTOS: 0.80-0.85 is excellent
```

---

## Common Issues & Solutions

### Issue 1: Model Only Predicts Class 0
```
Symptom: All predictions are class 0, accuracy ~49%
Causes:
  ✗ WeightedRandomSampler disabled
  ✗ Focal Loss not used
  ✗ Batch size too small (< 16)
  ✗ Number of training steps too low

Solutions:
  ✓ Enable WeightedRandomSampler
  ✓ Use Focal Loss (alpha=0.25, gamma=2.0)
  ✓ Increase batch size to 32+
  ✓ Increase num_epochs to 80+
```

### Issue 2: Minority Classes Have 0 samples in Validation
```
Symptom: Class 3 or 4 missing from val_fold
         Cannot compute F1 for that class

Causes:
  ✗ Stratified split failed
  ✗ num_folds too large (>5) with small minority classes

Solutions:
  ✓ Ensure StratifiedKFold preserves distribution
  ✓ Use num_folds = 5 (not 10)
  ✓ Check fold_assignments.json for class distribution
  ✓ If fold missing class, resample with random_state+1
```

### Issue 3: Validation F1 drops suddenly
```
Symptom: F1 was 0.80, suddenly drops to 0.60
         Loss is still decreasing

Causes:
  ✗ Learning rate too high
  ✗ No gradient clipping
  ✗ Batch normalization issues

Solutions:
  ✓ Check GRADIENT_CLIP = 1.0 (default is good)
  ✓ Reduce MAX_LR from 1e-3 to 5e-4
  ✓ Use warmup_epochs = 3-5
```

### Issue 4: Training is very slow
```
Symptom: 1 epoch takes > 1 minute

Causes:
  ✗ num_workers too high (default 0 is OK for Windows)
  ✗ Image loading from slow drive
  ✗ Image preprocessing is expensive

Solutions:
  ✓ Keep num_workers = 0 (Windows compatible)
  ✓ Use SSD for faster I/O
  ✓ Increase batch_size to 32-64 (more images per batch)
  ✓ Profile code to find bottleneck
```

---

## Cross-Fold Variance Analysis

### Good Variance (Stable Model)
```
Accuracy:       0.850 ± 0.008  (std < 1%)
F1 Macro:       0.820 ± 0.012
QWK:            0.760 ± 0.015

Interpretation: Model generalizes consistently
                Low variance across folds = robust
                Std < 2% indicates good stability
```

### High Variance (Unstable Model)
```
Accuracy:       0.850 ± 0.045  (std > 5%)
F1 Macro:       0.820 ± 0.068
QWK:            0.760 ± 0.085

Interpretation: Model's performance varies by fold
                High variance = overfitting or poor regularization
                Std > 5% indicates instability

Actions:
  ✓ Increase dropout (0.4-0.5)
  ✓ Add L2 regularization (weight_decay > 2e-4)
  ✓ Increase augmentation strength
  ✓ Reduce learning rate
```

---

## Reproducibility Checklist

- [ ] Set `SEED = 42` (or any fixed number)
- [ ] Set `torch.backends.cudnn.deterministic = True`
- [ ] Set `torch.backends.cudnn.benchmark = False`
- [ ] Save `fold_assignments.json`
- [ ] Document exact PyTorch version
- [ ] Document exact CUDA version
- [ ] Document GPU model (RTX 3080, A100, etc.)
- [ ] Save final `final_summary.json`
- [ ] Save all `best_model_fold_*.pth`
- [ ] Version control the notebook
- [ ] Document any custom modifications

---

## Paper Writing Checklist

### Methods Section
- [ ] Describe stratified K-Fold setup
- [ ] State random seed (42)
- [ ] Describe fold preservation strategy
- [ ] Explain WeightedRandomSampler rationale
- [ ] Describe Focal Loss parameters
- [ ] List all hyperparameters

### Results Section
- [ ] Table with Mean ± Std across folds
- [ ] Per-class F1 scores
- [ ] Fold-wise stability values
- [ ] QWK for ordinal evaluation
- [ ] ROC-AUC curves

### Supplementary Material
- [ ] Per-fold metric breakdown
- [ ] Fold assignments file
- [ ] Confusion matrices per fold
- [ ] ROC curves per fold
- [ ] Best model checkpoints

---

## Final Summary Table

| Aspect | Recommendation | Justification |
|--------|---|---|
| Validation Strategy | Stratified 5-Fold CV | Robust, publishable |
| Backbone | EfficientNet-B4 | Best accuracy/speed tradeoff |
| Batch Size | 32 | Balance between stability and memory |
| Learning Rate | 1e-3 | Pretrained model fine-tuning |
| Optimizer | AdamW | Handles pretrained parameters well |
| LR Scheduler | CosineAnnealingLR | Smooth decay, escape local minima |
| Loss Function | Focal Loss | Handle 21:1 class imbalance |
| Sampling | WeightedRandomSampler | Balanced class distribution per epoch |
| Epochs | 80 (with early stopping) | Sufficient for convergence |
| Dropout | 0.3 | Regularization without being excessive |
| Image Size | 224×224 | Standard for medical imaging |
| Seed | 42 | Reproducibility |

---

**Created**: February 26, 2026  
**Status**: Production-Ready  
**Support**: Verified on Windows 10/11, NVIDIA RTX 3080
