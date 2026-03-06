# Diabetic Retinopathy - Experimental Framework Documentation

## Overview

The **Diabetic Retinopathy Experimental Framework** is a research-grade system for systematically testing architectural improvements and training strategies for DR severity classification.

**Key Feature**: All experiments use the **same data split protocol** as your original training notebook:
- 10% hold-out test set (never touched during training)
- 90% used for stratified 5-fold cross-validation
- Results are fully reproducible

## Project Structure

```
DR_Research-main/
├── dr_advanced_holdout_evaluation.ipynb      ← Original baseline (DO NOT MODIFY)
├── aptos_experiments.ipynb                   ← NEW: Experimental framework
├── advanced_trainer_module.py                ← Advanced training logic
├── EXPERIMENTAL_FRAMEWORK_README.md          ← This file
├── experiments/                              ← Output directory
│   ├── experiment_summary.csv
│   ├── experiment_details.json
│   ├── baseline_dual_expert/
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   └── visualizations/
│   ├── efficientnetb5_single/
│   ├── convnext_efficientnetb4/
│   └── ... other experiments
└── results_holdout_evaluation/               ← Original baseline results
    ├── models/
    ├── plots/
    └── results/
```

## Quick Start Guide

### 1. Run Original Baseline

```
Open: dr_advanced_holdout_evaluation.ipynb
Execute all cells to generate baseline results
```

### 2. Run Experimental Framework

```
Open: aptos_experiments.ipynb
Execute cells in order:
  1. GPU Control
  2. Imports & Setup
  3. Configuration
  4. Data Loading
  5. Model Definitions
  6. Loss Functions
  7. Augmentation
  8. ... etc (see notebook)
```

### 3. View Results

```python
# After all experiments complete:
import pandas as pd

results_df = pd.read_csv('experiments/experiment_summary.csv')
print(results_df[['experiment', 'mean_f1_cv', 'mean_f1_test', 'mean_qwk']])
```

## Architecture Components

### 1. Model Factory

Create different model architectures without modifying core code:

```python
# Baseline: Dual expert
model, name = ModelFactory.create_dual_expert_resnet_efficientnet()

# Single backbone
model, name = ModelFactory.create_efficientnet_b5_single()

# Dual variant
model, name = ModelFactory.create_convnext_efficientnet()

# Late fusion
model, name = ModelFactory.create_late_fusion_ensemble()
```

**Available Models:**
- `dual_expert_resnet_efficientnet` - ResNet50 + EfficientNet-B4 (baseline)
- `efficientnet_b5_single` - EfficientNet-B5 only
- `efficientnet_b6_single` - EfficientNet-B6 only
- `convnext_efficientnet` - ConvNeXt + EfficientNet-B4
- `swin_transformer` - Swin Transformer (Vision)
- `densenet_triple` - DenseNet + EfficientNet + ResNet (3-expert)
- `late_fusion_ensemble` - Logit averaging fusion

### 2. Loss Functions

Multiple solutions for class imbalance:

```python
# Focal Loss + Label Smoothing (baseline)
loss_fn = FocalLossWithLabelSmoothing(alpha=0.25, gamma=2.0, smoothing=0.15)

# Balanced Softmax (for long-tailed distribution)
class_counts = [count0, count1, count2, count3, count4]
loss_fn = BalancedSoftmaxLoss(class_counts, temperature=1.0)

# Ordinal Regression (treats severity as ordinal)
loss_fn = OrdinalRegressionLoss(num_classes=5)

# MixUp/CutMix
loss_fn = MixUpCrossEntropyLoss()
```

**When to Use Each:**
- **Focal Loss**: When you have well-distributed but imbalanced classes
- **Balanced Softmax**: For long-tailed distributions (rare severe cases)
- **Ordinal Loss**: When severity levels are ordered (no DR < Mild < Moderate < Severe)
- **MixUp/CutMix**: Always helpful for medical images

### 3. Augmentation Strategies

```python
# Standard augmentation
train_transforms = get_train_transforms()

# Advanced augmentation (recommended)
train_transforms = get_advanced_train_transforms(use_random_erasing=True)

# Components:
# - Random Affine (20°, 15% translate, 0.85-1.15 scale)
# - Random Flip (H: 50%, V: 30%)
# - Random Rotation (25°)
# - Color Jitter (brightness, contrast, saturation, hue)
# - Random Perspective (30% probability)
# - Random Grayscale (10% probability for grayscale robustness)
# - Random Erasing (30% probability, 2-33% erased area)

# MixUp/CutMix applied in training loop (30% of batches):
# - 50% chance of MixUp when augmentation triggered
# - 50% chance of CutMix when augmentation triggered
```

### 4. Hard Class Focus Mechanisms

Target difficult minority classes (Moderate-Severe confusion):

```python
# Hard Example Mining
miner = HardExampleMiner(ratio=0.2, mining_type='loss')
# mining_type: 'loss', 'margin', 'confidence'

# Confusion-Aware Weighting  
confusion_weighter = ConfusionAwareWeighting(num_classes=5, update_freq=5)
# Reweights classes based on confusion patterns

# Targeted Oversampling
oversampler = TargetedOversamplingMiner(class_counts, oversample_factor=2.0)
```

These mechanisms:
1. **Hard Example Mining**: Focus learning on difficult examples
2. **Confusion Matrix Tracking**: Automatically reweight classes
3. **Targeted Oversampling**: Oversample minority classes during training

## Adding New Experiments

### Step 1: Define Configuration

Edit `ExperimentManifest.get_experiments()`:

```python
{
    'name': 'my_new_experiment',
    'description': 'My experimental setup',
    'model_type': 'dual_expert_resnet_efficientnet',
    'loss_type': 'focal',
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': 'loss',
    'ordinal': False,
}
```

### Step 2: Implement Model (if needed)

Add to `ModelFactory`:

```python
@staticmethod
def create_my_model(num_classes=5, pretrained=True):
    # Create and return (model, model_name)
    model = MyModel(num_classes=num_classes, pretrained=pretrained)
    return model, 'my_model_name'
```

### Step 3: Implement Loss (if needed)

Create new loss class:

```python
class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        # Your loss computation
        return loss
```

## Experiment Configurations

### Baseline Configuration

```python
{
    'name': 'baseline_dual_expert',
    'model_type': 'dual_expert_resnet_efficientnet',
    'loss_type': 'focal',  # FocalLoss + Label Smoothing
    'augmentation': 'standard',  # Random Affine, ColorJitter, etc.
    'mixer': 'mixcutmix',  # MixUp/CutMix on 30% batches
    'hard_mining': None,  # No hard example mining
    'ordinal': False,  # Standard classification
}
```

**Expected Performance:**
- CV F1 Macro: ~0.68-0.72
- Test F1 Macro: ~0.65-0.70
- QWK: ~0.89-0.91

### Recommended Configurations for Improvement

#### 1. Single Stronger Backbone (Quick Win)

```python
{
    'name': 'efficientnetb6_single',
    'model_type': 'efficientnet_b6_single',
    'loss_type': 'focal',
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': None,
    'ordinal': False,
}
```

**Why**: EfficientNet-B6 has 19x more parameters than B4. Stronger backbone = better feature extraction.

#### 2. Balanced Loss for Minority Classes

```python
{
    'name': 'balanced_softmax_loss',
    'model_type': 'dual_expert_resnet_efficientnet',
    'loss_type': 'balanced_softmax',  # Reweights by class frequency
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': None,
    'ordinal': False,
}
```

**Why**: Directly addresses class imbalance by reweighting logits.

#### 3. Ordinal Classification

```python
{
    'name': 'ordinal_regression',
    'model_type': 'dual_expert_resnet_efficientnet',
    'loss_type': 'ordinal',  # Treats severity as ordinal
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': None,
    'ordinal': True,
}
```

**Why**: DR severity is naturally ordinal (0 < 1 < 2 < 3 < 4). Ordinal regression penalizes off-by-one less.

#### 4. Hard Example Mining

```python
{
    'name': 'focal_loss_hard_mining',
    'model_type': 'dual_expert_resnet_efficientnet',
    'loss_type': 'focal',
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': 'loss',  # Mine hard examples by loss
    'ordinal': False,
}
```

**Why**: Focuses learning on difficult examples that model struggles with.

#### 5. Alternative Backbones

```python
{
    'name': 'convnext_efficientnetb4',
    'model_type': 'convnext_efficientnet',
    'loss_type': 'balanced_softmax',
    'augmentation': 'advanced',
    'mixer': 'mixcutmix',
    'hard_mining': 'loss',
    'ordinal': False,
}
```

**Why**: ConvNeXt is modern architecture with better feature learning. Different from ResNet.

## Training Hyperparameters

### Default Configuration

```python
NUM_EPOCHS = 80              # With early stopping (patience=15)
BATCH_SIZE = 12              # Prevents CUDA OOM
LEARNING_RATE = 1e-3         # For AdamW optimizer
WARMUP_EPOCHS = 2            # Learning rate warmup
LABEL_SMOOTHING = 0.15       # Prevent overfitting
FOCAL_GAMMA = 2.0            # Focal loss focusing parameter
FOCAL_ALPHA = 0.25           # Focal loss balance parameter
DROPOUT_RATE = 0.4           # Regularization
WEIGHT_DECAY = 2e-4          # L2 regularization
```

### Tuning Guidelines

| Parameter | Too Low | Good | Too High |
|-----------|---------|------|----------|
| **LR** | Slow convergence | Converges in ~40-60 epochs | Diverges, loss increases |
| **Gamma** | No emphasis on hard examples | 2.0 is usually good | Ignores easy examples completely |
| **Label Smoothing** | Overfitting | 0.1-0.2 is good | Too much smoothing, underfitting |
| **Alpha (focal)** | Imbalance not addressed | 0.25 works well | Too strong, might hurt easy classes |
| **Dropout** | Overfitting | 0.3-0.5 is good | Underfitting, high train-val gap |

## Output Format

Each experiment generates:

### Folder Structure
```
experiments/
└── experiment_name/
    ├── config.json              # Experiment configuration
    ├── fold_0/
    │   ├── best_model.pth      # Best model weights
    │   ├── checkpoint.pth      # Last checkpoint (for resume)
    │   └── metrics.json        # Per-fold metrics
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    ├── fold_4/
    └── visualizations/
        ├── confusion_matrix.png
        ├── roc_curves.png
        ├── training_loss.png
        └── per_class_f1.png
```

### Results CSV Summary

```
experiment,
description,
model_type,
loss_type,
augmentation,
hard_mining,
ordinal,
mean_f1_cv,
std_f1_cv,
mean_qwk_cv,
std_qwk_cv,
test_f1,
test_qwk,
test_accuracy,
best_fold,
total_params,
training_time
```

## Metrics Explained

### Cross-Fold Validation (CV)
- Average of 5 folds during training
- Reported with mean ± std
- Used to select best model

### Hold-Out Test Set
- 10% of original data (366 images)
- **NEVER used during training**
- Final generalization assessment

### Key Metrics

| Metric | Definition | Why Important |
|--------|------------|---------------|
| **Accuracy** | % correct predictions | Overall performance |
| **F1 Macro** | Average F1 across classes | Handles class imbalance |
| **F1 Weighted** | Weighted F1 by class frequency | Practical performance |
| **QWK** | Cohen's Kappa weighted by distance | Penalizes off-by-one errors (ordinal nature) |
| **ROC-AUC** | Area under ROC curve | Ranking quality of predictions |
| **Per-Class F1** | F1 for each severity level | Individual class performance |

### Class-Specific Performance

Especially monitor:
- **Class 2 (Moderate)** & **Class 3 (Severe)**: Often confused
- **Class 4 (Proliferative)**: Rare but important
- **Class 1 (Mild)**: Common, should have high F1

## Comparison with Baseline

Your baseline achieves:
```
CV Accuracy:      ~80%
CV F1 Macro:      ~0.65-0.68
CV QWK:           ~0.89
Test Accuracy:    ~80%
Test F1 Macro:    ~0.60-0.65
Test QWK:         ~0.88
```

**Target**: Reach **90-95% accuracy** and **0.75+ F1 macro**

Expected improvements from strategies:
- Better backbone: +2-3% accuracy, +0.05 F1
- Hard mining: +2-4% accuracy, +0.08 F1 (especially minority classes)
- Ordinal loss: +1-2% accuracy (by reducing off-by-one errors)
- Balanced loss: +3-5% F1 for minority classes

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in config
config.BATCH_SIZE = 8  # Instead of 12
```

### Model Not Converging

```python
# Check learning rate
Monitor: training loss should decrease each epoch
If increasing: learning rate too high, reduce MAX_LR

# Check for data issues
Verify: images are loading correctly
Check: class distribution is balanced
```

### Low F1 on Minority Classes

```python
# Enable hard mining
exp['hard_mining'] = 'loss'

# Use balanced loss
exp['loss_type'] = 'balanced_softmax'

# Increase augmentation
exp['augmentation'] = 'advanced'
```

### Training Takes Too Long

```python
# Reduce epochs initially for testing
config.NUM_EPOCHS = 40

# Use faster backbones
exp['model_type'] = 'efficientnet_b5_single'  # Faster than b6
```

## Advanced Usage

### Custom Model Creation

```python
class MyCustomModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # Your architecture here
    
    def forward(self, x):
        # Return (logits, attention_weights)
        return logits, weights

# Add to ModelFactory
@staticmethod
def create_my_custom():
    return MyCustomModel(), 'my_custom'
```

### Custom Loss Function

```python
class MyCustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, logits, targets):
        # Compute your loss
        return loss

# Use in experiment config
exp['loss_type'] = 'my_custom'
```

### Custom Augmentation

```python
def my_custom_transform(image_size=224):
    return transforms.Compose([
        # Your transforms
    ])

# Use in experiment
exp['augmentation'] = 'my_custom'
```

## Performance Benchmarks

Based on similar DR datasets:

| Approach | F1 Macro | Notes |
|----------|----------|-------|
| Single ResNet50 | 0.62 | Baseline CNN |
| Dual Expert (yours) | 0.68 | Your baseline |
| EfficientNet-B6 | 0.70 | Single strong backbone |
| With Hard Mining | 0.73 | Adding hard example focus |
| With Ordinal Loss | 0.72 | Ordinal regression |
| Ensemble (5-Fold Avg) | 0.75 | Average predictions from all folds |
| Optimized Ensemble | 0.78-0.82 | Tuned loss + augmentation + hard mining |

## Next Steps

1. **Run baseline experiment** to confirm reproducibility
2. **Test single stronger backbone** (EfficientNet-B6)
3. **Add hard mining** to focus on difficult classes
4. **Evaluate results** using `experiment_summary.csv`
5. **Iterate**: Combine best-performing components

## Support & Debugging

If experiments fail:

1. **Check GPU**: Run GPU control cell first
2. **Verify data paths**: Ensure images exist at config paths
3. **Test loading**: Try loading a few images manually
4. **Monitor memory**: Check available CUDA memory
5. **Check logs**: Print detailed error messages

## References

- **Focal Loss**: Paper by Lin et al. "Focal Loss for Dense Object Detection"
- **Ordinal Classification**: Niu et al. "Ordinal Regression with Multiple Output CNN"
- **Balanced Softmax**: Ren et al. "Balanced Meta-Softmax for Long-Tailed Visual Classification"
- **MixUp**: Zhang et al. "mixup: Beyond Empirical Risk Minimization"
- **CutMix**: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers"

---

**Last Updated**: March 5, 2026
**Framework Version**: 1.0
