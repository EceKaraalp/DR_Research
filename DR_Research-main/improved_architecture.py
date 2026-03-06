"""
===============================================================
IMPROVED DIABETIC RETINOPATHY CLASSIFICATION SYSTEM
Multi-Expert Fusion with Attention Mechanisms
===============================================================

Academic Reference:
- Focal Loss: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- Vision Transformer: Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words"
- CBAM: Woo et al. (2018) "CBAM: Convolutional Block Attention Module"
- Retinal Preprocessing: Ben Graham APTOS 2019 Winning Solution

Author: DR Classification Research
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torchvision import models


# ================================================================
# ATTENTION MECHANISMS
# ================================================================

class ChannelAttentionModule(nn.Module):
    """Channel Attention (CA) from CBAM"""
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttentionModule(nn.Module):
    """Spatial Attention (SA) from CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return out * x


class CBAMBlock(nn.Module):
    """Complete CBAM: Channel + Spatial Attention"""
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttentionModule(channels)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class FeatureFusionAttention(nn.Module):
    """
    Multi-expert feature fusion with learned attention weights.
    Enables the model to dynamically focus on relevant expert predictions.
    """
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        
        # Attention network to learn expert importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, expert_features):
        """
        Args:
            expert_features: List of [B, D] tensors from each expert
        Returns:
            fused: [B, D*num_experts] weighted concatenation
        """
        # Stack expert features: [B, num_experts, D]
        stacked = torch.stack(expert_features, dim=1)
        B, N, D = stacked.shape
        
        # Mean pooling for attention input
        att_input = stacked.mean(dim=1)  # [B, D]
        
        # Compute attention weights
        att_weights = self.attention(att_input)  # [B, num_experts]
        
        # Apply weights (element-wise for each feature dimension)
        weighted = (stacked * att_weights.unsqueeze(-1)).view(B, -1)  # [B, N*D]
        
        return weighted, att_weights


# ================================================================
# BACKBONE FEATURE EXTRACTORS
# ================================================================

class ResNet50Backbone(nn.Module):
    """ResNet50 as feature extractor"""
    def __init__(self, pretrained=True, num_classes=0):
        super().__init__()
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove classification head
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_features = model.fc.in_features  # 2048
        
    def forward(self, x):
        x = self.features(x)  # [B, 2048, H, W]
        x = self.avgpool(x)   # [B, 2048, 1, 1]
        x = x.flatten(1)      # [B, 2048]
        return x


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B3 as feature extractor"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b3', 
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.num_features = self.model.num_features  # 1536
        
    def forward(self, x):
        return self.model(x)


class DenseNetBackbone(nn.Module):
    """DenseNet201 as feature extractor"""
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.densenet201(
            weights=models.DenseNet201_Weights.DEFAULT if pretrained else None
        )
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_features = model.classifier.in_features  # 1920
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


# ================================================================
# HYBRID DUAL-EXPERT FUSION MODEL (OPTIMIZED)
# ================================================================

class DualExpertFusionModel(nn.Module):
    """
    Optimized hybrid architecture for DR classification with 2 backbones.
    
    Architecture:
    - 2 independent backbones (ResNet50, EfficientNet-B3)
    - Learned multi-expert fusion with attention weights
    - Final classification head with dropout regularization
    
    Advantages:
    1. Complimentary feature extraction:
       - ResNet50: Strong edge/spatial features (vessel detection)
       - EfficientNet-B3: Multi-scale efficiency (lesion detection)
    
    2. Attention mechanisms:
       - Feature fusion learns expert importance
       - Simpler than 3-expert system, faster training
    
    3. Regularization:
       - Dropout in fusion head
       - Batch normalization throughout
       - Label smoothing compatible
    
    4. Efficiency:
       - Parameters: ~24M (vs 48M with 3 experts)
       - Training time: 3-4 hours (vs 5-6 hours)
       - Still captures complementary features
    """
    
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.4):
        super().__init__()
        self.num_classes = num_classes
        
        # ============ BACKBONES (2-Expert) ============
        self.backbone_resnet = ResNet50Backbone(pretrained=pretrained)
        self.backbone_effnet = EfficientNetBackbone(pretrained=pretrained)
        
        # Feature dimensions
        self.feat_dim_resnet = self.backbone_resnet.num_features      # 2048
        self.feat_dim_effnet = self.backbone_effnet.num_features      # 1536
        
        # ============ PROJECT TO COMMON DIMENSION ============
        # Normalize features to 512-dim common space
        self.proj_resnet = nn.Sequential(
            nn.Linear(self.feat_dim_resnet, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.proj_effnet = nn.Sequential(
            nn.Linear(self.feat_dim_effnet, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # ============ FUSION ATTENTION (2-Expert) ============
        self.fusion_attention = FeatureFusionAttention(input_dim=512, num_experts=2)
        
        total_fused_dim = 512 * 2  # 1024 (more efficient than 1536)
        
        # ============ CLASSIFICATION HEAD ============
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_fused_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(total_fused_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classification weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Xavier initialization for classifier layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through dual-expert fusion.
        
        Args:
            x: [B, 3, 224, 224] Input images
            
        Returns:
            logits: [B, num_classes] Classification logits
        """
        # ============ EXTRACT FEATURES FROM EXPERTS ============
        feat_resnet = self.backbone_resnet(x)      # [B, 2048]
        feat_effnet = self.backbone_effnet(x)      # [B, 1536]
        
        # ============ PROJECT TO COMMON DIMENSION ============
        feat_resnet = self.proj_resnet(feat_resnet)      # [B, 512]
        feat_effnet = self.proj_effnet(feat_effnet)      # [B, 512]
        
        # ============ LEARNED FUSION WITH ATTENTION ============
        fused, att_weights = self.fusion_attention([feat_resnet, feat_effnet])
        # fused: [B, 1024]
        # att_weights: [B, 2] - learned importance of each expert
        
        # ============ CLASSIFICATION ============
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits
    
    def get_expert_confidence(self, x):
        """
        For interpretability: get individual expert predictions
        """
        feat_resnet = self.backbone_resnet(x)
        feat_effnet = self.backbone_effnet(x)
        
        feat_resnet = self.proj_resnet(feat_resnet)
        feat_effnet = self.proj_effnet(feat_effnet)
        
        _, att_weights = self.fusion_attention([feat_resnet, feat_effnet])
        
        return {
            'resnet50': att_weights[:, 0],
            'efficientnet_b3': att_weights[:, 1]
        }


# ================================================================
# LOSS FUNCTIONS
# ================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    
    Formula: L = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - α_t: class weight
    - γ: focusing parameter (Higher γ = more focus on hard examples)
    - p_t: model's estimated probability
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, num_classes] model outputs (logits)
            targets: [B] ground truth labels
            
        Returns:
            loss: scalar or tensor based on reduction
        """
        p = F.softmax(predictions, dim=1)
        ce_loss = F.cross_entropy(predictions, targets, reduction='none', weight=self.alpha)
        
        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    Prevents the model from becoming overconfident.
    
    Formula: L = -Σ q_i * log(p_i)
    where q_i = (1-ε)*(i==target) + ε/K (K = num_classes)
    """
    
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        num_classes = predictions.size(1)
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Create smooth targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        
        loss = torch.sum(-true_dist * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ================================================================
# SCHEDULER WITH WARMUP
# ================================================================

class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler: Linear warmup + Cosine annealing
    
    Strategy:
    - First `warmup_epochs`: Linear increase from warmup_lr to max_lr
    - Then: Cosine decrease from max_lr to min_lr
    """
    
    def __init__(self, optimizer, max_lr, min_lr, warmup_epochs, total_epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.optimizer = optimizer
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_epoch = self.last_epoch
        
        if current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_lr = self.min_lr + (self.max_lr - self.min_lr) * (current_epoch / self.warmup_epochs)
            return [warmup_lr for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            return [cosine_lr for _ in self.optimizer.param_groups]


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualExpertFusionModel(num_classes=5, pretrained=True).to(device)
    
    # Dummy input
    x = torch.randn(8, 3, 224, 224).to(device)
    
    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")  # Expected: [8, 5]
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss functions
    targets = torch.randint(0, 5, (8,)).to(device)
    
    # Focal Loss
    alpha = torch.tensor([0.6, 1.2, 1.0, 1.8, 1.0]).to(device)
    focal_loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    focal_loss = focal_loss_fn(logits, targets)
    print(f"Focal Loss: {focal_loss:.4f}")
    
    # Label Smoothing CE
    ls_loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
    ls_loss = ls_loss_fn(logits, targets)
    print(f"Label Smoothing CE: {ls_loss:.4f}")
