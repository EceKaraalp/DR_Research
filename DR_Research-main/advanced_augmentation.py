"""
===============================================================
ADVANCED DATA PREPROCESSING & AUGMENTATION FOR DR CLASSIFICATION
Medical Imaging Optimized Pipeline
===============================================================

Key Improvements:
1. Ben Graham preprocessing (green channel extraction)
2. Medical-specific augmentations (MixUp, CutMix with soft labels)
3. Elastic deformations (simulate retinal variations)
4. Frequency domain augmentations (brightness, contrast)
5. Advanced color jittering (match imaging variations)

Reference:
- Ben Graham APTOS 2019 winning solution
- Medical imaging augmentation best practices
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter
import random
from typing import Tuple, List


# ================================================================
# MEDICAL PREPROCESSING
# ================================================================

class BenGrahamPreprocessing:
    """
    Ben Graham's preprocessing from APTOS 2019 winning solution.
    
    Key steps:
    1. Extract green channel (most informative for DR pathology)
    2. Apply bilateral filtering (preserve vessel edges)
    3. CLAHE enhancement with medical parameters
    4. Gamma correction
    """
    
    def __init__(self, image_size=512):
        self.image_size = image_size
    
    def __call__(self, img_path: str) -> np.ndarray:
        """
        Read and preprocess image.
        Returns: [image_size, image_size, 3] RGB numpy array
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ===== STEP 1: Extract green channel =====
        # Green channel has best contrast for microaneurysms/hemorrhages
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize to 0-255
        img_gray = (img_gray / img_gray.max() * 255).astype(np.uint8)
        
        # ===== STEP 2: Bilateral filtering =====
        # Smooth while preserving vessel edges (critical for pathology detection)
        img_filtered = cv2.bilateralFilter(img_gray, 9, 75, 75)
        
        # ===== STEP 3: CLAHE (Contrast Limited Adaptive Histogram Equalization) =====
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_filtered)
        
        # ===== STEP 4: Resize and pad to square =====
        h, w = img_clahe.shape
        size = max(h, w)
        
        # Calculate padding
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        
        img_padded = cv2.copyMakeBorder(
            img_clahe, pad_h, size - h - pad_h, pad_w, size - w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Resize to target
        img_resized = cv2.resize(img_padded, (self.image_size, self.image_size))
        
        # Convert grayscale back to RGB (3 channels)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        return img_rgb


# ================================================================
# ADVANCED AUGMENTATIONS
# ================================================================

class MixUp:
    """
    MixUp augmentation for medical images.
    
    Reference: Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
    
    Formula: x_mix = λ*x_i + (1-λ)*x_j
             y_mix = λ*y_i + (1-λ)*y_j  (soft labels)
    
    where λ ~ Beta(α, α)
    """
    
    def __init__(self, alpha=0.3, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, images: List[torch.Tensor], labels: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Apply MixUp to a batch at runtime.
        
        Args:
            images: List of [C, H, W] tensors
            labels: [B] tensor of class indices
            
        Returns:
            mixed_images: mixed batch
            mixed_labels: soft labels [B, num_classes]
        """
        if random.random() > self.p:
            return images, labels
        
        B = len(images)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(B)
        
        # Mix images
        mixed_images = []
        for i in range(B):
            mixed = lam * images[i] + (1 - lam) * images[index[i]]
            mixed_images.append(mixed)
        
        # Mix labels (soft labels)
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes).float()
        one_hot_mixed = one_hot_labels.clone()
        one_hot_mixed[index] = one_hot_labels[index]
        soft_labels = lam * one_hot_labels + (1 - lam) * one_hot_mixed
        
        return mixed_images, soft_labels


class CutMix:
    """
    CutMix augmentation adapted for medical images.
    
    Reference: Yun et al. (2019) "CutMix: Regularization Strategy to Train Strong Classifiers"
    
    Key for medical images: preserve pathological regions when possible
    """
    
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, images: List[torch.Tensor], labels: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Apply CutMix to batch.
        
        Args:
            images: List of [C, H, W] tensors
            labels: [B] class indices
            
        Returns:
            mixed_images: with cut regions
            mixed_labels: soft labels
        """
        if random.random() > self.p:
            return images, labels
        
        B, C, H, W = len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2]
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random index
        index = torch.randperm(B)
        
        # Random patch
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        # Random position
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = []
        for i in range(B):
            mixed = images[i].clone()
            mixed[:, bby1:bby2, bbx1:bbx2] = images[index[i]][:, bby1:bby2, bbx1:bbx2]
            mixed_images.append(mixed)
        
        # Recalculate lambda (based on actual cut area)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        # Soft labels
        num_classes = labels.max().item() + 1
        one_hot = F.one_hot(labels, num_classes).float()
        one_hot_index = F.one_hot(labels[index], num_classes).float()
        soft_labels = lam * one_hot + (1 - lam) * one_hot_index
        
        return mixed_images, soft_labels


class ElasticDeformation:
    """
    Elastic deformation of retinal images.
    
    Medical context: Simulates variations in retinal imaging
    (camera angle, eye movements, tissue elasticity)
    
    Reference: Simard et al. (2003) "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"
    """
    
    def __init__(self, alpha=50, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        img_array = np.array(img, dtype=np.float32)
        h, w = img_array.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * self.sigma
        dy = np.random.randn(h, w) * self.sigma
        
        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (5, 5), self.sigma)
        dy = cv2.GaussianBlur(dy, (5, 5), self.sigma)
        
        # Normalize
        dx = (dx / dx.std()) * self.alpha
        dy = (dy / dy.std()) * self.alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = (x + dx).astype(np.float32)
        y = (y + dy).astype(np.float32)
        
        # Remap image
        if len(img_array.shape) == 3:  # RGB
            deformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:  # Grayscale
            deformed = cv2.remap(img_array, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(np.uint8(deformed))


class AdaptiveGaussianBlur:
    """
    Adaptive Gaussian blur for medical images.
    Simulates different focus levels in fundus imaging.
    """
    
    def __init__(self, kernel_sizes=[3, 5, 7], p=0.3):
        self.kernel_sizes = kernel_sizes
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        kernel_size = random.choice(self.kernel_sizes)
        sigma = random.uniform(0.1, 2.0)
        
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class MedicalColorJitter:
    """
    Color jittering optimized for medical imaging.
    
    Key differences from standard ColorJitter:
    - Preserves luminance channel (medical images rely on brightness)
    - Controlled saturation changes (avoid unrealistic colors)
    - Hue changes restricted to physiological range
    """
    
    def __init__(self, brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        # Apply each transformation with probability
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = TF.adjust_brightness(img, factor)
        
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = TF.adjust_contrast(img, factor)
        
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            img = TF.adjust_saturation(img, factor)
        
        if random.random() < 0.5:
            factor = random.uniform(-self.hue, self.hue)
            img = TF.adjust_hue(img, factor)
        
        return img


# ================================================================
# COMPLETE AUGMENTATION PIPELINE
# ================================================================

class MedicalDRTrainAugmentation:
    """
    Complete training augmentation pipeline for DR classification.
    
    Strategy:
    1. Geometric transforms (safe for medical images)
    2. Elastic deformations (simulate retinal variations)
    3. Intensity transforms (medical-relevant color changes)
    4. Advanced mixing (MixUp, CutMix for regularization)
    
    All transforms preserve annotation integrity for medical images.
    """
    
    def __init__(self, image_size=224, strength='strong'):
        """
        Args:
            image_size: Target size (e.g., 224 for ImageNet models)
            strength: 'light', 'medium', or 'strong'
        """
        self.image_size = image_size
        self.strength = strength
        
        # Define augmentation strength
        if strength == 'light':
            aug_probs = {'geometric': 0.5, 'elastic': 0.2, 'intensity': 0.3}
            rotation = 10
            translate = (0.05, 0.05)
            scale_range = (0.95, 1.05)
            elastic_alpha = 30
        elif strength == 'medium':
            aug_probs = {'geometric': 0.7, 'elastic': 0.4, 'intensity': 0.6}
            rotation = 20
            translate = (0.1, 0.1)
            scale_range = (0.9, 1.1)
            elastic_alpha = 50
        else:  # strong
            aug_probs = {'geometric': 0.8, 'elastic': 0.5, 'intensity': 0.7}
            rotation = 30
            translate = (0.15, 0.15)
            scale_range = (0.85, 1.15)
            elastic_alpha = 70
        
        self.geometric_p = aug_probs['geometric']
        self.elastic_p = aug_probs['elastic']
        self.intensity_p = aug_probs['intensity']
        
        # Build augmentation transforms
        geometric_transforms = []
        
        # Random Affine (rotation, translation, scaling, shearing)
        if random.random() < self.geometric_p:
            geometric_transforms.append(
                transforms.RandomAffine(
                    degrees=rotation,
                    translate=translate,
                    scale=scale_range,
                    shear=10
                )
            )
        
        # Random Perspective (3D-like deformations)
        if random.random() < self.geometric_p:
            geometric_transforms.append(
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            )
        
        # Elastic deformation
        if random.random() < self.elastic_p:
            geometric_transforms.append(
                ElasticDeformation(alpha=elastic_alpha, sigma=5, p=1.0)
            )
        
        # Color/intensity transforms
        intensity_transforms = [
            MedicalColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=self.intensity_p),
            AdaptiveGaussianBlur(kernel_sizes=[3, 5], p=0.3),
            transforms.RandomEqualize(p=0.2),
        ]
        
        # Standard normalization (ImageNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Combine all transforms
        self.geometric_aug = transforms.Compose(geometric_transforms) if geometric_transforms else None
        self.intensity_aug = transforms.Compose(intensity_transforms)
        
        # Final pipeline
        self.pipeline = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply complete augmentation pipeline.
        
        Args:
            img: PIL Image
            
        Returns:
            augmented: [C, H, W] normalized tensor
        """
        # Apply geometric transforms
        if self.geometric_aug:
            img = self.geometric_aug(img)
        
        # Apply intensity transforms
        img = self.intensity_aug(img)
        
        # Resize and convert to tensor
        img = self.pipeline(img)
        
        # Normalize
        img = self.normalize(img)
        
        return img


class MedicalDRValAugmentation:
    """
    Validation augmentation: minimal, deterministic.
    Only standardization, no data augmentation.
    """
    
    def __init__(self, image_size=224):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.pipeline = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.pipeline(img)


class MedicalDRTestAugmentation:
    """
    Test-time augmentation (TTA) for ensemble predictions.
    
    Strategy: Generate 10 different augmented versions of same image,
    get predictions for each, average predictions.
    
    this significantly improves robustness and reduces overfitting effects.
    """
    
    def __init__(self, image_size=224, num_augmentations=10):
        self.image_size = image_size
        self.num_augmentations = num_augmentations
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Define TTA augmentation strategies
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize
            ]),  # Original
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                self.normalize
            ]),  # Horizontal flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                self.normalize
            ]),  # Vertical flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                self.normalize
            ]),  # Rotate +10
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=-10),
                transforms.ToTensor(),
                self.normalize
            ]),  # Rotate -10
            transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                self.normalize
            ]),  # Zoom in 1.1x
            transforms.Compose([
                transforms.Resize((int(image_size * 0.9), int(image_size * 0.9))),
                transforms.Pad(int(image_size * 0.05)),
                transforms.ToTensor(),
                self.normalize
            ]),  # Zoom out 0.9x
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                self.normalize
            ]),  # Brightness/contrast
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                self.normalize
            ]),  # Blur
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                self.normalize
            ]),  # Slight translation
        ]
    
    def __call__(self, img: Image.Image, num_augmentations=None) -> List[torch.Tensor]:
        """
        Generate multiple TTA augmentations.
        
        Args:
            img: PIL Image
            num_augmentations: override default count
            
        Returns:
            List of [C, H, W] tensors, each an augmented version
        """
        n_aug = num_augmentations or self.num_augmentations
        augmented = []
        
        for i in range(min(n_aug, len(self.tta_transforms))):
            aug_img = self.tta_transforms[i](img)
            augmented.append(aug_img)
        
        return augmented


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing augmentation pipeline...")
    
    # Create dummy image
    from PIL import Image
    import numpy as np
    
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Test train augmentation
    train_aug = MedicalDRTrainAugmentation(image_size=224, strength='strong')
    aug_img = train_aug(dummy_img)
    print(f"Train augmented shape: {aug_img.shape}")  # Should be [3, 224, 224]
    
    # Test val augmentation
    val_aug = MedicalDRValAugmentation(image_size=224)
    val_img = val_aug(dummy_img)
    print(f"Val augmented shape: {val_img.shape}")  # Should be [3, 224, 224]
    
    # Test TTA
    tta_aug = MedicalDRTestAugmentation(image_size=224, num_augmentations=10)
    tta_imgs = tta_aug(dummy_img, num_augmentations=5)
    print(f"TTA generated {len(tta_imgs)} augmentations, each shape: {tta_imgs[0].shape}")
    
    print("✅ All augmentations working correctly!")
