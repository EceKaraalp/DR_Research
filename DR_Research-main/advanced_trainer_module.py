#!/usr/bin/env python3
"""
Diabetic Retinopathy Experimental Framework - Advanced Trainer Module

This module contains the advanced training loop, hard example mining,
and experiment execution logic for the DR classification framework.

Usage:
    python advanced_trainer_module.py
    
    Or import in notebook:
    exec(open('advanced_trainer_module.py').read())
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score

# ============================================================================
# ADVANCED TRAINER WITH HARD MINING & CHECKPOINT RESUME
# ============================================================================

class AdvancedExperimentTrainer:
    """Advanced trainer featuring hard example mining, confusion-aware weighting,<br>    and comprehensive checkpointing."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 loss_fn, config, fold=0, hard_miner=None, confusion_weighter=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.config = config
        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hard_miner = hard_miner
        self.confusion_weighter = confusion_weighter
        
        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.MAX_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
        
        if config.USE_SWA:
            from torch.optim.swa_utils import SWALR
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.SWA_LR, anneal_epochs=10)
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_qwk': [],
            'learning_rate': []
        }
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.start_epoch = 0
    
    def save_checkpoint(self, checkpoint_path, epoch):
        """Save training checkpoint for resuming"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'history': self.history,
        }
        if self.config.USE_SWA:
            checkpoint['swa_scheduler_state'] = self.swa_scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.best_metric = checkpoint['best_metric']
            self.patience_counter = checkpoint['patience_counter']
            self.history = checkpoint['history']
            self.start_epoch = checkpoint['epoch'] + 1
            
            if self.config.USE_SWA and 'swa_scheduler_state' in checkpoint:
                self.swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state'])
            
            print(f\"✅ Resumed from checkpoint (epoch {checkpoint['epoch'] + 1})\\n\")
            return True
        except Exception as e:
            print(f\"⚠️ Failed to load checkpoint: {e}\\nStarting fresh...\\n\")
            return False
    
    def train_epoch(self, epoch):
        \"\"\"Train for one epoch with optional hard mining\"\"\"
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        hard_example_buffer = []
        
        for batch in tqdm(self.train_loader, desc=f\"Fold{self.fold} E{epoch+1}\", leave=False):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            
            # Hard example mining
            if self.hard_miner and epoch > 10:  # Start after warmup
                loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
                hard_indices = self.hard_miner.mine(logits, labels, loss_per_sample)
                hard_example_buffer.extend(hard_indices.cpu().numpy().tolist())
                
                # Reweight loss for hard examples
                if len(hard_indices) > 0:
                    hard_loss = loss_per_sample[hard_indices].mean()
                    loss = 0.7 * loss + 0.3 * hard_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update confusion matrix for confusion-aware weighting
            if self.confusion_weighter:
                self.confusion_weighter.update_confusion(preds, labels.cpu().numpy())
        
        return total_loss / len(self.train_loader), np.mean([p == l for p, l in zip(all_preds, all_labels)])
    
    def validate(self, epoch):
        \"\"\"Validate on validation set\"\"\"
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = np.mean(all_preds == all_labels)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        
        return val_loss, val_acc, val_f1, val_qwk
    
    def fit(self, resume=False, checkpoint_dir=None):
        \"\"\"Main training loop\"\"\"
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold{self.fold}.pth') if checkpoint_dir else None
        best_model_path = os.path.join(checkpoint_dir, f'best_model_fold{self.fold}.pth') if checkpoint_dir else None
        
        if resume and checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            # Learning rate warmup
            if epoch < self.config.WARMUP_EPOCHS:
                lr = self.config.MIN_LR + (self.config.MAX_LR - self.config.MIN_LR) * (epoch / self.config.WARMUP_EPOCHS)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                if self.config.USE_SWA and epoch >= self.config.SWA_START:
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, val_qwk = self.validate(epoch)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_qwk'].append(val_qwk)
            self.history['learning_rate'].append(current_lr)
            
            if (epoch + 1) % 10 == 0:
                print(f\"E{epoch+1:3d} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | QWK: {val_qwk:.4f}\")
            
            # Early stopping
            if val_f1 > self.best_metric:
                self.best_metric = val_f1
                self.patience_counter = 0
                if best_model_path:
                    torch.save(self.model.state_dict(), best_model_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f\"⚠ Early stopping at epoch {epoch+1}\")
                    break
            
            # Save checkpoint periodically
            if checkpoint_path and (epoch + 1) % 5 == 0:
                self.save_checkpoint(checkpoint_path, epoch)
        
        # Cleanup
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        print(f\"✓ Fold {self.fold + 1} done (Best F1: {self.best_metric:.4f})\")
        return best_model_path
    
    def evaluate_test(self):
        \"\"\"Evaluate on test set\"\"\"
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f\"Testing Fold {self.fold + 1}\", leave=False):
                images = batch['image'].to(self.device)
                
                if 'label' in batch:
                    all_labels.extend(batch['label'].cpu().numpy())
                
                logits, _ = self.model(images)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().numpy())
        
        return (np.array(all_labels) if all_labels else None, 
                np.array(all_preds), 
                np.array(all_probs))


# ============================================================================
# EXPERIMENT RUNNER - ORCHESTRATE ALL EXPERIMENTS
# ============================================================================

class ExperimentRunner:
    \"\"\"Master runner for all experiments\"\"\"
    
    def __init__(self, experiments_manifest, base_config, device='cuda'):
        self.experiments = experiments_manifest
        self.config = base_config
        self.device = device
        self.results_summary = []
    
    def run_all_experiments(self, experiments_base_dir):
        \"\"\"Execute all experiments\"\"\"
        os.makedirs(experiments_base_dir, exist_ok=True)
        
        for experiment in self.experiments:
            print(f\"\\n{'='*70}\")
            print(f\"EXPERIMENT: {experiment['name']}\")
            print(f\"{'='*70}\")
            print(f\"{experiment['description']}\\n\")
            
            # Create experiment directory
            exp_dir = os.path.join(experiments_base_dir, experiment['name'])
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save experiment config
            with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
                json.dump(experiment, f, indent=2)
            
            # Run experiment
            results = self._run_single_experiment(experiment, exp_dir)
            self.results_summary.append(results)
            
            print(f\"\\n✓ Experiment '{experiment['name']}' completed\")
        
        # Save summary
        self._save_summary(experiments_base_dir)
    
    def _run_single_experiment(self, exp_config, exp_dir):
        \"\"\"Run a single experiment across all folds\"\"\"
        fold_results = []
        
        for fold_idx in range(self.config.NUM_FOLDS):
            print(f\"\\nFold {fold_idx + 1}/{self.config.NUM_FOLDS}...\")
            # Implementation would go here
            # Loading data, creating model, training, evaluation
        
        return {
            'experiment': exp_config['name'],
            'description': exp_config['description'],
            'fold_results': fold_results,
            'mean_f1': np.mean([f['f1'] for f in fold_results]),
            'mean_qwk': np.mean([f['qwk'] for f in fold_results]),
        }
    
    def _save_summary(self, base_dir):
        \"\"\"Save experiment summary\"\"\"
        summary_path = os.path.join(base_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.results_summary, f, indent=2)
        print(f\"\\n✓ Experiment summary saved: {summary_path}\")


print(\"✓ Advanced trainer module loaded\")\n