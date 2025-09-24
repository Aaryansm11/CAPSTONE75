# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from src.config import DEVICE, LR, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, EPOCHS
from src.utils import EarlyStopping, save_checkpoint, compute_metrics, format_time

logger = logging.getLogger(__name__)


class LossFunction:
    """Combined loss function for multimodal training."""
    
    def __init__(self, arrhythmia_weight: float = 1.0, stroke_weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None, 
                 label_smoothing: float = 0.0):
        self.arrhythmia_weight = arrhythmia_weight
        self.stroke_weight = stroke_weight
        self.label_smoothing = label_smoothing
        
        # Arrhythmia classification loss
        self.arrhythmia_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # Stroke regression loss
        self.stroke_criterion = nn.MSELoss()
        
    def __call__(self, predictions: Dict[str, torch.Tensor], 
                 arrhythmia_labels: torch.Tensor, 
                 stroke_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        arrhythmia_loss = self.arrhythmia_criterion(
            predictions['arrhythmia_logits'], arrhythmia_labels
        )
        
        stroke_loss = self.stroke_criterion(
            predictions['stroke_output'], stroke_labels
        )
        
        total_loss = (self.arrhythmia_weight * arrhythmia_loss + 
                     self.stroke_weight * stroke_loss)
        
        return {
            'total_loss': total_loss,
            'arrhythmia_loss': arrhythmia_loss,
            'stroke_loss': stroke_loss
        }


class Trainer:
    """Training manager for multimodal model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: str = DEVICE,
                 lr: float = LR,
                 weight_decay: float = WEIGHT_DECAY,
                 epochs: int = EPOCHS,
                 early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
                 arrhythmia_weight: float = 1.0,
                 stroke_weight: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 use_scheduler: bool = True,
                 use_amp: bool = True,
                 checkpoint_path: Optional[str] = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        
        self.loss_fn = LossFunction(
            arrhythmia_weight=arrhythmia_weight,
            stroke_weight=stroke_weight,
            class_weights=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=label_smoothing
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001
        )
        
        self.use_amp = use_amp and device.startswith('cuda')
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_arrhythmia_loss': [],
            'val_arrhythmia_loss': [],
            'train_stroke_loss': [],
            'val_stroke_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"Using AMP: {self.use_amp}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_arrhythmia_loss = 0.0
        total_stroke_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_arrhythmia_labels = []
        all_probabilities = []  # will hold batches, then concat
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for ecg, ppg, arr_labels, stroke_labels, _ in progress_bar:
            ecg, ppg = ecg.to(self.device), ppg.to(self.device)
            arr_labels, stroke_labels = arr_labels.to(self.device), stroke_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    predictions = self.model(ecg, ppg)
                    losses = self.loss_fn(predictions, arr_labels, stroke_labels)
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(ecg, ppg)
                losses = self.loss_fn(predictions, arr_labels, stroke_labels)
                losses['total_loss'].backward()
                self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            total_arrhythmia_loss += losses['arrhythmia_loss'].item()
            total_stroke_loss += losses['stroke_loss'].item()
            
            predicted_classes = torch.argmax(predictions['arrhythmia_logits'], dim=1)
            correct_predictions += (predicted_classes == arr_labels).sum().item()
            total_samples += arr_labels.size(0)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_arrhythmia_labels.extend(arr_labels.cpu().numpy())
            
            probs = torch.softmax(predictions['arrhythmia_logits'], dim=1)
            all_probabilities.append(probs.detach().cpu().numpy())  # 🔧 FIX (append, not extend)
            
            progress_bar.set_postfix({
                'Loss': f'{losses["total_loss"].item():.4f}',
                'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
            })
        
        # 🔧 Concatenate probs properly
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_arrhythmia_loss = total_arrhythmia_loss / len(self.train_loader)
        epoch_stroke_loss = total_stroke_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct_predictions / total_samples
        
        metrics = compute_metrics(
            np.array(all_arrhythmia_labels), 
            np.array(all_predictions),
            all_probabilities
        )
        epoch_f1 = metrics['f1_weighted'] * 100
        
        return {
            'loss': epoch_loss,
            'arrhythmia_loss': epoch_arrhythmia_loss,
            'stroke_loss': epoch_stroke_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1
        }


    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss, total_arrhythmia_loss, total_stroke_loss = 0.0, 0.0, 0.0
        correct_predictions, total_samples = 0, 0
        all_predictions, all_arrhythmia_labels, all_probabilities = [], [], []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for ecg, ppg, arr_labels, stroke_labels, _ in progress_bar:
                ecg, ppg = ecg.to(self.device), ppg.to(self.device)
                arr_labels, stroke_labels = arr_labels.to(self.device), stroke_labels.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(ecg, ppg)
                        losses = self.loss_fn(predictions, arr_labels, stroke_labels)
                else:
                    predictions = self.model(ecg, ppg)
                    losses = self.loss_fn(predictions, arr_labels, stroke_labels)
                
                total_loss += losses['total_loss'].item()
                total_arrhythmia_loss += losses['arrhythmia_loss'].item()
                total_stroke_loss += losses['stroke_loss'].item()
                
                predicted_classes = torch.argmax(predictions['arrhythmia_logits'], dim=1)
                correct_predictions += (predicted_classes == arr_labels).sum().item()
                total_samples += arr_labels.size(0)
                
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_arrhythmia_labels.extend(arr_labels.cpu().numpy())
                
                probs = torch.softmax(predictions['arrhythmia_logits'], dim=1)
                all_probabilities.append(probs.detach().cpu().numpy())  # 🔧 FIX (append, not extend)
                
                progress_bar.set_postfix({
                    'Loss': f'{losses["total_loss"].item():.4f}',
                    'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
                })
        
        # 🔧 Concatenate probs properly
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_arrhythmia_loss = total_arrhythmia_loss / len(self.val_loader)
        epoch_stroke_loss = total_stroke_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct_predictions / total_samples
        
        metrics = compute_metrics(
            np.array(all_arrhythmia_labels), 
            np.array(all_predictions),
            all_probabilities
        )
        epoch_f1 = metrics['f1_weighted'] * 100
        
        return {
            'loss': epoch_loss,
            'arrhythmia_loss': epoch_arrhythmia_loss,
            'stroke_loss': epoch_stroke_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1,
            'detailed_metrics': metrics
        }

    
    def train(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            if self.use_scheduler:
                self.scheduler.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_arrhythmia_loss'].append(train_metrics['arrhythmia_loss'])
            self.history['val_arrhythmia_loss'].append(val_metrics['arrhythmia_loss'])
            self.history['train_stroke_loss'].append(train_metrics['stroke_loss'])
            self.history['val_stroke_loss'].append(val_metrics['stroke_loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(current_lr)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} ({format_time(epoch_time)}) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val F1: {val_metrics['f1']:.2f}%, "
                f"LR: {current_lr:.6f}"
            )
            
            if self.checkpoint_path and (val_metrics['f1'] >= self.best_val_f1):
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.use_scheduler else None,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'history': self.history
                }
                if self.use_amp:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                save_checkpoint(checkpoint, self.checkpoint_path, is_best=True)
                logger.info(f"Checkpoint saved with Val F1: {val_metrics['f1']:.2f}%")
            
            if self.early_stopping(val_metrics['f1'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(total_time)}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Best validation F1: {self.best_val_f1:.2f}%")
        
        return self.history


def create_trainer(model: nn.Module, train_loader, val_loader, config: Dict = None) -> Trainer:
    """Factory function to create a trainer."""
    if config is None:
        config = {}
    
    class_weights = None
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.get('device', DEVICE),
        lr=config.get('lr', LR),
        weight_decay=config.get('weight_decay', WEIGHT_DECAY),
        epochs=config.get('epochs', EPOCHS),
        early_stopping_patience=config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE),
        arrhythmia_weight=config.get('arrhythmia_weight', 1.0),
        stroke_weight=config.get('stroke_weight', 0.5),
        class_weights=class_weights,
        label_smoothing=config.get('label_smoothing', 0.0),
        use_scheduler=config.get('use_scheduler', True),
        use_amp=config.get('use_amp', True),
        checkpoint_path=config.get('checkpoint_path', None)
    )


def make_loaders_from_multimodal_arrays(X_ecg_train, X_ppg_train, y_arr_train, 
                                       X_ecg_val, X_ppg_val, y_arr_val,
                                       transform_ecg=None, transform_ppg=None, 
                                       batch_size=64, num_workers=4, oversample=True):
    """Create DataLoaders from numpy arrays for training and validation."""
    from src.dataset import MultiModalDataset
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    train_dataset = MultiModalDataset(
        X_ecg_train, X_ppg_train, y_arr_train, 
        transform_ecg=transform_ecg, transform_ppg=transform_ppg
    )
    val_dataset = MultiModalDataset(
        X_ecg_val, X_ppg_val, y_arr_val,
        transform_ecg=transform_ecg, transform_ppg=transform_ppg
    )
    
    if oversample and len(np.unique(y_arr_train)) > 1:
        class_counts = np.bincount(y_arr_train)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[y_arr_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, 
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def train_multimodal(model, train_loader, val_loader, device="cpu", 
                    epochs=30, lr=1e-3, early_stop=6, 
                    weight_arr=1.0, weight_stroke=1.0):
    """Simple training function compatible with 1.py expectations."""
    config = {
        'device': device,
        'lr': lr,
        'epochs': epochs,
        'early_stopping_patience': early_stop,
        'arrhythmia_weight': weight_arr,
        'stroke_weight': weight_stroke,
        'checkpoint_path': 'models/multimodal_best.pth'
    }
    
    trainer = create_trainer(model, train_loader, val_loader, config)
    history = trainer.train()
    
    return model, {
        "train_loss": history['train_loss'],
        "val_loss": history['val_loss'], 
        "train_arr_acc": history['train_acc'],
        "val_arr_acc": history['val_acc']
    }
