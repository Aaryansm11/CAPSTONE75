# src/utils.py
import random
import os
import time
import datetime
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SEED, LOG_DIR, LOG_FORMAT, LOG_LEVEL, DEVICE


def seed_everything(seed: int = SEED) -> None:
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_dir: Union[str, Path] = LOG_DIR, 
                  name: str = "multimodal_pipeline",
                  level: str = LOG_LEVEL) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Clear any existing handlers
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(state: Dict[str, Any], filepath: Union[str, Path], 
                   is_best: bool = False) -> None:
    """Save model checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.parent / f"best_{filepath.name}"
        torch.save(state, best_path)


def load_checkpoint(filepath: Union[str, Path], 
                   device: str = DEVICE) -> Dict[str, Any]:
    """Load model checkpoint."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    return torch.load(filepath, map_location=device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_prob: Optional[np.ndarray] = None,
                   class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    metrics['precision_macro'] = precision.mean()
    metrics['recall_macro'] = recall.mean()
    metrics['f1_macro'] = f1.mean()
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Per-class details
    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics[f'{class_name}_precision'] = precision[i]
                metrics[f'{class_name}_recall'] = recall[i]
                metrics[f'{class_name}_f1'] = f1[i]
                metrics[f'{class_name}_support'] = support[i]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            # Force normalization so rows sum to 1
            row_sums = y_prob.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # prevent division by zero
            y_prob = y_prob / row_sums  

            if y_prob.shape[1] == 2:  # Binary
                auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc'] = auc
            else:  # Multi-class
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_macro'] = auc
        except Exception as e:
            logging.warning(f"Could not compute AUC: {e}")
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str = "Confusion Matrix",
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss curve
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
        axes[0].plot(history['val_loss'], label='Val Loss', color='red', alpha=0.7)
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve  
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', color='blue', alpha=0.7)
        axes[1].plot(history['val_acc'], label='Val Acc', color='red', alpha=0.7)
        axes[1].set_title('Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # F1 Score curve
    if 'train_f1' in history and 'val_f1' in history:
        axes[2].plot(history['train_f1'], label='Train F1', color='blue', alpha=0.7)
        axes[2].plot(history['val_f1'], label='Val F1', color='red', alpha=0.7)
        axes[2].set_title('F1 Score Curve')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                   class_names: List[str], title: str = "ROC Curves",
                   save_path: Optional[Union[str, Path]] = None,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot ROC curves for multi-class classification."""
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i < y_prob.shape[1]:
            # Convert to binary classification for each class
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            auc_score = roc_auc_score(y_binary, y_prob[:, i])
            
            plt.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                    label=f'{class_name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    plt.show()


def save_results_json(results: Dict[str, Any], 
                      filepath: Union[str, Path]) -> None:
    """Save results dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_results[key] = value.item()
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def load_results_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load results from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """Early stopping to prevent overfitting during training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_score: Current validation score (higher is better)
            model: Model to save best weights
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self._save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self._save_checkpoint(model)
            self.counter = 0
            
        return False
    
    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_memory() -> Dict[str, float]:
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "cached": 0}
    
    return {
        "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
    }