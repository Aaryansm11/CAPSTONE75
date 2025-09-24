# src/contrastive_trainer.py
"""
Contrastive learning trainer for self-supervised cardiac pattern discovery.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import DEVICE, LR, WEIGHT_DECAY, EPOCHS, MODEL_DIR
from src.models import MultiModalFusionNetwork, SignalBranchEncoder
from src.self_supervised import ContrastiveLoss, CrossModalConsistencyLoss, PatternDiscovery
from src.utils import save_checkpoint, EarlyStopping
from src.validation import ModelValidator

logger = logging.getLogger(__name__)


class SelfSupervisedEncoder(nn.Module):
    """Self-supervised encoder for learning cardiac representations."""

    def __init__(self,
                 ecg_seq_len: int = 3600,
                 ppg_seq_len: int = 3600,
                 embedding_dim: int = 256,
                 ecg_cnn_filters: List[int] = None,
                 ppg_cnn_filters: List[int] = None,
                 ecg_lstm_hidden: int = 128,
                 ppg_lstm_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        if ecg_cnn_filters is None:
            ecg_cnn_filters = [32, 64, 64, 128]
        if ppg_cnn_filters is None:
            ppg_cnn_filters = [16, 32, 32, 64]

        # ECG and PPG encoders
        self.ecg_encoder = SignalBranchEncoder(
            in_channels=1,
            cnn_filters=ecg_cnn_filters,
            lstm_hidden=ecg_lstm_hidden,
            dropout=dropout,
            use_attention=True
        )

        self.ppg_encoder = SignalBranchEncoder(
            in_channels=1,
            cnn_filters=ppg_cnn_filters,
            lstm_hidden=ppg_lstm_hidden,
            dropout=dropout,
            use_attention=True
        )

        # Projection heads for contrastive learning
        ecg_hidden = ecg_lstm_hidden * 2  # Bidirectional
        ppg_hidden = ppg_lstm_hidden * 2  # Bidirectional

        self.ecg_projection = nn.Sequential(
            nn.Linear(ecg_hidden, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.ppg_projection = nn.Sequential(
            nn.Linear(ppg_hidden, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Joint embedding for multimodal fusion
        self.joint_projection = nn.Sequential(
            nn.Linear(ecg_hidden + ppg_hidden, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, ecg_signal: torch.Tensor, ppg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for self-supervised learning."""

        # Encode signals
        ecg_out = self.ecg_encoder(ecg_signal)
        ppg_out = self.ppg_encoder(ppg_signal)

        ecg_embedding = ecg_out['embedding']
        ppg_embedding = ppg_out['embedding']

        # Project to embedding space
        ecg_projected = self.ecg_projection(ecg_embedding)
        ppg_projected = self.ppg_projection(ppg_embedding)

        # Joint embedding
        joint_features = torch.cat([ecg_embedding, ppg_embedding], dim=1)
        joint_projected = self.joint_projection(joint_features)

        return {
            'ecg_embedding': ecg_projected,
            'ppg_embedding': ppg_projected,
            'joint_embedding': joint_projected,
            'ecg_attention': ecg_out['attention_weights'],
            'ppg_attention': ppg_out['attention_weights'],
            'raw_ecg_features': ecg_embedding,
            'raw_ppg_features': ppg_embedding
        }

    def get_representations(self, ecg_signal: torch.Tensor, ppg_signal: torch.Tensor) -> torch.Tensor:
        """Get joint representations for clustering/analysis."""
        with torch.no_grad():
            outputs = self.forward(ecg_signal, ppg_signal)
            return outputs['joint_embedding']


class ContrastiveTrainer:
    """Trainer for self-supervised contrastive learning."""

    def __init__(self,
                 model: SelfSupervisedEncoder,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = DEVICE,
                 lr: float = LR,
                 weight_decay: float = WEIGHT_DECAY,
                 epochs: int = EPOCHS,
                 contrastive_weight: float = 1.0,
                 cross_modal_weight: float = 0.5,
                 temporal_weight: float = 0.3,
                 temperature: float = 0.07,
                 save_dir: Optional[Path] = None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir or MODEL_DIR

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.cross_modal_loss = CrossModalConsistencyLoss(temperature=0.1)
        self.temporal_loss = nn.MSELoss()  # For temporal consistency

        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.cross_modal_weight = cross_modal_weight
        self.temporal_weight = temporal_weight

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        # Validation
        self.validator = ModelValidator()

        logger.info(f"Initialized contrastive trainer with {sum(p.numel() for p in model.parameters())} parameters")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        contrastive_loss_sum = 0.0
        cross_modal_loss_sum = 0.0
        temporal_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            # Move to device
            ecg_view1 = batch['ecg_view1'].to(self.device)
            ecg_view2 = batch['ecg_view2'].to(self.device)
            ppg_view1 = batch['ppg_view1'].to(self.device)
            ppg_view2 = batch['ppg_view2'].to(self.device)
            patient_indices = batch['patient_idx']

            # Forward pass for both views
            outputs1 = self.model(ecg_view1, ppg_view1)
            outputs2 = self.model(ecg_view2, ppg_view2)

            # Contrastive loss between views
            contrastive_loss_ecg = self.contrastive_loss(
                outputs1['ecg_embedding'], outputs2['ecg_embedding']
            )
            contrastive_loss_ppg = self.contrastive_loss(
                outputs1['ppg_embedding'], outputs2['ppg_embedding']
            )
            contrastive_loss_joint = self.contrastive_loss(
                outputs1['joint_embedding'], outputs2['joint_embedding']
            )

            contrastive_loss = (contrastive_loss_ecg + contrastive_loss_ppg + contrastive_loss_joint) / 3

            # Cross-modal consistency loss
            cross_modal_loss1 = self.cross_modal_loss(
                outputs1['ecg_embedding'], outputs1['ppg_embedding']
            )
            cross_modal_loss2 = self.cross_modal_loss(
                outputs2['ecg_embedding'], outputs2['ppg_embedding']
            )
            cross_modal_loss = (cross_modal_loss1 + cross_modal_loss2) / 2

            # Temporal consistency loss (same patient should have similar representations)
            temporal_loss = self._compute_temporal_consistency_loss(
                outputs1, outputs2, patient_indices
            )

            # Combined loss
            total_batch_loss = (
                self.contrastive_weight * contrastive_loss +
                self.cross_modal_weight * cross_modal_loss +
                self.temporal_weight * temporal_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            cross_modal_loss_sum += cross_modal_loss.item()
            temporal_loss_sum += temporal_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_batch_loss.item():.4f}",
                'Contr': f"{contrastive_loss.item():.4f}",
                'Cross': f"{cross_modal_loss.item():.4f}",
                'Temp': f"{temporal_loss.item():.4f}"
            })

        return {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': contrastive_loss_sum / num_batches,
            'cross_modal_loss': cross_modal_loss_sum / num_batches,
            'temporal_loss': temporal_loss_sum / num_batches
        }

    def _compute_temporal_consistency_loss(self, outputs1: Dict, outputs2: Dict,
                                         patient_indices: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss for same patient samples."""
        if len(torch.unique(patient_indices)) < 2:
            return torch.tensor(0.0, device=self.device)

        # Group by patient
        unique_patients = torch.unique(patient_indices)
        consistency_losses = []

        for patient_id in unique_patients:
            patient_mask = patient_indices == patient_id
            if torch.sum(patient_mask) < 2:
                continue

            # Get embeddings for this patient
            patient_emb1 = outputs1['joint_embedding'][patient_mask]
            patient_emb2 = outputs2['joint_embedding'][patient_mask]

            # Compute consistency within patient
            if len(patient_emb1) > 1:
                # Mean embedding for this patient
                mean_emb1 = torch.mean(patient_emb1, dim=0, keepdim=True)
                mean_emb2 = torch.mean(patient_emb2, dim=0, keepdim=True)

                # Consistency loss
                consistency_loss = self.temporal_loss(mean_emb1, mean_emb2)
                consistency_losses.append(consistency_loss)

        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                ecg_view1 = batch['ecg_view1'].to(self.device)
                ecg_view2 = batch['ecg_view2'].to(self.device)
                ppg_view1 = batch['ppg_view1'].to(self.device)
                ppg_view2 = batch['ppg_view2'].to(self.device)

                outputs1 = self.model(ecg_view1, ppg_view1)
                outputs2 = self.model(ecg_view2, ppg_view2)

                # Compute validation loss (simplified)
                contrastive_loss = self.contrastive_loss(
                    outputs1['joint_embedding'], outputs2['joint_embedding']
                )
                cross_modal_loss = self.cross_modal_loss(
                    outputs1['ecg_embedding'], outputs1['ppg_embedding']
                )

                batch_loss = contrastive_loss + 0.5 * cross_modal_loss
                total_loss += batch_loss.item()
                num_batches += 1

        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0.0}

    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        logger.info(f"Starting contrastive training for {self.epochs} epochs")

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update scheduler
            self.scheduler.step()

            # Record metrics
            self.train_losses.append(train_metrics['total_loss'])
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - start_time

            # Logging
            log_msg = f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s) - "
            log_msg += f"Train Loss: {train_metrics['total_loss']:.4f}"

            if val_metrics:
                log_msg += f", Val Loss: {val_metrics['val_loss']:.4f}"
                current_val_loss = val_metrics['val_loss']
            else:
                current_val_loss = train_metrics['total_loss']

            logger.info(log_msg)

            # Save best model
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.save_checkpoint(epoch, is_best=True)

            # Early stopping
            if self.early_stopping(current_val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        logger.info("Training completed!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"contrastive_model_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / "contrastive_model_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint at epoch {epoch+1}")

    def extract_representations(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract learned representations from trained model."""
        self.model.eval()

        all_embeddings = []
        all_patient_indices = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting representations"):
                ecg = batch['ecg_view1'].to(self.device)  # Use first view
                ppg = batch['ppg_view1'].to(self.device)
                patient_indices = batch['patient_idx'].cpu().numpy()

                embeddings = self.model.get_representations(ecg, ppg)
                all_embeddings.append(embeddings.cpu().numpy())
                all_patient_indices.extend(patient_indices)

        embeddings_array = np.vstack(all_embeddings)
        patient_array = np.array(all_patient_indices)

        logger.info(f"Extracted {len(embeddings_array)} representations with dimension {embeddings_array.shape[1]}")

        return embeddings_array, patient_array

    def discover_patterns(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Discover cardiac patterns using trained representations."""
        embeddings, patient_indices = self.extract_representations(dataloader)

        # Initialize pattern discovery
        pattern_discoverer = PatternDiscovery()

        # Discover patterns
        results = pattern_discoverer.discover_patterns(embeddings)
        results['embeddings'] = embeddings
        results['patient_indices'] = patient_indices

        return results

    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Validation Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate
        axes[1].plot(self.learning_rates, label='Learning Rate', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()