# src/model_bridge.py
"""
Bridge module to integrate self-supervised trained model with existing pipeline.
Maintains compatibility with original MultiModalFusionNetwork interface.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path

from src.contrastive_trainer import SelfSupervisedEncoder
from src.models import MultiModalFusionNetwork
from src.clinical_explainer import ModelExplainer, create_clinical_report
from src.config import DEVICE, CLASS_NAMES, N_ARRHYTHMIA_CLASSES, STROKE_OUTPUT_DIM

logger = logging.getLogger(__name__)


class AdaptiveMultiModalNetwork(nn.Module):
    """
    Adaptive network that can use either traditional supervised training
    or self-supervised pre-trained encoder with clinical prediction heads.
    """

    def __init__(self,
                 ecg_seq_len: int,
                 ppg_seq_len: int,
                 use_pretrained_encoder: bool = False,
                 pretrained_encoder_path: Optional[Path] = None,
                 n_arrhythmia_classes: int = N_ARRHYTHMIA_CLASSES,
                 stroke_output_dim: int = STROKE_OUTPUT_DIM,
                 embedding_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize adaptive multimodal network.

        Args:
            ecg_seq_len: ECG sequence length
            ppg_seq_len: PPG sequence length
            use_pretrained_encoder: Whether to use self-supervised pre-trained encoder
            pretrained_encoder_path: Path to pre-trained encoder checkpoint
            n_arrhythmia_classes: Number of arrhythmia classes
            stroke_output_dim: Stroke prediction output dimension
            embedding_dim: Embedding dimension for encoders
            dropout: Dropout rate
        """
        super().__init__()

        self.use_pretrained_encoder = use_pretrained_encoder
        self.embedding_dim = embedding_dim
        self.n_arrhythmia_classes = n_arrhythmia_classes
        self.stroke_output_dim = stroke_output_dim

        if use_pretrained_encoder and pretrained_encoder_path:
            # Load pre-trained self-supervised encoder
            self.encoder = self._load_pretrained_encoder(pretrained_encoder_path)

            # Add clinical prediction heads
            self._add_clinical_heads(embedding_dim, dropout)

            logger.info("Initialized with pre-trained self-supervised encoder")
        else:
            # Use traditional supervised model
            self.encoder = MultiModalFusionNetwork(
                ecg_seq_len=ecg_seq_len,
                ppg_seq_len=ppg_seq_len,
                n_arrhythmia_classes=n_arrhythmia_classes,
                stroke_output_dim=stroke_output_dim,
                dropout=dropout
            )

            logger.info("Initialized with traditional supervised model")

        # Initialize explainer
        self.explainer = ModelExplainer(self)

    def _load_pretrained_encoder(self, checkpoint_path: Path) -> SelfSupervisedEncoder:
        """Load pre-trained self-supervised encoder."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

            # Extract model config
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                encoder = SelfSupervisedEncoder(**config)
            else:
                # Default config if not available
                encoder = SelfSupervisedEncoder(
                    ecg_seq_len=3600,
                    ppg_seq_len=3600,
                    embedding_dim=self.embedding_dim
                )

            # Load state dict
            if 'model_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                encoder.load_state_dict(checkpoint)

            # Freeze encoder weights (optional - can be fine-tuned)
            # for param in encoder.parameters():
            #     param.requires_grad = False

            return encoder

        except Exception as e:
            logger.error(f"Failed to load pre-trained encoder: {e}")
            raise

    def _add_clinical_heads(self, embedding_dim: int, dropout: float):
        """Add clinical prediction heads to pre-trained encoder."""

        # Arrhythmia classification head
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, self.n_arrhythmia_classes)
        )

        # Stroke risk prediction head
        self.stroke_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, self.stroke_output_dim)
        )

    def forward(self, ecg_signal: torch.Tensor, ppg_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass - compatible with original MultiModalFusionNetwork interface."""

        if self.use_pretrained_encoder:
            # Use pre-trained encoder + clinical heads
            encoder_output = self.encoder(ecg_signal, ppg_signal)

            # Get joint embeddings
            joint_embedding = encoder_output['joint_embedding']

            # Clinical predictions
            arrhythmia_logits = self.arrhythmia_head(joint_embedding)
            stroke_output = self.stroke_head(joint_embedding)

            return {
                'arrhythmia_logits': arrhythmia_logits,
                'stroke_output': stroke_output.squeeze(-1),
                'ecg_embedding': encoder_output.get('ecg_embedding'),
                'ppg_embedding': encoder_output.get('ppg_embedding'),
                'fused_features': joint_embedding,
                'ecg_attention': encoder_output.get('ecg_attention'),
                'ppg_attention': encoder_output.get('ppg_attention')
            }
        else:
            # Use traditional supervised model
            return self.encoder(ecg_signal, ppg_signal)

    def get_clinical_explanation(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray,
                               patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive clinical explanation of predictions."""

        self.eval()
        with torch.no_grad():
            # Convert to tensors
            ecg_tensor = torch.from_numpy(ecg_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)
            ppg_tensor = torch.from_numpy(ppg_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Get model predictions
            model_outputs = self.forward(ecg_tensor, ppg_tensor)

            # Generate clinical explanation
            explanation = self.explainer.explain_prediction(
                ecg_signal, ppg_signal, model_outputs
            )

            # Add patient ID if provided
            if patient_id:
                explanation['patient_id'] = patient_id

            return explanation

    def generate_clinical_report(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray,
                               patient_id: Optional[str] = None) -> str:
        """Generate clinical report for doctor validation."""

        explanation = self.get_clinical_explanation(ecg_signal, ppg_signal, patient_id)
        return create_clinical_report(explanation, patient_id)

    def predict_arrhythmia_with_explanation(self, ecg_signal: np.ndarray,
                                          ppg_signal: np.ndarray) -> Dict[str, Any]:
        """Predict arrhythmia with clinical explanation - main interface for existing pipeline."""

        self.eval()
        with torch.no_grad():
            # Convert to tensors
            ecg_tensor = torch.from_numpy(ecg_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)
            ppg_tensor = torch.from_numpy(ppg_signal).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # Get predictions
            outputs = self.forward(ecg_tensor, ppg_tensor)

            # Process outputs
            arrhythmia_probs = torch.softmax(outputs['arrhythmia_logits'], dim=-1)
            predicted_class = torch.argmax(arrhythmia_probs, dim=-1).item()
            confidence = torch.max(arrhythmia_probs).item()

            stroke_risk = torch.sigmoid(outputs['stroke_output']).item()

            # Get clinical explanation
            explanation = self.get_clinical_explanation(ecg_signal, ppg_signal)

            return {
                'predicted_class': predicted_class,
                'predicted_class_name': CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else 'Unknown',
                'confidence': confidence,
                'class_probabilities': arrhythmia_probs[0].cpu().numpy(),
                'stroke_risk_score': stroke_risk,
                'clinical_explanation': explanation,
                'embeddings': outputs.get('fused_features', torch.tensor([])).cpu().numpy(),
                'attention_weights': {
                    'ecg': outputs.get('ecg_attention', torch.tensor([])).cpu().numpy(),
                    'ppg': outputs.get('ppg_attention', torch.tensor([])).cpu().numpy()
                }
            }

    def batch_predict_with_explanations(self, ecg_batch: np.ndarray,
                                      ppg_batch: np.ndarray,
                                      patient_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Batch prediction with explanations for multiple patients."""

        results = []

        for i in range(len(ecg_batch)):
            patient_id = patient_ids[i] if patient_ids else f"patient_{i}"

            try:
                result = self.predict_arrhythmia_with_explanation(
                    ecg_batch[i], ppg_batch[i]
                )
                result['patient_id'] = patient_id
                result['success'] = True

            except Exception as e:
                logger.error(f"Prediction failed for patient {patient_id}: {e}")
                result = {
                    'patient_id': patient_id,
                    'success': False,
                    'error': str(e)
                }

            results.append(result)

        return results

    def save_model(self, save_path: Path, include_explanation_config: bool = True):
        """Save model with configuration for later loading."""

        save_data = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'use_pretrained_encoder': self.use_pretrained_encoder,
                'embedding_dim': self.embedding_dim,
                'n_arrhythmia_classes': self.n_arrhythmia_classes,
                'stroke_output_dim': self.stroke_output_dim
            }
        }

        if include_explanation_config:
            save_data['explanation_config'] = {
                'clinical_features_enabled': True,
                'stroke_risk_assessment_enabled': True,
                'attention_analysis_enabled': True
            }

        torch.save(save_data, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, model_path: Path, ecg_seq_len: int, ppg_seq_len: int) -> 'AdaptiveMultiModalNetwork':
        """Load saved model."""

        checkpoint = torch.load(model_path, map_location=DEVICE)
        config = checkpoint['model_config']

        # Create model
        model = cls(
            ecg_seq_len=ecg_seq_len,
            ppg_seq_len=ppg_seq_len,
            use_pretrained_encoder=config['use_pretrained_encoder'],
            n_arrhythmia_classes=config['n_arrhythmia_classes'],
            stroke_output_dim=config['stroke_output_dim'],
            embedding_dim=config['embedding_dim']
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {model_path}")
        return model


def create_model_for_existing_pipeline(ecg_seq_len: int = 3600,
                                     ppg_seq_len: int = 3600,
                                     pretrained_encoder_path: Optional[Path] = None) -> AdaptiveMultiModalNetwork:
    """
    Factory function to create model for integration with existing pipeline.

    Args:
        ecg_seq_len: ECG sequence length
        ppg_seq_len: PPG sequence length
        pretrained_encoder_path: Path to self-supervised pre-trained encoder (optional)

    Returns:
        AdaptiveMultiModalNetwork ready for use in existing pipeline
    """

    use_pretrained = pretrained_encoder_path is not None and pretrained_encoder_path.exists()

    model = AdaptiveMultiModalNetwork(
        ecg_seq_len=ecg_seq_len,
        ppg_seq_len=ppg_seq_len,
        use_pretrained_encoder=use_pretrained,
        pretrained_encoder_path=pretrained_encoder_path
    )

    logger.info(f"Created adaptive model for existing pipeline (pretrained: {use_pretrained})")
    return model


# Example usage in existing pipeline
def example_integration():
    """Example of how to integrate with existing pipeline."""

    # Option 1: Use traditional supervised model (no changes needed)
    model = create_model_for_existing_pipeline()

    # Option 2: Use self-supervised pre-trained model
    pretrained_path = Path("models/contrastive_model_best.pth")
    if pretrained_path.exists():
        model = create_model_for_existing_pipeline(
            pretrained_encoder_path=pretrained_path
        )

    # Use in existing pipeline (same interface)
    ecg_signal = np.random.randn(3600)  # 10 seconds at 360 Hz
    ppg_signal = np.random.randn(1250)  # 10 seconds at 125 Hz (will be resampled)

    # Get predictions with clinical explanations
    result = model.predict_arrhythmia_with_explanation(ecg_signal, ppg_signal)

    print(f"Predicted class: {result['predicted_class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Stroke risk: {result['stroke_risk_score']:.1%}")

    # Generate clinical report
    report = model.generate_clinical_report(ecg_signal, ppg_signal, "PATIENT_001")
    print(report)

    return model, result, report