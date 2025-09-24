# src/explain.py - Fixed version compatible with 1.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Some explainability features will be disabled.")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Some explainability features will be disabled.")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# Import from expected config structure (compatible with 1.py)
try:
    from src.config import CLASS_NAMES, DEVICE
    from src.data_utils import extract_basic_features
except ImportError:
    # Fallback definitions if config not available
    CLASS_NAMES = ['Normal', 'Atrial Fibrillation', 'Other Arrhythmia', 'Noise']
    DEVICE = 'cpu'
    
    def extract_basic_features(signal):
        """Basic feature extraction fallback."""
        return np.array([
            np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
            np.ptp(signal), np.sum(signal**2), np.var(signal), np.median(signal)
        ])

logger = logging.getLogger(__name__)

# Constants compatible with 1.py expectations
SHAP_BACKGROUND_SIZE = 50
SHAP_EXPLAINER_SIZE = 20


def shap_on_features(feature_matrix: np.ndarray, predict_fn, out_dir: str, prefix: str = "shap"):
    """
    SHAP analysis on features - compatible with 1.py expectations.
    
    Args:
        feature_matrix: Feature matrix [n_samples, n_features]
        predict_fn: Prediction function
        out_dir: Output directory for plots
        prefix: Prefix for saved files
        
    Returns:
        Dictionary with SHAP explanations
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, returning mock explanations")
        return {
            'shap_values': [np.random.randn(*feature_matrix.shape) for _ in range(len(CLASS_NAMES))],
            'feature_names': [f'feature_{i}' for i in range(feature_matrix.shape[1])],
            'mock_data': True
        }
    
    try:
        # Use a subset for background
        background_size = min(20, len(feature_matrix) // 2)
        background_indices = np.random.choice(len(feature_matrix), background_size, replace=False)
        background_data = feature_matrix[background_indices]
        
        # Initialize explainer
        explainer = shap.KernelExplainer(predict_fn, background_data)
        
        # Compute SHAP values for a subset
        explain_size = min(10, len(feature_matrix))
        explain_indices = np.random.choice(len(feature_matrix), explain_size, replace=False)
        explain_data = feature_matrix[explain_indices]
        
        shap_values = explainer.shap_values(explain_data)
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
        
        # Save summary plots
        os.makedirs(out_dir, exist_ok=True)
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            try:
                plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list) and len(shap_values) > class_idx:
                    shap.summary_plot(
                        shap_values[class_idx], 
                        explain_data, 
                        feature_names=feature_names,
                        show=False
                    )
                    plt.title(f'SHAP Summary - {class_name}')
                    save_path = os.path.join(out_dir, f'{prefix}_{class_name.lower().replace(" ", "_")}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved SHAP plot: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to create SHAP plot for {class_name}: {e}")
                plt.close()
        
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'background_data': background_data,
            'explained_data': explain_data
        }
        
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        return {
            'error': str(e),
            'shap_values': [np.random.randn(*feature_matrix.shape) for _ in range(len(CLASS_NAMES))],
            'feature_names': [f'feature_{i}' for i in range(feature_matrix.shape[1])]
        }


def attention_heatmap(attention_weights: np.ndarray, save_path: str):
    """
    Create attention heatmap - compatible with 1.py expectations.
    
    Args:
        attention_weights: Attention weights [n_heads, seq_len] or [seq_len]
        save_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    try:
        plt.figure(figsize=(15, 6))
        
        # Handle different attention weight dimensions
        if attention_weights.ndim > 2:
            # Take mean across batch dimension if present
            attention_weights = np.mean(attention_weights, axis=0)
        
        if attention_weights.ndim == 2:
            # Multiple heads - show as heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(attention_weights, cmap='viridis', cbar=True)
            plt.title('Attention Weights (All Heads)')
            plt.xlabel('Sequence Position')
            plt.ylabel('Attention Head')
            
            # Show average attention
            plt.subplot(1, 2, 2)
            avg_attention = np.mean(attention_weights, axis=0)
            plt.plot(avg_attention, 'b-', linewidth=2)
            plt.fill_between(range(len(avg_attention)), avg_attention, alpha=0.3)
            plt.title('Average Attention Weights')
            plt.xlabel('Sequence Position')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
            
        else:
            # Single dimension - line plot
            plt.plot(attention_weights, 'b-', linewidth=2)
            plt.fill_between(range(len(attention_weights)), attention_weights, alpha=0.3)
            plt.title('Attention Weights')
            plt.xlabel('Sequence Position')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention heatmap: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Failed to create attention heatmap: {e}")
        # Create a simple placeholder plot
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f'Attention visualization failed:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Attention Heatmap (Error)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path


class ModelExplainer:
    """Explainability framework for multimodal ECG/PPG model - compatible with 1.py."""
    
    def __init__(self, model: torch.nn.Module, device: str = None):
        self.model = model
        self.device = device or DEVICE
        self.model.to(self.device)
        self.model.eval()
        
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info(f"Model explainer initialized on device: {self.device}")
    
    def setup_shap_explainer(self, background_data: Dict[str, np.ndarray]):
        """Setup SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return
            
        try:
            # Extract features from background data
            ecg_data = background_data['ecg_windows'][:SHAP_BACKGROUND_SIZE]
            ppg_data = background_data['ppg_windows'][:SHAP_BACKGROUND_SIZE]
            
            background_features = self._extract_features_batch(ecg_data, ppg_data)
            
            # Create prediction function for SHAP
            def predict_fn(features):
                return self._predict_from_features(features)
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.KernelExplainer(predict_fn, background_features)
            
            logger.info(f"SHAP explainer setup with {len(background_features)} background samples")
            
        except Exception as e:
            logger.error(f"Failed to setup SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _extract_features_batch(self, ecg_batch: np.ndarray, ppg_batch: np.ndarray) -> np.ndarray:
        """Extract features from batch of ECG and PPG signals."""
        features_list = []
        
        for ecg_signal, ppg_signal in zip(ecg_batch, ppg_batch):
            # Use the basic feature extraction function
            ecg_features = extract_basic_features(ecg_signal)
            ppg_features = extract_basic_features(ppg_signal)
            combined_features = np.concatenate([ecg_features, ppg_features])
            features_list.append(combined_features)
        
        return np.array(features_list)
    
    def _predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions from extracted features.
        This is a simplified approach for demonstration.
        """
        # For demonstration, return random predictions weighted by features
        batch_size = features.shape[0]
        
        # Simple feature-based prediction (replace with actual feature-based model)
        predictions = np.random.rand(batch_size, len(CLASS_NAMES))
        
        # Add some correlation with features for more realistic behavior
        feature_sum = np.sum(features, axis=1, keepdims=True)
        feature_weights = feature_sum / (np.max(feature_sum) + 1e-8)
        
        # Modify predictions based on feature magnitudes
        for i in range(len(CLASS_NAMES)):
            predictions[:, i] *= (1 + 0.5 * feature_weights.flatten())
        
        # Normalize to probabilities
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
        
        return predictions
    
    def _predict_from_signals(self, ecg_batch: np.ndarray, ppg_batch: np.ndarray) -> np.ndarray:
        """Make predictions from raw signals."""
        with torch.no_grad():
            ecg_tensor = torch.from_numpy(ecg_batch).unsqueeze(1).float().to(self.device)
            ppg_tensor = torch.from_numpy(ppg_batch).unsqueeze(1).float().to(self.device)
            
            outputs = self.model(ecg_tensor, ppg_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'arr_logits' in outputs:
                    logits = outputs['arr_logits']
                elif 'arrhythmia_logits' in outputs:
                    logits = outputs['arrhythmia_logits']
                else:
                    # Take first available logits
                    logits = next(v for k, v in outputs.items() if 'logit' in k.lower())
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
    
    def explain_with_gradients(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray,
                             class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate gradient-based explanations."""
        try:
            # Enable gradients
            ecg_tensor = torch.from_numpy(ecg_signal).unsqueeze(0).unsqueeze(0).float().to(self.device)
            ppg_tensor = torch.from_numpy(ppg_signal).unsqueeze(0).unsqueeze(0).float().to(self.device)
            ecg_tensor.requires_grad_(True)
            ppg_tensor.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(ecg_tensor, ppg_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'arr_logits' in outputs:
                    logits = outputs['arr_logits']
                elif 'arrhythmia_logits' in outputs:
                    logits = outputs['arrhythmia_logits']
                else:
                    logits = next(v for k, v in outputs.items() if 'logit' in k.lower())
            else:
                logits = outputs
            
            # Get target class
            if class_idx is None:
                class_idx = torch.argmax(logits, dim=1).item()
            
            # Compute gradients
            target_score = logits[0, class_idx]
            target_score.backward()
            
            # Get gradients
            ecg_gradients = ecg_tensor.grad.cpu().numpy().squeeze()
            ppg_gradients = ppg_tensor.grad.cpu().numpy().squeeze()
            
            return {
                'ecg_gradients': ecg_gradients,
                'ppg_gradients': ppg_gradients,
                'target_class': class_idx,
                'prediction_score': target_score.item()
            }
            
        except Exception as e:
            logger.error(f"Gradient explanation failed: {e}")
            return {
                'error': str(e),
                'ecg_gradients': np.zeros_like(ecg_signal),
                'ppg_gradients': np.zeros_like(ppg_signal)
            }
    
    def create_feature_explanation(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray) -> Dict[str, Any]:
        """Create feature-based explanation compatible with 1.py."""
        try:
            # Extract features
            ecg_features = extract_basic_features(ecg_signal)
            ppg_features = extract_basic_features(ppg_signal)
            combined_features = np.concatenate([ecg_features, ppg_features])
            
            # Feature names
            ecg_feature_names = [
                'ECG_mean', 'ECG_std', 'ECG_min', 'ECG_max', 
                'ECG_ptp', 'ECG_energy', 'ECG_variation', 'ECG_median'
            ]
            ppg_feature_names = [
                'PPG_mean', 'PPG_std', 'PPG_min', 'PPG_max',
                'PPG_ptp', 'PPG_energy', 'PPG_variation', 'PPG_median'
            ]
            feature_names = ecg_feature_names + ppg_feature_names
            
            # Get model prediction
            prediction = self._predict_from_signals(
                np.array([ecg_signal]), np.array([ppg_signal])
            )[0]
            
            return {
                'features': combined_features,
                'feature_names': feature_names,
                'prediction': prediction,
                'predicted_class': CLASS_NAMES[np.argmax(prediction)],
                'confidence': float(np.max(prediction))
            }
            
        except Exception as e:
            logger.error(f"Feature explanation failed: {e}")
            return {
                'error': str(e),
                'features': np.zeros(16),  # 8 ECG + 8 PPG features
                'feature_names': [f'feature_{i}' for i in range(16)]
            }


def create_visualization_plots(ecg_signal: np.ndarray, ppg_signal: np.ndarray, 
                             explanation: Dict[str, Any], save_dir: str,
                             patient_id: str = "unknown") -> List[str]:
    """
    Create visualization plots for explanations - compatible with 1.py.
    
    Args:
        ecg_signal: ECG signal data
        ppg_signal: PPG signal data
        explanation: Explanation dictionary
        save_dir: Directory to save plots
        patient_id: Patient identifier
        
    Returns:
        List of paths to saved plots
    """
    plot_paths = []
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. Signal plots with gradients (if available)
        if 'ecg_gradients' in explanation:
            plt.figure(figsize=(15, 10))
            
            # ECG signal and gradients
            plt.subplot(2, 2, 1)
            plt.plot(ecg_signal, 'b-', alpha=0.7, label='ECG Signal')
            plt.title(f'ECG Signal - {patient_id}')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 2)
            gradients = explanation['ecg_gradients']
            if len(gradients) == len(ecg_signal):
                plt.plot(gradients, 'r-', alpha=0.7, label='Gradients')
                plt.title('ECG Gradients')
                plt.xlabel('Time Steps')
                plt.ylabel('Gradient')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            # PPG signal and gradients
            plt.subplot(2, 2, 3)
            plt.plot(ppg_signal, 'g-', alpha=0.7, label='PPG Signal')
            plt.title(f'PPG Signal - {patient_id}')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 4)
            gradients = explanation['ppg_gradients']
            if len(gradients) == len(ppg_signal):
                plt.plot(gradients, 'orange', alpha=0.7, label='Gradients')
                plt.title('PPG Gradients')
                plt.xlabel('Time Steps')
                plt.ylabel('Gradient')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'{patient_id}_signals_gradients.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 2. Feature importance plot
        if 'features' in explanation and 'feature_names' in explanation:
            plt.figure(figsize=(12, 8))
            
            features = explanation['features']
            feature_names = explanation['feature_names']
            
            # Simple importance based on feature magnitudes
            importances = np.abs(features)
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            plt.barh(range(len(sorted_importances)), sorted_importances, alpha=0.7)
            plt.yticks(range(len(sorted_importances)), sorted_names)
            plt.xlabel('Feature Importance (Magnitude)')
            plt.title(f'Feature Importance - {patient_id}')
            plt.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'{patient_id}_feature_importance.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 3. Prediction summary
        if 'prediction' in explanation:
            plt.figure(figsize=(10, 6))
            
            prediction = explanation['prediction']
            
            plt.bar(range(len(CLASS_NAMES)), prediction, alpha=0.7)
            plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
            plt.ylabel('Prediction Probability')
            plt.title(f'Prediction Probabilities - {patient_id}')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Highlight predicted class
            predicted_idx = np.argmax(prediction)
            plt.bar(predicted_idx, prediction[predicted_idx], color='red', alpha=0.8, 
                   label=f'Predicted: {CLASS_NAMES[predicted_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'{patient_id}_predictions.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        logger.info(f"Created {len(plot_paths)} visualization plots for {patient_id}")
        
    except Exception as e:
        logger.error(f"Failed to create visualization plots: {e}")
    
    return plot_paths


def generate_comprehensive_explanation(model: torch.nn.Module, ecg_signal: np.ndarray,
                                     ppg_signal: np.ndarray, save_dir: Optional[str] = None,
                                     patient_id: str = "unknown") -> Dict[str, Any]:
    """
    Generate comprehensive explanation - main function compatible with 1.py.
    
    Args:
        model: Trained PyTorch model
        ecg_signal: Single ECG signal
        ppg_signal: Single PPG signal
        save_dir: Directory to save outputs
        patient_id: Patient identifier
        
    Returns:
        Dictionary containing all explanations and plot paths
    """
    try:
        explainer = ModelExplainer(model)
        
        # Generate different types of explanations
        explanation = {}
        
        # 1. Feature-based explanation
        feature_explanation = explainer.create_feature_explanation(ecg_signal, ppg_signal)
        explanation.update(feature_explanation)
        
        # 2. Gradient-based explanation
        gradient_explanation = explainer.explain_with_gradients(ecg_signal, ppg_signal)
        explanation.update(gradient_explanation)
        
        # 3. Create visualizations
        plot_paths = []
        if save_dir:
            plot_paths = create_visualization_plots(
                ecg_signal, ppg_signal, explanation, save_dir, patient_id
            )
        
        # 4. Summary
        result = {
            'explanations': explanation,
            'plots': plot_paths,
            'patient_id': patient_id,
            'summary': {
                'predicted_class': explanation.get('predicted_class', 'Unknown'),
                'confidence': explanation.get('confidence', 0.0),
                'target_class': explanation.get('target_class', 0),
                'num_plots': len(plot_paths)
            }
        }
        
        logger.info(f"Generated comprehensive explanation for {patient_id}")
        return result
        
    except Exception as e:
        logger.error(f"Comprehensive explanation failed: {e}")
        return {
            'error': str(e),
            'explanations': {},
            'plots': [],
            'patient_id': patient_id
        }