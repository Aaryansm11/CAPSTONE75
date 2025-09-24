# src/self_supervised.py
"""
Self-supervised learning framework for ECG+PPG cardiac pattern discovery.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from pathlib import Path
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.config import (
    ECG_FS, PPG_FS, WINDOW_SECONDS, DEVICE, BATCH_SIZE,
    AUGMENTATION_PROB, NOISE_FACTOR, STRETCH_FACTOR
)
from src.data_utils import SignalProcessor, FeatureExtractor
from src.validation import SignalValidator

logger = logging.getLogger(__name__)


class SelfSupervisedDataset(Dataset):
    """Dataset for self-supervised learning on unlabeled ECG+PPG pairs."""

    def __init__(self,
                 ecg_signals: List[np.ndarray],
                 ppg_signals: List[np.ndarray],
                 patient_ids: Optional[List[str]] = None,
                 window_seconds: int = WINDOW_SECONDS,
                 ecg_fs: int = ECG_FS,
                 ppg_fs: int = PPG_FS,
                 augment: bool = True,
                 quality_filter: bool = True,
                 overlap: float = 0.5):
        """
        Initialize self-supervised dataset for unlabeled ECG+PPG pairs.

        Args:
            ecg_signals: List of ECG signal arrays (one per patient/record)
            ppg_signals: List of PPG signal arrays (one per patient/record)
            patient_ids: Optional patient identifiers
            window_seconds: Window size for segmentation
            ecg_fs: ECG sampling rate
            ppg_fs: PPG sampling rate
            augment: Whether to apply data augmentation
            quality_filter: Whether to filter low-quality signals
            overlap: Overlap between consecutive windows
        """
        self.window_seconds = window_seconds
        self.ecg_fs = ecg_fs
        self.ppg_fs = ppg_fs
        self.augment = augment
        self.quality_filter = quality_filter
        self.overlap = overlap

        self.processor = SignalProcessor()
        self.validator = SignalValidator() if quality_filter else None
        self.augmenter = ContrastiveAugmentation() if augment else None

        # Process and segment signals
        self.ecg_windows = []
        self.ppg_windows = []
        self.patient_indices = []
        self.window_quality_scores = []

        self._process_signals(ecg_signals, ppg_signals, patient_ids)

        logger.info(f"Created self-supervised dataset with {len(self.ecg_windows)} windows")

    def _process_signals(self, ecg_signals: List[np.ndarray],
                        ppg_signals: List[np.ndarray],
                        patient_ids: Optional[List[str]]):
        """Process and segment raw signals into windows."""

        if patient_ids is None:
            patient_ids = [f"patient_{i}" for i in range(len(ecg_signals))]

        window_samples = int(self.window_seconds * self.ecg_fs)
        step_samples = max(1, int(window_samples * (1.0 - self.overlap)))

        for patient_idx, (ecg, ppg) in enumerate(zip(ecg_signals, ppg_signals)):
            if ecg is None or ppg is None or len(ecg) == 0 or len(ppg) == 0:
                continue

            # Preprocess signals
            ecg_processed = self.processor.preprocess_ecg(ecg, self.ecg_fs)
            ppg_processed = self.processor.preprocess_ppg(ppg, self.ppg_fs)

            # Resample PPG to match ECG sampling rate
            ppg_resampled = self.processor.resample_signal(ppg_processed, self.ppg_fs, self.ecg_fs)

            # Ensure same length
            min_length = min(len(ecg_processed), len(ppg_resampled))
            ecg_aligned = ecg_processed[:min_length]
            ppg_aligned = ppg_resampled[:min_length]

            # Create overlapping windows
            for start in range(0, min_length - window_samples + 1, step_samples):
                end = start + window_samples

                ecg_window = ecg_aligned[start:end]
                ppg_window = ppg_aligned[start:end]

                # Quality filtering
                if self.quality_filter:
                    ecg_quality = self.validator.validate_signal_quality(ecg_window, self.ecg_fs)
                    ppg_quality = self.validator.validate_signal_quality(ppg_window, self.ecg_fs)

                    if not ecg_quality['is_good_quality'] or not ppg_quality['is_good_quality']:
                        continue

                    # Store quality scores for analysis
                    avg_quality = (ecg_quality['quality_score'] + ppg_quality['quality_score']) / 2
                    self.window_quality_scores.append(avg_quality)
                else:
                    self.window_quality_scores.append(1.0)

                self.ecg_windows.append(ecg_window.astype(np.float32))
                self.ppg_windows.append(ppg_window.astype(np.float32))
                self.patient_indices.append(patient_idx)

    def __len__(self):
        return len(self.ecg_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample for contrastive learning."""
        ecg = self.ecg_windows[idx]
        ppg = self.ppg_windows[idx]
        patient_idx = self.patient_indices[idx]

        # Create two augmented views for contrastive learning
        if self.augment:
            ecg_view1 = self.augmenter(ecg.copy())
            ecg_view2 = self.augmenter(ecg.copy())
            ppg_view1 = self.augmenter(ppg.copy())
            ppg_view2 = self.augmenter(ppg.copy())
        else:
            ecg_view1 = ecg_view2 = ecg
            ppg_view1 = ppg_view2 = ppg

        return {
            'ecg_view1': torch.from_numpy(ecg_view1).unsqueeze(0),  # Add channel dimension
            'ecg_view2': torch.from_numpy(ecg_view2).unsqueeze(0),
            'ppg_view1': torch.from_numpy(ppg_view1).unsqueeze(0),
            'ppg_view2': torch.from_numpy(ppg_view2).unsqueeze(0),
            'patient_idx': torch.tensor(patient_idx, dtype=torch.long),
            'window_idx': torch.tensor(idx, dtype=torch.long),
            'quality_score': torch.tensor(self.window_quality_scores[idx], dtype=torch.float)
        }

    def get_patient_groups(self) -> Dict[int, List[int]]:
        """Get grouping of windows by patient for temporal consistency."""
        patient_groups = {}
        for window_idx, patient_idx in enumerate(self.patient_indices):
            if patient_idx not in patient_groups:
                patient_groups[patient_idx] = []
            patient_groups[patient_idx].append(window_idx)
        return patient_groups


class ContrastiveAugmentation:
    """Augmentation techniques for contrastive learning on cardiac signals."""

    def __init__(self,
                 noise_std: float = NOISE_FACTOR,
                 stretch_factor: float = STRETCH_FACTOR,
                 amplitude_scale_range: Tuple[float, float] = (0.8, 1.2),
                 baseline_shift_range: Tuple[float, float] = (-0.1, 0.1),
                 dropout_prob: float = 0.1):

        self.noise_std = noise_std
        self.stretch_factor = stretch_factor
        self.amplitude_scale_range = amplitude_scale_range
        self.baseline_shift_range = baseline_shift_range
        self.dropout_prob = dropout_prob

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to signal."""
        augmented = signal.copy()

        # Gaussian noise
        if random.random() < 0.7:
            noise = np.random.normal(0, self.noise_std, len(augmented))
            augmented = augmented + noise

        # Amplitude scaling
        if random.random() < 0.5:
            scale = np.random.uniform(*self.amplitude_scale_range)
            augmented = augmented * scale

        # Baseline shift
        if random.random() < 0.3:
            shift = np.random.uniform(*self.baseline_shift_range)
            augmented = augmented + shift

        # Time stretching (mild)
        if random.random() < 0.3:
            stretch = np.random.uniform(1 - self.stretch_factor, 1 + self.stretch_factor)
            augmented = self._time_stretch(augmented, stretch)

        # Signal dropout (set random segments to zero)
        if random.random() < self.dropout_prob:
            augmented = self._signal_dropout(augmented)

        return augmented.astype(np.float32)

    def _time_stretch(self, signal: np.ndarray, factor: float) -> np.ndarray:
        """Apply time stretching to signal."""
        try:
            from scipy.interpolate import interp1d

            original_indices = np.linspace(0, len(signal) - 1, len(signal))
            new_length = int(len(signal) * factor)
            new_indices = np.linspace(0, len(signal) - 1, new_length)

            if new_length < 10:  # Avoid too short signals
                return signal

            f = interp1d(original_indices, signal, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
            stretched = f(new_indices)

            # Resize back to original length
            if len(stretched) > len(signal):
                # Truncate
                return stretched[:len(signal)]
            else:
                # Pad with edge values
                padded = np.pad(stretched, (0, len(signal) - len(stretched)),
                              mode='edge')
                return padded

        except Exception:
            return signal

    def _signal_dropout(self, signal: np.ndarray, max_dropout_ratio: float = 0.1) -> np.ndarray:
        """Apply signal dropout by masking random segments."""
        dropout_length = int(len(signal) * max_dropout_ratio * random.random())
        if dropout_length > 0:
            start_idx = random.randint(0, len(signal) - dropout_length)
            signal_copy = signal.copy()
            signal_copy[start_idx:start_idx + dropout_length] = 0
            return signal_copy
        return signal


class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning."""

    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings [batch_size, embedding_dim]
            embeddings2: Second set of embeddings [batch_size, embedding_dim]
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        batch_size = embeddings1.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Create positive and negative masks
        positive_mask = torch.eye(batch_size, device=embeddings1.device)
        negative_mask = 1 - positive_mask

        # Compute positive similarities (diagonal elements)
        positive_similarities = torch.diag(similarity_matrix)

        # Compute negative similarities (off-diagonal elements)
        negative_similarities = similarity_matrix * negative_mask

        # InfoNCE loss
        numerator = torch.exp(positive_similarities)
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)

        loss = -torch.mean(torch.log(numerator / (denominator + 1e-8)))

        return loss


class CrossModalConsistencyLoss(nn.Module):
    """Cross-modal consistency loss for ECG-PPG alignment."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, ecg_embeddings: torch.Tensor, ppg_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encourage ECG and PPG embeddings from the same heartbeat to be similar.
        """
        # Normalize embeddings
        ecg_norm = F.normalize(ecg_embeddings, dim=1)
        ppg_norm = F.normalize(ppg_embeddings, dim=1)

        # Compute cosine similarity
        similarities = self.cosine_sim(ecg_norm, ppg_norm)

        # Convert to positive similarities and apply temperature
        positive_similarities = (similarities + 1) / 2  # Scale to [0, 1]
        scaled_similarities = positive_similarities / self.temperature

        # Maximize similarity (minimize negative log likelihood)
        loss = -torch.mean(torch.log(scaled_similarities + 1e-8))

        return loss


class PatternDiscovery:
    """Pattern discovery pipeline using clustering on learned representations."""

    def __init__(self,
                 cluster_methods: List[str] = ['kmeans', 'dbscan'],
                 n_clusters_range: Tuple[int, int] = (3, 15),
                 feature_dim: int = 256):

        self.cluster_methods = cluster_methods
        self.n_clusters_range = n_clusters_range
        self.feature_dim = feature_dim
        self.best_clustering = None
        self.best_score = -1

    def discover_patterns(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Discover natural patterns in the embedding space.

        Args:
            embeddings: Learned embeddings [n_samples, embedding_dim]

        Returns:
            Dictionary with clustering results and analysis
        """
        results = {
            'optimal_clusters': None,
            'cluster_labels': None,
            'silhouette_score': -1,
            'cluster_centers': None,
            'cluster_analysis': {}
        }

        best_score = -1
        best_labels = None
        best_n_clusters = None

        # Try different numbers of clusters
        for n_clusters in range(*self.n_clusters_range):
            # K-means clustering
            if 'kmeans' in self.cluster_methods:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)

                    # Evaluate clustering quality
                    if len(set(labels)) > 1:  # Need at least 2 clusters
                        score = silhouette_score(embeddings, labels)

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_n_clusters = n_clusters
                            results['cluster_centers'] = kmeans.cluster_centers_

                except Exception as e:
                    logger.warning(f"K-means with {n_clusters} clusters failed: {e}")

        # Try DBSCAN for automatic cluster detection
        if 'dbscan' in self.cluster_methods:
            try:
                # Try different epsilon values
                for eps in [0.3, 0.5, 0.7, 1.0]:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    labels = dbscan.fit_predict(embeddings)

                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

                    if n_clusters_found >= 2 and n_clusters_found <= 20:
                        score = silhouette_score(embeddings, labels)

                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_n_clusters = n_clusters_found

            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {e}")

        # Store best results
        if best_labels is not None:
            results['optimal_clusters'] = best_n_clusters
            results['cluster_labels'] = best_labels
            results['silhouette_score'] = best_score
            results['cluster_analysis'] = self._analyze_clusters(embeddings, best_labels)

            logger.info(f"Discovered {best_n_clusters} optimal clusters with silhouette score {best_score:.3f}")

        return results

    def _analyze_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze discovered clusters."""
        unique_labels = np.unique(labels)
        analysis = {
            'cluster_sizes': {},
            'cluster_stats': {},
            'separation_quality': {}
        }

        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue

            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]

            analysis['cluster_sizes'][int(label)] = int(np.sum(cluster_mask))

            # Basic statistics
            analysis['cluster_stats'][int(label)] = {
                'mean': cluster_embeddings.mean(axis=0).tolist(),
                'std': cluster_embeddings.std(axis=0).tolist(),
                'size': int(len(cluster_embeddings))
            }

        return analysis


def create_self_supervised_dataloader(ecg_signals: List[np.ndarray],
                                     ppg_signals: List[np.ndarray],
                                     patient_ids: Optional[List[str]] = None,
                                     batch_size: int = BATCH_SIZE,
                                     **dataset_kwargs) -> DataLoader:
    """Create DataLoader for self-supervised learning."""

    dataset = SelfSupervisedDataset(
        ecg_signals=ecg_signals,
        ppg_signals=ppg_signals,
        patient_ids=patient_ids,
        **dataset_kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )