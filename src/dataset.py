# src/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Tuple, Callable, Dict, Any, List
import logging

from src.config import AUGMENTATION_PROB, NOISE_FACTOR, STRETCH_FACTOR

logger = logging.getLogger(__name__)


class SignalAugmentation:
    """Signal augmentation techniques for ECG and PPG data."""
    
    def __init__(self, noise_factor: float = NOISE_FACTOR, 
                 stretch_factor: float = STRETCH_FACTOR,
                 augmentation_prob: float = AUGMENTATION_PROB):
        self.noise_factor = noise_factor
        self.stretch_factor = stretch_factor
        self.augmentation_prob = augmentation_prob
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to signal."""
        if np.random.random() > self.augmentation_prob:
            return signal
            
        noise = np.random.normal(0, self.noise_factor, signal.shape)
        return (signal + noise).astype(np.float32)
    
    def time_stretch(self, signal: np.ndarray) -> np.ndarray:
        """Apply time stretching/compression."""
        if np.random.random() > self.augmentation_prob:
            return signal
            
        # Random stretch factor
        stretch = 1 + np.random.uniform(-self.stretch_factor, self.stretch_factor)
        
        # Simple linear interpolation for stretching
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(signal)-1, len(signal))
        new_indices = np.linspace(0, len(signal)-1, int(len(signal) * stretch))
        
        if len(new_indices) == 0:
            return signal
            
        # Interpolate
        f = interp1d(original_indices, signal, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        stretched_signal = f(new_indices)
        
        # Resize to original length
        if len(stretched_signal) > len(signal):
            # Truncate
            return stretched_signal[:len(signal)].astype(np.float32)
        else:
            # Pad with zeros
            padded = np.zeros(len(signal))
            padded[:len(stretched_signal)] = stretched_signal
            return padded.astype(np.float32)
    
    def amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """Apply random amplitude scaling."""
        if np.random.random() > self.augmentation_prob:
            return signal
            
        scale = np.random.uniform(0.8, 1.2)
        return (signal * scale).astype(np.float32)
    
    def baseline_shift(self, signal: np.ndarray) -> np.ndarray:
        """Apply random baseline shift."""
        if np.random.random() > self.augmentation_prob:
            return signal
            
        shift = np.random.uniform(-0.1, 0.1)
        return (signal + shift).astype(np.float32)
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        signal = self.add_noise(signal)
        signal = self.time_stretch(signal)
        signal = self.amplitude_scale(signal)
        signal = self.baseline_shift(signal)
        return signal


class MultiModalDataset(Dataset):
    """Dataset for paired ECG and PPG signals with arrhythmia and stroke labels."""
    
    def __init__(self, 
                 ecg_windows: np.ndarray,
                 ppg_windows: np.ndarray, 
                 arrhythmia_labels: np.ndarray,
                 stroke_labels: Optional[np.ndarray] = None,
                 patient_ids: Optional[List[str]] = None,
                 transform_ecg: Optional[Callable] = None,
                 transform_ppg: Optional[Callable] = None,
                 augment: bool = False):
        """
        Initialize multimodal dataset.
        
        Args:
            ecg_windows: ECG signal windows (N, seq_len)
            ppg_windows: PPG signal windows (N, seq_len)  
            arrhythmia_labels: Arrhythmia class labels (N,)
            stroke_labels: Stroke risk labels (N,) - optional
            patient_ids: Patient identifiers - optional
            transform_ecg: ECG preprocessing transform
            transform_ppg: PPG preprocessing transform
            augment: Whether to apply data augmentation
        """
        assert len(ecg_windows) == len(ppg_windows) == len(arrhythmia_labels)
        
        self.ecg_windows = ecg_windows.astype(np.float32)
        self.ppg_windows = ppg_windows.astype(np.float32)
        self.arrhythmia_labels = arrhythmia_labels.astype(np.int64)
        
        # Stroke labels (default to zeros if not provided)
        if stroke_labels is not None:
            assert len(stroke_labels) == len(arrhythmia_labels)
            self.stroke_labels = stroke_labels.astype(np.float32)
        else:
            self.stroke_labels = np.zeros(len(arrhythmia_labels), dtype=np.float32)
        
        self.patient_ids = patient_ids if patient_ids else [f"patient_{i}" for i in range(len(ecg_windows))]
        self.transform_ecg = transform_ecg
        self.transform_ppg = transform_ppg
        
        # Setup augmentation
        self.augment = augment
        if augment:
            self.augmenter = SignalAugmentation()
        
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Arrhythmia class distribution: {np.bincount(self.arrhythmia_labels)}")
    
    def __len__(self) -> int:
        return len(self.ecg_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get sample at index.
        
        Returns:
            ecg_tensor: ECG signal tensor (1, seq_len)
            ppg_tensor: PPG signal tensor (1, seq_len)  
            arrhythmia_label: Arrhythmia class label
            stroke_label: Stroke risk label
            patient_id: Patient identifier
        """
        # Get raw signals
        ecg_signal = self.ecg_windows[idx].copy()
        ppg_signal = self.ppg_windows[idx].copy()
        
        # Apply augmentation if enabled (only during training)
        if self.augment:
            ecg_signal = self.augmenter(ecg_signal)
            ppg_signal = self.augmenter(ppg_signal)
        
        # Apply transforms
        if self.transform_ecg:
            ecg_signal = self.transform_ecg(ecg_signal)
        if self.transform_ppg:
            ppg_signal = self.transform_ppg(ppg_signal)
        
        # Convert to tensors and add channel dimension
        ecg_tensor = torch.from_numpy(ecg_signal).unsqueeze(0)  # (1, seq_len)
        ppg_tensor = torch.from_numpy(ppg_signal).unsqueeze(0)  # (1, seq_len)
        arrhythmia_label = torch.tensor(self.arrhythmia_labels[idx], dtype=torch.long)
        stroke_label = torch.tensor(self.stroke_labels[idx], dtype=torch.float)
        
        return ecg_tensor, ppg_tensor, arrhythmia_label, stroke_label, self.patient_ids[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        class_counts = np.bincount(self.arrhythmia_labels)
        total_samples = len(self.arrhythmia_labels)
        
        # Calculate inverse frequency weights
        class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """Get sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        sample_weights = class_weights[self.arrhythmia_labels]
        return sample_weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self),
            'ecg_shape': self.ecg_windows.shape,
            'ppg_shape': self.ppg_windows.shape,
            'arrhythmia_classes': np.unique(self.arrhythmia_labels),
            'class_distribution': np.bincount(self.arrhythmia_labels),
            'stroke_label_range': (self.stroke_labels.min(), self.stroke_labels.max()),
            'ecg_signal_range': (self.ecg_windows.min(), self.ecg_windows.max()),
            'ppg_signal_range': (self.ppg_windows.min(), self.ppg_windows.max())
        }
        return stats


class PatientDataset(Dataset):
    """Dataset for patient-level data with multiple windows per patient."""
    
    def __init__(self,
                 patient_data: Dict[str, Dict[str, Any]],
                 transform_ecg: Optional[Callable] = None,
                 transform_ppg: Optional[Callable] = None):
        """
        Initialize patient dataset.
        
        Args:
            patient_data: Dictionary with patient_id as key and data dict as value
                         Each data dict should contain 'ecg_windows', 'ppg_windows', 
                         'arrhythmia_labels', 'stroke_labels'
        """
        self.patient_data = patient_data
        self.patient_ids = list(patient_data.keys())
        self.transform_ecg = transform_ecg
        self.transform_ppg = transform_ppg
        
        logger.info(f"Patient dataset initialized with {len(self.patient_ids)} patients")
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get patient data at index."""
        patient_id = self.patient_ids[idx]
        data = self.patient_data[patient_id]
        
        # Process all windows for this patient
        ecg_windows = data['ecg_windows']
        ppg_windows = data['ppg_windows']
        
        if self.transform_ecg:
            ecg_windows = np.array([self.transform_ecg(w) for w in ecg_windows])
        if self.transform_ppg:
            ppg_windows = np.array([self.transform_ppg(w) for w in ppg_windows])
        
        return {
            'patient_id': patient_id,
            'ecg_windows': torch.from_numpy(ecg_windows).unsqueeze(1),  # (N, 1, seq_len)
            'ppg_windows': torch.from_numpy(ppg_windows).unsqueeze(1),  # (N, 1, seq_len)
            'arrhythmia_labels': torch.from_numpy(data['arrhythmia_labels']).long(),
            'stroke_labels': torch.from_numpy(data['stroke_labels']).float(),
            'metadata': data.get('metadata', {})
        }


def create_data_loaders(train_dataset: MultiModalDataset,
                       val_dataset: MultiModalDataset,
                       test_dataset: Optional[MultiModalDataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       use_weighted_sampler: bool = True,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Whether to use weighted sampling for training
        pin_memory: Whether to pin memory for GPU training
    
    Returns:
        train_loader, val_loader, test_loader (optional)
    """
    
    # Training loader with optional weighted sampling
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Test loader (if provided)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    if test_loader:
        logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def collate_patient_data(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for patient-level data.
    Handles variable number of windows per patient.
    """
    # Group data by key
    collated = {}
    
    # Patient IDs
    collated['patient_ids'] = [item['patient_id'] for item in batch]
    
    # Variable length sequences - return as list of tensors
    collated['ecg_windows'] = [item['ecg_windows'] for item in batch]
    collated['ppg_windows'] = [item['ppg_windows'] for item in batch]
    collated['arrhythmia_labels'] = [item['arrhythmia_labels'] for item in batch]
    collated['stroke_labels'] = [item['stroke_labels'] for item in batch]
    
    # Metadata
    collated['metadata'] = [item['metadata'] for item in batch]
    
    return collated


class StratifiedDataSplitter:
    """Split data while maintaining class distribution."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, 
                 random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(self, ecg_windows: np.ndarray, ppg_windows: np.ndarray,
              arrhythmia_labels: np.ndarray, stroke_labels: Optional[np.ndarray] = None,
              patient_ids: Optional[List[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            train_data, val_data, test_data dictionaries
        """
        from sklearn.model_selection import train_test_split
        
        # Prepare indices for splitting
        indices = np.arange(len(ecg_windows))
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size,
            stratify=arrhythmia_labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        adjusted_val_size = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=adjusted_val_size,
            stratify=arrhythmia_labels[train_val_idx],
            random_state=self.random_state
        )
        
        # Create data dictionaries
        train_data = {
            'ecg_windows': ecg_windows[train_idx],
            'ppg_windows': ppg_windows[train_idx],
            'arrhythmia_labels': arrhythmia_labels[train_idx],
            'stroke_labels': stroke_labels[train_idx] if stroke_labels is not None else None,
            'patient_ids': [patient_ids[i] for i in train_idx] if patient_ids else None
        }
        
        val_data = {
            'ecg_windows': ecg_windows[val_idx],
            'ppg_windows': ppg_windows[val_idx],
            'arrhythmia_labels': arrhythmia_labels[val_idx],
            'stroke_labels': stroke_labels[val_idx] if stroke_labels is not None else None,
            'patient_ids': [patient_ids[i] for i in val_idx] if patient_ids else None
        }
        
        test_data = {
            'ecg_windows': ecg_windows[test_idx],
            'ppg_windows': ppg_windows[test_idx],
            'arrhythmia_labels': arrhythmia_labels[test_idx],
            'stroke_labels': stroke_labels[test_idx] if stroke_labels is not None else None,
            'patient_ids': [patient_ids[i] for i in test_idx] if patient_ids else None
        }
        
        # Log split statistics
        logger.info("Data split completed:")
        logger.info(f"  Train: {len(train_idx)} samples")
        logger.info(f"  Val: {len(val_idx)} samples")
        logger.info(f"  Test: {len(test_idx)} samples")
        
        for split_name, split_labels in [('Train', arrhythmia_labels[train_idx]),
                                       ('Val', arrhythmia_labels[val_idx]),
                                       ('Test', arrhythmia_labels[test_idx])]:
            class_dist = np.bincount(split_labels)
            logger.info(f"  {split_name} class distribution: {class_dist}")
        
        return train_data, val_data, test_data