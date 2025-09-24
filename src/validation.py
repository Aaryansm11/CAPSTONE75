# src/validation.py
"""
Comprehensive validation and error checking utilities for ECG+PPG analysis system.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from src.config import (
    ECG_FS, PPG_FS, MIN_SIGNAL_LENGTH, MAX_NOISE_RATIO, MIN_SNR_DB,
    CLASS_NAMES, N_ARRHYTHMIA_CLASSES
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SignalValidator:
    """Comprehensive signal validation utilities."""

    def __init__(self):
        self.min_signal_length = MIN_SIGNAL_LENGTH
        self.max_noise_ratio = MAX_NOISE_RATIO
        self.min_snr_db = MIN_SNR_DB

    def validate_signal_basic(self, signal: np.ndarray, signal_type: str = "ECG") -> Dict[str, bool]:
        """Basic signal validation checks."""
        checks = {
            'is_array': isinstance(signal, np.ndarray),
            'not_empty': signal is not None and len(signal) > 0,
            'is_numeric': True,
            'no_inf': True,
            'no_nan': True,
            'reasonable_length': True,
            'reasonable_range': True
        }

        if not checks['is_array'] or not checks['not_empty']:
            return checks

        try:
            # Check for numeric data
            signal_float = signal.astype(float)
            checks['is_numeric'] = True
        except (ValueError, TypeError):
            checks['is_numeric'] = False
            return checks

        # Check for inf/nan values
        checks['no_inf'] = not np.isinf(signal_float).any()
        checks['no_nan'] = not np.isnan(signal_float).any()

        # Check signal length
        min_samples = self.min_signal_length * (ECG_FS if signal_type == "ECG" else PPG_FS)
        checks['reasonable_length'] = len(signal) >= min_samples

        # Check signal range (basic sanity check)
        if signal_type == "ECG":
            # ECG typically ranges from -5mV to +5mV after preprocessing
            checks['reasonable_range'] = np.abs(signal_float).max() < 50.0
        else:  # PPG
            # PPG values should be reasonable after normalization
            checks['reasonable_range'] = np.abs(signal_float).max() < 100.0

        return checks

    def validate_signal_quality(self, signal: np.ndarray, fs: int) -> Dict[str, float]:
        """Advanced signal quality assessment."""
        if signal is None or len(signal) == 0:
            return {
                'snr_db': -np.inf,
                'noise_ratio': 1.0,
                'baseline_stability': -np.inf,
                'artifact_ratio': 1.0,
                'quality_score': 0.0,
                'is_good_quality': False
            }

        try:
            # Signal-to-noise ratio estimate
            signal_power = np.mean(signal ** 2)
            noise_estimate = np.var(np.diff(signal))  # High-frequency noise estimate
            snr_db = 10.0 * np.log10(max(signal_power / (noise_estimate + 1e-12), 1e-12))

            # Baseline stability
            from scipy import signal as scipy_signal
            detrended = scipy_signal.detrend(signal, type='linear')
            baseline_stability = 1.0 / (np.std(detrended) + 1e-6)

            # Artifact detection (outliers beyond 5 standard deviations)
            std_signal = np.std(signal)
            outliers = np.abs(signal) > 5 * std_signal
            artifact_ratio = np.sum(outliers) / len(signal)

            # Noise ratio (high frequency content)
            noise_ratio = noise_estimate / (signal_power + 1e-12)

            # Overall quality score (0-1)
            snr_score = min(max(snr_db / 20.0, 0), 1)  # Normalize SNR
            baseline_score = min(max(baseline_stability / 10.0, 0), 1)
            artifact_score = max(1 - artifact_ratio * 5, 0)
            noise_score = max(1 - noise_ratio * 10, 0)

            quality_score = (snr_score + baseline_score + artifact_score + noise_score) / 4.0

            is_good_quality = (
                snr_db > self.min_snr_db and
                artifact_ratio < self.max_noise_ratio and
                quality_score > 0.5
            )

            return {
                'snr_db': float(snr_db),
                'noise_ratio': float(noise_ratio),
                'baseline_stability': float(baseline_stability),
                'artifact_ratio': float(artifact_ratio),
                'quality_score': float(quality_score),
                'is_good_quality': bool(is_good_quality)
            }

        except Exception as e:
            logger.warning(f"Signal quality assessment failed: {e}")
            return {
                'snr_db': -np.inf,
                'noise_ratio': 1.0,
                'baseline_stability': -np.inf,
                'artifact_ratio': 1.0,
                'quality_score': 0.0,
                'is_good_quality': False
            }

    def validate_paired_signals(self, ecg: np.ndarray, ppg: np.ndarray) -> Dict[str, bool]:
        """Validate ECG+PPG signal pairs."""
        checks = {
            'both_valid': True,
            'length_compatible': True,
            'temporal_alignment': True,
            'cross_correlation': True
        }

        # Basic validation for both signals
        ecg_checks = self.validate_signal_basic(ecg, "ECG")
        ppg_checks = self.validate_signal_basic(ppg, "PPG")

        checks['both_valid'] = all(ecg_checks.values()) and all(ppg_checks.values())

        if not checks['both_valid']:
            return checks

        # Length compatibility (after resampling)
        ecg_duration = len(ecg) / ECG_FS
        ppg_duration = len(ppg) / PPG_FS
        duration_diff = abs(ecg_duration - ppg_duration)
        checks['length_compatible'] = duration_diff < 1.0  # Within 1 second

        # Basic temporal alignment check
        if len(ecg) > 100 and len(ppg) > 100:
            # Resample PPG to ECG rate for comparison
            from scipy.signal import resample
            ppg_resampled = resample(ppg, len(ecg))

            # Cross-correlation to check alignment
            correlation = np.corrcoef(ecg[:min(len(ecg), len(ppg_resampled))],
                                    ppg_resampled[:min(len(ecg), len(ppg_resampled))])[0, 1]
            checks['cross_correlation'] = not np.isnan(correlation) and correlation > 0.1
            checks['temporal_alignment'] = checks['cross_correlation']

        return checks


class DataValidator:
    """Dataset validation utilities."""

    def __init__(self):
        self.signal_validator = SignalValidator()

    def validate_dataset_structure(self, data_path: Union[str, Path]) -> Dict[str, bool]:
        """Validate dataset directory structure."""
        data_path = Path(data_path)

        checks = {
            'path_exists': data_path.exists(),
            'is_directory': data_path.is_dir(),
            'has_files': False,
            'proper_structure': True
        }

        if not checks['path_exists'] or not checks['is_directory']:
            return checks

        # Check for files
        files = list(data_path.rglob('*'))
        checks['has_files'] = len(files) > 0

        return checks

    def validate_batch_data(self, ecg_batch: np.ndarray, ppg_batch: np.ndarray,
                           labels: Optional[np.ndarray] = None) -> Dict[str, bool]:
        """Validate a batch of training data."""
        checks = {
            'consistent_batch_size': True,
            'valid_shapes': True,
            'valid_labels': True,
            'no_corrupted_samples': True
        }

        # Check batch consistency
        if ecg_batch.shape[0] != ppg_batch.shape[0]:
            checks['consistent_batch_size'] = False
            return checks

        # Check shapes
        if len(ecg_batch.shape) != 2 or len(ppg_batch.shape) != 2:
            checks['valid_shapes'] = False
            return checks

        # Check labels if provided
        if labels is not None:
            if len(labels) != ecg_batch.shape[0]:
                checks['valid_labels'] = False
                return checks

            # Check label range
            if isinstance(labels, np.ndarray) and labels.dtype in [np.int32, np.int64]:
                if np.any(labels < 0) or np.any(labels >= N_ARRHYTHMIA_CLASSES):
                    checks['valid_labels'] = False
                    return checks

        # Check for corrupted samples
        corrupted_count = 0
        for i in range(min(10, ecg_batch.shape[0])):  # Sample check
            ecg_quality = self.signal_validator.validate_signal_quality(ecg_batch[i], ECG_FS)
            ppg_quality = self.signal_validator.validate_signal_quality(ppg_batch[i], PPG_FS)

            if not ecg_quality['is_good_quality'] or not ppg_quality['is_good_quality']:
                corrupted_count += 1

        checks['no_corrupted_samples'] = corrupted_count < 3  # Allow some bad samples

        return checks


class ModelValidator:
    """Model validation utilities."""

    def validate_model_output(self, predictions: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate model predictions."""
        checks = {
            'has_required_keys': True,
            'valid_arrhythmia_logits': True,
            'valid_stroke_output': True,
            'consistent_batch_size': True
        }

        required_keys = ['arrhythmia_logits', 'stroke_output']
        for key in required_keys:
            if key not in predictions:
                checks['has_required_keys'] = False
                return checks

        # Check arrhythmia logits
        arrhythmia_logits = predictions['arrhythmia_logits']
        if not isinstance(arrhythmia_logits, np.ndarray):
            checks['valid_arrhythmia_logits'] = False
        elif len(arrhythmia_logits.shape) != 2 or arrhythmia_logits.shape[1] != N_ARRHYTHMIA_CLASSES:
            checks['valid_arrhythmia_logits'] = False
        elif np.isnan(arrhythmia_logits).any() or np.isinf(arrhythmia_logits).any():
            checks['valid_arrhythmia_logits'] = False

        # Check stroke output
        stroke_output = predictions['stroke_output']
        if not isinstance(stroke_output, np.ndarray):
            checks['valid_stroke_output'] = False
        elif len(stroke_output.shape) != 1:
            checks['valid_stroke_output'] = False
        elif np.isnan(stroke_output).any() or np.isinf(stroke_output).any():
            checks['valid_stroke_output'] = False

        # Check batch size consistency
        if checks['valid_arrhythmia_logits'] and checks['valid_stroke_output']:
            if arrhythmia_logits.shape[0] != stroke_output.shape[0]:
                checks['consistent_batch_size'] = False

        return checks


def validate_config():
    """Validate configuration parameters."""
    warnings_list = []
    errors_list = []

    # Check class names consistency
    if len(CLASS_NAMES) != N_ARRHYTHMIA_CLASSES:
        errors_list.append(f"CLASS_NAMES length ({len(CLASS_NAMES)}) doesn't match N_ARRHYTHMIA_CLASSES ({N_ARRHYTHMIA_CLASSES})")

    # Check sampling rates
    if ECG_FS <= 0 or PPG_FS <= 0:
        errors_list.append("Sampling rates must be positive")

    # Check signal processing parameters
    if MIN_SIGNAL_LENGTH <= 0:
        warnings_list.append("MIN_SIGNAL_LENGTH should be positive")

    if MAX_NOISE_RATIO < 0 or MAX_NOISE_RATIO > 1:
        warnings_list.append("MAX_NOISE_RATIO should be between 0 and 1")

    # Log warnings and errors
    for warning in warnings_list:
        logger.warning(f"Config warning: {warning}")

    for error in errors_list:
        logger.error(f"Config error: {error}")

    if errors_list:
        raise ValidationError(f"Configuration validation failed: {errors_list}")

    return len(warnings_list) == 0 and len(errors_list) == 0


def safe_import_check():
    """Check for potential import issues."""
    try:
        import torch
        import numpy as np
        import scipy
        import sklearn
        import matplotlib
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return False


# Validation decorators
def validate_signal_input(func):
    """Decorator to validate signal inputs."""
    def wrapper(signal, *args, **kwargs):
        validator = SignalValidator()
        checks = validator.validate_signal_basic(signal)

        if not all(checks.values()):
            failed_checks = [k for k, v in checks.items() if not v]
            logger.warning(f"Signal validation failed: {failed_checks}")

        return func(signal, *args, **kwargs)
    return wrapper


def validate_model_predictions(func):
    """Decorator to validate model predictions."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, dict):
            validator = ModelValidator()
            checks = validator.validate_model_output(result)

            if not all(checks.values()):
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"Model output validation failed: {failed_checks}")

        return result
    return wrapper