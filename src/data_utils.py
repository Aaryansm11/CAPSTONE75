# src/data_utils.py
"""
Data utilities for ECG + PPG pipeline.

Provides:
 - SignalProcessor: preprocessing helpers (filtering, resampling, R-peak detection, quality checks)
 - FeatureExtractor: time/freq/HRV/morphological feature extraction
 - DatasetLoader: high-level record loading + segmentation (annotation-based or sliding windows)
 - DataBalancer: SMOTE-based feature-space balancing utilities
 - align_and_segment_multimodal: convenience function to create paired windows from raw ECG+PPG

Also exposes top-level helper functions:
 - load_mit_bih_records
 - extract_windows_from_record
 - bandpass_filter
 - resample_signal
 - smote_on_concatenated_features

These helpers make imports in 1.py and other entrypoints straightforward.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from scipy.signal import butter, filtfilt, resample, find_peaks, detrend
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Import configuration constants
from src.config import (
    ECG_FS, PPG_FS, ECG_WINDOW, PPG_WINDOW, ECG_LOW_FREQ, ECG_HIGH_FREQ,
    PPG_LOW_FREQ, PPG_HIGH_FREQ, WINDOW_SECONDS, MIT_BIH_PATH,
    TIME_FEATURES, FREQ_FEATURES, HRV_FEATURES, MIN_SIGNAL_LENGTH,
    MAX_NOISE_RATIO, MIN_SNR_DB, SMOTE_K_NEIGHBORS, SMOTE_RANDOM_STATE
)

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Signal processing utilities for ECG and PPG signals."""

    def __init__(self):
        self.ecg_scaler = StandardScaler()
        self.ppg_scaler = StandardScaler()

    def bandpass_filter(self, signal_data: np.ndarray, low_freq: float,
                        high_freq: float, fs: int, order: int = 4) -> np.ndarray:
        """Apply bandpass filter to 1D signal and return float32 array."""
        if len(signal_data) < 4:
            return signal_data.astype(np.float32)

        nyquist = 0.5 * fs
        low = max(low_freq / nyquist, 1e-5)
        high = min(high_freq / nyquist, 0.99999)
        if low >= high:
            # fallback: small band around low_freq
            low = max(low - 1e-3, 1e-5)
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_data)
        return filtered.astype(np.float32)

    def notch_filter(self, signal_data: np.ndarray, notch_freq: float,
                     fs: int, quality_factor: float = 30.0) -> np.ndarray:
        """Apply notch filter (IIR) to remove line noise."""
        try:
            b, a = signal.iirnotch(notch_freq, quality_factor, fs)
            filtered = filtfilt(b, a, signal_data)
            return filtered.astype(np.float32)
        except Exception:
            # If iirnotch unavailable or fails, return original
            return signal_data.astype(np.float32)

    def remove_baseline_wander(self, signal_data: np.ndarray,
                               fs: int, cutoff: float = 0.5) -> np.ndarray:
        """High-pass filter to remove baseline wander."""
        if len(signal_data) < 4:
            return signal_data.astype(np.float32)
        nyquist = 0.5 * fs
        normal_cutoff = max(min(cutoff / nyquist, 0.99), 1e-5)
        b, a = butter(3, normal_cutoff, btype='high')
        filtered = filtfilt(b, a, signal_data)
        return filtered.astype(np.float32)

    def normalize_signal(self, signal_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize signal using zscore, minmax or robust (median/MAD)."""
        if len(signal_data) == 0:
            return signal_data.astype(np.float32)

        if method == 'zscore':
            mean_val = float(np.mean(signal_data))
            std_val = float(np.std(signal_data)) + 1e-8
            normalized = (signal_data - mean_val) / std_val
        elif method == 'minmax':
            min_val = float(np.min(signal_data))
            max_val = float(np.max(signal_data))
            normalized = (signal_data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median_val = float(np.median(signal_data))
            mad = float(np.median(np.abs(signal_data - median_val))) + 1e-8
            normalized = (signal_data - median_val) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized.astype(np.float32)

    def preprocess_ecg(self, ecg_signal: np.ndarray, fs: int = ECG_FS) -> np.ndarray:
        """Full ECG preprocessing pipeline (detrend, baseline removal, notch, bandpass, normalize)."""
        if ecg_signal is None or len(ecg_signal) == 0:
            return np.array([], dtype=np.float32)

        ecg = detrend(ecg_signal, type='constant')
        ecg = self.remove_baseline_wander(ecg, fs, cutoff=0.5)
        # try both 50 and 60 Hz to conservatively remove line noise if present
        ecg = self.notch_filter(ecg, 50.0, fs)
        ecg = self.notch_filter(ecg, 60.0, fs)
        ecg = self.bandpass_filter(ecg, ECG_LOW_FREQ, ECG_HIGH_FREQ, fs)
        ecg = self.normalize_signal(ecg, method='zscore')
        return ecg

    def preprocess_ppg(self, ppg_signal: np.ndarray, fs: int = PPG_FS) -> np.ndarray:
        """Full PPG preprocessing pipeline."""
        if ppg_signal is None or len(ppg_signal) == 0:
            return np.array([], dtype=np.float32)

        ppg = detrend(ppg_signal, type='constant')
        ppg = self.remove_baseline_wander(ppg, fs, cutoff=0.1)
        ppg = self.bandpass_filter(ppg, PPG_LOW_FREQ, PPG_HIGH_FREQ, fs)
        ppg = self.normalize_signal(ppg, method='zscore')
        return ppg

    def resample_signal(self, signal_data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
        """Resample 1D array from orig_fs to target_fs."""
        if signal_data is None or len(signal_data) == 0 or orig_fs == target_fs:
            return signal_data.astype(np.float32) if signal_data is not None else np.array([], dtype=np.float32)
        num_samples = int(len(signal_data) * (target_fs / orig_fs))
        if num_samples < 1:
            return np.array([], dtype=np.float32)
        out = resample(signal_data, num_samples)
        return out.astype(np.float32)

    def detect_r_peaks(self, ecg_signal: np.ndarray, fs: int = ECG_FS) -> np.ndarray:
        """Basic R-peak detection (peak height & distance)."""
        if ecg_signal is None or len(ecg_signal) < int(0.2 * fs):
            return np.array([], dtype=int)
        # Use absolute signal for peak prominence
        abs_sig = np.abs(ecg_signal)
        height_threshold = max(0.15 * np.max(abs_sig), 1e-3)
        min_distance = int(0.3 * fs)  # allow up to ~200 bpm minimum spacing
        peaks, _ = find_peaks(abs_sig, height=height_threshold, distance=min_distance)
        return peaks.astype(int)

    def calculate_heart_rate(self, r_peaks: np.ndarray, fs: int = ECG_FS) -> float:
        """Compute heart rate (BPM) from R-peaks positions (samples)."""
        if r_peaks is None or len(r_peaks) < 2:
            return 0.0
        rr = np.diff(r_peaks) / float(fs)
        mean_rr = np.mean(rr)
        return float(60.0 / mean_rr) if mean_rr > 0 else 0.0

    def assess_signal_quality(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Assess basic signal quality (SNR estimate, baseline drift, artifact ratio)."""
        if signal_data is None or len(signal_data) == 0:
            return {'snr_db': -np.inf, 'baseline_drift': np.inf, 'artifact_ratio': 1.0, 'is_good_quality': False}

        signal_power = float(np.mean(signal_data ** 2))
        noise_estimate = float(np.var(np.diff(signal_data)))
        snr_db = 10.0 * np.log10(max(signal_power / (noise_estimate + 1e-12), 1e-12))
        baseline_drift = float(np.std(signal.detrend(signal_data, type='linear')))
        artifact_ratio = float(np.sum(np.abs(signal_data) > 5 * np.std(signal_data)) / len(signal_data))
        is_good = (snr_db > MIN_SNR_DB) and (artifact_ratio < MAX_NOISE_RATIO)
        return {'snr_db': snr_db, 'baseline_drift': baseline_drift, 'artifact_ratio': artifact_ratio, 'is_good_quality': is_good}


class FeatureExtractor:
    """Extract features from ECG and PPG windows (time, frequency, HRV, morphology)."""

    def __init__(self):
        self.processor = SignalProcessor()

    def extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if signal_data is None or len(signal_data) == 0:
            return {feature: 0.0 for feature in TIME_FEATURES}

        features['mean'] = float(np.mean(signal_data))
        features['std'] = float(np.std(signal_data))
        features['min'] = float(np.min(signal_data))
        features['max'] = float(np.max(signal_data))
        features['range'] = float(features['max'] - features['min'])
        features['median'] = float(np.median(signal_data))
        features['skewness'] = float(skew(signal_data))
        features['kurtosis'] = float(kurtosis(signal_data))
        features['rms'] = float(np.sqrt(np.mean(signal_data ** 2)))
        features['var'] = float(np.var(signal_data))
        features['energy'] = float(np.sum(signal_data ** 2))
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossings'] = float(len(zero_crossings))
        return features

    def extract_frequency_domain_features(self, signal_data: np.ndarray, fs: int = ECG_FS) -> Dict[str, float]:
        features: Dict[str, float] = {}
        if signal_data is None or len(signal_data) < 4:
            return {feature: 0.0 for feature in FREQ_FEATURES}

        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, max(8, len(signal_data) // 4)))
        total_power = float(np.sum(psd) + 1e-12)
        features['spectral_centroid'] = float(np.sum(freqs * psd) / total_power)
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power))
        cumulative = np.cumsum(psd)
        roll_idx = np.where(cumulative >= 0.95 * cumulative[-1])[0]
        features['spectral_rolloff'] = float(freqs[roll_idx[0]] if len(roll_idx) > 0 else 0.0)

        # Spectral flux - measure of spectral change rate
        if len(psd) > 1:
            psd_diff = np.diff(psd)
            features['spectral_flux'] = float(np.sqrt(np.sum(psd_diff ** 2)))
        else:
            features['spectral_flux'] = 0.0

        # MFCC - simplified version (first coefficient)
        # For cardiac signals, we use a simple log-power approximation
        if len(psd) > 0:
            log_psd = np.log(psd + 1e-12)
            features['mfcc'] = float(np.mean(log_psd))
        else:
            features['mfcc'] = 0.0

        dom_idx = int(np.argmax(psd))
        features['dominant_frequency'] = float(freqs[dom_idx])

        # Power bands - clinical frequency bands for cardiac signals
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)  # Low frequency
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)   # High frequency
        vlf_mask = freqs <= 0.04                      # Very low frequency

        lf_power = float(np.sum(psd[lf_mask]))
        hf_power = float(np.sum(psd[hf_mask]))
        vlf_power = float(np.sum(psd[vlf_mask]))

        features['lf_power'] = lf_power / (total_power + 1e-12)
        features['hf_power'] = hf_power / (total_power + 1e-12)
        features['lf_hf_ratio'] = lf_power / (hf_power + 1e-12)
        features['power_bands'] = (vlf_power + lf_power + hf_power) / (total_power + 1e-12)

        return features

    def extract_hrv_features(self, ecg_signal: np.ndarray, fs: int = ECG_FS) -> Dict[str, float]:
        if ecg_signal is None or len(ecg_signal) == 0:
            return {f: 0.0 for f in HRV_FEATURES}
        r_peaks = self.processor.detect_r_peaks(ecg_signal, fs)
        if len(r_peaks) < 2:
            return {f: 0.0 for f in HRV_FEATURES}
        rr_intervals = np.diff(r_peaks) / float(fs) * 1000.0
        rr_diffs = np.diff(rr_intervals) if len(rr_intervals) > 1 else np.array([0.0])
        features = {
            'rr_mean': float(np.mean(rr_intervals)),
            'rr_std': float(np.std(rr_intervals)),
            'rr_rmssd': float(np.sqrt(np.mean((rr_diffs) ** 2))) if len(rr_diffs) > 0 else 0.0,
            'rr_pnn50': float(np.sum(np.abs(rr_diffs) > 50) / len(rr_diffs) * 100) if len(rr_diffs) > 0 else 0.0,
            'hr_mean': float(np.mean(60000.0 / rr_intervals)) if len(rr_intervals) > 0 else 0.0,
            'hr_std': float(np.std(60000.0 / rr_intervals)) if len(rr_intervals) > 0 else 0.0
        }
        return features

    def extract_morphological_features(self, ecg_signal: np.ndarray, fs: int = ECG_FS) -> Dict[str, float]:
        if ecg_signal is None or len(ecg_signal) == 0:
            return {'qrs_width': 0.0, 'p_wave_amplitude': 0.0, 't_wave_amplitude': 0.0}
        r_peaks = self.processor.detect_r_peaks(ecg_signal, fs)
        if len(r_peaks) == 0:
            return {'qrs_width': 0.0, 'p_wave_amplitude': 0.0, 't_wave_amplitude': 0.0}
        qrs_widths = []
        for peak in r_peaks:
            start = max(0, peak - int(0.05 * fs))
            end = min(len(ecg_signal), peak + int(0.05 * fs))
            qrs_segment = ecg_signal[start:end]
            qrs_widths.append(len(qrs_segment) / float(fs) * 1000.0)
        p_amp = np.mean([ecg_signal[max(0, p - int(0.2 * fs))] for p in r_peaks]) if len(r_peaks) > 0 else 0.0
        t_amp = np.mean([ecg_signal[min(len(ecg_signal) - 1, p + int(0.3 * fs))] for p in r_peaks]) if len(r_peaks) > 0 else 0.0
        return {'qrs_width': float(np.mean(qrs_widths)) if qrs_widths else 0.0, 'p_wave_amplitude': float(p_amp), 't_wave_amplitude': float(t_amp)}

    def extract_all_features(self, ecg_signal: np.ndarray, ppg_signal: Optional[np.ndarray] = None,
                             ecg_fs: int = ECG_FS, ppg_fs: int = PPG_FS) -> Dict[str, float]:
        all_features: Dict[str, float] = {}
        if ecg_signal is not None and len(ecg_signal) > 0:
            t = self.extract_time_domain_features(ecg_signal)
            f = self.extract_frequency_domain_features(ecg_signal, ecg_fs)
            h = self.extract_hrv_features(ecg_signal, ecg_fs)
            m = self.extract_morphological_features(ecg_signal, ecg_fs)
            for k, v in {**t, **f, **h, **m}.items():
                all_features[f'ecg_{k}'] = v
        if ppg_signal is not None and len(ppg_signal) > 0:
            t = self.extract_time_domain_features(ppg_signal)
            f = self.extract_frequency_domain_features(ppg_signal, ppg_fs)
            for k, v in {**t, **f}.items():
                all_features[f'ppg_{k}'] = v
        return all_features


class DatasetLoader:
    """Load and preprocess datasets (MIT-BIH style)."""

    def __init__(self):
        self.processor = SignalProcessor()
        self.feature_extractor = FeatureExtractor()
        self.arrhythmia_mapping = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
            'A': 1, 'a': 1, 'S': 1, 'J': 1,          # Supraventricular
            'V': 2, 'E': 2,                          # Ventricular
            'F': 3,                                  # Fusion
            '/': 4, 'f': 4, 'Q': 4, '?': 4           # Unknown/Other
        }

    def load_mit_bih_records(self, data_path: Union[str, Path] = MIT_BIH_PATH) -> List[str]:
        """Return list of record names from MIT-BIH `RECORDS` file."""
        data_path = Path(data_path)
        records_file = data_path / "RECORDS"
        if not records_file.exists():
            logger.warning("RECORDS file not found at %s", records_file)
            return []
        with open(records_file, 'r') as fh:
            records = [line.strip() for line in fh if line.strip()]
        logger.info("Found %d records in %s", len(records), records_file)
        return records

    def load_record_data(self, record_path: str) -> Tuple[np.ndarray, dict, np.ndarray, List[str]]:
        """Load signals and annotations from a WFDB record path (without extension)."""
        try:
            signals, fields = wfdb.rdsamp(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            return signals, fields, np.array(ann.sample, dtype=int), list(ann.symbol)
        except Exception as e:
            logger.error("Error loading record %s: %s", record_path, e)
            return np.array([]), {}, np.array([]), []

    def segment_signals(self, ecg_signal: np.ndarray, ppg_signal: Optional[np.ndarray] = None,
                        ecg_annotations: Optional[np.ndarray] = None,
                        annotation_symbols: Optional[List[str]] = None,
                        window_size: int = ECG_WINDOW, overlap: float = 0.5,
                        ecg_fs: int = ECG_FS, ppg_fs: int = PPG_FS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment ECG/PPG either by annotations or sliding windows."""
        if ecg_annotations is not None and annotation_symbols is not None:
            return self._segment_with_annotations(ecg_signal, ppg_signal, ecg_annotations, annotation_symbols, window_size, ecg_fs, ppg_fs)
        else:
            return self._segment_sliding_window(ecg_signal, ppg_signal, window_size, overlap, ecg_fs, ppg_fs)

    def _segment_with_annotations(self, ecg_signal: np.ndarray, ppg_signal: Optional[np.ndarray],
                                  ecg_annotations: np.ndarray, annotation_symbols: List[str],
                                  window_size: int, ecg_fs: int, ppg_fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ecg_windows, ppg_windows, labels = [], [], []
        for ann_sample, sym in zip(ecg_annotations, annotation_symbols):
            if sym not in self.arrhythmia_mapping:
                continue
            start = max(0, int(ann_sample - window_size // 2))
            end = start + window_size
            if end > len(ecg_signal):
                continue
            ecg_w = ecg_signal[start:end] if ecg_signal.ndim == 1 else ecg_signal[start:end, 0]
            ecg_windows.append(ecg_w.astype(np.float32))
            if ppg_signal is not None:
                ppg_res = self.processor.resample_signal(ppg_signal, ppg_fs, ecg_fs)
                if len(ppg_res) >= end:
                    ppg_w = ppg_res[start:end]
                else:
                    ppg_w = np.zeros(window_size, dtype=np.float32)
                ppg_windows.append(ppg_w.astype(np.float32))
            labels.append(self.arrhythmia_mapping[sym])
        if len(ecg_windows) == 0:
            return np.zeros((0, window_size), dtype=np.float32), np.zeros((0, window_size), dtype=np.float32), np.array([], dtype=int)
        return np.stack(ecg_windows, axis=0), np.stack(ppg_windows, axis=0) if ppg_windows else np.zeros((len(ecg_windows), window_size), dtype=np.float32), np.array(labels, dtype=int)

    def _segment_sliding_window(self, ecg_signal: np.ndarray, ppg_signal: Optional[np.ndarray],
                                window_size: int, overlap: float, ecg_fs: int, ppg_fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        step = max(1, int(window_size * (1.0 - overlap)))
        if ecg_signal.ndim > 1:
            ecg_vec = ecg_signal[:, 0]
        else:
            ecg_vec = ecg_signal
        ecg_windows, ppg_windows = [], []
        for start in range(0, len(ecg_vec) - window_size + 1, step):
            end = start + window_size
            ecg_w = ecg_vec[start:end].astype(np.float32)
            ecg_windows.append(ecg_w)
            if ppg_signal is not None:
                ppg_start = int(start * ppg_fs / ecg_fs)
                ppg_end = int(end * ppg_fs / ecg_fs)
                if ppg_end <= len(ppg_signal):
                    ppg_w = ppg_signal[ppg_start:ppg_end]
                    ppg_w = self.processor.resample_signal(ppg_w, ppg_fs, ecg_fs)
                    if len(ppg_w) != window_size:
                        ppg_w = np.resize(ppg_w, window_size)
                else:
                    ppg_w = np.zeros(window_size, dtype=np.float32)
                ppg_windows.append(ppg_w.astype(np.float32))
        if len(ecg_windows) == 0:
            return np.zeros((0, window_size), dtype=np.float32), np.zeros((0, window_size), dtype=np.float32), np.array([], dtype=int)
        ecg_arr = np.stack(ecg_windows, axis=0)
        ppg_arr = np.stack(ppg_windows, axis=0) if ppg_signal is not None and len(ppg_windows) > 0 else np.zeros((len(ecg_arr), window_size), dtype=np.float32)
        labels = np.zeros(len(ecg_arr), dtype=int)
        return ecg_arr, ppg_arr, labels


class DataBalancer:
    """Balance datasets via SMOTE on extracted features (not raw waveforms)."""

    def __init__(self, random_state: int = SMOTE_RANDOM_STATE):
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor()

    def extract_features_for_smote(self, ecg_windows: np.ndarray, ppg_windows: np.ndarray) -> np.ndarray:
        features = []
        for ecg_w, ppg_w in zip(ecg_windows, ppg_windows):
            ecg_feats = self.feature_extractor.extract_time_domain_features(ecg_w)
            ppg_feats = self.feature_extractor.extract_time_domain_features(ppg_w)
            combined = list(ecg_feats.values()) + list(ppg_feats.values())
            features.append(combined)
        return np.array(features, dtype=np.float32)

    def apply_smote(self, X_ecg: np.ndarray, X_ppg: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply SMOTE to feature space and duplicate nearest windows as synthetic proxies."""
        if len(y) == 0:
            return X_ecg, X_ppg, y
        X_feat = self.extract_features_for_smote(X_ecg, X_ppg)
        # determine k for SMOTE robustly
        min_class_count = np.min(np.bincount(y))
        k = min(SMOTE_K_NEIGHBORS, max(1, int(min_class_count - 1)))
        sm = SMOTE(k_neighbors=k, random_state=self.random_state)
        try:
            X_res, y_res = sm.fit_resample(X_feat, y)
        except Exception as e:
            logger.warning("SMOTE failed: %s", e)
            return X_ecg, X_ppg, y
        n_orig = len(y)
        n_new = len(y_res)
        if n_new == n_orig:
            return X_ecg, X_ppg, y
        # Simple replication strategy: map each resampled feature back to original nearest index
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_feat)
        distances, indices = nbrs.kneighbors(X_res)
        indices = indices.flatten()
        X_ecg_res = X_ecg[indices]
        X_ppg_res = X_ppg[indices]
        return X_ecg_res, X_ppg_res, np.array(y_res, dtype=int)

    def apply_combined_sampling(self, X_ecg: np.ndarray, X_ppg: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply SMOTETomek (SMOTE + Tomek links) strategy."""
        try:
            from imblearn.combine import SMOTETomek
        except Exception:
            logger.warning("imblearn.combine.SMOETTomek not available - falling back to apply_smote")
            return self.apply_smote(X_ecg, X_ppg, y)
        if len(y) == 0:
            return X_ecg, X_ppg, y
        X_feat = self.extract_features_for_smote(X_ecg, X_ppg)
        min_class_count = np.min(np.bincount(y))
        k = min(SMOTE_K_NEIGHBORS, max(1, int(min_class_count - 1)))
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=self.random_state), random_state=self.random_state)
        try:
            X_res, y_res = smt.fit_resample(X_feat, y)
        except Exception as e:
            logger.warning("SMOTETomek failed: %s", e)
            return X_ecg, X_ppg, y
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_feat)
        _, indices = nbrs.kneighbors(X_res)
        indices = indices.flatten()
        X_ecg_res = X_ecg[indices]
        X_ppg_res = X_ppg[indices]
        return X_ecg_res, X_ppg_res, np.array(y_res, dtype=int)


def align_and_segment_multimodal(ecg_signal: np.ndarray, ppg_signal: np.ndarray,
                                 ecg_fs: int = ECG_FS, ppg_fs: int = PPG_FS,
                                 window_seconds: int = WINDOW_SECONDS,
                                 overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to create paired ECG/PPG windows (quality-checked)."""
    proc = SignalProcessor()
    if ecg_signal is None or ppg_signal is None or len(ecg_signal) == 0 or len(ppg_signal) == 0:
        return np.zeros((0, int(window_seconds * ecg_fs))), np.zeros((0, int(window_seconds * ecg_fs)))
    ppg_res = proc.resample_signal(ppg_signal, ppg_fs, ecg_fs)
    window_samples = int(window_seconds * ecg_fs)
    step = max(1, int(window_samples * (1.0 - overlap)))
    min_len = min(len(ecg_signal), len(ppg_res))
    ecg_windows, ppg_windows = [], []
    for start in range(0, min_len - window_samples + 1, step):
        end = start + window_samples
        ecg_w = ecg_signal[start:end].astype(np.float32)
        ppg_w = ppg_res[start:end].astype(np.float32)
        eq = proc.assess_signal_quality(ecg_w)
        pq = proc.assess_signal_quality(ppg_w)
        if eq.get('is_good_quality', False) and pq.get('is_good_quality', False):
            ecg_windows.append(ecg_w)
            ppg_windows.append(ppg_w)
    if len(ecg_windows) == 0:
        logger.warning("align_and_segment_multimodal: no quality windows found")
        return np.zeros((0, window_samples), dtype=np.float32), np.zeros((0, window_samples), dtype=np.float32)
    return np.stack(ecg_windows, axis=0), np.stack(ppg_windows, axis=0)


# -------------------------
# Module-level helper APIs
# -------------------------
def load_mit_bih_records(data_path: Union[str, Path] = MIT_BIH_PATH) -> List[str]:
    """Module-level helper that returns MIT-BIH record names (wraps DatasetLoader)."""
    loader = DatasetLoader()
    return loader.load_mit_bih_records(data_path)


def extract_windows_from_record(record_path: str, lead_idx: int = 0,
                                window_size: int = ECG_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience to extract windows and labels from a single MIT-BIH record using annotations.
    Returns (X_windows, y_labels). If extraction fails returns empty arrays.
    """
    loader = DatasetLoader()
    signals, fields, ann_samples, ann_symbols = loader.load_record_data(record_path)
    if signals is None or signals.size == 0 or ann_samples.size == 0:
        return np.zeros((0, window_size), dtype=np.float32), np.array([], dtype=int)
    ecg_wins, ppg_wins, labels = loader._segment_with_annotations(signals, None, ann_samples, ann_symbols, window_size, ECG_FS, PPG_FS)
    return ecg_wins, labels


def bandpass_filter(signal_data: np.ndarray, low: float, high: float, fs: int) -> np.ndarray:
    """Top-level bandpass helper (wraps SignalProcessor)."""
    proc = SignalProcessor()
    return proc.bandpass_filter(signal_data, low, high, fs)


def resample_signal(signal_data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Top-level resample helper (wraps SignalProcessor)."""
    proc = SignalProcessor()
    return proc.resample_signal(signal_data, orig_fs, target_fs)


def smote_on_concatenated_features(X_ecg: np.ndarray, X_ppg: np.ndarray, y: np.ndarray, random_state: int = SMOTE_RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE on concatenated ECG+PPG basic features and return (features_resampled, y_resampled).
    This is meant for tabular/classical baselines and not for generating raw waveform windows.
    """
    balancer = DataBalancer(random_state=random_state)
    Xf = balancer.extract_features_for_smote(X_ecg, X_ppg)
    if len(y) == 0:
        return Xf, y
    min_count = np.min(np.bincount(y))
    k = min(SMOTE_K_NEIGHBORS, max(1, int(min_count - 1)))
    sm = SMOTE(k_neighbors=k, random_state=random_state)
    try:
        Xr, yr = sm.fit_resample(Xf, y)
        return Xr, yr
    except Exception as e:
        logger.warning("smote_on_concatenated_features failed: %s", e)
        return Xf, y
