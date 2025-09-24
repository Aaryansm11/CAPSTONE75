# src/config.py
import os
from pathlib import Path
import torch

# Project structure
ROOT = Path(__file__).parent.parent.absolute()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_PROCESSED, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Signal processing parameters
ECG_FS = 360  # ECG sampling rate (Hz)
PPG_FS = 125  # PPG sampling rate (Hz)
WINDOW_SECONDS = 10  # Window size for segmentation
ECG_WINDOW = ECG_FS * WINDOW_SECONDS
PPG_WINDOW = PPG_FS * WINDOW_SECONDS

# Signal filtering parameters
ECG_LOW_FREQ = 0.5
ECG_HIGH_FREQ = 40.0
PPG_LOW_FREQ = 0.5
PPG_HIGH_FREQ = 10.0

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 8
NUM_WORKERS = 4
SEED = 42

# Model parameters
CLASS_NAMES = ["N", "S", "V", "F", "U"]  # Normal, Supraventricular, Ventricular, Fusion, Unknown
N_ARRHYTHMIA_CLASSES = len(CLASS_NAMES)
STROKE_OUTPUT_DIM = 1  # Single regression output for stroke risk

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Data augmentation parameters
AUGMENTATION_PROB = 0.3
NOISE_FACTOR = 0.02
STRETCH_FACTOR = 0.1

# Cross-validation parameters
CV_FOLDS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# SMOTE parameters
SMOTE_K_NEIGHBORS = 5
SMOTE_RANDOM_STATE = SEED

# Explainability parameters
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAINER_SIZE = 200

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 8081
API_WORKERS = 1

# Report parameters
REPORT_TEMPLATE_DIR = ROOT / "templates"
REPORT_ASSETS_DIR = ROOT / "assets"

# File paths for datasets
MIT_BIH_PATH = DATA_RAW / "mit-bih-arrhythmia-database-1.0.0"
PTB_PATH = DATA_RAW / "ptb-diagnostic-ecg-database-1.0.0"
PHYSIONET_AF_PATH = DATA_RAW / "physionet-af-database"

# Model checkpoint paths
BEST_MODEL_PATH = MODEL_DIR / "best_multimodal_model.pth"
ARRHYTHMIA_MODEL_PATH = MODEL_DIR / "arrhythmia_classifier.pth"
STROKE_MODEL_PATH = MODEL_DIR / "stroke_predictor.pth"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Feature extraction parameters
EXTRACT_TIME_DOMAIN = True
EXTRACT_FREQUENCY_DOMAIN = True
EXTRACT_MORPHOLOGICAL = True
EXTRACT_HRV = True

# Time-domain features
TIME_FEATURES = [
    "mean", "std", "min", "max", "range", "median", "skewness", 
    "kurtosis", "rms", "var", "energy", "zero_crossings"
]

# Frequency-domain features  
FREQ_FEATURES = [
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_flux", "mfcc", "dominant_frequency", "power_bands"
]

# HRV features
HRV_FEATURES = [
    "rr_mean", "rr_std", "rr_rmssd", "rr_pnn50", "hr_mean", "hr_std"
]

# Model architecture parameters
ECG_CNN_FILTERS = [32, 64, 64, 128]
PPG_CNN_FILTERS = [16, 32, 32, 64]
ECG_LSTM_HIDDEN = 128
PPG_LSTM_HIDDEN = 64
FUSION_DIM = 256
DROPOUT_RATE = 0.3
ATTENTION_DIM = 128

# Performance thresholds
MIN_ACCURACY = 0.85
MIN_SENSITIVITY = 0.90
MIN_SPECIFICITY = 0.85
MIN_F1_SCORE = 0.80

# Data quality thresholds
MIN_SIGNAL_LENGTH = 5  # minimum seconds
MAX_NOISE_RATIO = 0.3
MIN_SNR_DB = 10

# Clinical thresholds
STROKE_RISK_LOW = 0.3
STROKE_RISK_MODERATE = 0.6
STROKE_RISK_HIGH = 0.8

AF_RISK_LOW = 0.2
AF_RISK_MODERATE = 0.5
AF_RISK_HIGH = 0.8