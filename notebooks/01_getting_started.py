# notebooks/01_getting_started.py
"""
ECG+PPG Multimodal Analysis - Getting Started Example

This notebook demonstrates how to use the ECG+PPG analysis system for:
1. Loading and preprocessing signals
2. Making predictions
3. Generating explanations
4. Creating reports

Convert to Jupyter notebook: jupytext --to notebook 01_getting_started.py
"""

# %% [markdown]
# # ECG+PPG Multimodal Cardiac Analysis - Getting Started
# 
# This notebook provides a comprehensive introduction to using the ECG+PPG analysis system.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the src directory to Python path
sys.path.append('../')

from src.config import DEVICE, CLASS_NAMES, MODEL_DIR
from src.utils import seed_everything, setup_logging
from src.data_utils import resample_signal, bandpass_filter, extract_basic_features
from src.models import MultiModalCNNLSTM
from src.report import ReportGenerator
from src.utils import load_checkpoint

# Set random seed for reproducibility
seed_everything(42)

print(f"Using device: {DEVICE}")
print(f"Available classes: {CLASS_NAMES}")

# %% [markdown]
# ## 2. Generate Sample ECG and PPG Data

# %%
def generate_sample_ecg(duration=10, fs=360, noise_level=0.1):
    """Generate synthetic ECG signal for demonstration."""
    t = np.arange(0, duration, 1/fs)
    
    # Basic ECG components
    heart_rate = 75  # BPM
    rr_interval = 60 / heart_rate
    
    ecg = np.zeros_like(t)
    
    for beat_start in np.arange(0, duration, rr_interval):
        beat_indices = (t >= beat_start) & (t < beat_start + 0.4)
        beat_time = t[beat_indices] - beat_start
        
        if len(beat_time) > 0:
            # Simplified ECG waveform
            p_wave = 0.1 * np.exp(-(beat_time - 0.05)**2 / 0.001)
            qrs_complex = 1.0 * np.exp(-(beat_time - 0.15)**2 / 0.0001) - 0.3 * np.exp(-(beat_time - 0.13)**2 / 0.00005)
            t_wave = 0.2 * np.exp(-(beat_time - 0.3)**2 / 0.002)
            
            ecg[beat_indices] += p_wave + qrs_complex + t_wave
    
    # Add some noise
    ecg += noise_level * np.random.randn(len(ecg))
    
    return ecg, t

def generate_sample_ppg(duration=10, fs=125, noise_level=0.05):
    """Generate synthetic PPG signal for demonstration."""
    t = np.arange(0, duration, 1/fs)
    
    # Basic PPG components
    heart_rate = 75  # BPM  
    rr_interval = 60 / heart_rate
    
    ppg = np.zeros_like(t)
    
    for beat_start in np.arange(0, duration, rr_interval):
        beat_indices = (t >= beat_start) & (t < beat_start + 0.6)
        beat_time = t[beat_indices] - beat_start
        
        if len(beat_time) > 0:
            # Simplified PPG pulse
            systolic = np.exp(-(beat_time - 0.2)**2 / 0.002)
            dicrotic = 0.3 * np.exp(-(beat_time - 0.4)**2 / 0.003)
            
            ppg[beat_indices] += systolic + dicrotic
    
    # Add baseline and noise
    baseline = 1.0 + 0.1 * np.sin(2 * np.pi * 0.1 * t)  # Respiratory variation
    ppg = baseline + 0.5 * ppg + noise_level * np.random.randn(len(ppg))
    
    return ppg, t

# Generate sample signals
ecg_signal, ecg_time = generate_sample_ecg(duration=10, fs=360)
ppg_signal, ppg_time = generate_sample_ppg(duration=10, fs=125)

print(f"ECG signal shape: {ecg_signal.shape}, duration: {len(ecg_signal)/360:.1f}s")
print(f"PPG signal shape: {ppg_signal.shape}, duration: {len(ppg_signal)/125:.1f}s")

# %% [markdown]
# ## 3. Visualize Raw Signals

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

# Plot ECG
ax1.plot(ecg_time, ecg_signal, 'b-', linewidth=1, label='ECG')
ax1.set_title('Sample ECG Signal', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude (mV)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot PPG  
ax2.plot(ppg_time, ppg_signal, 'r-', linewidth=1, label='PPG')
ax2.set_title('Sample PPG Signal', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Amplitude')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Signal Preprocessing

# %%
# Apply bandpass filtering
ecg_filtered = bandpass_filter(ecg_signal, low=0.5, high=40, fs=360)
ppg_filtered = bandpass_filter(ppg_signal, low=0.5, high=10, fs=125)

# Resample PPG to match ECG sampling rate
ppg_resampled = resample_signal(ppg_filtered, orig_fs=125, target_fs=360)

# Ensure same length
min_length = min(len(ecg_filtered), len(ppg_resampled))
ecg_processed = ecg_filtered[:min_length]
ppg_processed = ppg_resampled[:min_length]

# Normalize signals
ecg_normalized = (ecg_processed - np.mean(ecg_processed)) / (np.std(ecg_processed) + 1e-8)
ppg_normalized = (ppg_processed - np.mean(ppg_processed)) / (np.std(ppg_processed) + 1e-8)

print(f"Processed signals length: {len(ecg_normalized)} samples")
print(f"ECG stats - Mean: {np.mean(ecg_normalized):.4f}, Std: {np.std(ecg_normalized):.4f}")
print(f"PPG stats - Mean: {np.mean(ppg_normalized):.4f}, Std: {np.std(ppg_normalized):.4f}")

# %% [markdown]
# ## 5. Feature Extraction

# %%
# Extract basic features
ecg_features = extract_basic_features(ecg_normalized)
ppg_features = extract_basic_features(ppg_normalized)

feature_names_ecg = ['Mean', 'Std', 'Min', 'Max', 'PtP', 'Energy', 'Variation', 'Median']
feature_names_ppg = ['Mean', 'Std', 'Min', 'Max', 'PtP', 'Energy', 'Variation', 'Median']

# Create feature summary
features_df = pd.DataFrame({
    'ECG_Features': ecg_features,
    'PPG_Features': ppg_features,
    'Feature_Names': feature_names_ecg
}).round(4)

print("Extracted Features:")
print(features_df.to_string(index=False))

# Visualize features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.bar(range(len(ecg_features)), ecg_features, color='blue', alpha=0.7)
ax1.set_title('ECG Features')
ax1.set_xlabel('Feature Index')
ax1.set_ylabel('Value')
ax1.set_xticks(range(len(feature_names_ecg)))
ax1.set_xticklabels(feature_names_ecg, rotation=45)

ax2.bar(range(len(ppg_features)), ppg_features, color='red', alpha=0.7)
ax2.set_title('PPG Features')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Value')
ax2.set_xticks(range(len(feature_names_ppg)))
ax2.set_xticklabels(feature_names_ppg, rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Load Trained Model and Make Predictions

# %%
# Check if model exists
model_path = os.path.join(MODEL_DIR, 'multimodal_best.pth')

if os.path.exists(model_path):
    print("Loading trained model...")
    
    # Load checkpoint
    checkpoint = load_checkpoint(model_path, device=DEVICE)
    
    # Initialize model
    model = MultiModalCNNLSTM(
        seq_len_ecg=len(ecg_normalized),
        seq_len_ppg=len(ppg_normalized),
        n_classes_arr=len(CLASS_NAMES),
        stroke_output_dim=1
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model version: {checkpoint.get('version', 'unknown')}")
    
    # Prepare input tensors
    import torch
    ecg_tensor = torch.from_numpy(ecg_normalized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    ppg_tensor = torch.from_numpy(ppg_normalized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    print(f"Input tensor shapes - ECG: {ecg_tensor.shape}, PPG: {ppg_tensor.shape}")
    
    # Make predictions
    with torch.no_grad():
        outputs = model(ecg_tensor, ppg_tensor)
        
        # Get arrhythmia predictions
        arr_logits = outputs['arr_logits'].cpu().numpy()[0]
        arr_probs = torch.softmax(outputs['arr_logits'], dim=1).cpu().numpy()[0]
        
        # Get stroke risk
        stroke_risk = torch.sigmoid(outputs['stroke']).cpu().numpy()[0]
        
        # Predicted class
        predicted_idx = np.argmax(arr_probs)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = arr_probs[predicted_idx]
    
    print("\n=== PREDICTION RESULTS ===")
    print(f"Predicted Arrhythmia Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Stroke Risk: {stroke_risk:.4f} ({stroke_risk*100:.2f}%)")
    
    print(f"\nClass Probabilities:")
    for class_name, prob in zip(CLASS_NAMES, arr_probs):
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")

else:
    print(f"Model not found at {model_path}")
    print("Please run the training pipeline first: python 1.py")
    
    # Create mock predictions for demonstration
    arr_probs = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    stroke_risk = 0.25
    predicted_class = CLASS_NAMES[0]
    confidence = 0.6
    
    print("\n=== MOCK PREDICTIONS (for demonstration) ===")
    print(f"Predicted Arrhythmia Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Stroke Risk: {stroke_risk:.4f} ({stroke_risk*100:.2f}%)")

# %% [markdown]
# ## 7. Generate Patient Report

# %%
# Prepare data for report generation
patient_id = "DEMO_PATIENT_001"

predictions = {
    'predicted_class': predicted_class,
    'arrhythmia_probs': arr_probs.tolist(),
    'stroke_risk': float(stroke_risk),
    'confidence': float(confidence),
    'class_names': CLASS_NAMES
}

# Mock explanations for demonstration
explanations = {
    'shap_values': [np.random.randn(1, 16) for _ in range(len(CLASS_NAMES))],
    'feature_names': [
        'ECG_mean', 'ECG_std', 'ECG_min', 'ECG_max', 'ECG_ptp', 'ECG_energy', 'ECG_variation', 'ECG_median',
        'PPG_mean', 'PPG_std', 'PPG_min', 'PPG_max', 'PPG_ptp', 'PPG_energy', 'PPG_variation', 'PPG_median'
    ],
    'attention_weights': np.random.rand(8, 80)
}

metadata = {
    'model_version': 'v1.0',
    'ecg_fs': 360,
    'ppg_fs': 360,
    'signal_duration': len(ecg_normalized) / 360,
    'analysis_method': 'CNN-LSTM Multimodal Fusion',
    'preprocessing': 'Bandpass filtering, normalization'
}

# Generate report
try:
    report_generator = ReportGenerator('../reports')
    
    report_paths = report_generator.generate_complete_report(
        patient_id=patient_id,
        predictions=predictions,
        explanations=explanations,
        metadata=metadata,
        ecg_data=ecg_normalized,
        ppg_data=ppg_normalized,
        format_types=['html', 'pdf']
    )
    
    print("\n=== REPORT GENERATED ===")
    for format_type, path in report_paths.items():
        print(f"{format_type.upper()} Report: {path}")
    
except Exception as e:
    print(f"Report generation failed: {e}")
    print("This is normal if dependencies like weasyprint are not installed.")

# %% [markdown]
# ## 8. Risk Stratification Analysis

# %%
# Risk assessment based on predictions
def assess_risk_level(stroke_risk, arrhythmia_class, confidence):
    """Assess overall cardiovascular risk level."""
    
    high_risk_classes = ['V', 'F']  # Ventricular and Fusion beats
    moderate_risk_classes = ['S']   # Supraventricular beats
    
    risk_score = stroke_risk
    
    # Adjust based on arrhythmia type
    if arrhythmia_class in high_risk_classes:
        risk_score += 0.3
    elif arrhythmia_class in moderate_risk_classes:
        risk_score += 0.1
    
    # Adjust based on confidence
    if confidence < 0.7:
        risk_score += 0.1  # Less confident predictions increase risk
    
    risk_score = min(risk_score, 1.0)  # Cap at 1.0
    
    if risk_score > 0.7:
        return "HIGH", risk_score, [
            "Immediate cardiology consultation recommended",
            "Consider continuous cardiac monitoring",
            "Evaluate for anticoagulation therapy",
            "Lifestyle modifications including diet and exercise"
        ]
    elif risk_score > 0.3:
        return "MODERATE", risk_score, [
            "Regular cardiology follow-up recommended",
            "Monitor blood pressure and heart rate",
            "Consider stress testing if symptoms present",
            "Maintain healthy lifestyle habits"
        ]
    else:
        return "LOW", risk_score, [
            "Continue routine cardiac care",
            "Annual cardiovascular screening",
            "Maintain current medications as prescribed",
            "Regular exercise and healthy diet"
        ]

# Assess risk
risk_level, risk_score, recommendations = assess_risk_level(
    stroke_risk, predicted_class, confidence
)

print(f"\n=== RISK ASSESSMENT ===")
print(f"Overall Risk Level: {risk_level}")
print(f"Risk Score: {risk_score:.3f}")
print(f"\nRecommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# %% [markdown]
# ## 9. Summary and Next Steps

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

summary_data = {
    'Patient ID': patient_id,
    'Signal Duration': f"{len(ecg_normalized)/360:.1f} seconds",
    'Predicted Class': predicted_class,
    'Confidence': f"{confidence*100:.1f}%",
    'Stroke Risk': f"{stroke_risk*100:.1f}%",
    'Risk Level': risk_level,
    'Processing Device': DEVICE
}

for key, value in summary_data.items():
    print(f"{key:.<20} {value}")

print(f"\nNext Steps:")
print("1. Review the generated report for detailed analysis")
print("2. Validate predictions with clinical assessment")
print("3. Consider implementing continuous monitoring")
print("4. Schedule appropriate follow-up based on risk level")

print(f"\nFor production use:")
print("- Replace synthetic signals with real patient data")
print("- Validate model performance on your specific population")
print("- Integrate with hospital information systems")
print("- Implement proper clinical workflows")

# %%