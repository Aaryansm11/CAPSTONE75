# src/clinical_explainer.py
"""
Clinical explainability module for arrhythmia-stroke prediction model.
Designed for validation with cardiologists and clinical integration.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

from src.config import CLASS_NAMES, ECG_FS, PPG_FS
from src.data_utils import FeatureExtractor, SignalProcessor

logger = logging.getLogger(__name__)


class ClinicalRiskFeatures:
    """Extract clinically interpretable features for stroke risk assessment."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.signal_processor = SignalProcessor()

        # Clinical feature definitions based on 2024 guidelines
        self.clinical_features = {
            'rhythm_features': [
                'rr_mean', 'rr_std', 'rr_rmssd', 'rr_pnn50',  # HRV features
                'hr_mean', 'hr_std'  # Heart rate features
            ],
            'morphological_features': [
                'qrs_width', 'p_wave_amplitude', 't_wave_amplitude',  # ECG morphology
                'ecg_mean', 'ecg_std', 'ecg_skewness', 'ecg_kurtosis'  # Statistical features
            ],
            'frequency_features': [
                'ecg_lf_power', 'ecg_hf_power', 'ecg_lf_hf_ratio',  # HRV frequency
                'ecg_dominant_frequency', 'ecg_spectral_centroid'  # Spectral features
            ],
            'ppg_features': [
                'ppg_mean', 'ppg_std', 'ppg_skewness',  # PPG morphology
                'ppg_lf_power', 'ppg_hf_power', 'ppg_lf_hf_ratio'  # PPG frequency
            ],
            'regularity_features': [
                'rhythm_regularity', 'beat_consistency', 'morphology_stability'
            ]
        }

    def extract_clinical_features(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive clinical features for stroke risk assessment."""

        # Extract all features using existing extractor
        all_features = self.feature_extractor.extract_all_features(ecg_signal, ppg_signal)

        # Add clinical-specific features
        clinical_features = all_features.copy()

        # Heart rhythm regularity assessment
        r_peaks = self.signal_processor.detect_r_peaks(ecg_signal)
        if len(r_peaks) > 2:
            rr_intervals = np.diff(r_peaks) / float(ECG_FS) * 1000.0  # in ms

            # Rhythm regularity (inverse of coefficient of variation)
            if len(rr_intervals) > 1:
                cv_rr = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
                clinical_features['rhythm_regularity'] = float(1.0 / (cv_rr + 1e-6))

                # Beat consistency (percentage of RR intervals within 20% of mean)
                mean_rr = np.mean(rr_intervals)
                consistent_beats = np.sum(np.abs(rr_intervals - mean_rr) < 0.2 * mean_rr)
                clinical_features['beat_consistency'] = float(consistent_beats / len(rr_intervals))
            else:
                clinical_features['rhythm_regularity'] = 0.0
                clinical_features['beat_consistency'] = 0.0
        else:
            clinical_features['rhythm_regularity'] = 0.0
            clinical_features['beat_consistency'] = 0.0

        # ECG morphology stability
        if len(ecg_signal) > ECG_FS:  # At least 1 second
            segments = np.array_split(ecg_signal, min(5, len(ecg_signal) // ECG_FS))
            segment_means = [np.mean(seg) for seg in segments if len(seg) > 0]
            if len(segment_means) > 1:
                clinical_features['morphology_stability'] = float(1.0 / (np.std(segment_means) + 1e-6))
            else:
                clinical_features['morphology_stability'] = 1.0
        else:
            clinical_features['morphology_stability'] = 1.0

        return clinical_features

    def assess_arrhythmia_risk_factors(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Assess clinical arrhythmia risk factors based on extracted features."""

        risk_assessment = {
            'atrial_fibrillation_risk': 'low',
            'bradycardia_risk': 'low',
            'tachycardia_risk': 'low',
            'rhythm_irregularity_risk': 'low',
            'clinical_flags': [],
            'risk_scores': {}
        }

        # Heart rate assessment
        hr_mean = features.get('hr_mean', 70)
        if hr_mean < 50:
            risk_assessment['bradycardia_risk'] = 'high'
            risk_assessment['clinical_flags'].append('Bradycardia detected (HR < 50 bpm)')
        elif hr_mean < 60:
            risk_assessment['bradycardia_risk'] = 'moderate'
            risk_assessment['clinical_flags'].append('Borderline bradycardia (HR 50-60 bpm)')

        if hr_mean > 120:
            risk_assessment['tachycardia_risk'] = 'high'
            risk_assessment['clinical_flags'].append('Tachycardia detected (HR > 120 bpm)')
        elif hr_mean > 100:
            risk_assessment['tachycardia_risk'] = 'moderate'
            risk_assessment['clinical_flags'].append('Borderline tachycardia (HR 100-120 bpm)')

        # Rhythm irregularity assessment
        rhythm_regularity = features.get('rhythm_regularity', 100)
        rr_pnn50 = features.get('rr_pnn50', 0)

        if rhythm_regularity < 10 or rr_pnn50 > 20:
            risk_assessment['rhythm_irregularity_risk'] = 'high'
            risk_assessment['clinical_flags'].append('Significant rhythm irregularity detected')
        elif rhythm_regularity < 30 or rr_pnn50 > 10:
            risk_assessment['rhythm_irregularity_risk'] = 'moderate'
            risk_assessment['clinical_flags'].append('Mild rhythm irregularity detected')

        # Atrial fibrillation risk based on HRV and irregularity
        hrv_score = 0
        if features.get('rr_rmssd', 0) > 50:
            hrv_score += 1
        if rr_pnn50 > 15:
            hrv_score += 1
        if rhythm_regularity < 20:
            hrv_score += 1
        if features.get('beat_consistency', 1.0) < 0.7:
            hrv_score += 1

        if hrv_score >= 3:
            risk_assessment['atrial_fibrillation_risk'] = 'high'
            risk_assessment['clinical_flags'].append('High atrial fibrillation risk (HRV pattern)')
        elif hrv_score >= 2:
            risk_assessment['atrial_fibrillation_risk'] = 'moderate'
            risk_assessment['clinical_flags'].append('Moderate atrial fibrillation risk')

        # Risk scores (0-1 scale)
        risk_assessment['risk_scores'] = {
            'overall_arrhythmia_risk': min(hrv_score / 4.0, 1.0),
            'af_risk_score': min(hrv_score / 4.0, 1.0),
            'bradycardia_score': max(0, (60 - hr_mean) / 30) if hr_mean < 60 else 0,
            'tachycardia_score': max(0, (hr_mean - 100) / 50) if hr_mean > 100 else 0,
            'irregularity_score': max(0, (30 - rhythm_regularity) / 30) if rhythm_regularity < 30 else 0
        }

        return risk_assessment


class StrokeRiskPredictor:
    """Stroke risk prediction based on arrhythmia patterns and clinical features."""

    def __init__(self):
        self.clinical_extractor = ClinicalRiskFeatures()

        # CHA2DS2-VASc equivalent scoring (for comparison)
        self.traditional_risk_factors = {
            'age_75_plus': 2,
            'age_65_74': 1,
            'heart_failure': 1,
            'hypertension': 1,
            'stroke_tia_history': 2,
            'vascular_disease': 1,
            'diabetes': 1,
            'female_sex': 1  # Note: 2024 guidelines suggest CHA2DS2-VA without sex
        }

    def calculate_stroke_risk_from_arrhythmia(self, features: Dict[str, float],
                                            arrhythmia_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stroke risk based on arrhythmia patterns and clinical features."""

        stroke_risk = {
            'estimated_stroke_risk': 0.0,  # Annual stroke risk percentage
            'risk_category': 'low',
            'contributing_factors': [],
            'confidence_level': 0.0,
            'clinical_recommendations': []
        }

        base_risk = 0.0
        confidence_factors = []

        # Atrial fibrillation contribution (major risk factor)
        af_risk = arrhythmia_assessment['risk_scores']['af_risk_score']
        if af_risk > 0.7:
            base_risk += 4.0  # High AF risk adds ~4% annual stroke risk
            stroke_risk['contributing_factors'].append('High atrial fibrillation risk')
            confidence_factors.append(0.8)
        elif af_risk > 0.4:
            base_risk += 2.0  # Moderate AF risk
            stroke_risk['contributing_factors'].append('Moderate atrial fibrillation risk')
            confidence_factors.append(0.6)

        # Heart rate abnormalities
        hr_mean = features.get('hr_mean', 70)
        if hr_mean < 50 or hr_mean > 120:
            base_risk += 1.5
            stroke_risk['contributing_factors'].append(f'Heart rate abnormality ({hr_mean:.0f} bpm)')
            confidence_factors.append(0.7)

        # Rhythm irregularity
        irregularity_score = arrhythmia_assessment['risk_scores']['irregularity_score']
        if irregularity_score > 0.5:
            base_risk += irregularity_score * 2.0
            stroke_risk['contributing_factors'].append('Rhythm irregularity')
            confidence_factors.append(0.6)

        # HRV-based risk (established predictor)
        rr_rmssd = features.get('rr_rmssd', 30)
        if rr_rmssd > 60:  # Very high HRV can indicate AF
            base_risk += 1.0
            stroke_risk['contributing_factors'].append('Elevated heart rate variability')
            confidence_factors.append(0.5)

        # Beat consistency (specific to AF detection)
        beat_consistency = features.get('beat_consistency', 1.0)
        if beat_consistency < 0.6:
            base_risk += 1.5
            stroke_risk['contributing_factors'].append('Poor beat-to-beat consistency')
            confidence_factors.append(0.7)

        # Final risk calculation
        stroke_risk['estimated_stroke_risk'] = min(base_risk, 15.0)  # Cap at 15% annual risk

        # Risk categorization
        if stroke_risk['estimated_stroke_risk'] < 1.0:
            stroke_risk['risk_category'] = 'low'
        elif stroke_risk['estimated_stroke_risk'] < 3.0:
            stroke_risk['risk_category'] = 'moderate'
        else:
            stroke_risk['risk_category'] = 'high'

        # Confidence level
        if confidence_factors:
            stroke_risk['confidence_level'] = np.mean(confidence_factors)
        else:
            stroke_risk['confidence_level'] = 0.1  # Low confidence if no factors

        # Clinical recommendations
        if stroke_risk['risk_category'] == 'high':
            stroke_risk['clinical_recommendations'].extend([
                'Consider anticoagulation therapy evaluation',
                'Cardiology consultation recommended',
                'Continuous cardiac monitoring consideration'
            ])
        elif stroke_risk['risk_category'] == 'moderate':
            stroke_risk['clinical_recommendations'].extend([
                'Enhanced cardiac monitoring',
                'Risk factor modification',
                'Regular cardiology follow-up'
            ])
        else:
            stroke_risk['clinical_recommendations'].extend([
                'Routine cardiac monitoring',
                'Lifestyle modifications',
                'Annual cardiac risk assessment'
            ])

        return stroke_risk


class ModelExplainer:
    """Explainability module for the trained model predictions."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.clinical_extractor = ClinicalRiskFeatures()
        self.stroke_predictor = StrokeRiskPredictor()

    def explain_prediction(self, ecg_signal: np.ndarray, ppg_signal: np.ndarray,
                          model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Provide comprehensive explanation of model predictions."""

        explanation = {
            'clinical_features': {},
            'arrhythmia_assessment': {},
            'stroke_risk_analysis': {},
            'model_confidence': {},
            'attention_analysis': {},
            'clinical_interpretation': {},
            'validation_metrics': {}
        }

        # Extract clinical features
        clinical_features = self.clinical_extractor.extract_clinical_features(ecg_signal, ppg_signal)
        explanation['clinical_features'] = clinical_features

        # Assess arrhythmia risk
        arrhythmia_assessment = self.clinical_extractor.assess_arrhythmia_risk_factors(clinical_features)
        explanation['arrhythmia_assessment'] = arrhythmia_assessment

        # Calculate stroke risk
        stroke_risk = self.stroke_predictor.calculate_stroke_risk_from_arrhythmia(
            clinical_features, arrhythmia_assessment
        )
        explanation['stroke_risk_analysis'] = stroke_risk

        # Model confidence analysis
        if 'arrhythmia_logits' in model_outputs:
            arrhythmia_probs = torch.softmax(model_outputs['arrhythmia_logits'], dim=-1)
            max_prob = torch.max(arrhythmia_probs).item()
            predicted_class = torch.argmax(arrhythmia_probs).item()

            explanation['model_confidence'] = {
                'arrhythmia_confidence': float(max_prob),
                'predicted_class': CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else 'Unknown',
                'class_probabilities': {
                    CLASS_NAMES[i]: float(arrhythmia_probs[0, i])
                    for i in range(min(len(CLASS_NAMES), arrhythmia_probs.shape[1]))
                }
            }

        if 'stroke_output' in model_outputs:
            stroke_score = torch.sigmoid(model_outputs['stroke_output']).item()
            explanation['model_confidence']['stroke_risk_score'] = float(stroke_score)

        # Attention analysis
        if 'ecg_attention' in model_outputs and model_outputs['ecg_attention'] is not None:
            ecg_attention = model_outputs['ecg_attention'].detach().cpu().numpy()
            explanation['attention_analysis']['ecg_focus_regions'] = self._analyze_attention_weights(
                ecg_attention, 'ECG'
            )

        if 'ppg_attention' in model_outputs and model_outputs['ppg_attention'] is not None:
            ppg_attention = model_outputs['ppg_attention'].detach().cpu().numpy()
            explanation['attention_analysis']['ppg_focus_regions'] = self._analyze_attention_weights(
                ppg_attention, 'PPG'
            )

        # Clinical interpretation
        explanation['clinical_interpretation'] = self._generate_clinical_interpretation(
            explanation['arrhythmia_assessment'],
            explanation['stroke_risk_analysis'],
            explanation['model_confidence']
        )

        # Validation metrics for doctor review
        explanation['validation_metrics'] = self._calculate_validation_metrics(
            clinical_features, explanation['model_confidence']
        )

        return explanation

    def _analyze_attention_weights(self, attention_weights: np.ndarray, signal_type: str) -> Dict[str, Any]:
        """Analyze attention weights to identify clinically relevant regions."""

        if attention_weights.ndim > 2:
            # Average across batch and attention heads if present
            attention_weights = np.mean(attention_weights, axis=tuple(range(attention_weights.ndim - 1)))

        # Find peaks in attention
        from scipy.signal import find_peaks

        # Normalize attention weights
        attention_norm = (attention_weights - np.min(attention_weights)) / (
            np.max(attention_weights) - np.min(attention_weights) + 1e-8
        )

        # Find high attention regions
        high_attention_threshold = np.percentile(attention_norm, 80)
        peaks, _ = find_peaks(attention_norm, height=high_attention_threshold)

        # Time mapping (assuming ECG_FS sampling rate)
        fs = ECG_FS if signal_type == 'ECG' else ECG_FS  # PPG is resampled to ECG rate
        time_points = peaks / fs

        analysis = {
            'peak_indices': peaks.tolist(),
            'peak_times_seconds': time_points.tolist(),
            'peak_attention_values': attention_norm[peaks].tolist(),
            'mean_attention': float(np.mean(attention_norm)),
            'max_attention': float(np.max(attention_norm)),
            'attention_variance': float(np.var(attention_norm))
        }

        # Clinical interpretation of attention regions
        if signal_type == 'ECG':
            analysis['clinical_relevance'] = self._interpret_ecg_attention(time_points, peaks)
        else:
            analysis['clinical_relevance'] = self._interpret_ppg_attention(time_points, peaks)

        return analysis

    def _interpret_ecg_attention(self, time_points: np.ndarray, peak_indices: np.ndarray) -> List[str]:
        """Interpret ECG attention regions for clinical relevance."""
        interpretations = []

        # Basic rhythm analysis based on attention timing
        if len(time_points) > 1:
            intervals = np.diff(time_points)
            mean_interval = np.mean(intervals)

            if mean_interval < 0.5:  # Very frequent attention peaks
                interpretations.append("Model focuses on high-frequency cardiac events")
            elif mean_interval > 2.0:  # Infrequent peaks
                interpretations.append("Model identifies isolated cardiac events")
            else:
                interpretations.append("Model tracks regular cardiac rhythm patterns")

        # Check for clustered attention (potential arrhythmia focus)
        if len(peak_indices) > 2:
            clustering_score = np.std(np.diff(peak_indices)) / np.mean(np.diff(peak_indices))
            if clustering_score > 1.0:
                interpretations.append("Model detects irregular rhythm patterns")
            else:
                interpretations.append("Model focuses on consistent rhythm features")

        return interpretations

    def _interpret_ppg_attention(self, time_points: np.ndarray, peak_indices: np.ndarray) -> List[str]:
        """Interpret PPG attention regions for clinical relevance."""
        interpretations = []

        if len(time_points) > 1:
            intervals = np.diff(time_points)
            mean_interval = np.mean(intervals)

            if mean_interval < 1.0:
                interpretations.append("Model focuses on pulse wave morphology details")
            else:
                interpretations.append("Model tracks overall pulse patterns")

        if len(peak_indices) > 0:
            interpretations.append("Model utilizes peripheral vascular information")

        return interpretations

    def _generate_clinical_interpretation(self, arrhythmia_assessment: Dict[str, Any],
                                        stroke_risk: Dict[str, Any],
                                        model_confidence: Dict[str, Any]) -> Dict[str, str]:
        """Generate clinical interpretation for doctor validation."""

        interpretation = {
            'primary_finding': '',
            'clinical_significance': '',
            'recommended_actions': '',
            'monitoring_recommendations': '',
            'limitations': ''
        }

        # Primary finding
        predicted_class = model_confidence.get('predicted_class', 'Unknown')
        confidence = model_confidence.get('arrhythmia_confidence', 0.0)

        interpretation['primary_finding'] = (
            f"Model predicts {predicted_class} rhythm pattern with {confidence:.1%} confidence. "
            f"Estimated annual stroke risk: {stroke_risk['estimated_stroke_risk']:.1f}% "
            f"({stroke_risk['risk_category']} risk category)."
        )

        # Clinical significance
        significance_parts = []
        for flag in arrhythmia_assessment['clinical_flags']:
            significance_parts.append(flag)

        if stroke_risk['risk_category'] == 'high':
            significance_parts.append("High stroke risk requires immediate clinical attention")
        elif stroke_risk['risk_category'] == 'moderate':
            significance_parts.append("Moderate stroke risk warrants enhanced monitoring")

        interpretation['clinical_significance'] = '; '.join(significance_parts) if significance_parts else "No significant arrhythmia patterns detected"

        # Recommended actions
        interpretation['recommended_actions'] = '; '.join(stroke_risk['clinical_recommendations'])

        # Monitoring recommendations
        if stroke_risk['risk_category'] == 'high':
            interpretation['monitoring_recommendations'] = "Continuous cardiac monitoring, frequent follow-up, anticoagulation assessment"
        elif stroke_risk['risk_category'] == 'moderate':
            interpretation['monitoring_recommendations'] = "Regular cardiac monitoring, periodic ECG, lifestyle modifications"
        else:
            interpretation['monitoring_recommendations'] = "Routine cardiac assessment, annual screening"

        # Limitations
        interpretation['limitations'] = (
            f"Analysis based on single time window. Confidence level: {stroke_risk['confidence_level']:.1%}. "
            "Clinical correlation and additional testing recommended for definitive diagnosis."
        )

        return interpretation

    def _calculate_validation_metrics(self, clinical_features: Dict[str, float],
                                    model_confidence: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for clinical validation."""

        metrics = {
            'feature_consistency': 0.0,
            'clinical_plausibility': 0.0,
            'prediction_stability': 0.0,
            'overall_validity': 0.0
        }

        # Feature consistency (do extracted features make clinical sense?)
        hr_mean = clinical_features.get('hr_mean', 70)
        rr_mean = clinical_features.get('rr_mean', 800)

        # Check if HR and RR interval are consistent
        expected_hr = 60000 / rr_mean if rr_mean > 0 else 0
        hr_consistency = 1.0 - min(abs(hr_mean - expected_hr) / max(hr_mean, expected_hr, 1), 1.0)
        metrics['feature_consistency'] = hr_consistency

        # Clinical plausibility (are values within physiological ranges?)
        plausibility_score = 1.0
        if hr_mean < 20 or hr_mean > 200:
            plausibility_score *= 0.5  # Extreme HR values
        if clinical_features.get('qrs_width', 100) > 200:
            plausibility_score *= 0.7  # Very wide QRS

        metrics['clinical_plausibility'] = plausibility_score

        # Prediction stability (confidence-based)
        confidence = model_confidence.get('arrhythmia_confidence', 0.0)
        metrics['prediction_stability'] = confidence

        # Overall validity
        metrics['overall_validity'] = (
            metrics['feature_consistency'] * 0.3 +
            metrics['clinical_plausibility'] * 0.4 +
            metrics['prediction_stability'] * 0.3
        )

        return metrics


def create_clinical_report(explanation: Dict[str, Any], patient_id: str = None) -> str:
    """Generate a clinical report for doctor validation."""

    report = f"""
# Clinical AI Analysis Report
{f"Patient ID: {patient_id}" if patient_id else ""}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Primary Findings
{explanation['clinical_interpretation']['primary_finding']}

## Clinical Significance
{explanation['clinical_interpretation']['clinical_significance']}

## Detailed Analysis

### Arrhythmia Assessment
- Risk Category: {explanation['arrhythmia_assessment'].get('atrial_fibrillation_risk', 'Unknown')} AF risk
- Clinical Flags: {', '.join(explanation['arrhythmia_assessment'].get('clinical_flags', ['None']))}

### Stroke Risk Analysis
- Estimated Annual Risk: {explanation['stroke_risk_analysis']['estimated_stroke_risk']:.1f}%
- Risk Category: {explanation['stroke_risk_analysis']['risk_category'].title()}
- Confidence Level: {explanation['stroke_risk_analysis']['confidence_level']:.1%}
- Contributing Factors: {', '.join(explanation['stroke_risk_analysis']['contributing_factors'])}

### Model Predictions
- Predicted Rhythm: {explanation['model_confidence'].get('predicted_class', 'Unknown')}
- Confidence: {explanation['model_confidence'].get('arrhythmia_confidence', 0.0):.1%}

## Clinical Recommendations
{explanation['clinical_interpretation']['recommended_actions']}

## Monitoring Recommendations
{explanation['clinical_interpretation']['monitoring_recommendations']}

## Validation Metrics
- Feature Consistency: {explanation['validation_metrics']['feature_consistency']:.1%}
- Clinical Plausibility: {explanation['validation_metrics']['clinical_plausibility']:.1%}
- Overall Validity: {explanation['validation_metrics']['overall_validity']:.1%}

## Limitations
{explanation['clinical_interpretation']['limitations']}

---
*This AI analysis is intended to assist clinical decision-making and should not replace clinical judgment.
All findings should be validated with standard clinical assessments.*
"""

    return report