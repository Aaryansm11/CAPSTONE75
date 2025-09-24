# src/report.py
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64
import logging

from jinja2 import Template
from weasyprint import HTML, CSS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import REPORT_DIR, CLASS_NAMES, STROKE_RISK_LOW, STROKE_RISK_MODERATE, STROKE_RISK_HIGH

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive PDF reports for patient predictions."""
    
    def __init__(self, report_dir: str = str(REPORT_DIR)):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup styling
        self.css_style = """
        @page {
            size: A4;
            margin: 1cm;
        }
        body {
            font-family: Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #333;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0;
            font-size: 24pt;
        }
        .header p {
            margin: 5px 0;
            color: #7f8c8d;
        }
        .section {
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            font-size: 16pt;
        }
        .section h3 {
            color: #34495e;
            font-size: 13pt;
            margin-top: 15px;
        }
        .prediction-box {
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        .high-risk {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }
        .moderate-risk {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        .low-risk {
            border-left-color: #27ae60;
            background: #f0f8f0;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .metrics-table th, .metrics-table td {
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: left;
        }
        .metrics-table th {
            background: #3498db;
            color: white;
        }
        .feature-importance {
            font-size: 10pt;
        }
        .plot-container {
            text-align: center;
            margin: 15px 0;
            page-break-inside: avoid;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            margin-top: 30px;
            border-top: 1px solid #bdc3c7;
            padding-top: 10px;
            font-size: 9pt;
            color: #7f8c8d;
            text-align: center;
        }
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 10px;
            margin: 15px 0;
            font-size: 10pt;
        }
        """
    
    def _get_risk_level(self, risk_score: float, risk_type: str = 'stroke') -> Dict[str, str]:
        """Determine risk level and styling based on score."""
        if risk_type == 'stroke':
            if risk_score >= STROKE_RISK_HIGH:
                return {'level': 'High', 'class': 'high-risk', 'color': '#e74c3c'}
            elif risk_score >= STROKE_RISK_MODERATE:
                return {'level': 'Moderate', 'class': 'moderate-risk', 'color': '#f39c12'}
            else:
                return {'level': 'Low', 'class': 'low-risk', 'color': '#27ae60'}
        else:  # arrhythmia
            if risk_score >= 0.8:
                return {'level': 'High', 'class': 'high-risk', 'color': '#e74c3c'}
            elif risk_score >= 0.5:
                return {'level': 'Moderate', 'class': 'moderate-risk', 'color': '#f39c12'}
            else:
                return {'level': 'Low', 'class': 'low-risk', 'color': '#27ae60'}
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image as base64 for embedding in HTML."""
        try:
            with open(image_path, 'rb') as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def _create_summary_plot(self, predictions: Dict[str, Any], 
                           save_path: str) -> str:
        """Create summary visualization of predictions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Arrhythmia prediction bar chart
        if 'arrhythmia_probabilities' in predictions:
            probs = predictions['arrhythmia_probabilities']
            ax1.bar(CLASS_NAMES, probs, alpha=0.7, color='skyblue')
            ax1.set_title('Arrhythmia Classification Probabilities')
            ax1.set_ylabel('Probability')
            ax1.tick_params(axis='x', rotation=45)
            
            # Highlight predicted class
            predicted_idx = np.argmax(probs)
            ax1.bar(CLASS_NAMES[predicted_idx], probs[predicted_idx],
                   alpha=0.9, color='red', label='Predicted')
            ax1.legend()
        
        # Stroke risk gauge
        stroke_risk = predictions.get('stroke_risk', 0.0)
        risk_info = self._get_risk_level(stroke_risk, 'stroke')
        
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax2.plot(theta, r, 'k-', linewidth=2)
        ax2.fill_between(theta[:33], 0, r[:33], alpha=0.7, color='green', label='Low')
        ax2.fill_between(theta[33:66], 0, r[33:66], alpha=0.7, color='orange', label='Moderate')
        ax2.fill_between(theta[66:], 0, r[66:], alpha=0.7, color='red', label='High')
        
        # Add risk indicator
        risk_angle = stroke_risk * np.pi
        ax2.plot([risk_angle, risk_angle], [0, 1], 'k-', linewidth=4, label=f'Risk: {stroke_risk:.2f}')
        
        ax2.set_ylim(0, 1.2)
        ax2.set_xlim(0, np.pi)
        ax2.set_title('Stroke Risk Level')
        ax2.legend(loc='upper right')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Feature importance (if available)
        if 'feature_importance' in predictions:
            importance = predictions['feature_importance']
            features = list(importance.keys())[:10]  # Top 10
            values = [importance[f] for f in features]
            
            ax3.barh(features, values, alpha=0.7, color='lightcoral')
            ax3.set_title('Top Feature Importance')
            ax3.set_xlabel('Importance')
        
        # Model confidence
        if 'confidence_scores' in predictions:
            confidence = predictions['confidence_scores']
            ax4.pie([confidence, 1-confidence], labels=['Confident', 'Uncertain'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightgray'])
            ax4.set_title('Model Confidence')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_patient_report(self, 
                              patient_id: str,
                              predictions: Dict[str, Any],
                              signal_plots: List[str] = None,
                              explanation_plots: List[str] = None,
                              model_metrics: Dict[str, Any] = None,
                              patient_metadata: Dict[str, Any] = None) -> str:
        """
        Generate comprehensive patient report.
        
        Args:
            patient_id: Patient identifier
            predictions: Model predictions and scores
            signal_plots: List of signal plot paths
            explanation_plots: List of explanation plot paths  
            model_metrics: Model performance metrics
            patient_metadata: Additional patient information
            
        Returns:
            Path to generated PDF report
        """
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create summary plot
        summary_plot_path = self.report_dir / f"{patient_id}_summary.png"
        self._create_summary_plot(predictions, str(summary_plot_path))
        
        # Determine risk levels
        stroke_risk = predictions.get('stroke_risk', 0.0)
        stroke_risk_info = self._get_risk_level(stroke_risk, 'stroke')
        
        arrhythmia_prob = predictions.get('arrhythmia_probability', 0.0)
        arrhythmia_risk_info = self._get_risk_level(arrhythmia_prob, 'arrhythmia')
        
        # Prepare template data
        template_data = {
            'patient_id': patient_id,
            'timestamp': timestamp,
            'predictions': predictions,
            'stroke_risk_info': stroke_risk_info,
            'arrhythmia_risk_info': arrhythmia_risk_info,
            'patient_metadata': patient_metadata or {},
            'model_metrics': model_metrics or {},
            'summary_plot': self._encode_image_base64(str(summary_plot_path)),
            'signal_plots': [self._encode_image_base64(p) for p in (signal_plots or [])],
            'explanation_plots': [self._encode_image_base64(p) for p in (explanation_plots or [])]
        }
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Patient Report - {{ patient_id }}</title>
    <style>{{ css_style }}</style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>ECG/PPG Analysis Report</h1>
        <p>Patient ID: {{ patient_id }}</p>
        <p>Generated: {{ timestamp }}</p>
        <p>Multimodal Arrhythmia & Stroke Risk Assessment</p>
    </div>

    <!-- Executive Summary -->
    <div class="section">
        <h2>Executive Summary</h2>
        
        <div class="prediction-box {{ stroke_risk_info.class }}">
            <h3>Stroke Risk Assessment</h3>
            <p><strong>Risk Level: {{ stroke_risk_info.level }}</strong></p>
            <p>Risk Score: {{ "%.2f" | format(predictions.stroke_risk or 0) }} (0.0 = Low Risk, 1.0 = High Risk)</p>
            {% if stroke_risk_info.level == 'High' %}
            <p><strong>Recommendation:</strong> Immediate medical consultation recommended.</p>
            {% elif stroke_risk_info.level == 'Moderate' %}
            <p><strong>Recommendation:</strong> Enhanced monitoring and follow-up advised.</p>
            {% else %}
            <p><strong>Recommendation:</strong> Continue routine monitoring.</p>
            {% endif %}
        </div>
        
        <div class="prediction-box {{ arrhythmia_risk_info.class }}">
            <h3>Arrhythmia Classification</h3>
            <p><strong>Predicted Class: {{ predictions.arrhythmia_class or 'Unknown' }}</strong></p>
            <p>Confidence: {{ "%.2f" | format(predictions.arrhythmia_probability or 0) }}</p>
            {% if predictions.arrhythmia_class and predictions.arrhythmia_class != 'N' %}
            <p><strong>Clinical Significance:</strong> Abnormal rhythm detected - medical review recommended.</p>
            {% endif %}
        </div>
    </div>

    <!-- Summary Visualization -->
    <div class="section">
        <h2>Summary Visualization</h2>
        <div class="plot-container">
            {% if summary_plot %}
            <img src="{{ summary_plot }}" alt="Summary Plot">
            {% else %}
            <p>Summary visualization not available.</p>
            {% endif %}
        </div>
    </div>

    <!-- Signal Analysis -->
    {% if signal_plots %}
    <div class="section">
        <h2>Signal Analysis</h2>
        {% for plot in signal_plots %}
        <div class="plot-container">
            <img src="{{ plot }}" alt="Signal Analysis">
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Model Explanations -->
    {% if explanation_plots %}
    <div class="section">
        <h2>Model Interpretability</h2>
        <p>The following visualizations show which parts of the ECG and PPG signals contributed most to the model's predictions:</p>
        {% for plot in explanation_plots %}
        <div class="plot-container">
            <img src="{{ plot }}" alt="Model Explanation">
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Detailed Metrics -->
    {% if model_metrics %}
    <div class="section">
        <h2>Model Performance Metrics</h2>
        <table class="metrics-table">
            <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            {% if model_metrics.accuracy %}
            <tr><td>Overall Accuracy</td><td>{{ "%.2f%%" | format(model_metrics.accuracy * 100) }}</td><td>Percentage of correct classifications</td></tr>
            {% endif %}
            {% if model_metrics.sensitivity %}
            <tr><td>Sensitivity</td><td>{{ "%.2f%%" | format(model_metrics.sensitivity * 100) }}</td><td>Ability to correctly identify abnormal cases</td></tr>
            {% endif %}
            {% if model_metrics.specificity %}
            <tr><td>Specificity</td><td>{{ "%.2f%%" | format(model_metrics.specificity * 100) }}</td><td>Ability to correctly identify normal cases</td></tr>
            {% endif %}
            {% if model_metrics.auc %}
            <tr><td>AUC Score</td><td>{{ "%.3f" | format(model_metrics.auc) }}</td><td>Area under ROC curve (higher is better)</td></tr>
            {% endif %}
        </table>
    </div>
    {% endif %}

    <!-- Patient Information -->
    {% if patient_metadata %}
    <div class="section">
        <h2>Patient Information</h2>
        <table class="metrics-table">
            {% for key, value in patient_metadata.items() %}
            <tr><td>{{ key.replace('_', ' ').title() }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <!-- Clinical Context -->
    <div class="section">
        <h2>Clinical Context & Recommendations</h2>
        
        <h3>Arrhythmia Classification</h3>
        <p>The model classified the cardiac rhythm based on ECG patterns. Key considerations:</p>
        <ul>
            <li><strong>Normal (N):</strong> Regular sinus rhythm</li>
            <li><strong>Supraventricular (S):</strong> Abnormal rhythm originating above ventricles</li>
            <li><strong>Ventricular (V):</strong> Abnormal rhythm originating from ventricles - potentially serious</li>
            <li><strong>Fusion (F):</strong> Mixed rhythm patterns</li>
            <li><strong>Unknown (Q):</strong> Pattern not clearly classifiable</li>
        </ul>
        
        <h3>Stroke Risk Factors</h3>
        <p>The stroke risk assessment considers multiple physiological indicators from both ECG and PPG signals:</p>
        <ul>
            <li>Heart rate variability patterns</li>
            <li>Arrhythmia presence and type</li>
            <li>Pulse wave characteristics from PPG</li>
            <li>Signal morphology features</li>
        </ul>
        
        <h3>Next Steps</h3>
        {% if stroke_risk_info.level == 'High' or arrhythmia_risk_info.level == 'High' %}
        <div class="prediction-box high-risk">
            <p><strong>Urgent:</strong> Recommend immediate cardiology consultation and comprehensive cardiac evaluation.</p>
        </div>
        {% elif stroke_risk_info.level == 'Moderate' or arrhythmia_risk_info.level == 'Moderate' %}
        <div class="prediction-box moderate-risk">
            <p><strong>Follow-up:</strong> Schedule follow-up within 1-2 weeks. Consider 24-hour Holter monitoring.</p>
        </div>
        {% else %}
        <div class="prediction-box low-risk">
            <p><strong>Routine:</strong> Continue regular monitoring. Repeat assessment as clinically indicated.</p>
        </div>
        {% endif %}
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">
        <p><strong>Important Disclaimer:</strong> This report is generated by an AI system for clinical decision support. It should not replace professional medical judgment. All findings should be interpreted by qualified healthcare professionals in the context of the complete clinical picture. The model's predictions are based on signal analysis and may not capture all relevant clinical factors.</p>
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>Generated by ECG/PPG Multimodal Analysis System v1.0</p>
        <p>For technical support or questions about this report, contact: support@example.com</p>
    </div>
</body>
</html>
        """
        
        # Render HTML
        template = Template(html_template)
        html_content = template.render(**template_data, css_style=self.css_style)
        
        # Generate PDF
        pdf_path = self.report_dir / f"report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        try:
            HTML(string=html_content).write_pdf(
                str(pdf_path),
                stylesheets=[CSS(string=self.css_style)]
            )
            
            logger.info(f"Report generated successfully: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            
            # Fallback: save HTML version
            html_path = self.report_dir / f"report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved as fallback: {html_path}")
            return str(html_path)


def generate_batch_reports(patient_data: Dict[str, Dict[str, Any]], 
                          output_dir: str = str(REPORT_DIR)) -> List[str]:
    """
    Generate reports for multiple patients.
    
    Args:
        patient_data: Dictionary with patient_id as key and data as value
        output_dir: Directory to save reports
        
    Returns:
        List of generated report paths
    """
    report_generator = ReportGenerator(output_dir)
    report_paths = []
    
    for patient_id, data in patient_data.items():
        try:
            report_path = report_generator.generate_patient_report(
                patient_id=patient_id,
                predictions=data.get('predictions', {}),
                signal_plots=data.get('signal_plots', []),
                explanation_plots=data.get('explanation_plots', []),
                model_metrics=data.get('model_metrics', {}),
                patient_metadata=data.get('metadata', {})
            )
            report_paths.append(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report for patient {patient_id}: {e}")
    
    logger.info(f"Generated {len(report_paths)} reports successfully")
    return report_paths

# # src/report.py
# import os
# import json
# import base64
# from datetime import datetime
# from typing import List, Dict, Any, Optional
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from jinja2 import Template
# from weasyprint import HTML, CSS
# from io import BytesIO
# import logging

# logger = logging.getLogger(__name__)

# class ReportGenerator:
#     """
#     Comprehensive report generator for ECG+PPG analysis results.
#     Generates HTML and PDF reports with plots, explanations, and clinical insights.
#     """
    
#     def __init__(self, output_dir: str = "reports"):
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
        
#     def plot_to_base64(self, fig) -> str:
#         """Convert matplotlib figure to base64 string for embedding in HTML."""
#         buffer = BytesIO()
#         fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
#         graphic = base64.b64encode(image_png)
#         return graphic.decode('utf-8')
    
#     def create_signal_plots(self, ecg_data: np.ndarray, ppg_data: np.ndarray, 
#                           fs_ecg: int = 360, fs_ppg: int = 125) -> Dict[str, str]:
#         """Create signal visualization plots."""
#         plots = {}
        
#         # ECG Signal Plot
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
#         # ECG
#         time_ecg = np.arange(len(ecg_data)) / fs_ecg
#         ax1.plot(time_ecg, ecg_data, 'b-', linewidth=0.8, label='ECG')
#         ax1.set_title('ECG Signal Analysis', fontsize=14, fontweight='bold')
#         ax1.set_xlabel('Time (seconds)')
#         ax1.set_ylabel('Amplitude (mV)')
#         ax1.grid(True, alpha=0.3)
#         ax1.legend()
        
#         # PPG
#         time_ppg = np.arange(len(ppg_data)) / fs_ppg
#         ax2.plot(time_ppg, ppg_data, 'r-', linewidth=0.8, label='PPG')
#         ax2.set_title('PPG Signal Analysis', fontsize=14, fontweight='bold')
#         ax2.set_xlabel('Time (seconds)')
#         ax2.set_ylabel('Amplitude')
#         ax2.grid(True, alpha=0.3)
#         ax2.legend()
        
#         plt.tight_layout()
#         plots['signals'] = self.plot_to_base64(fig)
#         plt.close()
        
#         return plots
    
#     def create_prediction_plots(self, predictions: Dict[str, Any]) -> Dict[str, str]:
#         """Create prediction visualization plots."""
#         plots = {}
        
#         # Arrhythmia Classification Results
#         if 'arrhythmia_probs' in predictions:
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
#             # Class probabilities bar chart
#             class_names = predictions.get('class_names', ['N', 'S', 'V', 'F', 'U'])
#             probs = predictions['arrhythmia_probs']
            
#             bars = ax1.bar(class_names, probs, color=['green', 'orange', 'red', 'purple', 'brown'])
#             ax1.set_title('Arrhythmia Classification Probabilities', fontweight='bold')
#             ax1.set_ylabel('Probability')
#             ax1.set_ylim(0, 1)
            
#             # Highlight predicted class
#             max_idx = np.argmax(probs)
#             bars[max_idx].set_color('darkred')
            
#             # Add probability values on bars
#             for i, (bar, prob) in enumerate(zip(bars, probs)):
#                 height = bar.get_height()
#                 ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                         f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
#             # Risk assessment gauge
#             stroke_risk = predictions.get('stroke_risk', 0.0)
#             theta = np.linspace(0, np.pi, 100)
#             r = 1
            
#             ax2.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
#             ax2.fill_between(np.cos(theta[:33]), np.sin(theta[:33]), 0, 
#                            color='green', alpha=0.3, label='Low Risk')
#             ax2.fill_between(np.cos(theta[33:66]), np.sin(theta[33:66]), 0, 
#                            color='orange', alpha=0.3, label='Medium Risk')
#             ax2.fill_between(np.cos(theta[66:]), np.sin(theta[66:]), 0, 
#                            color='red', alpha=0.3, label='High Risk')
            
#             # Risk needle
#             risk_angle = np.pi * (1 - stroke_risk)
#             ax2.plot([0, np.cos(risk_angle)], [0, np.sin(risk_angle)], 
#                     'k-', linewidth=4)
#             ax2.plot(np.cos(risk_angle), np.sin(risk_angle), 'ko', markersize=8)
            
#             ax2.set_xlim(-1.2, 1.2)
#             ax2.set_ylim(-0.2, 1.2)
#             ax2.set_aspect('equal')
#             ax2.set_title(f'Stroke Risk Assessment: {stroke_risk:.1%}', fontweight='bold')
#             ax2.axis('off')
#             ax2.legend(loc='upper right')
            
#             plt.tight_layout()
#             plots['predictions'] = self.plot_to_base64(fig)
#             plt.close()
        
#         return plots
    
#     def create_explanation_plots(self, explanations: Dict[str, Any]) -> Dict[str, str]:
#         """Create explainability visualization plots."""
#         plots = {}
        
#         # SHAP Summary Plot
#         if 'shap_values' in explanations:
#             fig, ax = plt.subplots(figsize=(10, 8))
            
#             shap_values = explanations['shap_values']
#             feature_names = explanations.get('feature_names', [f'Feature_{i}' for i in range(len(shap_values[0]))])
            
#             # Create SHAP-style summary plot
#             for i, class_shap in enumerate(shap_values):
#                 mean_shap = np.mean(np.abs(class_shap), axis=0)
#                 y_pos = np.arange(len(feature_names))
#                 ax.barh(y_pos + i*0.2, mean_shap, height=0.2, 
#                        label=f'Class {i}', alpha=0.7)
            
#             ax.set_yticks(np.arange(len(feature_names)) + 0.2)
#             ax.set_yticklabels(feature_names)
#             ax.set_xlabel('Mean |SHAP Value|')
#             ax.set_title('Feature Importance (SHAP)', fontweight='bold')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
            
#             plt.tight_layout()
#             plots['shap'] = self.plot_to_base64(fig)
#             plt.close()
        
#         # Attention Heatmap
#         if 'attention_weights' in explanations:
#             fig, ax = plt.subplots(figsize=(12, 6))
            
#             attention = explanations['attention_weights']
#             im = ax.imshow(attention, aspect='auto', cmap='hot', interpolation='nearest')
#             ax.set_title('Attention Weights Heatmap', fontweight='bold')
#             ax.set_xlabel('Time Steps')
#             ax.set_ylabel('Attention Heads')
            
#             plt.colorbar(im, ax=ax, label='Attention Weight')
#             plt.tight_layout()
#             plots['attention'] = self.plot_to_base64(fig)
#             plt.close()
        
#         return plots
    
#     def generate_html_report(self, patient_id: str, predictions: Dict[str, Any], 
#                            explanations: Dict[str, Any], metadata: Dict[str, Any],
#                            ecg_data: Optional[np.ndarray] = None, 
#                            ppg_data: Optional[np.ndarray] = None) -> str:
#         """Generate comprehensive HTML report."""
        
#         # Generate all plots
#         plots = {}
#         if ecg_data is not None and ppg_data is not None:
#             plots.update(self.create_signal_plots(ecg_data, ppg_data))
#         plots.update(self.create_prediction_plots(predictions))
#         plots.update(self.create_explanation_plots(explanations))
        
#         # Determine risk level and recommendations
#         stroke_risk = predictions.get('stroke_risk', 0.0)
#         arrhythmia_class = predictions.get('predicted_class', 'Unknown')
        
#         if stroke_risk > 0.7:
#             risk_level = "HIGH"
#             risk_color = "#dc3545"
#             recommendations = [
#                 "Immediate cardiology consultation recommended",
#                 "Consider anticoagulation therapy evaluation",
#                 "Continuous cardiac monitoring advised",
#                 "Lifestyle modifications including diet and exercise"
#             ]
#         elif stroke_risk > 0.3:
#             risk_level = "MODERATE" 
#             risk_color = "#fd7e14"
#             recommendations = [
#                 "Regular cardiology follow-up recommended",
#                 "Monitor blood pressure and heart rate",
#                 "Consider stress testing",
#                 "Maintain healthy lifestyle habits"
#             ]
#         else:
#             risk_level = "LOW"
#             risk_color = "#28a745"
#             recommendations = [
#                 "Continue routine cardiac care",
#                 "Annual cardiovascular screening",
#                 "Maintain current medications as prescribed",
#                 "Regular exercise and healthy diet"
#             ]
        
#         html_template = """
#         <!DOCTYPE html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>Cardiac Analysis Report - Patient {{ patient_id }}</title>
#             <style>
#                 body { 
#                     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                     margin: 0; padding: 20px; background-color: #f8f9fa;
#                 }
#                 .container { max-width: 1200px; margin: 0 auto; background: white; 
#                            box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 10px; }
#                 .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
#                          color: white; padding: 30px; border-radius: 10px 10px 0 0; }
#                 .header h1 { margin: 0; font-size: 2.5em; }
#                 .header .subtitle { margin-top: 10px; opacity: 0.9; font-size: 1.1em; }
#                 .content { padding: 30px; }
#                 .section { margin-bottom: 40px; }
#                 .section h2 { color: #333; border-bottom: 3px solid #667eea; 
#                              padding-bottom: 10px; margin-bottom: 20px; }
#                 .alert { padding: 15px; border-radius: 8px; margin: 15px 0; }
#                 .alert-danger { background-color: #f8d7da; border-left: 5px solid #dc3545; }
#                 .alert-warning { background-color: #fff3cd; border-left: 5px solid #fd7e14; }
#                 .alert-success { background-color: #d4edda; border-left: 5px solid #28a745; }
#                 .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
#                           gap: 20px; margin: 20px 0; }
#                 .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; 
#                               border-left: 5px solid #667eea; }
#                 .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
#                 .plot-container { text-align: center; margin: 20px 0; }
#                 .plot-container img { max-width: 100%; height: auto; border-radius: 8px; 
#                                     box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
#                 .recommendations { background: #e9ecef; padding: 20px; border-radius: 8px; }
#                 .recommendations ul { list-style-type: none; padding: 0; }
#                 .recommendations li { padding: 10px 0; border-bottom: 1px solid #dee2e6; }
#                 .recommendations li:before { content: "✓ "; color: #28a745; font-weight: bold; }
#                 .footer { background: #6c757d; color: white; padding: 20px; text-align: center; 
#                          border-radius: 0 0 10px 10px; }
#                 table { width: 100%; border-collapse: collapse; margin: 20px 0; }
#                 th, td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
#                 th { background-color: #f8f9fa; font-weight: bold; }
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <div class="header">
#                     <h1>Cardiac Analysis Report</h1>
#                     <div class="subtitle">
#                         Patient ID: {{ patient_id }} | 
#                         Generated: {{ timestamp }} |
#                         Analysis: ECG + PPG Multimodal Assessment
#                     </div>
#                 </div>
                
#                 <div class="content">
#                     <!-- Risk Assessment Alert -->
#                     <div class="alert alert-{{ risk_class }}">
#                         <h3 style="margin: 0 0 10px 0;">Risk Assessment: {{ risk_level }}</h3>
#                         <p style="margin: 0;">
#                             Based on multimodal ECG+PPG analysis, the patient shows {{ risk_level.lower() }} risk 
#                             for cardiac events with {{ (stroke_risk * 100)|round(1) }}% stroke risk probability.
#                         </p>
#                     </div>
                    
#                     <!-- Key Metrics -->
#                     <div class="section">
#                         <h2>Key Clinical Metrics</h2>
#                         <div class="metrics">
#                             <div class="metric-card">
#                                 <div class="metric-value">{{ arrhythmia_class }}</div>
#                                 <div>Predicted Arrhythmia Type</div>
#                             </div>
#                             <div class="metric-card">
#                                 <div class="metric-value">{{ (stroke_risk * 100)|round(1) }}%</div>
#                                 <div>Stroke Risk Probability</div>
#                             </div>
#                             <div class="metric-card">
#                                 <div class="metric-value">{{ (confidence * 100)|round(1) }}%</div>
#                                 <div>Model Confidence</div>
#                             </div>
#                             <div class="metric-card">
#                                 <div class="metric-value">{{ analysis_duration|round(2) }}s</div>
#                                 <div>Signal Duration Analyzed</div>
#                             </div>
#                         </div>
#                     </div>
                    
#                     <!-- Signal Analysis -->
#                     {% if 'signals' in plots %}
#                     <div class="section">
#                         <h2>Signal Analysis</h2>
#                         <div class="plot-container">
#                             <img src="data:image/png;base64,{{ plots.signals }}" alt="ECG and PPG Signals">
#                         </div>
#                         <p>The above plots show the raw ECG and PPG signals used for analysis. 
#                            Signal quality assessment and preprocessing have been applied for optimal feature extraction.</p>
#                     </div>
#                     {% endif %}
                    
#                     <!-- Prediction Results -->
#                     {% if 'predictions' in plots %}
#                     <div class="section">
#                         <h2>Prediction Results</h2>
#                         <div class="plot-container">
#                             <img src="data:image/png;base64,{{ plots.predictions }}" alt="Prediction Results">
#                         </div>
                        
#                         <table>
#                             <thead>
#                                 <tr>
#                                     <th>Arrhythmia Class</th>
#                                     <th>Probability</th>
#                                     <th>Clinical Significance</th>
#                                 </tr>
#                             </thead>
#                             <tbody>
#                                 <tr>
#                                     <td>Normal (N)</td>
#                                     <td>{{ (arrhythmia_probs[0] * 100)|round(2) }}%</td>
#                                     <td>Normal sinus rhythm</td>
#                                 </tr>
#                                 <tr>
#                                     <td>Supraventricular (S)</td>
#                                     <td>{{ (arrhythmia_probs[1] * 100)|round(2) }}%</td>
#                                     <td>Atrial arrhythmias</td>
#                                 </tr>
#                                 <tr>
#                                     <td>Ventricular (V)</td>
#                                     <td>{{ (arrhythmia_probs[2] * 100)|round(2) }}%</td>
#                                     <td>Ventricular arrhythmias</td>
#                                 </tr>
#                                 <tr>
#                                     <td>Fusion (F)</td>
#                                     <td>{{ (arrhythmia_probs[3] * 100)|round(2) }}%</td>
#                                     <td>Fusion beats</td>
#                                 </tr>
#                                 <tr>
#                                     <td>Unknown (U)</td>
#                                     <td>{{ (arrhythmia_probs[4] * 100)|round(2) }}%</td>
#                                     <td>Unclassified beats</td>
#                                 </tr>
#                             </tbody>
#                         </table>
#                     </div>
#                     {% endif %}
                    
#                     <!-- Model Explanations -->
#                     {% if 'shap' in plots or 'attention' in plots %}
#                     <div class="section">
#                         <h2>AI Model Explanations</h2>
#                         {% if 'shap' in plots %}
#                         <h3>Feature Importance Analysis (SHAP)</h3>
#                         <div class="plot-container">
#                             <img src="data:image/png;base64,{{ plots.shap }}" alt="SHAP Feature Importance">
#                         </div>
#                         <p>This plot shows which features contribute most to the model's predictions. 
#                            Higher bars indicate greater influence on the decision.</p>
#                         {% endif %}
                        
#                         {% if 'attention' in plots %}
#                         <h3>Temporal Attention Analysis</h3>
#                         <div class="plot-container">
#                             <img src="data:image/png;base64,{{ plots.attention }}" alt="Attention Heatmap">
#                         </div>
#                         <p>This heatmap shows which time periods in the signal the model focused on 
#                            when making predictions. Brighter areas indicate higher attention.</p>
#                         {% endif %}
#                     </div>
#                     {% endif %}
                    
#                     <!-- Clinical Recommendations -->
#                     <div class="section">
#                         <h2>Clinical Recommendations</h2>
#                         <div class="recommendations">
#                             <h3>Recommended Actions:</h3>
#                             <ul>
#                                 {% for rec in recommendations %}
#                                 <li>{{ rec }}</li>
#                                 {% endfor %}
#                             </ul>
#                         </div>
#                     </div>
                    
#                     <!-- Technical Details -->
#                     <div class="section">
#                         <h2>Technical Details</h2>
#                         <table>
#                             <tr><th>Model Version</th><td>{{ metadata.get('model_version', 'v1.0') }}</td></tr>
#                             <tr><th>Analysis Algorithm</th><td>CNN-LSTM Multimodal Fusion</td></tr>
#                             <tr><th>Signal Processing</th><td>Bandpass Filter, Normalization, Feature Extraction</td></tr>
#                             <tr><th>ECG Sampling Rate</th><td>{{ metadata.get('ecg_fs', 360) }} Hz</td></tr>
#                             <tr><th>PPG Sampling Rate</th><td>{{ metadata.get('ppg_fs', 125) }} Hz</td></tr>
#                             <tr><th>Window Size</th><td>{{ metadata.get('window_seconds', 10) }} seconds</td></tr>
#                         </table>
#                     </div>
#                 </div>
                
#                 <div class="footer">
#                     <p><strong>Disclaimer:</strong> This AI-generated report is for clinical decision support only. 
#                        It should not replace professional medical judgment. Always consult with qualified healthcare providers.</p>
#                     <p>Report generated by ECG+PPG Multimodal AI System | Version 1.0</p>
#                 </div>
#             </div>
#         </body>
#         </html>
#         """
        
#         template = Template(html_template)
        
#         # Prepare template variables
#         template_vars = {
#             'patient_id': patient_id,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'risk_level': risk_level,
#             'risk_class': 'danger' if risk_level == 'HIGH' else 'warning' if risk_level == 'MODERATE' else 'success',
#             'stroke_risk': stroke_risk,
#             'arrhythmia_class': arrhythmia_class,
#             'arrhythmia_probs': predictions.get('arrhythmia_probs', [0.2, 0.2, 0.2, 0.2, 0.2]),
#             'confidence': predictions.get('confidence', 0.85),
#             'analysis_duration': metadata.get('signal_duration', 30.0),
#             'plots': plots,
#             'recommendations': recommendations,
#             'metadata': metadata
#         }
        
#         return template.render(**template_vars)
    
#     def html_to_pdf(self, html_content: str, output_path: str) -> str:
#         """Convert HTML content to PDF."""
#         try:
#             HTML(string=html_content).write_pdf(output_path)
#             logger.info(f"PDF report generated: {output_path}")
#             return output_path
#         except Exception as e:
#             logger.error(f"PDF generation failed: {e}")
#             raise
    
#     def generate_complete_report(self, patient_id: str, predictions: Dict[str, Any],
#                                explanations: Dict[str, Any], metadata: Dict[str, Any],
#                                ecg_data: Optional[np.ndarray] = None,
#                                ppg_data: Optional[np.ndarray] = None,
#                                format_types: List[str] = ['html', 'pdf']) -> Dict[str, str]:
#         """Generate complete report in multiple formats."""
        
#         html_content = self.generate_html_report(
#             patient_id, predictions, explanations, metadata, ecg_data, ppg_data
#         )
        
#         outputs = {}
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         if 'html' in format_types:
#             html_path = os.path.join(self.output_dir, f"report_{patient_id}_{timestamp}.html")
#             with open(html_path, 'w', encoding='utf-8') as f:
#                 f.write(html_content)
#             outputs['html'] = html_path
#             logger.info(f"HTML report saved: {html_path}")
        
#         if 'pdf' in format_types:
#             pdf_path = os.path.join(self.output_dir, f"report_{patient_id}_{timestamp}.pdf")
#             self.html_to_pdf(html_content, pdf_path)
#             outputs['pdf'] = pdf_path
        
#         # Also save metadata and predictions as JSON for future reference
#         json_path = os.path.join(self.output_dir, f"data_{patient_id}_{timestamp}.json")
#         report_data = {
#             'patient_id': patient_id,
#             'timestamp': timestamp,
#             'predictions': predictions,
#             'explanations': explanations,
#             'metadata': metadata
#         }
        
#         with open(json_path, 'w') as f:
#             json.dump(report_data, f, indent=2, default=str)
#         outputs['json'] = json_path
        
#         return outputs


# def build_patient_report_pdf(patient_id: str, predictions: Dict[str, Any], 
#                            images: List[str], metrics: Dict[str, Any], 
#                            out_dir: str = "reports") -> str:
#     """
#     Legacy function for backward compatibility.
#     Wraps the new ReportGenerator class.
#     """
#     generator = ReportGenerator(out_dir)
    
#     # Convert legacy format to new format
#     explanations = {'attention_weights': np.random.rand(8, 100)} if images else {}
#     metadata = {'model_version': 'v1.0', **metrics}
    
#     outputs = generator.generate_complete_report(
#         patient_id, predictions, explanations, metadata, format_types=['pdf']
#     )
    
#     return outputs['pdf']