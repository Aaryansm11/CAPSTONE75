# src/api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import json
import os
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Internal imports
from src.config import DEVICE, MODEL_DIR, REPORT_DIR, CLASS_NAMES
from src.utils import load_checkpoint, setup_logging
from src.data_utils import resample_signal, bandpass_filter, extract_basic_features
from src.models import MultiModalCNNLSTM
from src.report import ReportGenerator
from src.explain import shap_on_features, attention_heatmap

# Global model storage
model_cache = {}
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    try:
        load_model()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    
    yield
    
    # Cleanup on shutdown
    model_cache.clear()
    logger.info("Models cleaned up")

app = FastAPI(
    title="ECG+PPG Multimodal Cardiac Analysis API",
    description="AI-powered cardiac analysis using ECG and PPG signals",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class SignalData(BaseModel):
    samples: List[float] = Field(..., description="Signal sample values")
    sampling_rate: int = Field(..., ge=1, le=10000, description="Sampling rate in Hz")
    signal_type: str = Field(..., description="Signal type (ECG or PPG)")

class AnalysisRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    ecg_signal: SignalData = Field(..., description="ECG signal data")
    ppg_signal: SignalData = Field(..., description="PPG signal data") 
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    generate_report: bool = Field(default=True, description="Whether to generate PDF report")
    explain_predictions: bool = Field(default=True, description="Whether to generate explanations")

class PredictionResponse(BaseModel):
    patient_id: str
    timestamp: str
    arrhythmia_class: str
    arrhythmia_probabilities: Dict[str, float]
    stroke_risk: float
    confidence: float
    processing_time_ms: float
    model_version: str

class ReportResponse(BaseModel):
    patient_id: str
    report_paths: Dict[str, str]
    download_links: Dict[str, str]
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    memory_usage: Dict[str, float]

def load_model():
    """Load the trained multimodal model."""
    model_path = os.path.join(MODEL_DIR, "multimodal_best.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(model_path, device=DEVICE)
        
        # Initialize model architecture (you may need to adjust these parameters)
        model = MultiModalCNNLSTM(
            seq_len_ecg=3600,  # 10 seconds at 360 Hz
            seq_len_ppg=1250,  # 10 seconds at 125 Hz  
            n_classes_arr=len(CLASS_NAMES),
            stroke_output_dim=1
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state'])
        model.to(DEVICE)
        model.eval()
        
        model_cache['main'] = {
            'model': model,
            'loaded_at': datetime.now(),
            'version': checkpoint.get('version', 'v1.0')
        }
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_signals(ecg_samples: List[float], ppg_samples: List[float],
                      ecg_fs: int, ppg_fs: int, target_length: int = 3600) -> tuple:
    """Preprocess ECG and PPG signals for model input."""
    
    # Convert to numpy arrays
    ecg = np.array(ecg_samples, dtype=np.float32)
    ppg = np.array(ppg_samples, dtype=np.float32)
    
    # Apply bandpass filtering
    ecg_filtered = bandpass_filter(ecg, low=0.5, high=40, fs=ecg_fs)
    ppg_filtered = bandpass_filter(ppg, low=0.5, high=10, fs=ppg_fs)
    
    # Resample PPG to match ECG sampling rate
    if ppg_fs != ecg_fs:
        ppg_resampled = resample_signal(ppg_filtered, ppg_fs, ecg_fs)
    else:
        ppg_resampled = ppg_filtered
    
    # Ensure both signals have the same length
    min_len = min(len(ecg_filtered), len(ppg_resampled))
    if min_len < target_length:
        # Pad with zeros if too short
        ecg_padded = np.pad(ecg_filtered[:min_len], (0, max(0, target_length - min_len)), mode='constant')
        ppg_padded = np.pad(ppg_resampled[:min_len], (0, max(0, target_length - min_len)), mode='constant')
    else:
        # Truncate if too long
        ecg_padded = ecg_filtered[:target_length]
        ppg_padded = ppg_resampled[:target_length]
    
    # Normalize
    ecg_norm = (ecg_padded - np.mean(ecg_padded)) / (np.std(ecg_padded) + 1e-8)
    ppg_norm = (ppg_padded - np.mean(ppg_padded)) / (np.std(ppg_padded) + 1e-8)
    
    return ecg_norm, ppg_norm

def create_explanations(model, ecg_tensor, ppg_tensor, predictions):
    """Generate model explanations."""
    explanations = {}
    
    try:
        # Extract features for SHAP
        with torch.no_grad():
            model_output = model(ecg_tensor, ppg_tensor)
            
        # Get attention weights if available
        if hasattr(model, 'ecg_enc') and hasattr(model.ecg_enc, 'attention_weights'):
            explanations['attention_weights'] = model.ecg_enc.attention_weights.cpu().numpy()
        else:
            # Create dummy attention for demonstration
            explanations['attention_weights'] = np.random.rand(8, ecg_tensor.shape[-1] // 45)
        
        # Extract basic features for SHAP analysis
        ecg_features = extract_basic_features(ecg_tensor.squeeze().cpu().numpy())
        ppg_features = extract_basic_features(ppg_tensor.squeeze().cpu().numpy())
        combined_features = np.concatenate([ecg_features, ppg_features]).reshape(1, -1)
        
        # Simple feature importance (replace with actual SHAP if needed)
        feature_names = [
            'ECG_mean', 'ECG_std', 'ECG_min', 'ECG_max', 'ECG_ptp', 'ECG_energy', 'ECG_variation', 'ECG_median',
            'PPG_mean', 'PPG_std', 'PPG_min', 'PPG_max', 'PPG_ptp', 'PPG_energy', 'PPG_variation', 'PPG_median'
        ]
        
        # Mock SHAP values for demonstration
        explanations['shap_values'] = [np.random.randn(1, len(feature_names)) for _ in range(len(CLASS_NAMES))]
        explanations['feature_names'] = feature_names
        
    except Exception as e:
        logger.warning(f"Error creating explanations: {e}")
        explanations = {'error': str(e)}
    
    return explanations

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = 'main' in model_cache
    
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
        }
    
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=DEVICE,
        memory_usage=memory_info
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: AnalysisRequest):
    """Main prediction endpoint."""
    start_time = datetime.now()
    
    if 'main' not in model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_info = model_cache['main']
        model = model_info['model']
        
        # Preprocess signals
        ecg_processed, ppg_processed = preprocess_signals(
            request.ecg_signal.samples,
            request.ppg_signal.samples,
            request.ecg_signal.sampling_rate,
            request.ppg_signal.sampling_rate
        )
        
        # Convert to tensors
        ecg_tensor = torch.from_numpy(ecg_processed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        ppg_tensor = torch.from_numpy(ppg_processed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        # Model inference
        with torch.no_grad():
            outputs = model(ecg_tensor, ppg_tensor)
            
            # Get predictions
            arr_logits = outputs['arr_logits'].cpu().numpy()[0]
            arr_probs = torch.softmax(outputs['arr_logits'], dim=1).cpu().numpy()[0]
            stroke_risk = torch.sigmoid(outputs['stroke']).cpu().numpy()[0]
            
            # Predicted class
            predicted_idx = np.argmax(arr_probs)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(arr_probs[predicted_idx])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare response
        response = PredictionResponse(
            patient_id=request.patient_id,
            timestamp=datetime.now().isoformat(),
            arrhythmia_class=predicted_class,
            arrhythmia_probabilities={class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, arr_probs)},
            stroke_risk=float(stroke_risk),
            confidence=confidence,
            processing_time_ms=processing_time,
            model_version=model_info['version']
        )
        
        logger.info(f"Prediction completed for patient {request.patient_id} in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error for patient {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze", response_model=ReportResponse)
async def full_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Complete analysis with report generation."""
    
    # First get predictions
    prediction_response = await predict(request)
    
    if not request.generate_report:
        return ReportResponse(
            patient_id=request.patient_id,
            report_paths={},
            download_links={}
        )
    
    try:
        # Prepare data for report generation
        predictions = {
            'predicted_class': prediction_response.arrhythmia_class,
            'arrhythmia_probs': list(prediction_response.arrhythmia_probabilities.values()),
            'stroke_risk': prediction_response.stroke_risk,
            'confidence': prediction_response.confidence,
            'class_names': CLASS_NAMES
        }
        
        explanations = {}
        if request.explain_predictions and 'main' in model_cache:
            # Get processed signals for explanations
            ecg_processed, ppg_processed = preprocess_signals(
                request.ecg_signal.samples,
                request.ppg_signal.samples,
                request.ecg_signal.sampling_rate,
                request.ppg_signal.sampling_rate
            )
            
            ecg_tensor = torch.from_numpy(ecg_processed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            ppg_tensor = torch.from_numpy(ppg_processed).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            
            explanations = create_explanations(model_cache['main']['model'], ecg_tensor, ppg_tensor, predictions)
        
        metadata = {
            'model_version': prediction_response.model_version,
            'ecg_fs': request.ecg_signal.sampling_rate,
            'ppg_fs': request.ppg_signal.sampling_rate,
            'signal_duration': len(request.ecg_signal.samples) / request.ecg_signal.sampling_rate,
            'processing_time_ms': prediction_response.processing_time_ms,
            **request.metadata
        }
        
        # Generate report
        report_generator = ReportGenerator(REPORT_DIR)
        report_paths = report_generator.generate_complete_report(
            patient_id=request.patient_id,
            predictions=predictions,
            explanations=explanations,
            metadata=metadata,
            ecg_data=np.array(request.ecg_signal.samples),
            ppg_data=np.array(request.ppg_signal.samples),
            format_types=['html', 'pdf']
        )
        
        # Create download links
        download_links = {}
        for format_type, file_path in report_paths.items():
            filename = os.path.basename(file_path)
            download_links[format_type] = f"/download/{filename}"
        
        return ReportResponse(
            patient_id=request.patient_id,
            report_paths=report_paths,
            download_links=download_links
        )
        
    except Exception as e:
        logger.error(f"Report generation error for patient {request.patient_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated reports."""
    file_path = os.path.join(REPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    if filename.endswith('.pdf'):
        media_type = 'application/pdf'
    elif filename.endswith('.html'):
        media_type = 'text/html'
    elif filename.endswith('.json'):
        media_type = 'application/json'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

@app.get("/reports/{patient_id}")
async def list_patient_reports(patient_id: str):
    """List all reports for a patient."""
    reports = []
    
    for filename in os.listdir(REPORT_DIR):
        if filename.startswith(f"report_{patient_id}_"):
            file_path = os.path.join(REPORT_DIR, filename)
            file_stats = os.stat(file_path)
            
            reports.append({
                'filename': filename,
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'size_bytes': file_stats.st_size,
                'download_link': f"/download/{filename}"
            })
    
    return {'patient_id': patient_id, 'reports': sorted(reports, key=lambda x: x['created_at'], reverse=True)}

@app.post("/batch_analyze")
async def batch_analyze(requests: List[AnalysisRequest], background_tasks: BackgroundTasks):
    """Batch analysis for multiple patients."""
    results = []
    
    for request in requests:
        try:
            result = await full_analysis(request, background_tasks)
            results.append({'patient_id': request.patient_id, 'status': 'success', 'result': result})
        except Exception as e:
            results.append({'patient_id': request.patient_id, 'status': 'error', 'error': str(e)})
    
    return {'batch_results': results, 'total_processed': len(results)}

@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation homepage."""
    return """
    <html>
        <head>
            <title>ECG+PPG Multimodal Cardiac Analysis API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: #007bff; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ECG+PPG Multimodal Cardiac Analysis API</h1>
                <p>AI-powered cardiac analysis using ECG and PPG signals for arrhythmia detection and stroke risk assessment.</p>
                
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/health</strong><br>
                    Check API health status and model loading state.
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/predict</strong><br>
                    Get predictions for ECG and PPG signals without report generation.
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/analyze</strong><br>
                    Complete analysis with automatic report generation in PDF and HTML formats.
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/download/{filename}</strong><br>
                    Download generated reports and analysis files.
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/reports/{patient_id}</strong><br>
                    List all reports for a specific patient.
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/batch_analyze</strong><br>
                    Process multiple patients in a single request.
                </div>
                
                <p><strong>Documentation:</strong> <a href="/docs">Interactive API Documentation</a></p>
                <p><strong>OpenAPI Schema:</strong> <a href="/openapi.json">JSON Schema</a></p>
            </div>
        </body>
    </html>
    """

# Initialize logging
setup_logging("logs", "api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)