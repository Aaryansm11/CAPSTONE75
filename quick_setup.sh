#!/bin/bash
# quick_setup.sh - One-click setup for ECG+PPG system

echo "🫀 ECG+PPG MULTIMODAL CARDIAC ANALYSIS SYSTEM"
echo "=============================================="

# Check if conda environment exists
if conda env list | grep -q ecgppg; then
    echo "✅ Conda environment 'ecgppg' already exists"
else
    echo "🔧 Creating conda environment..."
    conda create -n ecgppg python=3.9 -y
fi

# Activate environment
echo "🔄 Activating environment..."
source activate ecgppg

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit plotly pandas numpy scikit-learn
pip install matplotlib seaborn jinja2 weasyprint
pip install fastapi uvicorn shap wfdb

# Check if model exists
if [ -f "models/multimodal_best.pth" ]; then
    echo "✅ Trained model found: models/multimodal_best.pth"
    MODEL_STATUS="READY"
else
    echo "⚠️  No trained model found. Options:"
    echo "   1. Download MIT-BIH data and run: python 1.py"
    echo "   2. Or continue with demo mode (synthetic data)"
    MODEL_STATUS="DEMO_MODE"
fi

# Create directories
echo "📁 Creating directory structure..."
mkdir -p data/raw data/processed models reports logs

echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo "Model Status: $MODEL_STATUS"
echo ""
echo "🚀 TO START THE SYSTEM:"
echo "   conda activate ecgppg"
echo "   streamlit run streamlit_app.py --server.port 8501"
echo ""
echo "📱 THEN OPEN: http://localhost:8501"
echo ""
echo "🔧 FOR FULL TRAINING:"
echo "   1. Download MIT-BIH from: https://physionet.org/content/mitdb/1.0.0/"
echo "   2. Extract to: data/raw/mit-bih-arrhythmia-database-1.0.0/"
echo "   3. Run: python 1.py"
echo ""
echo "📋 FOR API SERVER:"
echo "   uvicorn src.api:app --port 8081"
echo "   Open: http://localhost:8081/docs"