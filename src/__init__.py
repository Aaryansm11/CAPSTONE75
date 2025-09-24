# src/__init__.py
"""
ECG+PPG Multimodal Cardiac Analysis System

A comprehensive AI-powered system for cardiac analysis using ECG and PPG signals.
Features arrhythmia detection, stroke risk assessment, and automated report generation.
"""

__version__ = "1.0.0"
__author__ = "ECG-PPG Analysis Team"
__email__ = "support@ecg-ppg-analysis.com"
__license__ = "MIT"

# Import main components for easy access
try:
    from .config import *
    from .utils import seed_everything, setup_logging
    from .models import MultiModalFusionNetwork as MultiModalCNNLSTM, BranchEncoder
    from .dataset import MultiModalDataset
    from .report import ReportGenerator
    
    # Optional imports (may not be available in all environments)
    try:
        from .api import app as fastapi_app
    except ImportError:
        fastapi_app = None
    
except ImportError as e:
    # Handle cases where dependencies might not be installed
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")

# Version check for critical dependencies
def check_dependencies():
    """Check if all required dependencies are installed with correct versions."""
    import sys
    
    required_packages = {
        'torch': '2.5.0',
        'numpy': '2.2.0',
        'pandas': '2.3.0',
        'scikit-learn': '1.7.0',
        'fastapi': '0.98.0',
        'wfdb': '4.3.0'
    }
    
    missing_packages = []
    version_conflicts = []
    
    for package, min_version in required_packages.items():
        try:
            imported_package = __import__(package)
            if hasattr(imported_package, '__version__'):
                current_version = imported_package.__version__
                # Simple version comparison (could be enhanced)
                if current_version < min_version:
                    version_conflicts.append(f"{package}: {current_version} < {min_version}")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")
    
    if version_conflicts:
        import warnings
        warnings.warn(f"Version conflicts detected: {', '.join(version_conflicts)}")
    
    return True

# Module-level constants
SUPPORTED_SIGNAL_TYPES = ['ECG', 'PPG', 'EEG']  # Future expansion
SUPPORTED_FORMATS = ['WFDB', 'CSV', 'NPY', 'HDF5']
DEFAULT_SAMPLING_RATES = {'ECG': 360, 'PPG': 125, 'EEG': 256}

# Logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export key components
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Core classes
    'MultiModalCNNLSTM',
    'BranchEncoder', 
    'MultiModalDataset',
    'ReportGenerator',
    
    # Utilities
    'seed_everything',
    'setup_logging',
    'check_dependencies',
    
    # Constants
    'SUPPORTED_SIGNAL_TYPES',
    'SUPPORTED_FORMATS',
    'DEFAULT_SAMPLING_RATES',
    
    # API (if available)
    'fastapi_app',
]