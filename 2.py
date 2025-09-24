import numpy as np  # For numerical computations and array handling
import torch  # PyTorch main package
import torch.nn as nn  # Neural network components
import torch.optim as optim  # Optimization algorithms like Adam
from torch.utils.data import DataLoader, TensorDataset  # For batching and loading data
from sklearn.metrics import classification_report, confusion_matrix  # For evaluation metrics
from sklearn.preprocessing import StandardScaler  # For normalizing input features
from sklearn.model_selection import train_test_split  # For splitting dataset
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sn  # For heatmaps and other enhanced plots
from matplotlib.font_manager import FontProperties  # For font styling in plots
import pandas as pd  # For handling data in DataFrame format
import os  # For file and path operations
import wfdb  # For reading ECG data from PhysioNet
import logging  # For structured logging
import datetime  # For timestamps
import time  # For tracking execution time

# Configure logging to print and save to file
log_filename = f"ecg_model_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Define a custom CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTM, self).__init__()
        # 1D convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        # Max pooling and dropout
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout = nn.Dropout(0.45)
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(128, 210, batch_first=True)
        self.lstm2 = nn.LSTM(210, 190, batch_first=True)
        # Compute output size after pooling for linear layer
        conv_output_size = input_dim // 8
        self.fc1 = nn.Linear(190 * conv_output_size, 5)  # Hidden dense layer
        self.fc2 = nn.Linear(5, 5)  # Output layer (5 classes)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.001)

    def forward(self, x):
        # Pass through CNN layers
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        x = self.pool(x)  # Reduce sequence length
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, seq_len, features)
        x, _ = self.lstm1(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Remaining code explanation continued here
# Function to preprocess ECG data and train the CNN-LSTM model
def preprocess_and_train():
    logger.info("=" * 80)
    logger.info("Starting MIT-BIH Arrhythmia classification with CNN-LSTM model")
    logger.info("=" * 80)

    # Try importing wfdb, install if missing
    try:
        import wfdb
        logger.info(f"Using wfdb version {wfdb.__version__}")
    except ImportError:
        logger.info("wfdb library not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "wfdb"])
        import wfdb
        logger.info(f"Installed wfdb version {wfdb.__version__}")

    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load ECG data from MIT-BIH dataset folder
    data_path = 'mit-bih-arrhythmia-database-1.0.0'
    logger.info(f"Loading data from {data_path}...")
    X, y = load_mit_bih_data(data_path)

    logger.info(f"Loaded {len(X)} ECG segments with {len(np.unique(y))} classes")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Normalize each ECG segment individually
    logger.info("Normalizing data...")
    start_time = time.time()
    scaler = StandardScaler()
    X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])
    logger.info(f"Normalization completed in {time.time() - start_time:.2f}s")

    # Split into training and test sets (80/20 split)
    logger.info("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train the model using the standalone training function
    logger.info("Training CNN-LSTM model...")
    model = cnn_lstm_standalone(
        X_train, y_train, X_test, y_test, 
        epochs=2, 
        title="ECG Arrhythmia Classification"
    )

    logger.info("Training complete!")
    return model

# Main entry point for running the full pipeline
if __name__ == "__main__":
    preprocess_and_train()  # Call the main training function
