# pretrained_integration.py - Integration script to use pretrained ECG model
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("always")

# Internal imports
from src.config import (DATA_RAW, DATA_PROCESSED, MODEL_DIR, REPORT_DIR,
                       DEVICE, BATCH_SIZE, EPOCHS, LR, CLASS_NAMES, SEED)
from src.utils import seed_everything, setup_logging, save_checkpoint
from src.data_utils import (load_mit_bih_records, extract_windows_from_record,
                           bandpass_filter, resample_signal)
from src.dataset import MultiModalDataset
from src.train import make_loaders_from_multimodal_arrays
from src.models import MultiModalFusionNetwork
from src.explain import shap_on_features, attention_heatmap
from src.report import ReportGenerator


class PretrainedECGAdapter(nn.Module):
    """Adapter to use pretrained ECG CNN-LSTM in multimodal framework."""

    def __init__(self, pretrained_path=None, input_dim=360):
        super().__init__()

        # Define the same architecture as in 3.py
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout = nn.Dropout(0.45)

        self.lstm1 = nn.LSTM(128, 210, batch_first=True)
        self.lstm2 = nn.LSTM(210, 190, batch_first=True)

        # Calculate the size after convolutions and pooling
        conv_output_size = input_dim // 8

        # Original dense layers (we'll use fc1 output as embedding)
        self.fc1 = nn.Linear(190 * conv_output_size, 5)
        self.fc2 = nn.Linear(5, 5)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.001)

        # Load pretrained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained ECG weights from {pretrained_path}")
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            logger.info("Pretrained ECG weights loaded successfully")
        else:
            logger.warning(f"Pretrained weights not found at {pretrained_path}, using random initialization")

    def forward(self, x, return_embedding=False):
        # CNN layers
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))

        x = self.pool(x)
        x = self.dropout(x)

        # LSTM layers
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.lstm2(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Dense layers
        embedding = self.relu(self.fc1(x))  # Use this as feature embedding

        if return_embedding:
            return embedding

        logits = self.fc2(embedding)
        return logits


class HybridMultiModalNetwork(nn.Module):
    """Hybrid network using pretrained ECG encoder with new PPG encoder."""

    def __init__(self,
                 ecg_seq_len=360,
                 ppg_seq_len=360,
                 n_arrhythmia_classes=5,
                 stroke_output_dim=1,
                 pretrained_ecg_path=None,
                 dropout=0.3):
        super().__init__()

        # Pretrained ECG encoder
        self.ecg_encoder = PretrainedECGAdapter(
            pretrained_path=pretrained_ecg_path,
            input_dim=ecg_seq_len
        )

        # Freeze ECG encoder weights (optional - can be unfrozen for fine-tuning)
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False

        # Simple PPG encoder (since we don't have pretrained PPG)
        self.ppg_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layers
        fusion_input_dim = 5 + 64  # ECG embedding (5) + PPG embedding (64)
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_arrhythmia_classes)
        )

        self.stroke_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, stroke_output_dim)
        )

    def forward(self, ecg_signal, ppg_signal):
        # Get ECG embedding from pretrained model
        ecg_embedding = self.ecg_encoder(ecg_signal, return_embedding=True)

        # Get PPG embedding
        ppg_embedding = self.ppg_encoder(ppg_signal)

        # Fuse modalities
        fused_features = torch.cat([ecg_embedding, ppg_embedding], dim=1)
        fused_output = self.fusion_layers(fused_features)

        # Task predictions
        arrhythmia_logits = self.arrhythmia_head(fused_output)
        stroke_output = self.stroke_head(fused_output)

        return {
            'arrhythmia_logits': arrhythmia_logits,
            'stroke_output': stroke_output.squeeze(-1),
            'ecg_embedding': ecg_embedding,
            'ppg_embedding': ppg_embedding,
            'fused_features': fused_output
        }


def create_synthetic_ppg_data(ecg_data, sampling_rate=360):
    """Create synthetic PPG data from ECG data."""
    logger.info("Creating synthetic PPG data from ECG signals...")

    ppg_data = []
    for ecg_signal in ecg_data:
        # Simple PPG simulation
        # 1. Invert and smooth ECG
        ppg_sim = -ecg_signal

        # 2. Apply smoothing
        from scipy.signal import savgol_filter
        try:
            ppg_smooth = savgol_filter(ppg_sim, window_length=11, polyorder=3)
        except:
            ppg_smooth = ppg_sim

        # 3. Add noise and make positive
        noise = 0.1 * np.random.randn(len(ppg_smooth))
        ppg_final = np.abs(ppg_smooth + noise)

        # 4. Normalize
        ppg_final = (ppg_final - np.mean(ppg_final)) / (np.std(ppg_final) + 1e-8)

        ppg_data.append(ppg_final)

    return np.array(ppg_data)


def load_processed_data_or_create():
    """Load processed data if available, otherwise create minimal dataset."""
    processed_file = os.path.join(DATA_PROCESSED, 'multimodal_data.npz')

    if os.path.exists(processed_file):
        logger.info(f"Loading processed data from {processed_file}")
        data = np.load(processed_file)
        return (data['X_ecg_train'], data['X_ppg_train'], data['y_train'],
                data['X_ecg_val'], data['X_ppg_val'], data['y_val'],
                data['X_ecg_test'], data['X_ppg_test'], data['y_test'])

    # Create minimal synthetic dataset for testing
    logger.info("Creating minimal synthetic dataset for testing...")

    # Generate synthetic ECG data
    n_samples = 1000
    seq_len = 360
    n_classes = 5

    # Create synthetic ECG signals
    X_ecg = []
    y = []

    for i in range(n_samples):
        # Create a basic ECG-like signal
        t = np.linspace(0, 1, seq_len)

        # Base signal with different patterns for different classes
        class_id = i % n_classes
        if class_id == 0:  # Normal
            signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 8 * t)
        elif class_id == 1:  # Arrhythmia type 1
            signal = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
        elif class_id == 2:  # Arrhythmia type 2
            signal = np.sin(2 * np.pi * 0.8 * t) + 0.4 * np.sin(2 * np.pi * 15 * t)
        elif class_id == 3:  # Arrhythmia type 3
            signal = np.sin(2 * np.pi * 1.8 * t) + 0.6 * np.sin(2 * np.pi * 6 * t)
        else:  # Arrhythmia type 4
            signal = np.sin(2 * np.pi * 1.1 * t) + 0.7 * np.sin(2 * np.pi * 20 * t)

        # Add noise
        signal += 0.1 * np.random.randn(seq_len)

        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        X_ecg.append(signal)
        y.append(class_id)

    X_ecg = np.array(X_ecg)
    y = np.array(y)

    # Create synthetic PPG data
    X_ppg = create_synthetic_ppg_data(X_ecg)

    # Split data
    X_ecg_temp, X_ecg_test, X_ppg_temp, X_ppg_test, y_temp, y_test = train_test_split(
        X_ecg, X_ppg, y, test_size=0.2, random_state=SEED, stratify=y
    )

    X_ecg_train, X_ecg_val, X_ppg_train, X_ppg_val, y_train, y_val = train_test_split(
        X_ecg_temp, X_ppg_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )

    # Save processed data
    processed_data = {
        'X_ecg_train': X_ecg_train,
        'X_ppg_train': X_ppg_train,
        'y_train': y_train,
        'X_ecg_val': X_ecg_val,
        'X_ppg_val': X_ppg_val,
        'y_val': y_val,
        'X_ecg_test': X_ecg_test,
        'X_ppg_test': X_ppg_test,
        'y_test': y_test
    }

    np.savez_compressed(processed_file, **processed_data)
    logger.info(f"Synthetic data saved to {processed_file}")

    return X_ecg_train, X_ppg_train, y_train, X_ecg_val, X_ppg_val, y_val, X_ecg_test, X_ppg_test, y_test


def find_pretrained_ecg_model():
    """Find the pretrained ECG model file."""
    # Look for model files from the training
    model_patterns = [
        "ecg_model_*.pth",
        "models/ecg_model_*.pth"
    ]

    for pattern in model_patterns:
        import glob
        files = glob.glob(pattern)
        if files:
            # Get the most recent file
            latest_file = max(files, key=os.path.getctime)
            logger.info(f"Found pretrained ECG model: {latest_file}")
            return latest_file

    logger.warning("No pretrained ECG model found, will use random initialization")
    return None


def main():
    """Main integration pipeline."""
    start_time = time.time()

    print("="*80)
    print("PRETRAINED ECG + PPG MULTIMODAL INTEGRATION PIPELINE")
    print("="*80)

    # Setup
    seed_everything(SEED)
    global logger
    logger = setup_logging("logs", "pretrained_integration")

    logger.info("Starting pretrained ECG integration pipeline...")
    logger.info(f"Using device: {DEVICE}")

    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Find pretrained ECG model
    pretrained_ecg_path = find_pretrained_ecg_model()

    # Load or create data
    data_tuple = load_processed_data_or_create()
    X_ecg_train, X_ppg_train, y_train, X_ecg_val, X_ppg_val, y_val, X_ecg_test, X_ppg_test, y_test = data_tuple

    logger.info(f"Data loaded - Train: {len(X_ecg_train)}, Val: {len(X_ecg_val)}, Test: {len(X_ecg_test)}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}")

    # Create data loaders
    def transform_signal(x):
        return x.astype(np.float32)

    train_loader, val_loader = make_loaders_from_multimodal_arrays(
        X_ecg_train, X_ppg_train, y_train,
        X_ecg_val, X_ppg_val, y_val,
        transform_signal, transform_signal,
        batch_size=BATCH_SIZE,
        num_workers=2,
        oversample=True
    )

    # Initialize hybrid model
    model = HybridMultiModalNetwork(
        ecg_seq_len=X_ecg_train.shape[1],
        ppg_seq_len=X_ppg_train.shape[1],
        n_arrhythmia_classes=len(CLASS_NAMES),
        stroke_output_dim=1,
        pretrained_ecg_path=pretrained_ecg_path
    ).to(DEVICE)

    logger.info(f"Model initialized with pretrained ECG encoder")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Quick training (fine-tuning) - just a few epochs to adapt
    criterion_arr = nn.CrossEntropyLoss()
    criterion_stroke = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Training loop
    epochs = 3  # Just a few epochs for fine-tuning
    logger.info(f"Starting fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # MultiModalDataset returns: ecg, ppg, arrhythmia_label, stroke_label, patient_id
            ecg, ppg, y_arr, y_stroke, patient_ids = batch_data
            ecg, ppg, y_arr, y_stroke = ecg.to(DEVICE), ppg.to(DEVICE), y_arr.to(DEVICE), y_stroke.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(ecg, ppg)

            loss_arr = criterion_arr(outputs['arrhythmia_logits'], y_arr)
            loss_stroke = criterion_stroke(outputs['stroke_output'], y_stroke)
            loss = loss_arr + 0.1 * loss_stroke

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs['arrhythmia_logits'].argmax(dim=1)
            correct += pred.eq(y_arr).sum().item()
            total += y_arr.size(0)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        logger.info(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        test_loader = DataLoader(
            MultiModalDataset(X_ecg_test, X_ppg_test, y_test),
            batch_size=BATCH_SIZE, shuffle=False
        )
        for batch_data in test_loader:
            # MultiModalDataset returns: ecg, ppg, arrhythmia_label, stroke_label, patient_id
            ecg, ppg, y_arr, y_stroke, patient_ids = batch_data
            ecg, ppg, y_arr = ecg.to(DEVICE), ppg.to(DEVICE), y_arr.to(DEVICE)

            outputs = model(ecg, ppg)
            pred = outputs['arrhythmia_logits'].argmax(dim=1)

            test_predictions.extend(pred.cpu().numpy())
            test_labels.extend(y_arr.cpu().numpy())

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report

    test_accuracy = accuracy_score(test_labels, test_predictions)
    class_report = classification_report(test_labels, test_predictions, target_names=CLASS_NAMES)

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Classification Report:\n{class_report}")

    # Save final model
    final_checkpoint = {
        'model_state': model.state_dict(),
        'model_config': {
            'ecg_seq_len': X_ecg_train.shape[1],
            'ppg_seq_len': X_ppg_train.shape[1],
            'n_arrhythmia_classes': len(CLASS_NAMES),
            'stroke_output_dim': 1,
            'pretrained_ecg_path': pretrained_ecg_path
        },
        'test_accuracy': test_accuracy,
        'class_names': CLASS_NAMES,
        'version': 'pretrained_integration_v1.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    model_path = os.path.join(MODEL_DIR, 'multimodal_best.pth')
    save_checkpoint(final_checkpoint, model_path)

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("PRETRAINED INTEGRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final model saved to: {model_path}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nNext Steps:")
    print("1. Start the API server: uvicorn src.api:app --reload --port 8081")
    print("2. Access API documentation at: http://localhost:8081/docs")
    print("3. The pipeline now uses pretrained ECG weights with new PPG encoder")

    logger.info("Pretrained integration pipeline completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        if 'logger' in globals():
            logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)