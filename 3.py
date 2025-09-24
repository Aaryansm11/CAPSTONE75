import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.font_manager import FontProperties
import pandas as pd
import os
import wfdb
import logging
import datetime
import time

# Configure logging
log_filename = f"ecg_model_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout = nn.Dropout(0.45)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(128, 210, batch_first=True)
        self.lstm2 = nn.LSTM(210, 190, batch_first=True)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_dim // 8  # After 3 pooling layers with stride 2
        
        # Dense layers
        self.fc1 = nn.Linear(190 * conv_output_size, 5)
        self.fc2 = nn.Linear(5, 5)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.001)
        
    def forward(self, x):
        # CNN layers
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        
        x = self.pool(x)
        x = self.dropout(x)
        
        # LSTM layers
        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, features)
        x, _ = self.lstm1(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        x, _ = self.lstm2(x)
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def cnn_lstm_standalone(X_train, Y_train, X_test, Y_test, epochs=2, title="CNN-LSTM Model"):
    logger.info("Starting CNN-LSTM model training")
    logger.info(f"Input shapes - X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log class distribution
    train_class_dist = np.bincount(Y_train)
    test_class_dist = np.bincount(Y_test)
    logger.info(f"Training set class distribution: {train_class_dist}")
    logger.info(f"Test set class distribution: {test_class_dist}")
    
    # Reshape input
    X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
    
    # Convert to PyTorch tensors
    logger.info("Converting data to PyTorch tensors")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    # Create dataloaders with optimizations
    batch_size = 128  # Increased batch size for GPU
    logger.info(f"Creating DataLoaders with batch size {batch_size}")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model and move to GPU
    logger.info(f"Initializing model with input dimension {X_train.shape[2]}")
    model = CNNLSTM(X_train.shape[2])
    model = model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger.info("Using Adam optimizer with learning rate 0.001 and CrossEntropyLoss")
    
    # Training loop
    train_acc_history = []
    val_acc_history = []
    loss_history = []
    
    logger.info(f"Starting training for {epochs} epochs")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        logger.info(f"Epoch {epoch+1}/{epochs} - Training phase")
        batch_times = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}, "
                           f"Batch time: {batch_times[-1]:.4f}s")
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_acc_history.append(train_acc)
        loss_history.append(train_loss)
        
        logger.info(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, "
                  f"Avg batch time: {avg_batch_time:.4f}s")
        
        # Validation phase
        logger.info(f"Epoch {epoch+1}/{epochs} - Validation phase")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        class_correct = [0] * 5  # Assuming 5 classes (N, S, V, F, U)
        class_total = [0] * 5
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if label < len(class_total):  # Safety check
                        class_total[label] += 1
                        if label == pred:
                            class_correct[label] += 1
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        val_acc_history.append(val_acc)
        
        # Per-class accuracy
        class_accuracy = []
        for i in range(5):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accuracy.append(f"Class {i}: {class_acc:.2f}%")
            else:
                class_accuracy.append(f"Class {i}: N/A")
        
        logger.info(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        logger.info(f"  Per-class accuracy: {', '.join(class_accuracy)}")
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logger.info(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # Plot Accuracy
    font = FontProperties()
    font.set_family('serif bold')
    font.set_style('oblique')
    font.set_weight('semibold')
    
    plt.figure(figsize=(12, 12))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs', fontproperties=font, fontsize=12)
    plt.ylabel('Accuracy', fontproperties=font, fontsize=12)
    plt.legend(loc='lower right')
    plt.title(title, fontproperties=font, fontsize=16)
    
    # Save plot
    accuracy_plot_path = f"accuracy_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(accuracy_plot_path)
    logger.info(f"Saved accuracy plot to {accuracy_plot_path}")
    plt.show()
    
    # Plot Loss
    plt.figure(figsize=(12, 12))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epochs', fontproperties=font, fontsize=12)
    plt.ylabel('Loss', fontproperties=font, fontsize=12)
    plt.legend(loc='upper right')
    plt.title(f"{title} - Loss", fontproperties=font, fontsize=16)
    
    # Save plot
    loss_plot_path = f"loss_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(loss_plot_path)
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.show()
    
    # Evaluation
    logger.info("Performing final model evaluation")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    unique_labels = sorted(list(set(all_labels)))
    target_names = ["N", "V", "F"]  # Only using the classes that exist in our data
    class_report = classification_report(all_labels, all_predictions, target_names=target_names)
    logger.info(f"Classification Report:\n{class_report}")
    print(class_report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    plt.figure(figsize=(12, 10))
    sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt='.1g', square=True, linewidths=2, linecolor='white')
    plt.title("Confusion Matrix Heat Map", fontproperties=font, fontsize=16)
    
    # Save confusion matrix
    cm_path = f"confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    plt.show()
    
    # Save model
    model_path = f"ecg_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    return model

def load_mit_bih_data(path, classes=None, window_size=360, overlap=0.5):
    """
    Load MIT-BIH data and annotations.
    
    Args:
        path: Path to MIT-BIH database
        classes: Dictionary mapping beat types to class indices
        window_size: Size of the sliding window (in samples)
        overlap: Overlap between consecutive windows (0-1)
        
    Returns:
        X: ECG signal windows
        y: Beat type labels
    """
    if classes is None:
        # Using N, S, V, F, U as your 5 classes
        classes = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'U': 4}
    
    logger.info(f"Loading MIT-BIH data with window size {window_size} and class mapping: {classes}")
    
    # Read RECORDS file to get list of records
    with open(os.path.join(path, 'RECORDS'), 'r') as f:
        records = [line.strip() for line in f]
    
    logger.info(f"Found {len(records)} records in the dataset")
    
    X, y = [], []
    record_stats = {}
    
    for record in records:
        # Read record data
        try:
            logger.info(f"Processing record {record}...")
            signals, fields = wfdb.rdsamp(os.path.join(path, record))
            annotations = wfdb.rdann(os.path.join(path, record), 'atr')
            
            # Get MLII lead (usually the first lead)
            signal = signals[:, 0]
            
            # Get beat annotations
            beat_types = annotations.symbol
            beat_locs = annotations.sample
            
            record_beat_counts = {beat_type: 0 for beat_type in classes.keys()}
            total_beats = 0
            
            # Process each beat
            for i, (loc, beat_type) in enumerate(zip(beat_locs, beat_types)):
                # Skip if beat type is not in our classes
                if beat_type not in classes:
                    continue
                
                # Extract window centered on beat
                start = max(0, loc - window_size // 2)
                end = min(len(signal), loc + window_size // 2)
                
                # Skip if window is too short
                if end - start < window_size:
                    continue
                
                window = signal[start:end]
                if len(window) == window_size:
                    X.append(window)
                    y.append(classes[beat_type])
                    record_beat_counts[beat_type] += 1
                    total_beats += 1
            
            record_stats[record] = {
                "total_beats": total_beats,
                "beat_counts": record_beat_counts
            }
            
            logger.info(f"  Record {record}: Extracted {total_beats} beats - {record_beat_counts}")
            
        except Exception as e:
            logger.error(f"Error processing record {record}: {e}")
            continue
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    logger.info(f"Data loading complete. Total samples: {len(X_array)}")
    logger.info(f"Class distribution: {np.bincount(y_array)}")
    
    return X_array, y_array

def preprocess_and_train():
    """
    Load data, preprocess, and train the CNN-LSTM model
    """
    logger.info("=" * 80)
    logger.info("Starting MIT-BIH Arrhythmia classification with CNN-LSTM model")
    logger.info("=" * 80)
    
    # Check if wfdb is installed
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
    
    # Load data
    data_path = 'mit-bih-arrhythmia-database-1.0.0'
    logger.info(f"Loading data from {data_path}...")
    X, y = load_mit_bih_data(data_path)
    
    logger.info(f"Loaded {len(X)} ECG segments with {len(np.unique(y))} classes")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Normalize data
    logger.info("Normalizing data...")
    start_time = time.time()
    scaler = StandardScaler()
    X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])
    logger.info(f"Normalization completed in {time.time() - start_time:.2f}s")
    
    # Split into train and test sets
    logger.info("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train CNN-LSTM model
    logger.info("Training CNN-LSTM model...")
    model = cnn_lstm_standalone(
        X_train, y_train, X_test, y_test, 
        epochs=2, 
        title="ECG Arrhythmia Classification"
    )
    
    logger.info("Training complete!")
    return model

if __name__ == "__main__":
    preprocess_and_train()