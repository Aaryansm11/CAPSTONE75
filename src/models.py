# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

from src.config import (
    N_ARRHYTHMIA_CLASSES, STROKE_OUTPUT_DIM, DROPOUT_RATE,
    ECG_CNN_FILTERS, PPG_CNN_FILTERS, ECG_LSTM_HIDDEN, PPG_LSTM_HIDDEN,
    FUSION_DIM, ATTENTION_DIM
)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AttentionModule(nn.Module):
    """Self-attention module for signal processing."""
    
    def __init__(self, input_dim: int, attention_dim: int = ATTENTION_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.output_proj = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)  # (batch, seq_len, attention_dim)
        K = self.key(x)    # (batch, seq_len, attention_dim)
        V = self.value(x)  # (batch, seq_len, attention_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        output = self.output_proj(attended)
        
        return output, attention_weights


class CNNEncoder(nn.Module):
    """1D CNN encoder for signal processing."""
    
    def __init__(self, in_channels: int = 1, filters: list = None, 
                 kernel_size: int = 5, dropout: float = DROPOUT_RATE):
        super().__init__()
        
        if filters is None:
            filters = [32, 64, 64, 128]
            
        layers = []
        current_channels = in_channels
        
        for filter_size in filters:
            layers.extend([
                nn.Conv1d(current_channels, filter_size, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filter_size),
                nn.LeakyReLU(0.01, inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.Dropout1d(dropout)
            ])
            current_channels = filter_size
        
        self.cnn = nn.Sequential(*layers)
        self.out_channels = current_channels
    
    def forward(self, x):
        return self.cnn(x)


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = DROPOUT_RATE):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_size = hidden_size * 2  # Bidirectional
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)
        # output: (batch, seq_len, hidden_size * 2)
        # Use mean pooling over sequence dimension
        pooled_output = output.mean(dim=1)  # (batch, hidden_size * 2)
        return pooled_output, output


class SignalBranchEncoder(nn.Module):
    """Combined CNN-LSTM encoder for signal processing."""
    
    def __init__(self, in_channels: int = 1, cnn_filters: list = None,
                 lstm_hidden: int = 128, dropout: float = DROPOUT_RATE,
                 use_attention: bool = True):
        super().__init__()
        
        # CNN feature extraction
        self.cnn = CNNEncoder(in_channels, cnn_filters, dropout=dropout)
        
        # LSTM temporal modeling
        self.lstm = LSTMEncoder(
            input_size=self.cnn.out_channels,
            hidden_size=lstm_hidden,
            dropout=dropout
        )
        
        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(self.lstm.output_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, filters, seq_len')
        
        # Transpose for LSTM: (batch, seq_len', filters)
        lstm_input = cnn_out.transpose(1, 2)
        
        # LSTM processing
        pooled_out, sequence_out = self.lstm(lstm_input)
        
        # Optional attention
        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(sequence_out)
            # Global average pooling of attended features
            pooled_out = attended_out.mean(dim=1)
        
        final_output = self.dropout(pooled_out)
        
        return {
            'embedding': final_output,
            'sequence': sequence_out,
            'attention_weights': attention_weights
        }


class MultiModalFusionNetwork(nn.Module):
    """Multimodal fusion network for ECG and PPG signals."""
    
    def __init__(self, 
                 ecg_seq_len: int,
                 ppg_seq_len: int,
                 n_arrhythmia_classes: int = N_ARRHYTHMIA_CLASSES,
                 stroke_output_dim: int = STROKE_OUTPUT_DIM,
                 ecg_cnn_filters: list = None,
                 ppg_cnn_filters: list = None,
                 ecg_lstm_hidden: int = ECG_LSTM_HIDDEN,
                 ppg_lstm_hidden: int = PPG_LSTM_HIDDEN,
                 fusion_dim: int = FUSION_DIM,
                 dropout: float = DROPOUT_RATE,
                 use_attention: bool = True):
        super().__init__()
        
        # Default filter configurations
        if ecg_cnn_filters is None:
            ecg_cnn_filters = ECG_CNN_FILTERS
        if ppg_cnn_filters is None:
            ppg_cnn_filters = PPG_CNN_FILTERS
        
        # Signal encoders
        self.ecg_encoder = SignalBranchEncoder(
            in_channels=1,
            cnn_filters=ecg_cnn_filters,
            lstm_hidden=ecg_lstm_hidden,
            dropout=dropout,
            use_attention=use_attention
        )
        
        self.ppg_encoder = SignalBranchEncoder(
            in_channels=1,
            cnn_filters=ppg_cnn_filters,
            lstm_hidden=ppg_lstm_hidden,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Calculate fusion input dimension
        ecg_embed_dim = ecg_lstm_hidden * 2  # Bidirectional LSTM
        ppg_embed_dim = ppg_lstm_hidden * 2  # Bidirectional LSTM
        fusion_input_dim = ecg_embed_dim + ppg_embed_dim
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        fused_dim = fusion_dim // 2
        
        # Arrhythmia classification head
        self.arrhythmia_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_arrhythmia_classes)
        )
        
        # Stroke risk prediction head
        self.stroke_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, stroke_output_dim)
        )
        
    def forward(self, ecg_signal, ppg_signal):
        # Encode signals
        ecg_out = self.ecg_encoder(ecg_signal)
        ppg_out = self.ppg_encoder(ppg_signal)
        
        # Extract embeddings
        ecg_embedding = ecg_out['embedding']
        ppg_embedding = ppg_out['embedding']
        
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
            'fused_features': fused_output,
            'ecg_attention': ecg_out['attention_weights'],
            'ppg_attention': ppg_out['attention_weights']
        }


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for sequences."""
    
    def __init__(self, input_dim: int, model_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, ff_dim: int = 512, dropout: float = DROPOUT_RATE):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = model_dim
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer
        output = self.transformer(x)  # (batch, seq_len, model_dim)
        
        # Global pooling
        pooled = output.mean(dim=1)  # (batch, model_dim)
        
        return pooled, output


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
        
    def forward(self, ecg_signal, ppg_signal):
        outputs = []
        
        for model in self.models:
            output = model(ecg_signal, ppg_signal)
            outputs.append(output)
        
        # Weighted average of predictions
        arrhythmia_logits = torch.stack([out['arrhythmia_logits'] for out in outputs])
        stroke_outputs = torch.stack([out['stroke_output'] for out in outputs])
        
        # Apply weights
        weighted_arrhythmia = torch.sum(arrhythmia_logits * self.weights.view(-1, 1, 1), dim=0)
        weighted_stroke = torch.sum(stroke_outputs * self.weights.view(-1, 1), dim=0)
        
        return {
            'arrhythmia_logits': weighted_arrhythmia,
            'stroke_output': weighted_stroke,
            'individual_outputs': outputs
        }


def create_model(model_type: str = 'multimodal_cnn_lstm', **kwargs):
    """Factory function to create models."""
    
    if model_type == 'multimodal_cnn_lstm':
        return MultiModalFusionNetwork(**kwargs)
    elif model_type == 'transformer':
        # Create transformer-based model (placeholder for future implementation)
        raise NotImplementedError("Transformer model not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def initialize_weights(model):
    """Initialize model weights using appropriate strategies."""
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    def __init__(self, **kwargs):
        # Default configurations
        self.ecg_seq_len = kwargs.get('ecg_seq_len', 3600)
        self.ppg_seq_len = kwargs.get('ppg_seq_len', 1250)
        self.n_arrhythmia_classes = kwargs.get('n_arrhythmia_classes', N_ARRHYTHMIA_CLASSES)
        self.stroke_output_dim = kwargs.get('stroke_output_dim', STROKE_OUTPUT_DIM)
        
        # CNN configurations
        self.ecg_cnn_filters = kwargs.get('ecg_cnn_filters', ECG_CNN_FILTERS)
        self.ppg_cnn_filters = kwargs.get('ppg_cnn_filters', PPG_CNN_FILTERS)
        
        # LSTM configurations
        self.ecg_lstm_hidden = kwargs.get('ecg_lstm_hidden', ECG_LSTM_HIDDEN)
        self.ppg_lstm_hidden = kwargs.get('ppg_lstm_hidden', PPG_LSTM_HIDDEN)
        
        # Fusion configurations
        self.fusion_dim = kwargs.get('fusion_dim', FUSION_DIM)
        self.dropout = kwargs.get('dropout', DROPOUT_RATE)
        self.use_attention = kwargs.get('use_attention', True)
        
        # Training configurations
        self.use_class_weights = kwargs.get('use_class_weights', True)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'ecg_seq_len': self.ecg_seq_len,
            'ppg_seq_len': self.ppg_seq_len,
            'n_arrhythmia_classes': self.n_arrhythmia_classes,
            'stroke_output_dim': self.stroke_output_dim,
            'ecg_cnn_filters': self.ecg_cnn_filters,
            'ppg_cnn_filters': self.ppg_cnn_filters,
            'ecg_lstm_hidden': self.ecg_lstm_hidden,
            'ppg_lstm_hidden': self.ppg_lstm_hidden,
            'fusion_dim': self.fusion_dim,
            'dropout': self.dropout,
            'use_attention': self.use_attention,
            'use_class_weights': self.use_class_weights,
            'label_smoothing': self.label_smoothing
        }