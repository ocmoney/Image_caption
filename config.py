import torch
from dataclasses import dataclass

@dataclass
class DecoderConfig:
    # Model architecture parameters by default
    d_model: int = 768  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 2048  # Dimension of feedforward network
    dropout: float = 0.1  # Dropout rate
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01  # L2 regularization
    
    # Dataset parameters
    train_fraction: float = 0.1  # Fraction of training data to use (0.0 to 1.0)
    validation_fraction: float = 0.1  # Fraction of validation data to use
    
    # Optional parameters
    device: str = None  # If None, will use CUDA if available, else CPU
    seed: int = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        # Validate fractions
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be between 0.0 and 1.0")
        if not 0.0 < self.validation_fraction <= 1.0:
            raise ValueError("validation_fraction must be between 0.0 and 1.0")
        
        # Set device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set random seed
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.seed)

# Single configuration instance - modify these values as needed
config = DecoderConfig(
    # Model architecture
    d_model=768,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    
    # Training
    batch_size=8,
    learning_rate=1e-4,
    num_epochs=10,
    weight_decay=0.01,
    
    # Dataset
    train_fraction=0.001,# Use all training data
    validation_fraction=0.001,  # Use all validation data
    
    # Optional
    device=None,  # Auto-select device
    seed=42
) 