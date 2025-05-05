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
    batch_size: int = 8  # Changed from 32 to 8 to match instance value
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01  # L2 regularization
    
    # Dataset parameters
    initial_data_fraction: float = 0.1  # Fraction of total data to use initially
    train_fraction: float = 0.9  # Fraction of initial data to use for training (rest for testing)
    validation_fraction: float = 0.1  # Fraction of validation data to use
    max_caption_length: int = 32  # Maximum length of captions
    
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
    initial_data_fraction=0.1,  # Use 10% of total data
    train_fraction=0.9,  # Split that 10% into 90% train, 10% test
    validation_fraction=0.1,  # 10% for validation
    
    # Optional
    device=None,  # Auto-select device
    seed=42
) 