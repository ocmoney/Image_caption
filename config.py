import torch
from dataclasses import dataclass

@dataclass
class DecoderConfig:
    # Model architecture parameters
    d_model: int = 768  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 2048  # Dimension of the feedforward network
    dropout: float = 0.1  # Dropout rate
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 128  # Increased batch size for better training efficiency
    
    # Dataset parameters
    train_fraction: float = 0.9  # Fraction of training set to use for training
    max_caption_length: int = 32  # Maximum length of captions
    
    # Device configuration
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optional parameters
    seed: int = 42  # Random seed for reproducibility
    
    def __post_init__(self):
        # Validate train_fraction
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be between 0.0 and 1.0")
        
        # Set random seed
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

# Single configuration instance
config = DecoderConfig() 