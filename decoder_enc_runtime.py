import torch
import torch.nn as nn
from torch.nn import functional as F
from config import config  # Import config from config.py
from datasets import DatasetDict, load_from_disk
import wandb
import tempfile
import os
import time
from tqdm import tqdm

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=50260):  # GPT-2's vocabulary size + 3 special tokens
        super().__init__()
        self.d_model = d_model
        
        # Multi-head attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary size
        self.output_projection = nn.Linear(d_model, vocab_size)