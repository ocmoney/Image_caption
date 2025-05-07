import torch
import torch.nn as nn
from torch.nn import functional as F
from config import config
import wandb
import time
from tqdm import tqdm
from Liam_dataset import Flickr30k

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=50260):
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
        
    def create_caption_mask(self, seq_len):
        # Create a mask for the caption part (after image patches)
        mask = torch.ones(seq_len, seq_len, device=self.self_attn.in_proj_weight.device) * float('-inf')
        # Allow attention to image patches
        mask[:, :16*768] = 0  # 16 patches × 768 dimensions
        # Create causal mask for caption
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def forward(self, tgt, memory):
        """
        Args:
            tgt: Target sequence (caption) [seq_len, batch_size, d_model]
            memory: Memory from encoder (image patches + caption) [seq_len, batch_size, d_model]
        """
        # Get sequence length for mask creation
        seq_len = tgt.size(0)
        
        # Create attention mask for caption part
        tgt_mask = self.create_caption_mask(seq_len)
        
        # Ensure all tensors are on the same device
        tgt = tgt.to(self.self_attn.in_proj_weight.device)
        memory = memory.to(self.self_attn.in_proj_weight.device)
        
        # Self-attention block
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        # Feedforward network
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear1(tgt2)
        tgt2 = F.relu(tgt2)
        tgt2 = self.dropout(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout(tgt2)
        
        # Project to vocabulary size
        output = self.output_projection(tgt)
        
        return output

def process_batch(dataset, start_idx, batch_size):
    """Process a batch of sequences from the dataset"""
    sequences = []
    for i in range(batch_size):
        idx = start_idx + i
        if idx >= len(dataset):
            break
        sequence = dataset[idx]
        sequences.append(sequence)
    return torch.stack(sequences)

def train_decoder(config=config):
    # Initialize wandb run
    wandb.init(project="flickr30k", name="decoder-training")
    
    # Print device information
    print(f"\nUsing device: {config.device}")
    if config.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = Flickr30k(split="train")
    test_dataset = Flickr30k(split="test")
    
    # Calculate number of batches
    train_batches = len(train_dataset) // config.batch_size
    test_batches = len(test_dataset) // config.batch_size
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} examples ({train_batches} batches)")
    print(f"Testing: {len(test_dataset)} examples ({test_batches} batches)")
    
    # Initialize model
    model = TransformerDecoder(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        vocab_size=50260
    ).to(config.device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(config.device)
    
    # Start timing
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_tokens = 0
        
        # Process training data in batches
        for batch_idx in tqdm(range(train_batches), desc="Training"):
            # Get batch data
            start_idx = batch_idx * config.batch_size
            sequences = process_batch(train_dataset, start_idx, config.batch_size)
            sequences = sequences.to(config.device)
            
            # Permute for transformer [seq_len, batch_size, d_model]
            sequences = sequences.permute(1, 0, 2)
            
            # Forward pass
            outputs = model(sequences, sequences)
            
            # Get only the caption part of the outputs (last 25 tokens)
            caption_outputs = outputs[-25:, :, :]
            
            # Reshape outputs for loss calculation
            caption_outputs = caption_outputs.permute(1, 0, 2)
            caption_outputs = caption_outputs.reshape(-1, caption_outputs.size(-1))
            
            # Create dummy labels for now (we'll need to modify this based on actual caption data)
            caption_labels = torch.zeros(caption_outputs.size(0), dtype=torch.long, device=config.device)
            
            # Calculate loss
            loss = criterion(caption_outputs, caption_labels)
            
            # Calculate accuracy
            predictions = torch.argmax(caption_outputs, dim=-1)
            correct = (predictions == caption_labels).sum().item()
            train_correct += correct
            train_tokens += predictions.numel()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        avg_train_loss = train_loss / train_batches
        train_accuracy = (train_correct / train_tokens) * 100 if train_tokens > 0 else 0
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_tokens = 0
        
        with torch.no_grad():
            for batch_idx in tqdm(range(test_batches), desc="Testing"):
                # Get batch data
                start_idx = batch_idx * config.batch_size
                sequences = process_batch(test_dataset, start_idx, config.batch_size)
                sequences = sequences.to(config.device)
                
                # Permute for transformer
                sequences = sequences.permute(1, 0, 2)
                
                # Forward pass
                outputs = model(sequences, sequences)
                
                # Get only the caption part of the outputs
                caption_outputs = outputs[-25:, :, :]
                
                # Reshape outputs
                caption_outputs = caption_outputs.permute(1, 0, 2)
                caption_outputs = caption_outputs.reshape(-1, caption_outputs.size(-1))
                
                # Create dummy labels
                caption_labels = torch.zeros(caption_outputs.size(0), dtype=torch.long, device=config.device)
                
                # Calculate loss
                loss = criterion(caption_outputs, caption_labels)
                
                # Calculate accuracy
                predictions = torch.argmax(caption_outputs, dim=-1)
                correct = (predictions == caption_labels).sum().item()
                test_correct += correct
                test_tokens += predictions.numel()
                
                test_loss += loss.item()
        
        # Calculate test metrics
        avg_test_loss = test_loss / test_batches
        test_accuracy = (test_correct / test_tokens) * 100 if test_tokens > 0 else 0
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "epoch": epoch
        })
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Testing  - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Calculate and log total runtime
    runtime = time.time() - start_time
    wandb.log({"total_runtime": runtime})
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    
    # Finish wandb run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    model = train_decoder()
