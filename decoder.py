import torch
import torch.nn as nn
from torch.nn import functional as F
from config import config  # Import config from config.py
from datasets import DatasetDict, load_from_disk
import wandb
import tempfile
import os
import time

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
        
    def create_caption_mask(self, seq_len):
        # Create a mask for the caption part (after image patches)
        mask = torch.ones(seq_len, seq_len) * float('-inf')
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

def load_processed_dataset():
    # Initialize wandb
    wandb.init(project="flickr30k", name="decoder-training")
    
    try:
        # Try to download the artifacts
        train_artifact = wandb.use_artifact("processed_flickr30k_train:latest")
        val_artifact = wandb.use_artifact("processed_flickr30k_val:latest")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the artifacts
            train_dir = train_artifact.download(root=tmp_dir)
            val_dir = val_artifact.download(root=tmp_dir)
            print(f"Artifacts downloaded to: {tmp_dir}")
            
            # Load the tensors
            train_data = torch.load(os.path.join(train_dir, 'train_dataset.pt'))
            val_data = torch.load(os.path.join(val_dir, 'val_dataset.pt'))
            
            print("Successfully loaded processed dataset from wandb!")
            print(f"Training set size: {train_data['decoder_inputs'].size(1)} examples")
            print(f"Validation set size: {val_data['decoder_inputs'].size(1)} examples")
            return train_data, val_data
            
    except Exception as e:
        print(f"Could not load dataset from wandb: {e}")
        print("Please run upload_to_wandb.py first to process and upload the dataset.")
        raise

def train_decoder(train_data, val_data, config=config):
    # Initialize wandb run
    wandb.init(project="flickr30k", name="decoder-training")
    
    # Start timing
    start_time = time.time()
    
    # Split training data according to train_fraction
    total_examples = train_data['decoder_inputs'].size(1)
    train_size = int(config.train_fraction * total_examples)
    
    # Split the data
    train_data = {
        'decoder_inputs': train_data['decoder_inputs'][:, :train_size, :],
        'caption_input_ids': train_data['caption_input_ids'][:train_size, :],
        'caption_labels': train_data['caption_labels'][:train_size, :]
    }
    
    print(f"\nUsing {train_size} examples ({config.train_fraction*100}%) of the training set")
    
    # Initialize model and move to device
    model = TransformerDecoder(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        vocab_size=50260  # GPT-2's vocabulary size + 3 special tokens
    ).to(config.device)
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        
        # Process training data
        total_loss = 0
        num_batches = 0
        total_correct = 0
        total_tokens = 0
        
        # Process in batches
        batch_size = config.batch_size
        num_examples = train_data['decoder_inputs'].size(1)
        
        for i in range(0, num_examples, batch_size):
            # Get batch
            batch_end = min(i + batch_size, num_examples)
            current_batch_size = batch_end - i
            
            # Get decoder inputs [seq_len, batch_size, d_model]
            decoder_inputs = train_data['decoder_inputs'][:, i:batch_end, :].to(config.device)
            
            # Get labels [batch_size, seq_len]
            caption_labels = train_data['caption_labels'][i:batch_end, :].to(config.device)
            
            # Forward pass
            outputs = model(decoder_inputs, decoder_inputs)  # [seq_len, batch_size, vocab_size]
            
            # Get only the caption part of the outputs (last config.max_caption_length tokens)
            caption_outputs = outputs[-config.max_caption_length:, :, :]  # [max_caption_length, batch_size, vocab_size]
            
            # Reshape outputs and labels for loss calculation
            caption_outputs = caption_outputs.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
            
            # Calculate loss
            loss = criterion(caption_outputs.reshape(-1, caption_outputs.size(-1)), 
                           caption_labels.reshape(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(caption_outputs, dim=-1)  # [batch_size, seq_len]
            correct = (predictions == caption_labels).sum().item()
            total_correct += correct
            total_tokens += predictions.numel()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Calculate and log total runtime
    runtime = time.time() - start_time
    wandb.log({"total_runtime": runtime})
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    
    # Finish wandb run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    # Load processed dataset from wandb
    train_data, val_data = load_processed_dataset()
    
    # Train the decoder
    model = train_decoder(train_data, val_data)
