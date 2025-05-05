import torch
import torch.nn as nn
from torch.nn import functional as F
from config import config  # Import config from config.py
from datasets import DatasetDict, load_from_disk
import wandb
import tempfile
import os

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
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
        
        # Output layer
        self.output_layer = nn.Linear(d_model, d_model)
        
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
        
        # Final output layer
        output = self.output_layer(tgt)
        
        return output

def load_processed_dataset():
    # Initialize wandb
    wandb.init(project="flickr30k-captioning", name="decoder-training")
    
    try:
        # Try to download the artifact
        artifact = wandb.use_artifact("processed_flickr30k:latest")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the artifact
            artifact_dir = artifact.download(root=tmp_dir)
            print(f"Artifact downloaded to: {artifact_dir}")
            
            # Load the dataset directly from the artifact directory
            # The dataset is stored in the root of the artifact with a train subdirectory
            processed_dataset = load_from_disk(artifact_dir)
            
            print("Successfully loaded processed dataset from wandb!")
            print(f"Dataset size: {len(processed_dataset['train'])} examples")
            return processed_dataset
            
    except Exception as e:
        print(f"Could not load dataset from wandb: {e}")
        print("Please run upload_to_wandb.py first to process and upload the dataset.")
        raise

def train_decoder(processed_dataset, config=config):
    # Ask user for split preference
    print("\nChoose data split option:")
    print("1. Use pre-split data (train/val/test)")
    print("2. Split training data (90% train, 10% test)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        # Split training data into train/test
        train_size = int(0.9 * len(processed_dataset['train']))
        train_data = processed_dataset['train'].select(range(train_size))
        test_data = processed_dataset['train'].select(range(train_size, len(processed_dataset['train'])))
        processed_dataset = DatasetDict({
            'train': train_data,
            'test': test_data
        })
        print(f"\nSplit training data into:")
        print(f"- Training: {len(train_data)} examples")
        print(f"- Testing: {len(test_data)} examples")
    else:
        print("\nUsing pre-split data")
    
    # Initialize model and move to device
    model = TransformerDecoder(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
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
        
        # Process each split based on user choice
        splits = ['train', 'test'] if choice == "2" else ['train', 'val']
        for split in splits:
            # Get the dataset for this split
            dataset = processed_dataset[split]
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(dataset), config.batch_size):
                batch = dataset[i:i+config.batch_size]
                
                # Convert batch data to tensors and move to device
                decoder_inputs = torch.tensor(batch['decoder_inputs'], dtype=torch.float32).to(config.device)
                caption_labels = torch.tensor(batch['caption_labels'], dtype=torch.long).to(config.device)
                
                # Forward pass
                outputs = model(decoder_inputs, decoder_inputs)
                
                # Calculate loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               caption_labels.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss per batch
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{config.num_epochs}, {split} Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    # Load processed dataset from wandb
    processed_dataset = load_processed_dataset()
    
    # Train the decoder
    model = train_decoder(processed_dataset)
