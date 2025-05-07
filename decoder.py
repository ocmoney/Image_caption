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

def load_processed_dataset():
    # Initialize wandb
    wandb.init(project="flickr30k", name="decoder-training")
    
    try:
        # Try to download the artifact
        artifact = wandb.use_artifact("processed_flickr30k:latest")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the artifact
            artifact_dir = artifact.download(root=tmp_dir)
            print(f"Artifact downloaded to: {artifact_dir}")
            
            # Load the tensors
            data = torch.load(os.path.join(artifact_dir, 'processed_dataset.pt'))
            
            # Get shapes
            num_images = data['image_embeddings'].size(0)  # [num_images, 196, 768]
            total_captions = data['caption_embeddings'].size(1)  # [caption_len, total_captions, 768]
            
            # Create decoder inputs by expanding image embeddings for each caption
            decoder_inputs = []
            for img_idx, (start_idx, end_idx) in enumerate(data['image_to_caption_map']):
                # Get image embeddings for this image
                img_emb = data['image_embeddings'][img_idx]  # [196, 768]
                # Get caption embeddings for this image's captions
                cap_emb = data['caption_embeddings'][:, start_idx:end_idx, :]  # [caption_len, num_captions, 768]
                # Expand image embeddings for each caption
                img_emb_expanded = img_emb.unsqueeze(1).expand(-1, end_idx - start_idx, -1)  # [196, num_captions, 768]
                # Concatenate along sequence dimension
                decoder_inputs.append(torch.cat([img_emb_expanded, cap_emb], dim=0))  # [seq_len, num_captions, 768]
            
            # Concatenate all decoder inputs
            decoder_inputs = torch.cat(decoder_inputs, dim=1)  # [seq_len, total_captions, 768]
            
            # Create the format expected by the training code
            processed_data = {
                'decoder_inputs': decoder_inputs,
                'caption_input_ids': data['caption_input_ids'],
                'caption_labels': data['caption_labels']
            }
            
            print("Successfully loaded processed dataset from wandb!")
            print(f"Dataset size: {processed_data['decoder_inputs'].size(1)} examples")
            return processed_data
            
    except Exception as e:
        print(f"Could not load dataset from wandb: {e}")
        print("Please run upload_to_wandb.py first to process and upload the dataset.")
        raise

def train_decoder(processed_data, config=config):
    # Initialize wandb run
    wandb.init(project="flickr30k", name="decoder-training")
    
    # Print device information
    print(f"\nUsing device: {config.device}")
    if config.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Set GPU memory allocation
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    
    # Start timing
    start_time = time.time()
    
    # Calculate total examples and split size
    total_examples = processed_data['decoder_inputs'].size(1)
    train_size = int(config.train_fraction * total_examples)
    
    print(f"\nSplitting {total_examples} examples into train/test sets...")
    
    # Move all input data to GPU at once
    print("Moving input data to GPU...")
    processed_data = {k: v.to(config.device, non_blocking=True) for k, v in processed_data.items()}
    
    # Split the data with progress bar
    train_data = {
        'decoder_inputs': torch.zeros((processed_data['decoder_inputs'].size(0), train_size, processed_data['decoder_inputs'].size(2)), device=config.device),
        'caption_input_ids': torch.zeros((train_size, processed_data['caption_input_ids'].size(1)), device=config.device),
        'caption_labels': torch.zeros((train_size, processed_data['caption_labels'].size(1)), device=config.device)
    }
    
    test_data = {
        'decoder_inputs': torch.zeros((processed_data['decoder_inputs'].size(0), total_examples - train_size, processed_data['decoder_inputs'].size(2)), device=config.device),
        'caption_input_ids': torch.zeros((total_examples - train_size, processed_data['caption_input_ids'].size(1)), device=config.device),
        'caption_labels': torch.zeros((total_examples - train_size, processed_data['caption_labels'].size(1)), device=config.device)
    }
    
    # Copy data with progress bar
    print("\nLoading data to GPU...")
    for i in tqdm(range(total_examples), desc="Splitting data"):
        if i < train_size:
            train_data['decoder_inputs'][:, i, :] = processed_data['decoder_inputs'][:, i, :]
            train_data['caption_input_ids'][i, :] = processed_data['caption_input_ids'][i, :]
            train_data['caption_labels'][i, :] = processed_data['caption_labels'][i, :]
        else:
            test_idx = i - train_size
            test_data['decoder_inputs'][:, test_idx, :] = processed_data['decoder_inputs'][:, i, :]
            test_data['caption_input_ids'][test_idx, :] = processed_data['caption_input_ids'][i, :]
            test_data['caption_labels'][test_idx, :] = processed_data['caption_labels'][i, :]
    
    # Clear original data from GPU
    del processed_data
    torch.cuda.empty_cache()
    
    print(f"\nSplit data into:")
    print(f"- Training: {train_size} examples")
    print(f"- Testing: {total_examples - train_size} examples")
    
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
    criterion = nn.CrossEntropyLoss().to(config.device)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        
        # Process each split
        for split_name, split_data in [('train', train_data), ('test', test_data)]:
            total_loss = 0
            num_batches = 0
            total_correct = 0
            total_tokens = 0
            
            # Process in batches
            batch_size = 32  # Increased from 2 to 32 to better utilize GPU memory
            num_examples = split_data['decoder_inputs'].size(1)
            
            print(f"\nProcessing {num_examples} examples in batches of {batch_size}")
            
            for i in tqdm(range(0, num_examples, batch_size), desc=f"Training {split_name}"):
                # Get batch
                batch_end = min(i + batch_size, num_examples)
                current_batch_size = batch_end - i
                
                # Get decoder inputs [seq_len, batch_size, d_model]
                decoder_inputs = split_data['decoder_inputs'][:, i:batch_end, :]
                
                # Get labels [batch_size, seq_len]
                caption_labels = split_data['caption_labels'][i:batch_end, :]
                
                # Forward pass
                outputs = model(decoder_inputs, decoder_inputs)  # [seq_len, batch_size, vocab_size]
                
                # Get only the caption part of the outputs (last config.max_caption_length tokens)
                caption_outputs = outputs[-config.max_caption_length:, :, :]  # [max_caption_length, batch_size, vocab_size]
                
                # Reshape outputs and labels for loss calculation
                caption_outputs = caption_outputs.permute(1, 0, 2)  # [batch_size, seq_len, vocab_size]
                
                # Debug prints for first batch
                if i == 0:
                    print(f"\nDevice and type information:")
                    print(f"Device: {config.device}")
                    print(f"caption_outputs: {caption_outputs.dtype}, device: {caption_outputs.device}")
                    print(f"caption_labels: {caption_labels.dtype}, device: {caption_labels.device}")
                    print(f"Output shape: {caption_outputs.shape}")
                    print(f"Labels shape: {caption_labels.shape}")
                
                # Convert labels to long type for GPU
                if config.device == 'cuda':
                    caption_labels = caption_labels.long()
                
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
                
                # Clear GPU cache periodically
                if num_batches % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate metrics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
            
            # Log metrics to wandb
            wandb.log({
                f"{split_name}_loss": avg_loss,
                f"{split_name}_accuracy": accuracy,
                "epoch": epoch
            })
            
            print(f"Epoch {epoch+1}/{config.num_epochs}, {split_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Calculate and log total runtime
    runtime = time.time() - start_time
    wandb.log({"total_runtime": runtime})
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    
    # Finish wandb run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    # Load processed dataset from wandb
    processed_data = load_processed_dataset()
    
    # Train the decoder
    model = train_decoder(processed_data)
