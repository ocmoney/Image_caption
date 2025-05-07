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
    wandb.init(project="image-caption", name="decoder-training")
    
    try:
        # Get chunk artifacts
        api = wandb.Api()
        
        # List of chunks to try loading
        train_chunks = [1,2,3,4,5,6,7,8,9,10]  # Training chunks
        test_chunk = 46  # Test chunk
        
        print("Loading chunks from wandb...")
        
        # Create a single temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download all artifacts at once
            artifacts = {}
            for chunk_num in train_chunks + [test_chunk]:
                try:
                    artifact = api.artifact(f"olliecumming3-machine-learning-institute/image-caption/flickr30k_sequences_chunk_{chunk_num}:latest")
                    artifact_dir = artifact.download(root=tmp_dir)
                    artifacts[chunk_num] = artifact_dir
                    print(f"Downloaded chunk {chunk_num}")
                except Exception as e:
                    print(f"Could not download chunk {chunk_num}: {e}")
                    if chunk_num == test_chunk:
                        raise
            
            # Load all chunks in parallel
            train_sequences = []
            test_sequences = []
            
            # Load test chunk
            test_file = os.path.join(artifacts[test_chunk], f"sequences_chunk_{test_chunk}.pt")
            test_sequences = torch.load(test_file)
            print(f"Loaded test chunk {test_chunk}")
            
            # Load training chunks
            for chunk_num in train_chunks:
                if chunk_num in artifacts:
                    chunk_file = os.path.join(artifacts[chunk_num], f"sequences_chunk_{chunk_num}.pt")
                    chunk_sequences = torch.load(chunk_file)
                    train_sequences.extend(chunk_sequences)
                    print(f"Loaded training chunk {chunk_num}")
            
            if not train_sequences:
                raise ValueError("Could not load any training data")
            
            print(f"\nTotal sequences loaded:")
            print(f"Training: {len(train_sequences)} examples")
            print(f"Testing: {len(test_sequences)} examples")
            
            # Process training data
            train_sequences = torch.stack(train_sequences)
            if len(train_sequences.shape) == 4:
                train_sequences = train_sequences.squeeze(1)
            if train_sequences.size(1) == 768:
                train_sequences = train_sequences.permute(0, 2, 1)
            
            # Process test data
            test_sequences = torch.stack(test_sequences)
            if len(test_sequences.shape) == 4:
                test_sequences = test_sequences.squeeze(1)
            if test_sequences.size(1) == 768:
                test_sequences = test_sequences.permute(0, 2, 1)
            
            # Use batch size of 64
            batch_size = 64
            
            # Process training data
            train_examples = train_sequences.size(0)
            train_batches = train_examples // batch_size
            if train_batches == 0:
                raise ValueError(f"Not enough training examples ({train_examples}) for batch size {batch_size}")
            train_sequences = train_sequences[:train_batches * batch_size]
            train_sequences = train_sequences.view(train_batches, batch_size, train_sequences.size(1), train_sequences.size(2))
            
            # Process test data
            test_examples = test_sequences.size(0)
            test_batches = test_examples // batch_size
            if test_batches == 0:
                raise ValueError(f"Not enough test examples ({test_examples}) for batch size {batch_size}")
            test_sequences = test_sequences[:test_batches * batch_size]
            test_sequences = test_sequences.view(test_batches, batch_size, test_sequences.size(1), test_sequences.size(2))
            
            # Get sequence dimensions
            seq_len = train_sequences.size(2)
            embedding_dim = train_sequences.size(3)
            caption_len = seq_len - 196  # 196 is the number of image patches
            
            # Create the format expected by the training code
            processed_data = {
                'decoder_inputs': torch.cat([train_sequences, test_sequences], dim=0),  # Combine for initial processing
                'caption_input_ids': torch.zeros((train_batches + test_batches, batch_size, caption_len), dtype=torch.long),
                'caption_labels': torch.zeros((train_batches + test_batches, batch_size, caption_len), dtype=torch.long)
            }
            
            print("\nSuccessfully loaded processed dataset from wandb!")
            print(f"Training: {train_batches} batches ({train_batches * batch_size} examples)")
            print(f"Testing: {test_batches} batches ({test_batches * batch_size} examples)")
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
    
    # Get number of batches and calculate train/test split
    num_batches = processed_data['decoder_inputs'].size(0)
    train_batches = int(config.train_fraction * num_batches)
    
    print(f"\nSplitting {num_batches} batches into train/test sets...")
    
    # Move all input data to GPU at once
    print("Moving input data to GPU...")
    processed_data = {k: v.to(config.device, non_blocking=True) for k, v in processed_data.items()}
    
    # Split the data into train and test sets
    train_data = {
        'decoder_inputs': processed_data['decoder_inputs'][:train_batches],
        'caption_input_ids': processed_data['caption_input_ids'][:train_batches],
        'caption_labels': processed_data['caption_labels'][:train_batches]
    }
    
    test_data = {
        'decoder_inputs': processed_data['decoder_inputs'][train_batches:],
        'caption_input_ids': processed_data['caption_input_ids'][train_batches:],
        'caption_labels': processed_data['caption_labels'][train_batches:]
    }
    
    # Clear original data from GPU
    del processed_data
    torch.cuda.empty_cache()
    
    print(f"\nSplit data into:")
    print(f"- Training: {train_batches} batches ({train_batches * config.batch_size} examples)")
    print(f"- Testing: {num_batches - train_batches} batches ({(num_batches - train_batches) * config.batch_size} examples)")
    
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
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_tokens = 0
        
        # Process training data
        for batch_idx in tqdm(range(len(train_data['decoder_inputs'])), desc="Training"):
            # Get batch data
            decoder_inputs = train_data['decoder_inputs'][batch_idx]  # [batch_size, seq_len, 768]
            caption_labels = train_data['caption_labels'][batch_idx]  # [batch_size, caption_len]
            
            # Permute decoder inputs for transformer [seq_len, batch_size, d_model]
            decoder_inputs = decoder_inputs.permute(1, 0, 2)
            
            # Forward pass
            outputs = model(decoder_inputs, decoder_inputs)  # [seq_len, batch_size, vocab_size]
            
            # Get only the caption part of the outputs (last 25 tokens - 24 + SOS token)
            caption_outputs = outputs[-25:, :, :]  # [25, batch_size, vocab_size]
            
            # Reshape outputs and labels for loss calculation
            caption_outputs = caption_outputs.permute(1, 0, 2)  # [batch_size, 25, vocab_size]
            
            # Convert labels to long type for GPU
            if config.device == 'cuda':
                caption_labels = caption_labels.long()
            
            # Calculate loss - reshape to match dimensions
            batch_size = caption_outputs.size(0)
            seq_len = caption_outputs.size(1)
            vocab_size = caption_outputs.size(2)
            
            # Reshape outputs and labels
            caption_outputs = caption_outputs.reshape(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            caption_labels = caption_labels.reshape(-1)  # [batch_size * seq_len]
            
            # Calculate loss
            loss = criterion(caption_outputs, caption_labels)
            
            # Calculate accuracy
            predictions = torch.argmax(caption_outputs, dim=-1)  # [batch_size * seq_len]
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
        avg_train_loss = train_loss / len(train_data['decoder_inputs'])
        train_accuracy = (train_correct / train_tokens) * 100 if train_tokens > 0 else 0
        
        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        test_correct = 0
        test_tokens = 0
        
        # Process test data
        with torch.no_grad():  # Disable gradient computation
            for batch_idx in tqdm(range(len(test_data['decoder_inputs'])), desc="Testing"):
                # Get batch data
                decoder_inputs = test_data['decoder_inputs'][batch_idx]
                caption_labels = test_data['caption_labels'][batch_idx]
                
                # Permute decoder inputs for transformer
                decoder_inputs = decoder_inputs.permute(1, 0, 2)
                
                # Forward pass
                outputs = model(decoder_inputs, decoder_inputs)
                
                # Get only the caption part of the outputs
                caption_outputs = outputs[-25:, :, :]
                
                # Reshape outputs and labels
                caption_outputs = caption_outputs.permute(1, 0, 2)
                
                # Convert labels to long type for GPU
                if config.device == 'cuda':
                    caption_labels = caption_labels.long()
                
                # Reshape for loss calculation
                caption_outputs = caption_outputs.reshape(-1, caption_outputs.size(-1))
                caption_labels = caption_labels.reshape(-1)
                
                # Calculate loss
                loss = criterion(caption_outputs, caption_labels)
                
                # Calculate accuracy
                predictions = torch.argmax(caption_outputs, dim=-1)
                correct = (predictions == caption_labels).sum().item()
                test_correct += correct
                test_tokens += predictions.numel()
                
                test_loss += loss.item()
        
        # Calculate test metrics
        avg_test_loss = test_loss / len(test_data['decoder_inputs'])
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
    # Load processed dataset from wandb
    processed_data = load_processed_dataset()
    
    # Train the decoder
    model = train_decoder(processed_data)
