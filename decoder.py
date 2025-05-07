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
        # Get specific chunk artifact
        api = wandb.Api()
        artifact = api.artifact("olliecumming3-machine-learning-institute/image-caption/flickr30k_sequences_chunk_46:latest")
        
        print(f"Found chunk artifact: {artifact.name}")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the artifact
            artifact_dir = artifact.download(root=tmp_dir)
            print(f"Downloaded {artifact.name} to: {artifact_dir}")
            
            # Load the chunk file
            chunk_file = os.path.join(artifact_dir, "sequences_chunk_46.pt")
            all_sequences = torch.load(chunk_file)
            print(f"\nInitial loaded data type: {type(all_sequences)}")
            if isinstance(all_sequences, list):
                print(f"Number of sequences in list: {len(all_sequences)}")
                print(f"First sequence shape: {all_sequences[0].shape}")
            
            # Convert to tensor and ensure correct shape [batch_size, sequence_length, embedding_dim]
            all_sequences = torch.stack(all_sequences)  # [num_examples, seq_len, 768]
            print(f"\nAfter stack - all_sequences shape: {all_sequences.shape}")
            
            # If we have an extra dimension, squeeze it out
            if len(all_sequences.shape) == 4:  # [batch_size, 1, seq_len, embedding_dim]
                print(f"\nRemoving extra dimension - before: {all_sequences.shape}")
                all_sequences = all_sequences.squeeze(1)  # Remove the extra dimension
                print(f"After squeeze - all_sequences shape: {all_sequences.shape}")
            
            # If dimensions are swapped, transpose them
            if all_sequences.size(1) == 768:  # If sequence length and embedding dim are swapped
                print(f"\nBefore permute - all_sequences shape: {all_sequences.shape}")
                all_sequences = all_sequences.permute(0, 2, 1)  # Swap dimensions to [batch_size, seq_len, 768]
                print(f"After permute - all_sequences shape: {all_sequences.shape}")
            
            # Reshape to match config batch size
            batch_size = config.batch_size
            num_examples = all_sequences.size(0)
            seq_len = all_sequences.size(1)
            embedding_dim = all_sequences.size(2)
            
            # Calculate how many complete batches we can make
            num_complete_batches = num_examples // batch_size
            if num_complete_batches == 0:
                raise ValueError(f"Not enough examples ({num_examples}) for batch size {batch_size}")
            
            # Take only complete batches
            all_sequences = all_sequences[:num_complete_batches * batch_size]
            all_sequences = all_sequences.view(num_complete_batches, batch_size, seq_len, embedding_dim)
            
            print(f"\nAfter reshaping to batches:")
            print(f"Number of complete batches: {num_complete_batches}")
            print(f"Batch size: {batch_size}")
            print(f"Sequence length: {seq_len}")
            print(f"Embedding dimension: {embedding_dim}")
            print(f"Final shape: {all_sequences.shape}")
            
            # Split sequence into image patches and caption
            # First 196 tokens are image patches (14x14 grid)
            image_patches = all_sequences[:, :, :196, :]  # [num_batches, batch_size, 196, 768]
            caption_embeddings = all_sequences[:, :, 196:, :]  # [num_batches, batch_size, caption_len, 768]
            
            # Get shapes
            caption_len = caption_embeddings.size(2)
            
            print(f"\nShape information:")
            print(f"Number of batches: {num_complete_batches}")
            print(f"Caption length: {caption_len}")
            print(f"Image patches shape: {image_patches.shape}")
            print(f"Caption embeddings shape: {caption_embeddings.shape}")
            
            # Create decoder inputs by concatenating image patches and caption
            decoder_inputs = all_sequences  # [num_batches, batch_size, seq_len, 768]
            
            # Create input_ids and labels for captions
            # These will be generated during training from the caption embeddings
            caption_input_ids = torch.zeros((num_complete_batches, batch_size, caption_len), dtype=torch.long)
            caption_labels = torch.zeros((num_complete_batches, batch_size, caption_len), dtype=torch.long)
            
            # Create the format expected by the training code
            processed_data = {
                'decoder_inputs': decoder_inputs,  # [num_batches, batch_size, seq_len, 768]
                'caption_input_ids': caption_input_ids,  # [num_batches, batch_size, caption_len]
                'caption_labels': caption_labels  # [num_batches, batch_size, caption_len]
            }
            
            print("\nFinal processed data shapes:")
            for k, v in processed_data.items():
                print(f"{k}: {v.shape}")
            
            print("\nSuccessfully loaded processed dataset from wandb!")
            print(f"Dataset size: {num_complete_batches * batch_size} examples")
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
    
    # Print tensor shapes for debugging
    print("\nInput tensor shapes:")
    for k, v in processed_data.items():
        print(f"{k}: {v.shape}")
    
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
    
    # Print tensor shapes after splitting
    print("\nTrain data tensor shapes:")
    for k, v in train_data.items():
        print(f"{k}: {v.shape}")
    print("\nTest data tensor shapes:")
    for k, v in test_data.items():
        print(f"{k}: {v.shape}")
    
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
            
            # Debug prints for first batch
            if batch_idx == 0:
                print(f"\nTraining batch shapes:")
                print(f"caption_outputs: {caption_outputs.shape}")
                print(f"caption_labels: {caption_labels.shape}")
            
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
                
                # Debug prints for first batch
                if batch_idx == 0:
                    print(f"\nTest batch shapes:")
                    print(f"caption_outputs: {caption_outputs.shape}")
                    print(f"caption_labels: {caption_labels.shape}")
                
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
