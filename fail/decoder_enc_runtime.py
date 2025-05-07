import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
    
    def forward(self, x):
        """
        Args:
            x: Input sequence [seq_len, batch_size, d_model]
        """
        # Get sequence length for mask creation
        seq_len = x.size(0)
        
        # Create attention mask for caption part
        tgt_mask = self.create_caption_mask(seq_len)
        
        # Ensure tensor is on the correct device
        x = x.to(self.self_attn.in_proj_weight.device)
        
        # Self-attention block
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2, attn_mask=tgt_mask)[0]
        x = x + self.dropout(x2)
        
        # Feedforward network
        x2 = self.norm2(x)
        x2 = self.linear1(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.linear2(x2)
        x = x + self.dropout(x2)
        
        # Project to vocabulary size
        output = self.output_projection(x)
        
        return output

def train(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        images = batch["image"].to(device)                # [B, 3, H, W]
        caption_input = batch["caption_input"].to(device) # [B, T]
        caption_label = batch["caption_label"].to(device) # [B, T]

        # Forward pass
        logits = model(images, caption_input)             # [B, T, vocab_size]
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == caption_label).sum().item()
        total_correct += correct
        total_tokens += predictions.numel()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader), (total_correct / total_tokens) * 100 if total_tokens > 0 else 0

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            images = batch["image"].to(device)
            caption_input = batch["caption_input"].to(device)
            caption_label = batch["caption_label"].to(device)
            
            # Forward pass
            logits = model(images, caption_input)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == caption_label).sum().item()
            total_correct += correct
            total_tokens += predictions.numel()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader), (total_correct / total_tokens) * 100 if total_tokens > 0 else 0

def train_decoder(config=config):
    # Initialize wandb run
    wandb.init(project="flickr30k", name="decoder-training")
    
    # Print device information
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = Flickr30k(split="train")
    test_dataset = Flickr30k(split="test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} examples")
    print(f"Testing: {len(test_dataset)} examples")
    
    # Initialize model
    model = TransformerDecoder(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        vocab_size=50260
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Start timing
    start_time = time.time()
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        
        # Log training metrics
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "epoch": epoch
        })
        
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Only run testing after first epoch
        if epoch == 0:
            # Evaluation phase
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
            
            # Log test metrics
            wandb.log({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            })
            
            print(f"Testing  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Calculate and log total runtime
    runtime = time.time() - start_time
    wandb.log({"total_runtime": runtime})
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    
    # Finish wandb run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    model = train_decoder()
