import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataset import ImageTextDataset
from model import CaptionGenerator
import wandb
import itertools
import os
from safetensors.torch import save_file


def compute_accuracy(output, target, pad_token_id):
    output = output.argmax(dim=2)
    mask = target != pad_token_id
    output = output[mask]
    target = target[mask]
    correct_predictions = (output == target).sum().item()
    total_relevant_elements = target.numel()
    return correct_predictions / total_relevant_elements

if __name__ == "__main__":
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU" if torch.cuda.is_available() else "CPU")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Training hyperparameters
    num_epochs = 10
    num_heads = 6
    num_layers = 8
    learning_rate = 1e-4
    batch_size = 64
    img_seq_len = 50
    text_seq_len = 24

    # Initialize wandb for experiment tracking
    wandb.init(project="captioner", config={
        "num_epochs": num_epochs,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    })

    # Initialize datasets
    train_dataset = ImageTextDataset(split="train")
    test_dataset = ImageTextDataset(split="test")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = CaptionGenerator(num_heads=num_heads, num_layers=num_layers, tokenizer=train_dataset.tokenizer, device=device)
    criterion = CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Add model architecture to wandb config
    wandb.config.update({"model": str(model)})

    # Training loop
    interval = 100
    for epoch in range(num_epochs):
        # Initialize metrics for this epoch
        epoch_loss = []
        epoch_test_loss = []
        epoch_test_accuracy = []

        # Training phase
        model.train()
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for i, batch in enumerate(train_progress):
            # Get batch data
            images, text_tokens, output_tokens, padding_mask = batch
            
            # Move data to device
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            output_tokens = output_tokens.to(device)
            padding_mask = padding_mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images, text_tokens, padding_mask)
            loss = criterion(logits.transpose(1, 2), output_tokens)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record training loss
            epoch_loss.append(loss.item())
            train_progress.set_postfix({"loss": loss.item()})
            
            # Log metrics
            wandb.log({
                "train_loss": loss.item(),
                "step": i + epoch * len(train_dataloader)
            })
            
            # Print sample predictions periodically
            if i % interval == 0:
                print("\nSample predictions:")
                for j in range(3):
                    print("Generated:", train_dataset.tokenizer.decode(torch.argmax(logits[j], dim=-1).squeeze().tolist()))
                    print("Ground truth:", train_dataset.tokenizer.decode(output_tokens[j].squeeze().tolist()))

        # Evaluation phase
        model.eval()
        test_progress = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        with torch.no_grad():
            for batch in test_progress:
                # Get test batch data
                images, text_tokens, output_tokens, padding_mask = batch
                
                # Move data to device
                images = images.to(device)
                text_tokens = text_tokens.to(device)
                output_tokens = output_tokens.to(device)
                padding_mask = padding_mask.to(device)
                
                # Forward pass
                logits = model(images, text_tokens, padding_mask)
                test_loss = criterion(logits.transpose(1, 2), output_tokens)
                test_accuracy = compute_accuracy(logits, output_tokens, train_dataset.tokenizer.pad_token_id)
                
                # Record metrics
                epoch_test_loss.append(test_loss.item())
                epoch_test_accuracy.append(test_accuracy)
                test_progress.set_postfix({
                    "loss": test_loss.item(),
                    "accuracy": test_accuracy
                })
                
                # Log metrics
                wandb.log({
                    "test_loss": test_loss.item(),
                    "test_accuracy": test_accuracy,
                    "step": i + epoch * len(train_dataloader)
                })
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {torch.tensor(epoch_loss).mean().item():.4f}")
        print(f"Test Loss: {torch.tensor(epoch_test_loss).mean().item():.4f}")
        print(f"Test Accuracy: {torch.tensor(epoch_test_accuracy).mean().item():.4f}")
        
        # Save checkpoint
        model_path = f"model/{wandb.run.name}/transformer_{epoch}.safetensors"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)
        
        # Log epoch metrics
        wandb.log({
            "epoch_train_loss": torch.tensor(epoch_loss).mean().item(),
            "epoch_test_loss": torch.tensor(epoch_test_loss).mean().item(),
            "epoch_test_accuracy": torch.tensor(epoch_test_accuracy).mean().item(),
            "epoch": epoch
        })
    
    wandb.finish() 