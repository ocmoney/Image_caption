from dataset import ImageTextDataset
from model import CaptionGenerator
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import wandb
import torch.nn as nn
import itertools
import os
from safetensors.torch import save_file
    

def compute_accuracy(output, target):
    output = output.argmax(dim=2)
    mask = target != 0
    output = output[mask]
    target = target[mask]
    correct_predictions = (output == target).sum().item()
    total_relevant_elements = target.numel()
    return correct_predictions / total_relevant_elements

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    num_epochs = 10
    num_heads = 4
    num_layers = 3
    learning_rate = 1e-4
    batch_size = 64
    img_seq_len = 50
    text_seq_len = 24
    dataset = ImageTextDataset(split="train")
    test_dataset = ImageTextDataset(split="test")

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = CaptionGenerator(num_heads=num_heads, num_layers=num_layers, tokenizer=dataset.tokenizer, device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="captioner", config={
        "num_epochs": num_epochs,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model": str(model)
    })

    interval = 100
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_test_loss = []
        epoch_test_accuracy = []
        epoch_test_location_accuracy = []

        for i, (batch, test_batch) in enumerate(zip(dataloader, itertools.cycle(test_dataloader))):
            model.train()
            image_token, input_text, output_text, mask = batch
            image_token = image_token.to(device)
            input_text = input_text.to(device)
            output_text = output_text.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(image_token, input_text, mask)[:, -text_seq_len:, :] # only consider the generated text
            loss = criterion(output.transpose(1, 2), output_text)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_image_token, test_input_text, test_output_text, test_mask = test_batch
                test_image_token = test_image_token.to(device)
                test_input_text = test_input_text.to(device)
                test_output_text = test_output_text.to(device)
                test_mask = test_mask.to(device)
                test_output = model(test_image_token, test_input_text, test_mask)[:, -text_seq_len:, :] # only consider the generated text
                test_loss = criterion(test_output.transpose(1, 2), test_output_text)
                test_accuracy = compute_accuracy(test_output, test_output_text)
                
            epoch_loss.append(loss.item())
            epoch_test_loss.append(test_loss.item())
            epoch_test_accuracy.append(test_accuracy)
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "test_accuracy": test_accuracy
            })

            print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(dataloader)}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}", end="")
            
            if i != 0 and i % interval == 0:
                print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(dataloader)}], Train Loss: {torch.tensor(epoch_loss[-interval:]).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss[-interval:]).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy[-interval:]).mean().item():.4f}")
            
        print(f"\n[Epoch {epoch+1}/{num_epochs}], Train Loss: {torch.tensor(epoch_loss).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy).mean().item():.4f}")
        model_path = f"model/{wandb.run.name}/transformer_{epoch}.safetensors"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)
