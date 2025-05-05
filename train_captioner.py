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

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    num_epochs = 10
    num_heads = 4
    num_layers = 3
    learning_rate = 1e-3
    batch_size = 64

    dataset = ImageTextDataset(split="train")
    test_dataset = ImageTextDataset(split="test")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    model = CaptionGenerator(num_heads=num_heads, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
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
            image, input_text, output_text, mask = batch
            input_text = input_text.to(device)
            output_text = output_text.to(device)
            mask = mask.to(device)
            image = image.to(device)

            optimizer.zero_grad()
            output = model(input_text, image, mask)
            loss = criterion(output.transpose(1, 2), output_text)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_image, test_input_text, test_output_text, test_mask = test_batch
                test_output = model(test_input_text, test_image, test_mask)
                test_loss = criterion(test_output.transpose(1, 2), test_output_text)
                test_accuracy = (test_output.argmax(dim=2) == test_output_text).float().mean()
                
            epoch_loss.append(loss.item())
            epoch_test_loss.append(test_loss.item())
            epoch_test_accuracy.append(test_accuracy.item())
            wandb.log({
                "train_loss": loss.item(),
                "test_loss": test_loss.item(),
                "test_accuracy": test_accuracy.item(),
            })

            print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(dataloader)}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy.item():.4f}", end="")
            
            if i != 0 and i % interval == 0:
                print(f"\r[Epoch {epoch+1}/{num_epochs}], [Step {i+1}/{len(dataloader)}], Train Loss: {torch.tensor(epoch_loss[-interval:]).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss[-interval:]).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy[-interval:]).mean().item():.4f}")
            
        print(f"\n[Epoch {epoch+1}/{num_epochs}], Train Loss: {torch.tensor(epoch_loss).mean().item():.4f}, Test Loss: {torch.tensor(epoch_test_loss).mean().item():.4f}, Test Accuracy: {torch.tensor(epoch_test_accuracy).mean().item():.4f}")
        model_path = f"model/{wandb.run.name}/transformer_{epoch}.safetensors"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)
