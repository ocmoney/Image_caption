# training/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datasets import load_dataset
from dataset import Flickr30kDataset
from caption_model import CaptioningModel

def train(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)                # [B, 3, H, W]
        caption_input = batch["caption_input"].to(device) # [B, T]
        caption_label = batch["caption_label"].to(device) # [B, T]

        optimizer.zero_grad()
        logits = model(images, caption_input)             # [B, T, vocab_size]
        
        loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            caption_input = batch["caption_input"].to(device)
            caption_label = batch["caption_label"].to(device)

            logits = model(images, caption_input)
            loss = criterion(logits.view(-1, logits.size(-1)), caption_label.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset from Hugging Face
    full_data = load_dataset("nlphuji/flickr30k", split="test")

    # Split into train/val/test (80/10/10)
    train_size = int(0.8 * len(full_data))
    val_size   = int(0.1 * len(full_data))
    test_size  = len(full_data) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_data,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Dataset wrapping
    train_dataset = Flickr30kDataset(dataset_split=train_data)
    val_dataset   = Flickr30kDataset(dataset_split=val_data)
    test_dataset  = Flickr30kDataset(dataset_split=test_data)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)

    # Model setup
    vocab_size = train_dataset.tokenizer.vocab_size
    model = CaptioningModel(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pad_id = train_dataset.tokenizer.pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(1, 21):
        print(f"\nEpoch {epoch}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

if __name__ == "__main__":
    main()