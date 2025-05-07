import transformers
import torch
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

class Flickr30k(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ds = load_dataset("nlphuji/flickr30k", split="test")
        self.split = self.ds.filter(lambda s: s["split"]==(split))
        self.image_processor = transformers.AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert_model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")

    def __len__(self):
        return 5 * len(self.split)

    def __getitem__(self, idx: int):
        # Calculate which image and caption we want
        image_idx = idx // 5  # Which image (0 to len(self.split)-1)
        caption_idx = idx % 5  # Which caption (0 to 4)
        
        # Get image and process it
        image = self.split[image_idx]["image"]
        inputs = self.image_processor([image], return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        image_embedding = outputs.last_hidden_state[0]  # Remove batch dimension
        
        # Get and process caption
        caption = self.split[image_idx]["caption"][caption_idx]
        inputs = self.tokenizer([caption], return_tensors="pt", padding="max_length", max_length=24, truncation=True)
        
        # Create input and target sequences for caption
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        attention_mask = inputs["attention_mask"][0]
        
        # Create target sequence (shifted by one position)
        target_ids = input_ids[1:].clone()
        target_ids = torch.cat([target_ids, torch.tensor([self.tokenizer.pad_token_id])])
        
        return {
            "image": image_embedding,  # [seq_len, d_model]
            "caption_input": input_ids,  # [seq_len]
            "caption_label": target_ids,  # [seq_len]
            "attention_mask": attention_mask  # [seq_len]
        }

if __name__ == '__main__':
    # Create dataset
    print("Creating dataset...")
    dataset = Flickr30k(split="train")
    print(f"Dataset size: {len(dataset)} examples")
    
    # Test getting a single example
    print("\nTesting single example:")
    example = dataset[0]
    print(f"Image embedding shape: {example['image'].shape}")
    print(f"Caption input shape: {example['caption_input'].shape}")
    print(f"Caption label shape: {example['caption_label'].shape}")
    print(f"Attention mask shape: {example['attention_mask'].shape}")
    
    # Test DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"Images: {batch['image'].shape}")
        print(f"Caption inputs: {batch['caption_input'].shape}")
        print(f"Caption labels: {batch['caption_label'].shape}")
        print(f"Attention masks: {batch['attention_mask'].shape}")
        break