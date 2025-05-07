# data/dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from encoder_vit import ImagePatcher
from tokenizer import CaptionTokenizer
import random


class Flickr30kDataset(Dataset):
    def __init__(self,
                 dataset_split,  # <- pass torch.utils.data.Subset here
                 tokenizer_model_path="data/tokenizer/spm.model",
                 max_caption_len=30,
                 image_size=224,
                 patch_size=16,
                 embed_dim=512):
        
        self.dataset = dataset_split
        self.tokenizer = CaptionTokenizer(tokenizer_model_path, max_length=max_caption_len)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.image_patcher = ImagePatcher(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        captions = item["caption"]
        image_tensor = self.transform(image)

        caption = random.choice(captions)
        caption_input = self.tokenizer.encode(caption)
        caption_label = caption_input[1:] + [self.tokenizer.pad_id]

        return {
            "image": image_tensor,
            "caption_input": torch.tensor(caption_input, dtype=torch.long),
            "caption_label": torch.tensor(caption_label, dtype=torch.long)
        }

if __name__ == "__main__":
    from datasets import load_dataset
    from torch.utils.data import random_split

    raw_dataset = load_dataset("nlphuji/flickr30k", split="test")

    # Use just a subset for testing
    subset, _ = random_split(raw_dataset, [1000, len(raw_dataset) - 1000])

    dataset = Flickr30kDataset(dataset_split=subset)
    sample = dataset[0]

    print("Image shape:", sample["image"].shape)             # [3, 224, 224]
    print("Caption input:", sample["caption_input"])         # [30]
    print("Caption label:", sample["caption_label"])         # [30]