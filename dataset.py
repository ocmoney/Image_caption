from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor
import torch
from torchvision import transforms
from functools import lru_cache
to_tensor = transforms.ToTensor()

@lru_cache(maxsize=None)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True, extra_special_tokens={"image_end_token": "<img_end>"})

class ImageTextDataset(Dataset):
    def __init__(self, split="test"):
        self.dataset = load_dataset("nlphuji/flickr30k", split="test").train_test_split(test_size=0.1, seed=42)[split]
        self.tokenizer = get_tokenizer()
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.processor(item["image"], return_tensors="pt")["pixel_values"][0]
        text = item["caption"][torch.randint(0, len(item["caption"]), (1,))]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=24)
        text_token = inputs["input_ids"][0]
        output_text_token = torch.cat([text_token, torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)])[1:]
        mask = text_token == self.tokenizer.pad_token_id
        return image, text_token, output_text_token, mask
    
    def get_image(self, idx):
        return self.dataset[idx]["image"]
    
if __name__ == "__main__":
    torch.manual_seed(42)
    dataset = ImageTextDataset(split="train")
    print(dataset[0])
    
    image, input_text_token, text_token, mask = dataset[200]
    print("Image shape: ", image.shape)
    print("Input text token shape: ", input_text_token.shape)
    print("Text token shape: ", text_token.shape)
    print("Mask shape: ", mask.shape)

    # Turn the input_text_token into a string
    input_text_token = input_text_token.tolist()
    input_text_token = [dataset.tokenizer.decode(token) for token in input_text_token]
    print(input_text_token)

    # Turn the text_token into a string
    text_token = text_token.tolist()
    text_token = [dataset.tokenizer.decode(token) for token in text_token]
    print(text_token)

    image = dataset.get_image(200)
    image.save("test.jpeg")