from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor, ViTModel, Siglip2TextModel
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

to_tensor = transforms.ToTensor()

class ImageTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

    def __call__(self, image: list[Image.Image]):
        inputs = self.processor(image, return_tensors="pt")
        return self.model(**inputs, output_hidden_states=True).last_hidden_state
    
class TextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        model = Siglip2TextModel.from_pretrained("google/siglip2-base-patch16-224")
        self.embeddings = model.get_input_embeddings()


    def __call__(self, text: torch.Tensor):
        return self.embeddings(text)

class ImageTextDataset(Dataset):
    def __init__(self, split="test"):
        self.dataset = load_dataset("nlphuji/flickr30k", split="test").train_test_split(test_size=0.1)[split]
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224", use_fast=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        text = self.dataset[idx]["caption"][torch.randint(0, len(self.dataset[idx]["caption"]), (1,))]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        text_token = inputs["input_ids"][0]
        input_text_token = torch.cat([torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long), text_token])[:-1]
        mask = input_text_token == 0
        return image, input_text_token, text_token, mask
    
if __name__ == "__main__":
    torch.manual_seed(42)
    dataset = ImageTextDataset(split="train")
    print(dataset[0])
    
    image, input_text_token, text_token, mask = dataset[0]

    image_tokenizer = ImageTokenizer()
    image_token = image_tokenizer([image])
    print(image_token.shape)

    text_embedding = TextEmbedding()
    text_tokens = text_embedding(input_text_token.unsqueeze(0))
    print(text_tokens.shape)

