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
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def __call__(self, image: list[Image.Image]):
        inputs = self.processor(image, return_tensors="pt")
        return self.model(**inputs, output_hidden_states=True).last_hidden_state
    
class TextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        model = Siglip2TextModel.from_pretrained("google/siglip2-base-patch16-224")
        start_embedding = model.get_input_embeddings()
        embedding_dim = start_embedding.embedding_dim
        vocab_size = start_embedding.num_embeddings
        self.embeddings = nn.Embedding(vocab_size + 1, embedding_dim)
        with torch.no_grad():
            self.embeddings.weight[:vocab_size, :] = start_embedding.weight.data


    def __call__(self, text: torch.Tensor):
        return self.embeddings(text)

class ImageTextDataset(Dataset):
    def __init__(self, split="test"):
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224", extra_special_tokens={"image_separator": "<img_separator>"})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        text = self.dataset[idx]["caption"][torch.randint(0, len(self.dataset[idx]["caption"]), (1,))]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        text_token = inputs["input_ids"][0]
        input_text_token = torch.cat([torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long), text_token])[:-1]
        text_token = torch.cat([torch.tensor([self.tokenizer.image_separator_id], dtype=torch.long), text_token])
        input_text_token = torch.cat([torch.tensor([self.tokenizer.image_separator_id], dtype=torch.long), input_text_token])
        mask = input_text_token == 0
        return image, input_text_token, text_token, mask
    
if __name__ == "__main__":
    dataset = ImageTextDataset()
    print(dataset[0])
    
    image, input_text_token, text_token, mask = dataset[0]

    image_tokenizer = ImageTokenizer()
    image_token = image_tokenizer([image])
    print(image_token.shape)

    text_embedding = TextEmbedding()
    text_tokens = text_embedding(input_text_token.unsqueeze(0))
    print(text_tokens.shape)

