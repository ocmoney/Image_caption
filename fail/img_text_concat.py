from datasets import load_dataset
import torch
from transformers import ViTImageProcessorFast, ViTModel, AutoTokenizer, GPT2Model
from config import config
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self, split="train"):
        # Load dataset
        self.dataset = load_dataset("nlphuji/flickr30k", split="test")
        self.split = self.dataset.filter(lambda x: x["split"] == split)
        
        # Initialize models
        self.vit_processor = ViTImageProcessorFast.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_model = GPT2Model.from_pretrained("gpt2")
        
        # Move models to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_model = self.vit_model.to(self.device)
        self.text_model = self.text_model.to(self.device)
        
        # Add special tokens
        special_tokens = {
            'additional_special_tokens': ['<sos>', '<eos>'],
            'pad_token': '<pad>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.tokenizer))
        
        # Set max caption length
        self.max_caption_length = config.max_caption_length

    def __len__(self):
        return 5 * len(self.split)  # 5 captions per image

    def __getitem__(self, idx):
        # Calculate which image and caption to use
        image_idx = idx // 5
        caption_idx = idx % 5
        
        # Get single example
        example = self.split[image_idx]
        image = example['image']
        caption = example['caption'][caption_idx]  # Get single caption
        
        # Process image
        with torch.no_grad():
            # Resize image to 224x224
            img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(224, 224))[0]
            
            # Get image embeddings
            inputs = self.vit_processor(images=[img_tensor], return_tensors="pt").to(self.device)
            outputs = self.vit_model(**inputs)
            image_embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            
            # Add special tokens to caption
            caption = f"<sos> {caption} <eos>"
            
            # Tokenize caption
            inputs = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_caption_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get text embeddings
            outputs = self.text_model(**inputs)
            caption_embeddings = outputs.last_hidden_state[:, :self.max_caption_length, :]
            
            # Get SOS token embedding
            sos_tokens = self.tokenizer(["<sos>"], return_tensors="pt").to(self.device)
            sos_embedding = self.text_model(**sos_tokens).last_hidden_state[:, 0, :]
            sos_embedding = sos_embedding.unsqueeze(1)
            
            # Stack embeddings
            decoder_inputs = torch.cat([
                image_embeddings,  # [1, num_patches, 768]
                sos_embedding,     # [1, 1, 768]
                caption_embeddings # [1, max_caption_length, 768]
            ], dim=1)
            
            # Transpose to match transformer's expected format
            decoder_inputs = decoder_inputs.transpose(0, 1)
            
            # Clear memory
            torch.cuda.empty_cache()
            
            return decoder_inputs

def process_dataset():
    print("Processing dataset...")
    dataset = ImageCaptionDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(f"Processed {i * 32} examples...")
            torch.cuda.empty_cache()
        # Just print shape to verify
        if i == 0:
            print(f"Batch shape: {batch.shape}")
            break

if __name__ == "__main__":
    process_dataset()
