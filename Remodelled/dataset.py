from datasets import load_dataset  # HuggingFace datasets library for loading datasets
from torch.utils.data import Dataset  # PyTorch's Dataset class for custom datasets
from transformers import AutoTokenizer, AutoImageProcessor  # For text and image processing
import torch  # PyTorch main library
from torchvision import transforms  # For image transformations
from functools import lru_cache  # For caching the tokenizer

# Convert PIL images to PyTorch tensors
to_tensor = transforms.ToTensor()

# Cache the tokenizer to avoid reloading it multiple times
@lru_cache(maxsize=None)
def get_tokenizer():
    # Initialize BERT tokenizer with custom image end token
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", 
                                       use_fast=True, 
                                       extra_special_tokens={"image_end_token": "<img_end>"})

# Custom dataset class for image-text pairs
class ImageTextDataset(Dataset):
    def __init__(self, split="test"):
        # Load Flickr30k dataset and split into train/test
        self.dataset = load_dataset("nlphuji/flickr30k", 
                                  split="test").train_test_split(test_size=0.1, 
                                                               seed=42)[split]
        # Initialize tokenizer for text processing
        self.tokenizer = get_tokenizer()
        # Initialize CLIP image processor for image processing
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", 
                                                          use_fast=True)

    def __len__(self):
        # Return total number of samples in dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get a single item from the dataset
        item = self.dataset[idx]
        # Process image using CLIP processor
        image = self.processor(item["image"], return_tensors="pt")["pixel_values"][0]
        # Randomly select one caption from multiple available captions
        text = item["caption"][torch.randint(0, len(item["caption"]), (1,))]
        # Tokenize the text with padding and truncation
        inputs = self.tokenizer(text, 
                              return_tensors="pt", 
                              padding='max_length', 
                              truncation=True, 
                              max_length=24)
        # Get input token IDs
        text_token = inputs["input_ids"][0]
        # Create output tokens by shifting input tokens and adding padding
        output_text_token = torch.cat([text_token, 
                                     torch.tensor([self.tokenizer.pad_token_id], 
                                                dtype=torch.long)])[1:]
        # Create mask for padding tokens
        mask = text_token == self.tokenizer.pad_token_id
        # Return processed image, input tokens, output tokens, and mask
        return image, text_token, output_text_token, mask
    
    def get_image(self, idx):
        # Helper method to get raw image without processing
        return self.dataset[idx]["image"]
    
# Test code
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    # Create dataset instance
    dataset = ImageTextDataset(split="train")
    # Print first item
    print(dataset[0])
    
    # Get a specific item (index 200)
    image, input_text_token, output_text_token, mask = dataset[200]
    # Print shapes of all components
    print("the mask is", mask)
    print("Image shape: ", image.shape)
    print("Input text token shape: ", input_text_token.shape)
    print("Output text token shape: ", output_text_token.shape)
    print("Mask shape: ", mask.shape)

    # Convert input tokens to text for visualization
    input_text_token = input_text_token.tolist() # Convert to list
    input_text_token = [dataset.tokenizer.decode(token) for token in input_text_token] # Decode tokens to text
    print(input_text_token)

    # Convert output tokens to text for visualization
    text_token = output_text_token.tolist()
    text_token = [dataset.tokenizer.decode(token) for token in text_token]
    print(text_token)

    # Save a sample image
    image = dataset.get_image(200)
    image.save("test.jpeg") 
