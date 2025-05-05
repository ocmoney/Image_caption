from datasets import load_dataset
import torch
from transformers import ViTImageProcessorFast, ViTModel, AutoTokenizer, GPT2Model
from config import config
import numpy as np

# Load dataset
dataset = load_dataset("nlphuji/flickr30k")

# Initialize models
vit_processor = ViTImageProcessorFast.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text_model = GPT2Model.from_pretrained("gpt2")

# Move models to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vit_model = vit_model.to(device)
text_model = text_model.to(device)

# Add special tokens to tokenizer and resize model embeddings
special_tokens = {
    'additional_special_tokens': ['<sos>', '<eos>'],
    'pad_token': '<pad>'
}
tokenizer.add_special_tokens(special_tokens)
text_model.resize_token_embeddings(len(tokenizer))

# Set fixed max caption length
max_caption_length = 32

def get_patch_embeddings(images):
    with torch.no_grad():
        # First resize images to 224x224 if needed
        resized_images = []
        for img in images:
            # Convert PIL image to tensor and resize
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(224, 224))[0]
            resized_images.append(img_tensor)
        
        # Process all images
        inputs = vit_processor(images=resized_images, return_tensors="pt").to(device)
        outputs = vit_model(**inputs)
        
        # Get all patches (excluding CLS token)
        # This gives us 196 patches (14x14 grid) of 768 dimensions each
        patches = outputs.last_hidden_state[:, 1:, :]
        
        return patches

def get_token_embeddings(captions):
    with torch.no_grad():
        inputs = tokenizer(
            captions,
            padding='max_length',
            max_length=max_caption_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        outputs = text_model(**inputs)
        return outputs.last_hidden_state[:, :max_caption_length, :]

def debug_single_example():
    print("\nDebugging single example:")
    
    # Load a single example
    single_example = dataset['test'][0]
    print(f"Original image shape: {single_example['image'].size}")
    print(f"All captions: {single_example['caption']}")
    
    # First expand the example to include all caption pairs
    expanded_data = []
    for caption in single_example['caption']:
        expanded_data.append({
            'image': single_example['image'],
            'caption': caption
        })
    
    # Process all pairs
    image_embeddings = get_patch_embeddings([expanded_data[0]['image']])
    print(f"\nImage embeddings shape: {image_embeddings.shape}")
    
    # Process all captions
    decoder_inputs = [f"<sos> {cap}" for cap in single_example['caption']]
    caption_embeddings = get_token_embeddings(decoder_inputs)
    print(f"Caption embeddings shape: {caption_embeddings.shape}")
    
    # Process captions for input and target separately
    # Input to decoder: <sos> + caption
    decoder_inputs = [f"<sos> {cap}" for cap in single_example['caption']]
    # Target for loss: caption + <eos>
    decoder_targets = [f"{cap} <eos>" for cap in single_example['caption']]
    
    # Tokenize captions with fixed length
    input_tokens = tokenizer(
        decoder_inputs,
        padding='max_length',
        max_length=max_caption_length,
        truncation=True,
        return_tensors="pt"
    )
    target_tokens = tokenizer(
        decoder_targets,
        padding='max_length',
        max_length=max_caption_length,
        truncation=True,
        return_tensors="pt"
    )
    
    print("\nToken shapes:")
    print(f"Input tokens shape: {input_tokens['input_ids'].shape}")
    print(f"Target tokens shape: {target_tokens['input_ids'].shape}")
    
    # Calculate sequence length
    num_patches = image_embeddings.size(1)  # Number of patches from ViT
    seq_len = num_patches + max_caption_length  # Total sequence length
    print(f"\nSequence length calculation:")
    print(f"Number of patches: {num_patches}")
    print(f"Max caption length: {max_caption_length}")
    print(f"Total sequence length: {seq_len}")
    
    # Create decoder inputs by stacking image and caption embeddings
    batch_size = len(expanded_data)  # Should be 5
    
    # Stack image patches and caption tokens along sequence dimension
    # image_embeddings shape: [batch_size, num_patches, 768]
    # caption_embeddings shape: [batch_size, max_caption_length, 768]
    decoder_inputs = torch.cat([
        image_embeddings,  # [batch_size, num_patches, 768]
        caption_embeddings # [batch_size, max_caption_length, 768]
    ], dim=1)  # Stack along sequence dimension
    
    # Transpose to match transformer's expected format
    # [batch_size, seq_len, 768] -> [seq_len, batch_size, 768]
    decoder_inputs = decoder_inputs.transpose(0, 1)
    
    print("\nFinal shapes:")
    print(f"Decoder inputs shape: {decoder_inputs.shape}")
    print(f"Input tokens shape: {input_tokens['input_ids'].shape}")
    print(f"Target tokens shape: {target_tokens['input_ids'].shape}")
    
    # Verify the shapes
    assert decoder_inputs.size(0) == seq_len, f"Sequence length mismatch: {decoder_inputs.size(0)} != {seq_len}"
    assert decoder_inputs.size(1) == batch_size, f"Batch size mismatch: {decoder_inputs.size(1)} != {batch_size}"
    assert decoder_inputs.size(2) == 768, f"Embedding dimension mismatch: {decoder_inputs.size(2)} != 768"
    
    print("\nAll shape assertions passed!")
    
    # Convert to numpy array for dataset compatibility
    decoder_inputs = decoder_inputs.cpu().numpy()
    print(f"\nNumpy array shape: {decoder_inputs.shape}")

if __name__ == "__main__":
    debug_single_example() 