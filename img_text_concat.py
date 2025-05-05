from datasets import load_dataset
import torch
from transformers import ViTImageProcessorFast, ViTModel, AutoTokenizer, GPT2Model
from config import config  # Import config from config.py
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
    'pad_token': '<pad>'  # Add padding token
}
tokenizer.add_special_tokens(special_tokens)
text_model.resize_token_embeddings(len(tokenizer))

# Set fixed max caption length
max_caption_length = 32  # Increased from 22 to 32 to accommodate longer captions

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
        # This gives us 196 patches (14x14 grid) of 768 dimensions each, this means 14 patches of 16x16 resolution
        patches = outputs.last_hidden_state[:, 1:, :]
        
        return patches

def get_token_embeddings(captions):
    with torch.no_grad():
        # Add special tokens and tokenize all captions
        captions = [f"<sos> {cap} <eos>" for cap in captions] 
        inputs = tokenizer(
            captions, 
            padding='max_length',
            max_length=max_caption_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        outputs = text_model(**inputs)
        # Truncate embeddings to max_caption_length
        return outputs.last_hidden_state[:, :max_caption_length, :]

def process_batch(batch):
    # Get image embeddings in batch
    image_embeddings = get_patch_embeddings(batch['image'])
    caption_embeddings = get_token_embeddings([f"<sos> {cap}" for cap in batch['caption']])
    
    # Process captions for input and target separately
    # Input to decoder: <sos> + caption
    decoder_inputs = [f"<sos> {cap}" for cap in batch['caption']]
    # Target for loss: caption + <eos>
    decoder_targets = [f"{cap} <eos>" for cap in batch['caption']]
    
    # Tokenize captions in batch with fixed length
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
    
    # Create decoder inputs by stacking image and caption embeddings
    batch_size = len(batch['caption'])  # This is now the expanded batch size
    
    # Calculate sequence length
    num_patches = image_embeddings.size(1)  # Number of patches from ViT
    seq_len = num_patches + 1 + max_caption_length  # Total sequence length (patches + sos + caption)
    
    # Get the <sos> token embedding for each example in the batch
    sos_tokens = tokenizer(["<sos>"] * batch_size, return_tensors="pt").to(device)
    sos_embedding = text_model(**sos_tokens).last_hidden_state[:, 0, :]  # [batch_size, 768]
    sos_embedding = sos_embedding.unsqueeze(1)  # [batch_size, 1, 768]
    
    # Debug prints for tensor shapes
    print(f"image_embeddings shape: {image_embeddings.shape}")
    print(f"sos_embedding shape: {sos_embedding.shape}")
    print(f"caption_embeddings shape: {caption_embeddings.shape}")
    
    # Stack image patches, sos token, and caption tokens along sequence dimension
    # image_embeddings shape: [batch_size, num_patches, 768]
    # sos_embedding shape: [batch_size, 1, 768]
    # caption_embeddings shape: [batch_size, max_caption_length, 768]
    decoder_inputs = torch.cat([
        image_embeddings,  # [batch_size, num_patches, 768]
        sos_embedding,     # [batch_size, 1, 768]
        caption_embeddings # [batch_size, max_caption_length, 768]
    ], dim=1)  # Stack along sequence dimension
    
    # Transpose to match transformer's expected format
    # [batch_size, seq_len, 768] -> [seq_len, batch_size, 768]
    decoder_inputs = decoder_inputs.transpose(0, 1)
    
    # Verify the shapes
    assert decoder_inputs.size(0) == seq_len, f"Sequence length mismatch: {decoder_inputs.size(0)} != {seq_len}"
    assert decoder_inputs.size(1) == batch_size, f"Batch size mismatch: {decoder_inputs.size(1)} != {batch_size}"
    assert decoder_inputs.size(2) == 768, f"Embedding dimension mismatch: {decoder_inputs.size(2)} != 768"
    
    return {
        'decoder_inputs': decoder_inputs,  # Shape: [seq_len, batch_size, 768]
        'caption_input_ids': input_tokens['input_ids'],  # Input to decoder: <sos> + caption
        'caption_labels': target_tokens['input_ids']     # Target for loss: caption + <eos>
    }

def process_dataset():
    print("Processing dataset...")
    # Process only the train split with specified fraction
    processed_data = []

    # Process train split only
    print("Processing train split...")
    # Calculate how many train examples we need
    total_train = sum(1 for x in dataset['test'] if x['split'] == 'train')
    num_samples = int(total_train * config.train_fraction)
    print(f"Taking {num_samples} examples ({config.train_fraction*100}% of train split)...")

    # Collect only the indices we need
    train_indices = []
    for i, x in enumerate(dataset['test']):
        if x['split'] == 'train':
            train_indices.append(i)
            if len(train_indices) >= num_samples:
                break

    train_dataset = dataset['test'].select(train_indices)

    # Collect all examples first
    all_examples = []
    for example in train_dataset:
        for caption in example['caption']:
            all_examples.append({
                'image': example['image'],
                'caption': caption
            })
    
    # Process in batches of 8
    batch_size = 8
    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i:i+batch_size]
        # Create batch of images and captions
        batch_data = {
            'image': [ex['image'] for ex in batch],
            'caption': [ex['caption'] for ex in batch]
        }
        # Process the batch
        processed = process_batch(batch_data)
        processed_data.append(processed)
        print(f"Processed batch {i//batch_size + 1}/{(len(all_examples) + batch_size - 1)//batch_size}")
    
    print(f"Processed {len(all_examples)} examples in {len(processed_data)} batches")
    return processed_data

def debug_single_example():
    # Load a single example
    single_example = dataset['test'][0]
    print("\nDebugging single example:")
    print(f"Image shape: {single_example['image'].size}")
    print(f"Caption: {single_example['caption']}")
    
    # Process single example
    image_embeddings = get_patch_embeddings([single_example['image']])
    print(f"\nImage embeddings shape: {image_embeddings.shape}")
    
    # Process caption
    decoder_input = f"<sos> {single_example['caption'][0]}"
    caption_embeddings = get_token_embeddings([decoder_input])
    print(f"Caption embeddings shape: {caption_embeddings.shape}")
    
    # Calculate sizes
    image_size = 196 * 768
    caption_size = max_caption_length * 768
    combined_size = image_size + caption_size
    print(f"\nCalculated sizes:")
    print(f"Image size: {image_size}")
    print(f"Caption size: {caption_size}")
    print(f"Combined size: {combined_size}")
    
    # Flatten embeddings
    img_flat = image_embeddings[0].reshape(-1)
    cap_flat = caption_embeddings[0].reshape(-1)
    print(f"\nFlattened sizes:")
    print(f"Image flat size: {img_flat.size(0)}")
    print(f"Caption flat size: {cap_flat.size(0)}")
    
    # Try concatenation
    combined = torch.cat([img_flat, cap_flat])
    print(f"\nCombined size: {combined.size(0)}")
    print(f"Expected size: {combined_size}")
    
    return combined.size(0) == combined_size

if __name__ == "__main__":
    # First run debug test
    print("Running debug test...")
    debug_single_example()
    
    # Then process full dataset
    print("\nProcessing full dataset...")
    processed_data = process_dataset()

