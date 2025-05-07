import wandb
from tqdm import tqdm
from Liam_dataset import Flickr30k
import time
import torch
import os

def get_last_uploaded_chunk():
    """Get the number of the last uploaded chunk"""
    try:
        api = wandb.Api()
        artifacts = api.artifacts(name="image-caption/flickr30k_sequences_chunk_*", type_name="dataset")
        if not artifacts:
            return 0
        # Get the highest chunk number
        chunk_numbers = [int(art.name.split('_')[-1]) for art in artifacts]
        return max(chunk_numbers)
    except Exception as e:
        print(f"Error checking existing chunks: {e}")
        return 0

def upload_sequences_to_wandb(split="train", chunk_size=200):
    """
    Upload sequences to wandb in chunks using artifacts
    
    Args:
        split (str): Dataset split to use ("train", "test", etc.)
        chunk_size (int): Number of examples to process in each chunk
    """
    # Initialize wandb
    print("Initializing wandb...")
    run = wandb.init(project="image-caption", name="flickr30k-dataset")
    
    # Create dataset
    print("Creating dataset...")
    dataset = Flickr30k(split=split)
    total_examples = len(dataset)
    print(f"Dataset size: {total_examples} examples")
    
    # Get the last uploaded chunk
    last_chunk = get_last_uploaded_chunk()
    print(f"Last uploaded chunk: {last_chunk}")
    
    # Create a folder for chunks
    chunks_dir = "flickr30k_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Process in chunks, starting from where we left off
    for chunk_start in range(last_chunk * chunk_size, total_examples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        chunk_num = chunk_start // chunk_size + 1
        print(f"\nProcessing examples {chunk_start} to {chunk_end}...")
        
        # Create a list to store sequences for this chunk
        chunk_sequences = []
        
        # Process current chunk
        for idx in tqdm(range(chunk_start, chunk_end), desc=f"Processing chunk {chunk_num}"):
            # Get sequence
            sequence = dataset[idx]
            chunk_sequences.append(sequence)
        
        # Save chunk to file in the chunks directory
        chunk_file = os.path.join(chunks_dir, f"sequences_chunk_{chunk_num}.pt")
        torch.save(chunk_sequences, chunk_file)
        
        # Calculate chunk size in MB
        chunk_size_mb = sum(seq.element_size() * seq.nelement() for seq in chunk_sequences) / 1024 / 1024
        
        # Create a new artifact for this chunk
        artifact = wandb.Artifact(
            name=f"flickr30k_sequences_chunk_{chunk_num}",
            type="dataset",
            description=f"Flickr30k image-text sequences chunk {chunk_num}"
        )
        
        # Add chunk to artifact
        print(f"\nUploading chunk {chunk_num} to wandb...")
        print(f"Chunk size: {chunk_size_mb:.2f} MB")
        print("This may take several minutes...")
        
        artifact.add_file(chunk_file)
        run.log_artifact(artifact)
        print(f"Chunk {chunk_num} uploaded successfully")
        
        # Clean up local file
        os.remove(chunk_file)
    
    # Clean up chunks directory
    os.rmdir(chunks_dir)
    
    # Finish wandb run
    run.finish()
    print("Done!")

if __name__ == "__main__":
    upload_sequences_to_wandb()
