import wandb
import torch
from img_text_concat import process_dataset
import tempfile
import os

def upload_dataset():
    # Initialize wandb
    wandb.init(project="flickr30k", job_type="dataset_upload")
    
    # Process the dataset
    processed_data = process_dataset()
    
    # Create a temporary directory to save the tensors
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the tensors directly
        torch.save({
            'decoder_inputs': torch.stack([x['decoder_inputs'] for x in processed_data]),
            'caption_input_ids': torch.stack([x['caption_input_ids'] for x in processed_data]),
            'caption_labels': torch.stack([x['caption_labels'] for x in processed_data])
        }, os.path.join(temp_dir, 'processed_dataset.pt'))
        
        # Create a new artifact
        artifact = wandb.Artifact(
            name="processed_flickr30k",
            type="dataset",
            description="Processed Flickr30k dataset with image patches and caption embeddings"
        )
        
        # Add the saved tensors to the artifact
        artifact.add_file(os.path.join(temp_dir, 'processed_dataset.pt'))
        
        # Upload the artifact
        wandb.log_artifact(artifact)
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    upload_dataset() 