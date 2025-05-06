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
        # Separate train and validation data
        train_data = {
            'decoder_inputs': [],
            'caption_input_ids': [],
            'caption_labels': []
        }
        val_data = {
            'decoder_inputs': [],
            'caption_input_ids': [],
            'caption_labels': []
        }
        
        # First half is train split, second half is validation split
        split_point = len(processed_data) // 2
        
        # Combine batches into single tensors for each split
        for i, batch in enumerate(processed_data):
            if i < split_point:
                train_data['decoder_inputs'].append(batch['decoder_inputs'])
                train_data['caption_input_ids'].append(batch['caption_input_ids'])
                train_data['caption_labels'].append(batch['caption_labels'])
            else:
                val_data['decoder_inputs'].append(batch['decoder_inputs'])
                val_data['caption_input_ids'].append(batch['caption_input_ids'])
                val_data['caption_labels'].append(batch['caption_labels'])
        
        # Concatenate tensors for each split
        train_tensors = {
            'decoder_inputs': torch.cat(train_data['decoder_inputs'], dim=1),
            'caption_input_ids': torch.cat(train_data['caption_input_ids'], dim=0),
            'caption_labels': torch.cat(train_data['caption_labels'], dim=0)
        }
        
        val_tensors = {
            'decoder_inputs': torch.cat(val_data['decoder_inputs'], dim=1),
            'caption_input_ids': torch.cat(val_data['caption_input_ids'], dim=0),
            'caption_labels': torch.cat(val_data['caption_labels'], dim=0)
        }
        
        # Save the tensors
        torch.save(train_tensors, os.path.join(temp_dir, 'train_dataset.pt'))
        torch.save(val_tensors, os.path.join(temp_dir, 'val_dataset.pt'))
        
        # Create artifacts
        train_artifact = wandb.Artifact(
            name="processed_flickr30k_train",
            type="dataset",
            description="Processed Flickr30k training split with image patches and caption embeddings"
        )
        
        val_artifact = wandb.Artifact(
            name="processed_flickr30k_val",
            type="dataset",
            description="Processed Flickr30k validation split with image patches and caption embeddings"
        )
        
        # Add files to artifacts
        train_artifact.add_file(os.path.join(temp_dir, 'train_dataset.pt'))
        val_artifact.add_file(os.path.join(temp_dir, 'val_dataset.pt'))
        
        # Upload artifacts
        wandb.log_artifact(train_artifact, aliases=["latest"])
        wandb.log_artifact(val_artifact, aliases=["latest"])
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    upload_dataset() 