import wandb
import torch
from img_text_concat import process_dataset
import tempfile
import os
import math

def upload_dataset():
    # Initialize wandb
    wandb.init(project="flickr30k", job_type="dataset_upload")
    
    # Process the dataset
    processed_data = process_dataset()
    
    # Create a temporary directory to save the tensors
    with tempfile.TemporaryDirectory() as temp_dir:
        # Store unique image embeddings
        image_embeddings_dict = {}  # Map image index to embeddings
        all_caption_embeddings = []
        all_caption_input_ids = []
        all_caption_labels = []
        image_to_caption_map = []  # Map image index to caption indices
        
        for batch_idx, batch in enumerate(processed_data):
            # Split the sequence into image and caption parts
            decoder_inputs = batch['decoder_inputs']  # [seq_len, num_captions, d_model]
            image_embeddings = decoder_inputs[:196, 0, :]  # Get image embeddings for first caption only
            caption_embeddings = decoder_inputs[196:, :, :]  # Get all caption embeddings
            
            # Store image embeddings if not already stored
            if batch_idx not in image_embeddings_dict:
                image_embeddings_dict[batch_idx] = image_embeddings
            
            # Store caption data
            all_caption_embeddings.append(caption_embeddings)
            all_caption_input_ids.append(batch['caption_input_ids'])
            all_caption_labels.append(batch['caption_labels'])
            
            # Map this image to its caption indices
            start_idx = len(all_caption_embeddings) - caption_embeddings.size(1)
            end_idx = len(all_caption_embeddings)
            image_to_caption_map.append((start_idx, end_idx))
        
        # Convert to tensors
        final_image_embeddings = torch.stack([emb for emb in image_embeddings_dict.values()], dim=0)  # [num_images, 196, 768]
        final_caption_embeddings = torch.cat(all_caption_embeddings, dim=1)  # [caption_len, total_captions, 768]
        final_caption_input_ids = torch.cat(all_caption_input_ids, dim=0)  # [total_captions, max_len]
        final_caption_labels = torch.cat(all_caption_labels, dim=0)  # [total_captions, max_len]
        
        # Save the tensors
        torch.save({
            'image_embeddings': final_image_embeddings,  # [num_images, 196, 768]
            'caption_embeddings': final_caption_embeddings,  # [caption_len, total_captions, 768]
            'caption_input_ids': final_caption_input_ids,  # [total_captions, max_len]
            'caption_labels': final_caption_labels,  # [total_captions, max_len]
            'image_to_caption_map': image_to_caption_map  # List of (start_idx, end_idx) tuples
        }, os.path.join(temp_dir, 'processed_dataset.pt'))
        
        # Create a new artifact with the same name to overwrite
        artifact = wandb.Artifact(
            name="processed_flickr30k",
            type="dataset",
            description="Processed Flickr30k dataset with image patches and caption embeddings"
        )
        
        # Add the saved tensors to the artifact
        artifact.add_file(os.path.join(temp_dir, 'processed_dataset.pt'))
        
        # Upload the artifact with overwrite=True
        wandb.log_artifact(artifact, aliases=["latest"])
        
        # Log dataset statistics
        wandb.log({
            "total_images": final_image_embeddings.size(0),
            "total_captions": final_caption_embeddings.size(1),
            "image_patches": final_image_embeddings.size(1),
            "caption_length": final_caption_embeddings.size(0),
            "embedding_dim": final_image_embeddings.size(2),
            "max_caption_length": final_caption_input_ids.size(1),
            "storage_size_mb": os.path.getsize(os.path.join(temp_dir, 'processed_dataset.pt')) / (1024 * 1024)
        })
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    upload_dataset() 