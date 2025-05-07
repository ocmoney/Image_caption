import transformers
import torch
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

class Flickr30k(torch.utils.data.Dataset):
    def __init__(self,split):
        self.ds = load_dataset("nlphuji/flickr30k", split="test")
        self.split = self.ds.filter(lambda s: s["split"]==(split))
        self.image_processor = transformers.AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.bert_model = transformers.BertModel.from_pretrained("google-bert/bert-base-uncased")


    def __len__(self):
        return 5*len(self.split)

    def __getitem__(self, idx: int):
        q = idx // 5
        r = idx % 5
        image_data =self.split[q]["image"]
        text_data = self.split[q]["caption"][r]
        image_embeddings = self.image(image_data)
        text_embeddings = self.text(text_data)
        seq = self.concat_sequence(text=text_embeddings,image=image_embeddings)
        return seq 
    
    def image(self,image_data):
        inputs = self.image_processor(image_data, return_tensors="pt")
        with torch.no_grad():
            #the dictionary unpacking operator
            outputs = self.vit_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
    
    def text(self,text_data):
        inputs = self.tokenizer(text_data, return_tensors="pt", padding="max_length", max_length=24, truncation=True )
        outputs = self.bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
    
    def concat_sequence(self,image,text):
        result = torch.cat([image, text], dim=1)
        return result        

if __name__ == '__main__':
    # Initialize wandb
    print("Initializing wandb...")
    wandb.init(project="image-caption", name="flickr30k-dataset")
    
    # Create dataset
    print("Creating dataset...")
    dataset = Flickr30k(split="train")
    total_examples = len(dataset)
    print(f"Dataset size: {total_examples} examples")
    
    # Create a table to store all sequences
    columns = ["image", "caption", "sequence"]
    table = wandb.Table(columns=columns)
    
    # Process in chunks of 1000
    chunk_size = 1000
    for chunk_start in range(0, total_examples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        print(f"\nProcessing examples {chunk_start} to {chunk_end}...")
        
        # Process current chunk
        for idx in tqdm(range(chunk_start, chunk_end), desc=f"Chunk {chunk_start//chunk_size + 1}"):
            q = idx // 5
            r = idx % 5
            
            # Get original data
            image = dataset.split[q]["image"]
            caption = dataset.split[q]["caption"][r]
            
            # Get sequence
            sequence = dataset[idx]
            
            # Add to table
            table.add_data(
                wandb.Image(image),
                caption,
                sequence  # Store raw tensor
            )
        
        # Log current chunk
        print(f"Uploading chunk {chunk_start//chunk_size + 1} to wandb...")
        wandb.log({"sequences": table})
        print(f"Chunk {chunk_start//chunk_size + 1} uploaded")
    
    # Finish wandb run
    wandb.finish()
    print("Done!")