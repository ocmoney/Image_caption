import os
import torch
from datasets import DatasetDict
import pickle

def load_processed_dataset():
    # Check if processed dataset exists
    if os.path.exists('processed_dataset.pkl'):
        print("Loading existing processed dataset...")
        with open('processed_dataset.pkl', 'rb') as f:
            processed_dataset = pickle.load(f)
        print("Dataset loaded successfully!")
    else:
        print("No existing processed dataset found. Processing new dataset...")
        from img_text_concat import process_dataset
        processed_dataset = process_dataset()
        
        # Save the processed dataset
        print("Saving processed dataset...")
        with open('processed_dataset.pkl', 'wb') as f:
            pickle.dump(processed_dataset, f)
        print("Dataset saved successfully!")
    
    return processed_dataset

if __name__ == "__main__":
    processed_dataset = load_processed_dataset() 