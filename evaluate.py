# Import required modules
from model import CaptionGenerator
from safetensors.torch import load_file
from dataset import ImageTextDataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_captioner import compute_accuracy

# Function to create input tensors for the model
def create_inputs(input_text, tokenizer):
    # Convert input text to tensor and create mask
    text_token = torch.tensor(input_text, dtype=torch.long)
    mask = text_token == tokenizer.pad_token_id
    return text_token.unsqueeze(0), mask.unsqueeze(0)

# Function to perform temperature sampling for text generation
def temperature_sampling(logits, temperature=1.0, top_k=20):
    # Get top-k logits and apply temperature
    topk = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(topk.values / temperature, dim=-1)
    # Sample from the distribution
    indices = torch.multinomial(probs, 1)
    return topk.indices.gather(dim=-1, index=indices)

# Function to generate captions for an image
def predict_caption(model, image, tokenizer, max_length=24):
    # Initialize with CLS token and padding
    text = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * 23
    text_token, mask = create_inputs(text, tokenizer)

    # Generate tokens one by one
    for j in range(max_length-1):
        # Get model predictions
        output = model(image, text_token, mask)[:, -max_length:, :]
        output = torch.argmax(output[0, j], dim=-1)
        text[j+1] = output.item()
        text_token, mask = create_inputs(text, tokenizer)

        # Stop if we generate a separator token
        if output == tokenizer.sep_token_id:
            break

    # Process the generated text
    text = text[1:]  # Remove CLS token
    if tokenizer.sep_token_id in text:
        text = text[:text.index(tokenizer.sep_token_id)]  # Remove everything after SEP token

    return tokenizer.decode(text)

if __name__ == "__main__":
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model configuration
    run_name = "cosmic-yogurt-52"
    epoch = 10
    max_length = 24

    # Initialize test dataset and dataloader
    test_dataset = ImageTextDataset(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    tokenizer = test_dataset.tokenizer

    # Load trained model
    model = CaptionGenerator(num_heads=6, num_layers=8, tokenizer=tokenizer)
    model.load_state_dict(load_file(f"model/{run_name}/transformer_{epoch-1}.safetensors"))
    model.eval()

    # Evaluate on test set
    for i, batch in enumerate(test_dataloader):
        if i > 10:  # Limit to first 10 examples
            break

        # Get batch data
        image, input_tokens, output_tokens, real_mask = batch
        # Save image for visualization
        pil_image = test_dataset.get_image(i)
        pil_image.save(f"images/{i}.jpg")

        # Generate caption
        text = predict_caption(model, image, tokenizer)

        # Print results
        print(f"Prediction {i}:\n{text}")

        # Get forced decoding results
        output = model(image, input_tokens, real_mask)[:, -max_length:, :]
        text_output = torch.argmax(output, dim=-1)
        text_output = tokenizer.decode(text_output.squeeze().tolist())
        print(f"Force feeding prediction {i}:\n{text_output}")
        print("Accuracy", compute_accuracy(output, output_tokens, tokenizer.pad_token_id))
        print(f"Ground truth {i}:\n{tokenizer.decode(output_tokens.squeeze().tolist())}")
        print()