from model import CaptionGenerator
from safetensors.torch import load_file
from dataset import ImageTextDataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def create_inputs(input_text, tokenizer):
    text_token = torch.tensor(input_text, dtype=torch.long)
    mask = text_token == tokenizer.pad_token_id
    return text_token.unsqueeze(0), mask.unsqueeze(0)

def temperature_sampling(logits, temperature=1.0, top_k=20):
    topk = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(topk.values / temperature, dim=-1)
    indices = torch.multinomial(probs, 1)
    return topk.indices.gather(dim=-1, index=indices)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = "cosmic-yogurt-52"
    epoch = 10
    max_length = 24

    test_dataset = ImageTextDataset(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    tokenizer = test_dataset.tokenizer

    model = CaptionGenerator(num_heads=6, num_layers=8, tokenizer=tokenizer)
    model.load_state_dict(load_file(f"model/{run_name}/transformer_{epoch-1}.safetensors"))
    model.eval()

    for i, batch in enumerate(test_dataloader):
        if i > 10:
            break

        image, input_tokens, output_tokens, real_mask = batch
        pil_image = test_dataset.get_image(i)
        pil_image.save(f"images/{i}.jpg")

        text = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * 23
        text_token, mask = create_inputs(text, tokenizer)

        for j in range(max_length):
            output = model(image, text_token, mask)
            output = temperature_sampling(output[0, -max_length-j], temperature=0.7, top_k=20)
            text[j] = output.item()
            text_token, mask = create_inputs(text, tokenizer)

            if output == tokenizer.sep_token_id:
                break

        print(f"Prediction {i}:\n{tokenizer.decode(text)}")

        output = model(image, input_tokens, real_mask)
        output = torch.argmax(output, dim=-1)
        output = tokenizer.decode(output.squeeze().tolist())
        print(f"Force feeding prediction {i}:\n{output}")
        print(f"Ground truth {i}:\n{tokenizer.decode(output_tokens.squeeze().tolist())}")
        print()