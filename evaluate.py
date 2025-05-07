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

    run_name = "valiant-leaf-47"
    epoch = 10
    max_length = 24

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)
    model = CaptionGenerator(num_heads=4, num_layers=3, tokenizer=tokenizer)
    model.load_state_dict(load_file(f"model/{run_name}/transformer_{epoch-1}.safetensors"))
    model.eval()

    test_dataset = ImageTextDataset(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(test_dataloader):
        if i > 10:
            break

        image, _, _, _ = batch
        pil_image = test_dataset.get_image(i)
        pil_image.save(f"images/{i}.jpg")

        text = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * 23
        text_token, mask = create_inputs(text, tokenizer)

        for j in range(max_length):
            output = model(image, text_token, mask)
            output = temperature_sampling(output[0, -max_length-j], temperature=1.0, top_k=20)
            text[j] = output.item()
            text_token, mask = create_inputs(text, tokenizer)

            if output == tokenizer.sep_token_id:
                break

        print(tokenizer.decode(text))