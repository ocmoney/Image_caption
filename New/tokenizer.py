# data/tokenizer.py

import sentencepiece as spm
import os
from datasets import load_dataset

def save_all_captions_to_txt(output_path="data/tokenizer/captions.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ✅ Make sure the folder exists

    dataset = load_dataset("nlphuji/flickr30k", split="test")  # This is the only available full set

    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            for caption in sample["caption"]:
                f.write(caption.strip() + "\n")

    print(f"✅ Saved all captions to: {output_path}")


def train_sentencepiece(input_path="data/tokenizer/captions.txt", model_prefix="data/tokenizer/spm", vocab_size=8000):
    spm.SentencePieceTrainer.train(
        input=f"{input_path}",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        model_type="unigram",  # You can also try "bpe"
        user_defined_symbols=["<pad>", "<sos>", "<eos>"]
    )
    print("✅ SentencePiece tokenizer trained and saved.")

class CaptionTokenizer:
    def __init__(self, model_path="data/tokenizer/spm.model", max_length=30):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.max_length = max_length
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.sos_id = self.sp.piece_to_id("<sos>")
        self.eos_id = self.sp.piece_to_id("<eos>")

    def encode(self, text):
        tokens = self.sp.encode(text, out_type=int)
        tokens = [self.sos_id] + tokens + [self.eos_id]
        if len(tokens) < self.max_length:
            tokens += [self.pad_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return tokens

    def decode(self, token_ids):
        if self.eos_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.eos_id) + 1]
        return self.sp.decode(token_ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

# Example usage
if __name__ == "__main__":
    tokenizer = CaptionTokenizer()
    example = "a man in red is riding a bike"
    ids = tokenizer.encode(example)
    print("Encoded:", ids)
    print("Decoded:", tokenizer.decode(ids))


# if __name__ == "__main__":
#     save_all_captions_to_txt()
#     train_sentencepiece()