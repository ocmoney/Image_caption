# models/caption_model.py
import torch
import torch.nn as nn
from models.encoder_vit import ImagePatcher
from models.decoder_transformer import TransformerDecoder


class CaptioningModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len=30,
                 image_size=224,
                 patch_size=16,
                 embed_dim=512,
                 num_heads=8,
                 ff_dim=2048,
                 num_layers=6,
                 dropout=0.1):
        super().__init__()

        self.encoder = ImagePatcher(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, image_tensor, caption_input):
        """
        Inputs:
            image_tensor: [B, 3, H, W]
            caption_input: [B, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        # Encode image
        image_tokens = self.encoder(image_tensor.to(caption_input.device))  # [B, S, D]
        memory = image_tokens.transpose(0, 1)  # [S, B, D]

        # Generate caption
        tgt_mask = self.decoder.generate_square_subsequent_mask(caption_input.size(1)).to(caption_input.device)
        logits = self.decoder(caption_input, memory, tgt_mask=tgt_mask)  # [B, T, vocab_size]
        return logits

if __name__ == "__main__":
    model = CaptioningModel(vocab_size=150000)
    dummy_image = torch.randn(2, 3, 224, 224)       # [B, 3, 224, 224]
    dummy_caption = torch.randint(0, 8000, (2, 30)) # [B, T]
    output = model(dummy_image, dummy_caption)
    print("Output shape:", output.shape)  # [2, 30, 8000]
