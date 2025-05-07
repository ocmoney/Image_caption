# models/encoder_vit.py

import torch
import torch.nn as nn
import math

class ImagePatcher(nn.Module):
    """
    Splits an image into patches, flattens, embeds, and adds positional encoding.
    Example: 224x224 image with 16x16 patches → 196 tokens (14x14 grid)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: linear projection of flattened patches
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Positional embeddings (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        Input:
        - x: [batch_size, channels, height, width]
        
        Output:
        - tokens: [batch_size, num_patches, embed_dim]
        """
        B, C, H, W = x.shape

        patch_size = int(math.sqrt(self.patch_dim // C))
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)  # [B, C, N, P, P]
        x = x.permute(0, 2, 1, 3, 4)                                # [B, N, C, P, P]
        x = x.flatten(2)                                            # [B, N, C*P*P]

        tokens = self.patch_embed(x)                                # [B, N, embed_dim]
        tokens = tokens + self.pos_embed                            # add positional encoding

        return tokens


def test_image_patcher():
    model = ImagePatcher(img_size=224, patch_size=16, in_channels=3, embed_dim=512)

    dummy_image = torch.randn(1, 3, 224, 224)  # [batch, channels, H, W]

    tokens = model(dummy_image)
    print("Output shape:", tokens.shape)  # Expect: [1, 196, 512]

if __name__ == "__main__":
    test_image_patcher()