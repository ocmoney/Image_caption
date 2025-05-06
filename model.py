import torch
import torch.nn as nn
from dataset import TextEmbedding, ImageTokenizer
from PIL import Image

class AttentionHead(nn.Module):
    def __init__(self, x_dim, y_dim, head_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.head_dim = head_dim
        self.query = nn.Linear(x_dim, head_dim)
        self.key = nn.Linear(y_dim, head_dim)
        self.value = nn.Linear(y_dim, head_dim)
        
    def forward(self, x, y, mask):
        query = self.query(x) # [batch_size, num_tokens, head_dim]
        key = self.key(y)
        value = self.value(y)
        
        attention_weights = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) # [batch_size, num_tokens, num_tokens]
        attention_weights = attention_weights.masked_fill(torch.logical_not(mask.bool()), float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        weighted_value = torch.bmm(attention_weights, value)

        return weighted_value # [batch_size, num_tokens, head_dim]

class MultiHeadAttention(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim, num_heads):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionHead(x_dim, y_dim, output_dim // num_heads) for _ in range(num_heads)])

    def forward(self, x, y, mask):
        return torch.cat([head(x, y, mask) for head in self.heads], dim=-1)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, embedding_dim, embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x, mask):
        x_attn = self.multi_head_attention(x, x, mask)
        x = self.norm1(x + x_attn)
        x_ff = self.feed_forward(x)
        x = self.norm2(x + x_ff)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList([SelfAttentionBlock(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class CaptionGenerator(nn.Module):
    def __init__(self, num_heads, num_layers, img_seq_len=197, text_seq_len=24, device="cpu"):
        super().__init__()
        self.text_embedding = TextEmbedding().embeddings
        self.embedding_dim = self.text_embedding.embedding_dim
        self.vocab_size = self.text_embedding.num_embeddings
        self.device = device
        for param in self.text_embedding.parameters():
            param.requires_grad = False

        # Create a separator token to split the image and text tokens
        self.sep_embedding = nn.Embedding(1, self.embedding_dim)

        # Create a base mask to prevent the decoder from attending to future tokens
        self.base_mask = torch.tril(torch.ones(img_seq_len + text_seq_len + 1, img_seq_len + text_seq_len + 1, device=self.device), diagonal=0)
        # self.base_mask[:img_seq_len + 1, :] = 0 # Prevent the decoder from attending to the image tokens

        self.decoder = TransformerDecoder(self.embedding_dim, num_heads, num_layers)
        self.output_layer = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.to(device)

    def build_mask(self, text_mask):
        # Create a mask to prevent the decoder from attending to padding tokens
        mask = self.base_mask.clone().unsqueeze(0).expand(text_mask.shape[0], -1, -1)
        image_mask = torch.ones(text_mask.shape[0], self.base_mask.shape[0] - text_mask.shape[1], dtype=torch.bool, device=self.device)
        combined_mask = torch.cat([image_mask, text_mask], dim=1)
        expanded_mask = mask.masked_fill(torch.logical_not(combined_mask.unsqueeze(1)), 0)
        return expanded_mask

    def preprocess(self, image_token, text_token):
        text_token = self.text_embedding(text_token)
        sep_token = self.sep_embedding(torch.tensor([[0]], dtype=torch.long, device=self.device)).expand(text_token.shape[0], -1, -1)
        tokens = torch.cat([image_token, sep_token, text_token], dim=1)
        return tokens

    def forward(self, image_token, text_token, text_mask):
        tokens = self.preprocess(image_token, text_token)
        mask = self.build_mask(text_mask)
        tokens = self.decoder(tokens, mask)
        return self.softmax(self.output_layer(tokens))
    
if __name__ == "__main__":
    caption_generator = CaptionGenerator(num_heads=4, num_layers=3, img_seq_len=10, text_seq_len=10)
    text_mask = torch.ones(1, 10, dtype=torch.bool)
    text_mask[0, 3:] = False
    attention_mask = caption_generator.build_mask(text_mask)
    assert attention_mask.shape == (1, 21, 21)
    assert attention_mask[0, 0, 0] == 0
    assert attention_mask[0, 0, 10] == 0
    assert attention_mask[0, 11, 0] == 1
    assert attention_mask[0, 11, 11] == 1
    assert attention_mask[0, 11, 12] == 0
    assert attention_mask[0, 19, 12] == 1
    assert attention_mask[0, 19, 19] == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    caption_generator = CaptionGenerator(num_heads=4, num_layers=3, device=device)
    image_token = torch.randn(1, 197, 768).to(device)
    text_token = torch.randint(0, 100, (1, 24)).to(device)
    text_mask = torch.ones(1, 25, dtype=torch.bool).to(device)
    text_mask[10:] = False
    print(caption_generator(image_token, text_token, text_mask).shape)
    

