# Import required libraries
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, AutoTokenizer
from torchtune.modules import RotaryPositionalEmbeddings
from dataset import get_tokenizer

# Positional encoding layer to add position information to embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=10):
        super().__init__()
        # Create learnable position embeddings
        self.embedding = nn.Embedding(max_len, embedding_dim)

    def forward(self, x):
        # Generate position indices and add them to input embeddings
        positions = torch.arange(0, x.shape[1], device=x.device)
        return self.embedding(positions) + x

# Single attention head implementation
class AttentionHead(nn.Module):
    def __init__(self, x_dim, y_dim, head_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.head_dim = head_dim
        # Linear projections for query, key, and value
        self.query = nn.Linear(x_dim, head_dim)
        self.key = nn.Linear(y_dim, head_dim)
        self.value = nn.Linear(y_dim, head_dim)
        
    def forward(self, x, y, mask):
        # Project inputs to query, key, value
        query = self.query(x) # [batch_size, num_tokens, head_dim]
        key = self.key(y)
        value = self.value(y)
        
        # Compute attention scores and apply mask
        attention_weights = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = attention_weights.masked_fill(torch.logical_not(mask.bool()), float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        # Compute weighted sum of values
        weighted_value = torch.bmm(attention_weights, value)

        return weighted_value

# Multi-head attention implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim, num_heads):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        # Create multiple attention heads
        self.heads = nn.ModuleList([AttentionHead(x_dim, y_dim, output_dim // num_heads) for _ in range(num_heads)])

    def forward(self, x, y, mask):
        # Concatenate outputs from all heads
        return torch.cat([head(x, y, mask) for head in self.heads], dim=-1)
    
# Multi-head attention with Rotary Positional Embeddings (RoPE)
class MultiHeadAttentionWithROPE(nn.Module):
    def __init__(self, x_dim, output_dim, num_heads):
        super().__init__()
        self.x_dim = x_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        # Initialize RoPE for positional encoding
        self.rope = RotaryPositionalEmbeddings(self.head_dim, 222)
        # Combined projection for query, key, value
        self.qkv_proj = nn.Linear(x_dim, output_dim * 3)

    def forward(self, x, mask):
        # Project input to query, key, value
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], self.num_heads, 3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # Apply RoPE to query and key
        q = self.rope(q)
        k = self.rope(k)
        # Reshape for attention computation
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_logits = attn_logits.masked_fill(torch.logical_not(mask.bool()), float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        # Compute weighted sum of values
        weighted_value = torch.matmul(attn_weights, v)
        weighted_value = weighted_value.permute(0, 2, 1, 3)
        weighted_value = weighted_value.reshape(x.shape[0], x.shape[1], self.output_dim)
        return weighted_value

# Self-attention block with residual connections and layer normalization
class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        # Multi-head attention with RoPE
        self.multi_head_attention = MultiHeadAttentionWithROPE(embedding_dim, embedding_dim, num_heads)
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x, mask):
        # Self-attention with residual connection
        x_attn = self.multi_head_attention(x, mask)
        x = self.norm1(x + x_attn)
        # Feed-forward with residual connection
        x_ff = self.feed_forward(x)
        x = self.norm2(x + x_ff)
        return x
    
# Transformer decoder with multiple self-attention blocks
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Stack of self-attention blocks
        self.layers = nn.ModuleList([SelfAttentionBlock(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        # Process through each layer
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
# Main caption generation model
class CaptionGenerator(nn.Module):
    def __init__(self, num_heads, num_layers, tokenizer, img_seq_len=50, text_seq_len=24, device="cpu"):
        super().__init__()
        # Load and initialize BERT model for text processing
        text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        text_model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.text_embedding = text_model.get_input_embeddings()
        # Load and initialize CLIP model for image processing
        self.image_tokenizer = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.embedding_dim = self.text_embedding.embedding_dim
        self.vocab_size = self.text_embedding.num_embeddings
        self.device = device
        # Freeze CLIP parameters
        for param in self.image_tokenizer.parameters():
            param.requires_grad = False

        # Create attention mask for decoder
        self.base_mask = torch.tril(torch.ones(img_seq_len + text_seq_len + 1, img_seq_len + text_seq_len + 1, device=self.device), diagonal=0)
        self.base_mask[:, :img_seq_len + 1] = 1 # Allow image to attend to itself

        # Initialize decoder and output layer
        self.decoder = TransformerDecoder(self.embedding_dim, num_heads, num_layers)
        self.output_layer = nn.Linear(self.embedding_dim, self.vocab_size)

        # Move model to specified device
        self.to(device)

    def build_mask(self, text_mask):
        # Create attention mask for decoder
        mask = self.base_mask.clone().unsqueeze(0).expand(text_mask.shape[0], -1, -1)
        image_mask = torch.zeros(text_mask.shape[0], self.base_mask.shape[0] - text_mask.shape[1], dtype=torch.bool, device=self.device)
        combined_mask = torch.cat([image_mask, text_mask], dim=1)
        expanded_mask = mask.masked_fill(combined_mask.unsqueeze(1), 0)
        return expanded_mask

    def preprocess(self, image, text_token):
        # Process text and image inputs
        text_token = self.text_embedding(text_token)
        image_token = self.image_tokenizer(pixel_values=image, output_hidden_states=True).last_hidden_state
        # Add separator token between image and text
        sep_token = self.text_embedding(torch.tensor([[self.tokenizer.image_end_token_id]], dtype=torch.long, device=self.device)).expand(text_token.shape[0], -1, -1)
        tokens = torch.cat([image_token, sep_token, text_token], dim=1)
        return tokens

    def forward(self, image, text_token, text_mask):
        # Process inputs and generate captions
        tokens = self.preprocess(image, text_token)
        mask = self.build_mask(text_mask)
        tokens = self.decoder(tokens, mask)
        return self.output_layer(tokens)
    
# Test code
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = get_tokenizer()
    caption_generator = CaptionGenerator(num_heads=4, num_layers=3, tokenizer=tokenizer, img_seq_len=10, text_seq_len=10)
    
    # Test mask creation
    text_mask = torch.ones(1, 10, dtype=torch.bool)
    text_mask[0, :3] = False
    attention_mask = caption_generator.build_mask(text_mask)
    
    # Verify mask dimensions and values
    assert attention_mask.shape == (1, 21, 21)
    assert attention_mask[0, 0, 0] == 1
    assert attention_mask[0, 0, 10] == 1
    assert attention_mask[0, 0, 11] == 0
    assert attention_mask[0, 11, 0] == 1
    assert attention_mask[0, 11, 11] == 1
    assert attention_mask[0, 11, 12] == 0
    assert attention_mask[0, 19, 12] == 1
    assert attention_mask[0, 19, 19] == 0

    # Test model forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_generator = CaptionGenerator(num_heads=4, num_layers=3, tokenizer=tokenizer, device=device)
    image = torch.randn(1, 3, 224, 224).to(device)
    text_token = torch.randint(0, 100, (1, 24)).to(device)
    text_mask = torch.ones(1, 25, dtype=torch.bool).to(device)
    text_mask[0, :24] = False
    print(caption_generator(image, text_token, text_mask).shape)
    

