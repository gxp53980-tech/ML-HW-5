import torch
import torch.nn as nn

class SimpleTransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8, dim_ff=256):
        """
        d_model  : embedding dimension
        num_heads: number of attention heads
        dim_ff   : hidden dimension in feed-forward network
        """
        super(SimpleTransformerEncoderBlock, self).__init__()

        # --- Multi-head Self-Attention ---
        # batch_first=True → input shape: (batch_size, seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # --- Feed-Forward Network (FFN) ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        # --- Layer Normalization ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # (Optional) Dropout (can help in practice)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # ------- 1. Multi-Head Self-Attention -------
        # Q = K = V = x for self-attention
        attn_output, attn_weights = self.self_attn(x, x, x)
        # Residual connection + dropout
        x = x + self.dropout1(attn_output)
        # LayerNorm
        x = self.norm1(x)

        # ------- 2. Feed-Forward Network -------
        ff_output = self.ffn(x)
        # Residual connection + dropout
        x = x + self.dropout2(ff_output)
        # LayerNorm
        x = self.norm2(x)

        # x: (batch_size, seq_len, d_model)
        # attn_weights: (batch_size, num_heads, seq_len, seq_len)
        return x, attn_weights


if __name__ == "__main__":
    # ---- Initialize dimensions (part a) ----
    d_model = 64      # embedding size
    num_heads = 8     # must divide d_model evenly
    dim_ff = 256      # FFN hidden size

    encoder_block = SimpleTransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        dim_ff=dim_ff
    )

    # ---- Create dummy batch (part c) ----
    batch_size = 32
    seq_len = 10

    # Input: (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    out, attn_weights = encoder_block(x)

    print("Input shape:     ", x.shape)
    print("Output shape:    ", out.shape)
    print("Attention shape: ", attn_weights.shape)
    # Expected:
    # Input shape:      torch.Size([32, 10, 64])
    # Output shape:     torch.Size([32, 10, 64])
    # Attention shape:  torch.Size([32, 8, 10, 10])
