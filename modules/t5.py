import torch
import torch.nn as nn
from .base import Module

class T5LayerNorm(Module):
    """
    RMSNorm: Root Mean Square Normalization.
    T5 drops the shift (beta) and mean-centering of standard LayerNorm,
    keeping only the scaling (gamma) based on the root mean square.
    This is computationally cheaper and performs identically.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # Calculate root mean square without subtracting the mean
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

class RelativePositionBias(Module):
    """
    Instead of adding sinusoidal waves to the bottom input embeddings, 
    T5 calculates the literal scalar distance between two tokens (e.g. token J is 3 steps right of I).
    It maps that distance to a learned bucket (an embedding), and adds it directly 
    to the Attention logits before the softmax pass.
    """
    def __init__(self, num_heads, num_buckets=32):
        super().__init__()
        self.num_buckets = num_buckets
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _compute_buckets(self, relative_position):
        # Simplified bucketing logic for educational purposes:
        # We cleanly clamp the relative scalar distances into the available bucket memory.
        # e.g., distances from -16 to +15 map to buckets 0 to 31.
        half = self.num_buckets // 2
        clamped = relative_position.clamp(-half, half - 1)
        return clamped + half # Shift to positive indices

    def forward(self, q_len, k_len):
        # Create a grid of index distances
        # e.g., if q_len=3, k_len=3 -> k_pos - q_pos generates a matrix of integer distances
        q_pos = torch.arange(q_len)[:, None]
        k_pos = torch.arange(k_len)[None, :]
        relative_position = k_pos - q_pos 
        
        # Map scalar distances to bucket indices
        bucket_indices = self._compute_buckets(relative_position)
        
        # Lookup the learned bias for each bucket (q_len, k_len, num_heads)
        values = self.relative_attention_bias(bucket_indices.to(self.relative_attention_bias.weight.device))
        
        # Reshape to (1, num_heads, q_len, k_len) to natively broadcast across all batches
        return values.permute(2, 0, 1).unsqueeze(0)

class T5Attention(Module):
    """
    Custom Attention block that natively accepts a position_bias mathematical injection.
    T5 also actively removes all additive biases from Linear layers to save memory.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Bias is strictly False
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None, position_bias=None):
        batch, q_len, _ = q.size()
        k_len = k.size(1)
        
        Q = self.w_q(q).view(batch, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch, k_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # The defining feature of T5 Attention: Injecting the relative spatial map
        if position_bias is not None:
            scores = scores + position_bias
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        
        out = context.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.w_o(out)


class T5(Module):
    """
    Text-to-Text Transfer Transformer (T5).
    A unified Encoder-Decoder architecture combining RMSNorm, Relative Biases, 
    and Weight Tying to cast every NLP problem as text-in, text-out.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        
        # Shared vocabulary space
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        
        # Instead of absolute sinusoidal waves attached to embeddings,
        # T5 uses a single relative bias generator shared across the encoder
        self.encoder_relative_bias = RelativePositionBias(num_heads)
        self.decoder_relative_bias = RelativePositionBias(num_heads)
        
        # We instantiate a single standard T5 Block to represent the core architectural shift
        self.encoder_attention = T5Attention(d_model, num_heads)
        self.encoder_norm = T5LayerNorm(d_model)
        
        self.decoder_attention = T5Attention(d_model, num_heads)
        self.decoder_norm = T5LayerNorm(d_model)
        
        self.final_linear = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying (Massive memory optimization)
        # The final projection uses the exact same memory pointer as the input embeddings
        self.final_linear.weight = self.shared_embedding.weight
        
    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor):
        # 1. Embeddings (No absolute positional math is added here!)
        enc_x = self.shared_embedding(source_ids)
        dec_x = self.shared_embedding(target_ids)
        
        batch, enc_len, _ = enc_x.size()
        batch, dec_len, _ = dec_x.size()
        
        # 2. Generate the spatial matrix biases based purely on relative sequence lengths
        enc_bias = self.encoder_relative_bias(enc_len, enc_len)
        dec_bias = self.decoder_relative_bias(dec_len, dec_len)
        
        # 3. Native self-attention passes
        enc_out = self.encoder_attention(enc_x, enc_x, enc_x, position_bias=enc_bias)
        enc_out = self.encoder_norm(enc_x + enc_out)
        
        # 4. In a full stack, Decoder would do Masked Self-Attention + Cross-Attention
        # Here we just show the root spatial injection implementation
        dec_out = self.decoder_attention(dec_x, dec_x, dec_x, position_bias=dec_bias)
        dec_out = self.decoder_norm(dec_x + dec_out)
        
        # 5. Egress through the tied vocabulary
        logits = self.final_linear(dec_out)
        return logits
