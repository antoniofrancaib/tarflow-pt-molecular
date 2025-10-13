#!/usr/bin/env python3
"""
Scalable transformer-based autoregressive normalizing flow
Designed to handle arbitrary input dimensions (10D → 50D → 100D → molecular)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with causal masking"""
    
    def __init__(self, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q, k, v: [batch_size, num_heads, seq_len, head_dim]
            mask: [seq_len, seq_len] causal mask
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for autoregressive flows"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False) 
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [seq_len, seq_len] causal mask
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        q = self.w_q(x)  # [batch_size, seq_len, embed_dim]
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output = self.attention(q, k, v, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions"""
    
    def __init__(self, embed_dim: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class ScalableAutoregressiveFlowLayer(nn.Module):
    """Scalable autoregressive flow layer using transformer architecture"""
    
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
            
        # Input embedding: each coordinate becomes a token
        self.input_embedding = nn.Linear(1, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=input_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for scale and shift parameters
        self.scale_head = nn.Linear(embed_dim, 1)
        self.shift_head = nn.Linear(embed_dim, 1)
        
        # Create causal mask
        mask = torch.tril(torch.ones(input_dim, input_dim))
        self.register_buffer('causal_mask', mask)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters for stable training"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize output heads to small values for stability
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation
        Args:
            x: [batch_size, input_dim]
        Returns:
            z: transformed tensor
            log_det: log determinant of Jacobian
        """
        batch_size, seq_len = x.shape
        assert seq_len == self.input_dim
        
        # Embed each coordinate as a token: [batch_size, seq_len, 1] -> [batch_size, seq_len, embed_dim]
        x_embed = self.input_embedding(x.unsqueeze(-1))
        
        # Add positional encoding
        x_embed = self.pos_encoding(x_embed)
        
        # Apply transformer layers with causal masking
        hidden = x_embed
        for layer in self.transformer_layers:
            hidden = layer(hidden, self.causal_mask)
        
        # Get transformation parameters
        log_scale = self.scale_head(hidden).squeeze(-1)  # [batch_size, seq_len]
        shift = self.shift_head(hidden).squeeze(-1)      # [batch_size, seq_len]
        
        # Apply autoregressive masking: x_i depends only on x_{<i}
        # Shift parameters to ensure autoregressive property
        log_scale_masked = torch.cat([
            torch.zeros(batch_size, 1, device=x.device),
            log_scale[:, :-1]
        ], dim=1)
        
        shift_masked = torch.cat([
            torch.zeros(batch_size, 1, device=x.device),
            shift[:, :-1]
        ], dim=1)
        
        # Apply transformation: z = x * exp(log_scale) + shift
        scale = torch.exp(log_scale_masked)
        z = x * scale + shift_masked
        
        # Log determinant is sum of log scales
        log_det = log_scale_masked.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation (autoregressive sampling)
        Memory-efficient version using pre-allocated tensor.
        
        Args:
            z: latent tensor [batch_size, input_dim]
        Returns:
            x: reconstructed input
            log_det: log determinant of Jacobian
        """
        batch_size = z.shape[0]
        # Pre-allocate output tensor (more memory efficient than list + cat)
        x = torch.zeros_like(z)
        log_det_total = torch.zeros(batch_size, device=z.device)
        
        # Sequentially reconstruct each dimension
        for i in range(self.input_dim):
            if i == 0:
                # First dimension: no transformation (direct copy, not in-place op on input)
                x[:, 0] = z[:, 0]
            else:
                # For dimension i, use previous dimensions x[:, :i]
                # Create a NEW tensor from the slice (not a view)
                x_partial = x[:, :i].clone().unsqueeze(-1)  # [batch_size, i, 1]
                x_partial_embed = self.input_embedding(x_partial)
                x_partial_embed = self.pos_encoding(x_partial_embed)
                
                # Apply transformer layers (only up to position i-1)
                hidden = x_partial_embed
                mask = self.causal_mask[:i, :i] if i > 1 else None
                
                for layer in self.transformer_layers:
                    hidden = layer(hidden, mask)
                
                # Get parameters for current dimension from last position
                log_scale_i = self.scale_head(hidden[:, -1:]).squeeze(-1)  # [batch_size, 1]
                shift_i = self.shift_head(hidden[:, -1:]).squeeze(-1)      # [batch_size, 1]
                
                # Inverse transformation for dimension i
                scale_i = torch.exp(log_scale_i)
                # Compute to temporary variable first, then assign
                x_i_new = (z[:, i] - shift_i.squeeze(-1)) / scale_i.squeeze(-1)
                x[:, i] = x_i_new  # Assign to pre-allocated tensor (OK because x not in computation graph yet)
                
                # Accumulate log determinant (non-in-place)
                log_det_total = log_det_total - log_scale_i.squeeze(-1)
        
        return x, log_det_total


class ScalableTransformerFlow(nn.Module):
    """Complete scalable transformer-based normalizing flow"""
    
    def __init__(self,
                 input_dim: int,
                 num_flow_layers: int = 6,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_transformer_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_flow_layers = num_flow_layers
        
        # Stack of autoregressive flow layers
        self.flow_layers = nn.ModuleList([
            ScalableAutoregressiveFlowLayer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_transformer_layers,
                dropout=dropout
            )
            for _ in range(num_flow_layers)
        ])
        
        # Learnable permutation layers between flows
        self.permutations = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False)
            for _ in range(num_flow_layers - 1)
        ])
        
        # Initialize permutation matrices as random orthogonal matrices
        self._init_permutations()
        
    def _init_permutations(self):
        """Initialize permutation matrices as orthogonal transformations"""
        for perm in self.permutations:
            # Initialize as random orthogonal matrix
            nn.init.orthogonal_(perm.weight)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow
        Args:
            x: input tensor [batch_size, input_dim]
        Returns:
            z: latent tensor
            log_det: total log determinant
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for i, flow_layer in enumerate(self.flow_layers):
            # Apply autoregressive transformation
            z, log_det = flow_layer(z)
            log_det_total = log_det_total + log_det  # Non-in-place addition
            
            # Apply permutation (except for last layer)
            if i < len(self.permutations):
                z = self.permutations[i](z)
                # Permutation doesn't change log determinant (assuming orthogonal)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through the flow
        Args:
            z: latent tensor
        Returns:
            x: reconstructed input  
            log_det: total log determinant
        """
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        # Apply layers in reverse order
        for i in reversed(range(len(self.flow_layers))):
            # Reverse permutation first (except for last layer)
            if i < len(self.permutations):
                # Use pseudo-inverse for permutation (should be close to transpose for orthogonal)
                try:
                    perm_inv = torch.linalg.inv(self.permutations[i].weight)
                    x = F.linear(x, perm_inv)
                except:
                    # Fallback to transpose
                    x = F.linear(x, self.permutations[i].weight.T)
            
            # Apply inverse autoregressive transformation
            x, log_det = self.flow_layers[i].inverse(x)
            log_det_total = log_det_total + log_det  # Non-in-place addition
        
        return x, log_det_total
    
    def forward_with_logdet(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation with log determinant (alias for forward).
        Used by molecular PT trainer.
        
        Args:
            x: input tensor [batch_size, input_dim]
        Returns:
            z: transformed tensor
            log_det: log determinant of Jacobian
        """
        return self.forward(x)
    
    def inverse_with_logdet(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation with log determinant (alias for inverse).
        Used by molecular PT trainer.
        
        Args:
            z: latent tensor [batch_size, input_dim]
        Returns:
            x: reconstructed input
            log_det: log determinant of Jacobian
        """
        return self.inverse(z)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the model
        Args:
            x: input tensor
        Returns:
            log_prob: log probability
        """
        z, log_det = self.forward(x)
        
        # Standard Gaussian log probability in latent space
        log_prob_z = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        
        return log_prob_z + log_det
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the model
        Args:
            num_samples: number of samples
            device: device to generate samples on
        Returns:
            samples: generated samples
        """
        # Sample from standard Gaussian in latent space
        z = torch.randn(num_samples, self.input_dim, device=device)
        
        # Transform through inverse flow
        x, _ = self.inverse(z)
        
        return x


if __name__ == "__main__":
    # Test the scalable transformer flow
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test different dimensions
    for dim in [10, 25, 50]:
        print(f"\nTesting {dim}D flow...")
        
        model = ScalableTransformerFlow(
            input_dim=dim,
            num_flow_layers=4,
            embed_dim=64,
            num_heads=8,
            num_transformer_layers=3
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(32, dim, device=device)
        z, log_det = model.forward(x)
        
        print(f"Forward: {x.shape} -> {z.shape}, log_det: {log_det.shape}")
        
        # Test inverse pass
        x_recon, log_det_inv = model.inverse(z)
        reconstruction_error = torch.mean((x - x_recon)**2)
        
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        
        # Test sampling
        samples = model.sample(16, device)
        print(f"Sample shape: {samples.shape}")
        
        # Test log probability
        log_prob = model.log_prob(x)
        print(f"Log prob shape: {log_prob.shape}, mean: {log_prob.mean():.3f}")
    
    print("\nScalable transformer flow test completed!")
