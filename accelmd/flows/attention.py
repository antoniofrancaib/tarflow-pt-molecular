"""Attention mechanisms for autoregressive molecular coordinate flows.

Adapted from Apple's TARFlow with molecular-specific modifications.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, List

__all__ = ["Attention", "MLP", "AttentionBlock"]


class Attention(nn.Module):
    """Multi-head attention with causal masking for autoregressive generation."""
    
    def __init__(self, in_channels: int, head_channels: int):
        super().__init__()
        assert in_channels % head_channels == 0
        
        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        
        # For sampling mode with KV caching
        self.sample = False
        self.k_cache: Dict[str, List[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: Dict[str, List[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        temp: float = 1.0, 
        which_cache: str = 'cond'
    ) -> torch.Tensor:
        """Apply attention with optional causal masking.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, C]
        mask : torch.Tensor, optional
            Attention mask (typically causal)
        temp : float
            Temperature for attention softmax
        which_cache : str
            Cache key for sampling mode
            
        Returns
        -------
        torch.Tensor
            Attended output of shape [B, T, C]
        """
        B, T, C = x.size()
        x_norm = self.norm(x.float()).type(x.dtype)
        
        # Compute Q, K, V
        q, k, v = self.qkv(x_norm).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)
        # q, k, v are now [B, num_heads, T, head_dim]
        
        # Handle KV caching for sampling
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # concat along sequence dim
            v = torch.cat(self.v_cache[which_cache], dim=2)
        
        # Scaled dot-product attention
        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        
        # Use PyTorch's optimized SDPA if available
        try:
            x_attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale
            )
        except AttributeError:
            # Fallback implementation
            attn = torch.einsum('bhid,bhjd->bhij', q * self.sqrt_scale, k * self.sqrt_scale) / temp
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = attn.float().softmax(dim=-1).type(attn.dtype)
            x_attn = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Reshape back to [B, T, C]
        x_attn = x_attn.transpose(1, 2).reshape(B, T, C)
        
        # Output projection
        return self.proj(x_attn)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.main = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        attn_temp: float = 1.0, 
        which_cache: str = 'cond'
    ) -> torch.Tensor:
        """Apply transformer block with residual connections.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, T, C]
        attn_mask : torch.Tensor, optional
            Attention mask
        attn_temp : float
            Attention temperature
        which_cache : str
            Cache key for sampling
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, T, C]
        """
        # Self-attention with residual connection
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        
        # MLP with residual connection
        x = x + self.mlp(x)
        
        return x
