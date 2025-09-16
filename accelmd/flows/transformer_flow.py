"""Core autoregressive transformer flow for molecular coordinates.

Adapted from Apple's TARFlow for molecular PT swap applications.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .permutations import Permutation, PermutationIdentity, PermutationFlip
from .attention import AttentionBlock
from .coordinate_embedding import CoordinateEmbedding, TemperatureConditioning

__all__ = ["MetaBlock", "TransformerFlow"]


class MetaBlock(nn.Module):
    """Autoregressive transformer block for coordinate sequences.
    
    Adapted from TARFlow's MetaBlock for molecular coordinates.
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        temperature_conditioning: bool = False,
    ):
        super().__init__()
        
        # Input projection
        self.proj_in = nn.Linear(in_channels, channels)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 0.01)
        
        # Temperature conditioning (optional)
        if temperature_conditioning:
            self.temp_conditioning = TemperatureConditioning(channels)
        else:
            self.temp_conditioning = None
        
        # Transformer blocks
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(channels, head_dim, expansion) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, output_dim)
        
        # Initialize output projection to very small random values for stability
        # Start with tiny transformations that can grow during training
        nn.init.normal_(self.proj_out.weight.data, mean=0.0, std=0.001)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias.data)
        
        # Permutation strategy
        self.permutation = permutation
        
        # Causal mask for autoregressive generation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))

    def forward(
        self, 
        x: torch.Tensor, 
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through meta block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape [B, N, 3]
        source_temp : float, optional
            Source temperature for conditioning
        target_temp : float, optional
            Target temperature for conditioning
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (transformed_coords, log_det)
        """
        # Apply permutation
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        
        x_in = x
        
        # Input projection and positional embedding
        x = self.proj_in(x) + pos_embed
        
        # Add temperature conditioning if available
        if self.temp_conditioning is not None and source_temp is not None and target_temp is not None:
            temp_cond = self.temp_conditioning(source_temp, target_temp, x.shape[0], x.device)
            x = x + temp_cond  # Broadcast along sequence dimension
        
        # Apply transformer blocks with causal masking
        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        
        # Output projection
        x = self.proj_out(x)
        
        # Autoregressive shift - but handle single atom case properly
        if x.shape[1] > 1:
            # Multi-atom: standard autoregressive masking
            x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        else:
            # Single atom: use a simple coupling layer approach for stability
            # Split coordinates and only transform half
            x_in_flat = x_in.view(x.shape[0], -1)  # [B, coord_dim]
            coord_dim = x_in_flat.shape[-1]
            
            if coord_dim > 1:
                # Split coordinates: transform second half based on first half
                split_dim = coord_dim // 2
                x1, x2 = x_in_flat[:, :split_dim], x_in_flat[:, split_dim:]
                
                # Use network output to transform only x2
                x_net = x.view(x.shape[0], -1)  # [B, output_dim]
                
                if self.nvp:
                    # Scale and translate x2
                    log_s = x_net[:, :split_dim].tanh()  # Bounded scaling
                    t = x_net[:, split_dim:split_dim*2]
                    x2_new = x2 * torch.exp(log_s) + t
                    log_det = log_s.sum(dim=1)
                else:
                    # Only translate x2  
                    t = x_net[:, :split_dim]
                    x2_new = x2 + t
                    log_det = torch.zeros(x.shape[0], device=x.device)
                
                # Combine transformed coordinates
                x_out = torch.cat([x1, x2_new], dim=1).view_as(x_in)
                return x_out, log_det
            else:
                # Single coordinate: return unchanged (identity)
                return x_in, torch.zeros(x.shape[0], device=x.device)
        
        if self.nvp:
            # Non-volume preserving: split into scale and translation
            xa, xb = x.chunk(2, dim=-1)
        else:
            # Volume preserving: only translation
            xb = x
            xa = torch.zeros_like(x)
        
        # Apply transformation: (x - xb) * exp(-xa)
        scale = (-xa.float()).exp().type(xa.dtype)
        transformed = self.permutation((x_in - xb) * scale, inverse=True)
        
        # Log determinant
        log_det = -xa.mean(dim=[1, 2])
        
        return transformed, log_det

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single reverse step for autoregressive sampling."""
        # Get i-th patch but keep sequence dimension
        x_in = x[:, i:i+1]
        
        # Project and add positional embedding
        x_embed = self.proj_in(x_in) + pos_embed[i:i+1]
        
        # Add temperature conditioning
        if self.temp_conditioning is not None and source_temp is not None and target_temp is not None:
            temp_cond = self.temp_conditioning(source_temp, target_temp, x.shape[0], x.device)
            x_embed = x_embed + temp_cond
        
        # Apply transformer blocks (with KV caching)
        for block in self.attn_blocks:
            x_embed = block(x_embed, attn_temp=attn_temp, which_cache=which_cache)
        
        # Output projection
        x_out = self.proj_out(x_embed)
        
        if self.nvp:
            xa, xb = x_out.chunk(2, dim=-1)
        else:
            xb = x_out
            xa = torch.zeros_like(x_out)
        
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        """Enable/disable sampling mode with KV caching."""
        for m in self.modules():
            if hasattr(m, 'sample'):
                m.sample = flag
                if hasattr(m, 'k_cache'):
                    m.k_cache = {'cond': [], 'uncond': []}
                if hasattr(m, 'v_cache'):
                    m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        guidance: float = 0,
        attn_temp: float = 1.0,
    ) -> torch.Tensor:
        """Reverse (sampling) pass through the block."""
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        
        self.set_sample_mode(True)
        T = x.size(1)
        
        for i in range(x.size(1) - 1):
            # Conditional sampling
            xa, xb = self.reverse_step(x, pos_embed, i, source_temp, target_temp, which_cache='cond')
            
            # Optional unconditional guidance
            if guidance > 0:
                xa_u, xb_u = self.reverse_step(x, pos_embed, i, None, None, attn_temp=attn_temp, which_cache='uncond')
                xa = xa + guidance * (xa - xa_u)
                xb = xb + guidance * (xb - xb_u)
            
            # Apply transformation to next coordinate
            scale = xa[:, 0].float().exp().type(xa.dtype)  # Remove sequence dim
            x[:, i + 1] = x[:, i + 1] * scale + xb[:, 0]
        
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class TransformerFlow(nn.Module):
    """Autoregressive transformer flow for molecular coordinates.
    
    Main TARFlow model adapted for molecular PT swap applications.
    """
    
    def __init__(
        self,
        num_atoms: int,
        coordinate_dim: int = 3,
        channels: int = 256,
        num_blocks: int = 4,
        layers_per_block: int = 2,
        nvp: bool = True,
        temperature_conditioning: bool = True,
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 32,
    ):
        super().__init__()
        
        self.coordinate_dim = coordinate_dim
        self.num_atoms = num_atoms
        self.channels = channels
        
        # Coordinate embedding
        self.coord_embedding = CoordinateEmbedding(
            coordinate_dim=coordinate_dim,
            embed_dim=coordinate_dim,  # Keep same dimension for compatibility
            atom_vocab_size=atom_vocab_size,
            atom_embed_dim=atom_embed_dim if atom_vocab_size > 0 else 0,
        )
        
        # Permutation strategies (alternate between identity and flip)
        permutations = [PermutationIdentity(num_atoms), PermutationFlip(num_atoms)]
        
        # Build meta blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels=coordinate_dim,
                    channels=channels,
                    num_patches=num_atoms,
                    permutation=permutations[i % 2],
                    num_layers=layers_per_block,
                    nvp=nvp,
                    temperature_conditioning=temperature_conditioning,
                )
            )
        
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, 
        coordinates: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the flow.
        
        Parameters
        ----------
        coordinates : torch.Tensor
            Input coordinates of shape [B, N, 3]
        atom_types : torch.Tensor, optional
            Atom type indices
        source_temp : float, optional
            Source temperature
        target_temp : float, optional  
            Target temperature
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (output_coordinates, total_log_det)
        """
        x = coordinates
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        
        # Apply coordinate embedding (currently just pass through)
        # x = self.coord_embedding(x, atom_types)
        
        # Apply transformer blocks
        for block in self.blocks:
            x, log_det = block(x, source_temp, target_temp)
            total_log_det = total_log_det + log_det
        
        return x, total_log_det

    def reverse(
        self,
        coordinates: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
        source_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        guidance: float = 0,
        attn_temp: float = 1.0,
    ) -> torch.Tensor:
        """Reverse (sampling) pass through the flow."""
        x = coordinates
        
        # Apply blocks in reverse order
        for block in reversed(self.blocks):
            x = block.reverse(x, source_temp, target_temp, guidance, attn_temp)
        
        return x

    def log_prob(self, coordinates: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute log probability of coordinates."""
        _, log_det = self.forward(coordinates, **kwargs)
        
        # Base log probability (standard normal)
        base_log_prob = -0.5 * coordinates.pow(2).sum(dim=[1, 2])
        
        return base_log_prob + log_det
