"""Coordinate embedding utilities for molecular TARFlow.

Handles conversion between molecular coordinates and autoregressive sequence format.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

__all__ = ["CoordinateEmbedding", "TemperatureConditioning"]


class CoordinateEmbedding(nn.Module):
    """Embed molecular coordinates for autoregressive processing.
    
    Converts [B, N, 3] coordinates to patch-like sequences that can be
    processed autoregressively by transformer blocks.
    """
    
    def __init__(
        self, 
        coordinate_dim: int = 3,
        embed_dim: int = 128,
        atom_vocab_size: int = 4,  # H, C, N, O
        atom_embed_dim: int = 32,
    ):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.embed_dim = embed_dim
        
        # Coordinate projection
        self.coord_proj = nn.Linear(coordinate_dim, embed_dim - atom_embed_dim)
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(atom_vocab_size, atom_embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        coordinates: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embed coordinates and atom types.
        
        Parameters
        ----------
        coordinates : torch.Tensor
            Molecular coordinates of shape [B, N, 3]
        atom_types : torch.Tensor, optional
            Atom type indices of shape [B, N] or [1, N]
            
        Returns
        -------
        torch.Tensor
            Embedded coordinates of shape [B, N, embed_dim]
        """
        B, N, _ = coordinates.shape
        
        # Project coordinates
        coord_embed = self.coord_proj(coordinates)  # [B, N, embed_dim - atom_embed_dim]
        
        # Add atom type embeddings if provided
        if atom_types is not None:
            if atom_types.shape[0] == 1 and B > 1:
                # Broadcast atom types across batch
                atom_types = atom_types.expand(B, -1)
            
            atom_embed = self.atom_embedding(atom_types)  # [B, N, atom_embed_dim]
            
            # Concatenate coordinate and atom embeddings
            embedded = torch.cat([coord_embed, atom_embed], dim=-1)  # [B, N, embed_dim]
        else:
            # Pad with zeros if no atom types provided
            zero_pad = torch.zeros(B, N, self.embed_dim - coord_embed.shape[-1], 
                                 device=coordinates.device, dtype=coordinates.dtype)
            embedded = torch.cat([coord_embed, zero_pad], dim=-1)
        
        # Apply layer normalization
        return self.layer_norm(embedded)


class TemperatureConditioning(nn.Module):
    """Temperature pair conditioning for PT swaps.
    
    Embeds source and target temperatures as conditioning information
    for the autoregressive flow.
    """
    
    def __init__(self, embed_dim: int = 128, temp_embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.temp_embed_dim = temp_embed_dim
        
        # Temperature embedding (learnable)
        self.temp_proj = nn.Sequential(
            nn.Linear(2, temp_embed_dim),  # 2 for [source_temp, target_temp]
            nn.ReLU(),
            nn.Linear(temp_embed_dim, temp_embed_dim)
        )
        
        # Conditioning projection
        self.cond_proj = nn.Linear(temp_embed_dim, embed_dim)
    
    def forward(self, source_temp: float, target_temp: float, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create temperature conditioning.
        
        Parameters
        ----------
        source_temp : float
            Source temperature
        target_temp : float
            Target temperature  
        batch_size : int
            Batch size
        device : torch.device
            Device for tensors
            
        Returns
        -------
        torch.Tensor
            Temperature conditioning of shape [B, 1, embed_dim]
        """
        # Create temperature tensor
        # Handle both scalar and tensor inputs
        if torch.is_tensor(source_temp):
            if source_temp.numel() == 1:
                source_temp = source_temp.item()
            else:
                source_temp = source_temp[0].item()  # Take first element if batch
        if torch.is_tensor(target_temp):
            if target_temp.numel() == 1:
                target_temp = target_temp.item()
            else:
                target_temp = target_temp[0].item()  # Take first element if batch
                
        temp_tensor = torch.tensor(
            [[source_temp, target_temp]], 
            device=device, 
            dtype=torch.float32
        ).expand(batch_size, -1)  # [B, 2]
        
        # Embed temperatures
        temp_embed = self.temp_proj(temp_tensor)  # [B, temp_embed_dim]
        
        # Project to embedding dimension
        temp_cond = self.cond_proj(temp_embed)  # [B, embed_dim]
        
        # Add sequence dimension for broadcasting
        return temp_cond.unsqueeze(1)  # [B, 1, embed_dim]


def coordinates_to_patches(coordinates: torch.Tensor, patch_size: int = 1) -> torch.Tensor:
    """Convert coordinates to patch format for autoregressive processing.
    
    Simple implementation where each atom (or group of atoms) becomes a patch.
    
    Parameters
    ----------
    coordinates : torch.Tensor
        Coordinates of shape [B, N, 3]
    patch_size : int
        Number of atoms per patch
        
    Returns
    -------
    torch.Tensor
        Patches of shape [B, N//patch_size, patch_size*3]
    """
    B, N, C = coordinates.shape
    
    if patch_size == 1:
        # Each atom is a patch
        return coordinates  # [B, N, 3]
    else:
        # Group atoms into patches
        if N % patch_size != 0:
            # Pad if necessary
            pad_size = patch_size - (N % patch_size)
            padding = torch.zeros(B, pad_size, C, device=coordinates.device, dtype=coordinates.dtype)
            coordinates = torch.cat([coordinates, padding], dim=1)
            N = N + pad_size
        
        # Reshape to patches
        patches = coordinates.view(B, N // patch_size, patch_size * C)
        return patches


def patches_to_coordinates(patches: torch.Tensor, patch_size: int = 1, original_num_atoms: Optional[int] = None) -> torch.Tensor:
    """Convert patches back to coordinates.
    
    Parameters
    ----------
    patches : torch.Tensor
        Patches of shape [B, num_patches, patch_dim]
    patch_size : int
        Number of atoms per patch
    original_num_atoms : int, optional
        Original number of atoms (for padding removal)
        
    Returns
    -------
    torch.Tensor
        Coordinates of shape [B, N, 3]
    """
    B, num_patches, patch_dim = patches.shape
    
    if patch_size == 1:
        # Each patch is an atom
        return patches  # [B, N, 3]
    else:
        # Reshape patches back to coordinates
        C = 3  # coordinate dimensions
        coordinates = patches.view(B, num_patches * patch_size, C)
        
        # Remove padding if necessary
        if original_num_atoms is not None:
            coordinates = coordinates[:, :original_num_atoms, :]
        
        return coordinates
