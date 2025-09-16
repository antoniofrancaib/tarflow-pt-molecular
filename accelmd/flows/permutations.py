"""Permutation strategies for autoregressive coordinate processing.

Adapted from Apple's TARFlow for molecular coordinate sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Permutation", "PermutationIdentity", "PermutationFlip", "PermutationRandom"]


class Permutation(nn.Module):
    """Base class for coordinate permutation strategies."""

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        """Apply permutation to tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to permute
        dim : int
            Dimension along which to permute
        inverse : bool
            Whether to apply inverse permutation
            
        Returns
        -------
        torch.Tensor
            Permuted tensor
        """
        raise NotImplementedError('Implement in subclass')


class PermutationIdentity(Permutation):
    """Identity permutation (no change)."""
    
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    """Flip the sequence order."""
    
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class PermutationRandom(Permutation):
    """Fixed random permutation for molecular coordinates.
    
    Useful for breaking spatial locality assumptions in molecular sequences.
    """
    
    def __init__(self, seq_length: int, seed: int = 42):
        super().__init__(seq_length)
        # Create fixed random permutation
        torch.manual_seed(seed)
        self.register_buffer('perm_indices', torch.randperm(seq_length))
        self.register_buffer('inv_perm_indices', torch.argsort(self.perm_indices))
    
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        if inverse:
            indices = self.inv_perm_indices
        else:
            indices = self.perm_indices
        
        return torch.index_select(x, dim, indices)
