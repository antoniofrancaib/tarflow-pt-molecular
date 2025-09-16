"""PT-specific TARFlow wrapper for temperature swap applications.

This module provides a wrapper around the core TransformerFlow that includes
the target distributions and PT-specific functionality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .transformer_flow import TransformerFlow
from ..targets import build_target

__all__ = ["PTTARFlow"]


class PTTARFlow(nn.Module):
    """TARFlow model for PT coordinate swaps.
    
    Wrapper around TransformerFlow that includes target distributions
    and PT-specific methods for swap acceptance computation.
    """
    
    def __init__(
        self,
        num_atoms: int,
        source_temperature: float,
        target_temperature: float,
        target_name: str = "dipeptide",
        target_kwargs: Optional[dict] = None,
        channels: int = 256,
        num_blocks: int = 4,
        layers_per_block: int = 2,
        nvp: bool = True,
        atom_vocab_size: int = 4,
        atom_embed_dim: int = 32,
        coordinate_dim: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.source_temperature = source_temperature
        self.target_temperature = target_temperature
        self.device = torch.device(device)
        
        # Build target distributions
        target_kwargs = target_kwargs or {}
        
        self.source_target = build_target(
            target_name,
            temperature=source_temperature,
            device="cpu",  # Keep targets on CPU for energy computation
            **target_kwargs
        )
        
        self.target_target = build_target(
            target_name,
            temperature=target_temperature,
            device="cpu",
            **target_kwargs
        )
        
        # Core transformer flow
        self.flow = TransformerFlow(
            coordinate_dim=coordinate_dim,
            num_atoms=num_atoms,
            channels=channels,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            nvp=nvp,
            temperature_conditioning=True,
            atom_vocab_size=atom_vocab_size,
            atom_embed_dim=atom_embed_dim,
        ).to(self.device)
        
        # Store target properties for compatibility
        self.base_low = self.source_target
        self.base_high = self.target_target

    def forward(
        self,
        coordinates: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
        reverse: bool = False,
        return_log_det: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward/reverse pass through the flow.
        
        Parameters
        ----------
        coordinates : torch.Tensor
            Input coordinates of shape [B, N, 3]
        atom_types : torch.Tensor, optional
            Atom type indices
        reverse : bool
            Whether to apply reverse transformation
        return_log_det : bool
            Whether to return log determinant
            
        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Transformed coordinates, optionally with log determinant
        """
        coordinates = coordinates.to(self.device)
        if atom_types is not None:
            atom_types = atom_types.to(self.device)
        
        if reverse:
            # Reverse transformation (target -> source)
            output = self.flow.reverse(
                coordinates,
                atom_types=atom_types,
                source_temp=self.target_temperature,
                target_temp=self.source_temperature,
            )
            
            if return_log_det:
                # For reverse, need to compute log det via forward pass
                _, log_det = self.flow.forward(
                    output,
                    atom_types=atom_types,
                    source_temp=self.target_temperature,
                    target_temp=self.source_temperature,
                )
                return output, -log_det  # Negative for reverse
            return output
        else:
            # Forward transformation (source -> target)
            output, log_det = self.flow.forward(
                coordinates,
                atom_types=atom_types,
                source_temp=self.source_temperature,
                target_temp=self.target_temperature,
            )
            
            if return_log_det:
                return output, log_det
            return output

    def transform(
        self,
        coordinates: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform coordinates and return log determinant.
        
        Compatibility method for existing PT swap code.
        """
        return self.forward(coordinates, reverse=reverse, return_log_det=True)

    def log_likelihood(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        reverse: bool = False,
        atom_types: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute log likelihood for coordinate transformation.
        
        Parameters
        ----------
        x_coords : torch.Tensor
            Source coordinates
        y_coords : torch.Tensor
            Target coordinates
        reverse : bool
            Direction of transformation
        atom_types : torch.Tensor, optional
            Atom type information
            
        Returns
        -------
        torch.Tensor
            Log likelihood values
        """
        x_coords = x_coords.to(self.device)
        y_coords = y_coords.to(self.device)
        
        if reverse:
            # Compute y -> x transformation
            x_pred, log_det = self.forward(y_coords, atom_types, reverse=True, return_log_det=True)
            
            # Log likelihood: log p(x|y) = log p_source(x) + log|J|
            x_flat = x_pred.view(x_pred.shape[0], -1).cpu()
            log_p_source = self.source_target.log_prob(x_flat)
            
            return log_p_source + log_det.cpu()
        else:
            # Compute x -> y transformation
            y_pred, log_det = self.forward(x_coords, atom_types, reverse=False, return_log_det=True)
            
            # Log likelihood: log p(y|x) = log p_target(y) + log|J|
            y_flat = y_pred.view(y_pred.shape[0], -1).cpu()
            log_p_target = self.target_target.log_prob(y_flat)
            
            return log_p_target + log_det.cpu()

    def sample(
        self,
        coordinates: torch.Tensor,
        atom_types: Optional[torch.Tensor] = None,
        guidance: float = 0.0,
        attn_temp: float = 1.0,
    ) -> torch.Tensor:
        """Sample from the flow with optional guidance.
        
        Parameters
        ----------
        coordinates : torch.Tensor
            Initial coordinates (noise or actual coordinates)
        atom_types : torch.Tensor, optional
            Atom type information
        guidance : float
            Guidance strength for conditional sampling
        attn_temp : float
            Attention temperature
            
        Returns
        -------
        torch.Tensor
            Sampled coordinates
        """
        coordinates = coordinates.to(self.device)
        if atom_types is not None:
            atom_types = atom_types.to(self.device)
        
        return self.flow.reverse(
            coordinates,
            atom_types=atom_types,
            source_temp=self.source_temperature,
            target_temp=self.target_temperature,
            guidance=guidance,
            attn_temp=attn_temp,
        )
