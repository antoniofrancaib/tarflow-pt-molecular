"""Molecular Parallel Tempering dataset for cross-temperature transport.

Loads alanine dipeptide trajectories at different temperatures and provides
paired samples for training normalizing flows to learn temperature-to-temperature
transformations.
"""

import os
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


class MolecularPTDataset(Dataset):
    """Dataset for molecular cross-temperature transport.
    
    Loads pre-equilibrated molecular configurations from parallel tempering
    simulations and provides paired samples for training flows.
    
    Attributes:
        source_coords: Configurations at source temperature, shape [N, 69]
        target_coords: Configurations at target temperature, shape [N, 69]
        source_temp: Source temperature in Kelvin
        target_temp: Target temperature in Kelvin
        normalization_stats: Dict with 'mean' and 'std' for denormalization
    """
    
    def __init__(
        self,
        data_path: str = "datasets/AA/pt_AA.pt",
        source_temp_idx: int = 0,
        target_temp_idx: int = 1,
        normalize: bool = True,
        normalize_mode: str = "per_atom",
    ):
        """Initialize molecular PT dataset.
        
        Args:
            data_path: Path to PT trajectory file [n_temps, n_replicas, n_steps, 69]
            source_temp_idx: Index of source temperature (0=300K, 1=450K, 2=670K, 3=1000K)
            target_temp_idx: Index of target temperature
            normalize: Whether to normalize coordinates
            normalize_mode: 'per_atom' (69 independent) or 'global' (single mean/std)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"PT trajectory not found: {data_path}")
        
        # Load trajectory: [n_temps, n_replicas, n_steps, 69]
        data = torch.load(data_path, weights_only=False)
        
        if data.ndim != 4 or data.shape[-1] != 69:
            raise ValueError(f"Expected shape [n_temps, n_replicas, n_steps, 69], got {data.shape}")
        
        # Extract source and target configurations
        # Shape: [n_replicas, n_steps, 69] → flatten to [N, 69]
        source_data = data[source_temp_idx].reshape(-1, 69)
        target_data = data[target_temp_idx].reshape(-1, 69)
        
        self.source_coords = source_data
        self.target_coords = target_data
        
        # Temperature mapping (assumes default ladder)
        temp_ladder = [300.0, 450.0, 670.0, 1000.0]
        self.source_temp = temp_ladder[source_temp_idx]
        self.target_temp = temp_ladder[target_temp_idx]
        
        self.normalize_mode = normalize_mode
        self.normalization_stats = {}
        
        # Apply normalization
        if normalize:
            self._normalize_coordinates()
        
        print(f"Loaded MolecularPTDataset: {self.source_temp}K → {self.target_temp}K")
        print(f"  Source samples: {len(self.source_coords)}")
        print(f"  Target samples: {len(self.target_coords)}")
        print(f"  Normalization: {normalize_mode if normalize else 'none'}")
    
    def _normalize_coordinates(self):
        """Apply per-atom or global normalization to coordinates."""
        if self.normalize_mode == "per_atom":
            # Compute statistics per coordinate (69 independent normalizations)
            # Use source distribution statistics (transport FROM source)
            mean = self.source_coords.mean(dim=0, keepdim=True)  # [1, 69]
            std = self.source_coords.std(dim=0, keepdim=True)    # [1, 69]
            
            # Avoid division by zero (shouldn't happen for molecular coords)
            std = torch.clamp(std, min=1e-8)
            
        elif self.normalize_mode == "global":
            # Single mean/std across all coordinates
            mean = self.source_coords.mean()
            std = self.source_coords.std()
            std = max(std.item(), 1e-8)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        
        # Store for denormalization
        self.normalization_stats = {
            "mean": mean,
            "std": std,
            "mode": self.normalize_mode,
        }
        
        # Normalize both source and target with SOURCE statistics
        self.source_coords = (self.source_coords - mean) / std
        self.target_coords = (self.target_coords - mean) / std
    
    def denormalize(self, coords: Tensor) -> Tensor:
        """Convert normalized coordinates back to nanometers.
        
        Args:
            coords: Normalized coordinates, shape [..., 69]
            
        Returns:
            denormalized: Coordinates in nm, same shape as input
        """
        if not self.normalization_stats:
            return coords
        
        mean = self.normalization_stats["mean"].to(coords.device)
        std = self.normalization_stats["std"].to(coords.device)
        
        return coords * std + mean
    
    def get_betas(self) -> Tuple[float, float]:
        """Get inverse temperatures β = 1/(kB·T) in mol/kJ.
        
        Returns:
            (beta_source, beta_target) tuple
        """
        k_b = 8.314462618e-3  # kJ/(mol·K)
        beta_source = 1.0 / (k_b * self.source_temp)
        beta_target = 1.0 / (k_b * self.target_temp)
        return beta_source, beta_target
    
    def __len__(self) -> int:
        """Return number of samples (minimum of source/target)."""
        return min(len(self.source_coords), len(self.target_coords))
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get paired source and target configurations.
        
        Args:
            idx: Sample index
            
        Returns:
            (source_coord, target_coord) tuple, each shape [69]
        """
        return self.source_coords[idx], self.target_coords[idx]
    
    def get_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample a random batch from both distributions.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            (source_batch, target_batch) tuple, each shape [batch_size, 69]
        """
        indices = torch.randint(0, len(self), (batch_size,))
        source_batch = self.source_coords[indices]
        target_batch = self.target_coords[indices]
        return source_batch, target_batch

