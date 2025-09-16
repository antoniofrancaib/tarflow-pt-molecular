"""Simple dataset for distribution mapping experiments.

Dataset for training TARFlow to map between GMM and Two-Moons distributions.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

from ..targets import build_target

__all__ = ["SimpleDistributionDataset"]


class SimpleDistributionDataset(Dataset):
    """Dataset for training distribution mapping with TARFlow.
    
    Generates paired samples from source and target distributions for
    supervised learning of coordinate transformations.
    """
    
    def __init__(
        self,
        source_target_name: str = "gmm",
        target_target_name: str = "twomoons", 
        n_samples: int = 10000,
        source_temperature: float = 1.0,
        target_temperature: float = 1.0,
        device: str = "cpu",
        source_kwargs: Optional[Dict] = None,
        target_kwargs: Optional[Dict] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_samples = n_samples
        self.device = torch.device(device)
        self.source_temperature = source_temperature
        self.target_temperature = target_temperature
        
        # Set random seed for reproducible datasets
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Build source and target distributions
        source_kwargs = source_kwargs or {}
        target_kwargs = target_kwargs or {}
        
        self.source_dist = build_target(
            source_target_name,
            temperature=source_temperature,
            device=device,
            **source_kwargs
        )
        
        self.target_dist = build_target(
            target_target_name,
            temperature=target_temperature,
            device=device,
            **target_kwargs
        )
        
        # Pre-generate all samples for consistent training
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate all coordinate pairs for the dataset."""
        # Generate source samples
        self.source_coords = self.source_dist.sample(self.n_samples)  # [N, 2]
        
        # Generate target samples  
        self.target_coords = self.target_dist.sample(self.n_samples)  # [N, 2]
        
        # Reshape to [N, 1, 2] format expected by PTTARFlow (num_atoms=1)
        self.source_coords = self.source_coords.unsqueeze(1)  # [N, 1, 2]
        self.target_coords = self.target_coords.unsqueeze(1)  # [N, 1, 2]
        
        print(f"Generated {self.n_samples} samples:")
        print(f"  Source shape: {self.source_coords.shape}")
        print(f"  Target shape: {self.target_coords.shape}")
        print(f"  Source range: [{self.source_coords.min():.2f}, {self.source_coords.max():.2f}]")
        print(f"  Target range: [{self.target_coords.min():.2f}, {self.target_coords.max():.2f}]")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample pair compatible with existing training infrastructure.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with same format as PTTemperaturePairDataset for compatibility
        """
        return {
            "source_coords": self.source_coords[idx],  # [1, 2]
            "target_coords": self.target_coords[idx],  # [1, 2]
            "source_temp": torch.tensor(self.source_temperature, dtype=torch.float32),
            "target_temp": torch.tensor(self.target_temperature, dtype=torch.float32),
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader compatibility."""
        # Stack samples into batches
        source_coords = torch.stack([item["source_coords"] for item in batch])  # [B, 1, 2]
        target_coords = torch.stack([item["target_coords"] for item in batch])  # [B, 1, 2]
        source_temp = torch.stack([item["source_temp"] for item in batch])  # [B]
        target_temp = torch.stack([item["target_temp"] for item in batch])  # [B]
        
        return {
            "source_coords": source_coords,
            "target_coords": target_coords,
            "source_temp": source_temp,
            "target_temp": target_temp,
        }
    
    def visualize_samples(self, n_viz: int = 1000):
        """Visualize source and target distributions.
        
        Parameters
        ----------
        n_viz : int
            Number of samples to plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        n_viz = min(n_viz, len(self))
        
        # Get samples for visualization
        source_viz = self.source_coords[:n_viz, 0, :].numpy()  # [n_viz, 2]
        target_viz = self.target_coords[:n_viz, 0, :].numpy()  # [n_viz, 2]
        
        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Source distribution
        ax1.scatter(source_viz[:, 0], source_viz[:, 1], alpha=0.6, s=1)
        ax1.set_title("Source Distribution (GMM)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Target distribution  
        ax2.scatter(target_viz[:, 0], target_viz[:, 1], alpha=0.6, s=1)
        ax2.set_title("Target Distribution (Two-Moons)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def get_temperature_info(self):
        """Return temperature information for compatibility with PT interface."""
        return {
            "source_temperature": getattr(self.source_dist, 'temperature', 1.0),
            "target_temperature": getattr(self.target_dist, 'temperature', 1.0),
            "source_beta": getattr(self.source_dist, 'beta', 1.0),
            "target_beta": getattr(self.target_dist, 'beta', 1.0),
        }
