#!/usr/bin/env python3
"""
Checkerboard target distribution for normalizing flow training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class CheckerboardDistribution:
    """
    8x8 Checkerboard distribution with alternating high/low density regions
    Creates a grid pattern with sharp boundaries between regions
    """
    
    def __init__(self, 
                 grid_size: int = 8,
                 domain_size: float = 4.0,
                 high_density: float = 1.0,
                 low_density: float = 0.1):
        """
        Args:
            grid_size: Number of squares per side (8x8 = 64 squares total)
            domain_size: Size of the domain [-domain_size, domain_size]
            high_density: Relative density of "white" squares
            low_density: Relative density of "black" squares
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.high_density = high_density
        self.low_density = low_density
        self.square_size = (2 * domain_size) / grid_size
        
        # Compute normalization constant
        total_area = (2 * domain_size) ** 2
        high_squares = (grid_size * grid_size) // 2
        low_squares = (grid_size * grid_size) - high_squares
        
        unnormalized_integral = (high_squares * high_density + low_squares * low_density) * (self.square_size ** 2)
        self.normalization = 1.0 / unnormalized_integral
        
        print(f"Checkerboard: {grid_size}x{grid_size} grid, domain ±{domain_size}")
        print(f"High density regions: {high_squares}, Low density regions: {low_squares}")
        print(f"Normalization constant: {self.normalization:.4f}")
    
    def _get_square_indices(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get grid square indices for points"""
        # Map from [-domain_size, domain_size] to [0, grid_size-1]
        i = torch.floor((x + self.domain_size) / self.square_size).long()
        j = torch.floor((y + self.domain_size) / self.square_size).long()
        
        # Clamp to valid range
        i = torch.clamp(i, 0, self.grid_size - 1)
        j = torch.clamp(j, 0, self.grid_size - 1)
        
        return i, j
    
    def _is_high_density_square(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Check if square (i,j) is a high density square (white in checkerboard)"""
        return (i + j) % 2 == 0
    
    def _is_in_domain(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Check if points are within the domain"""
        return (torch.abs(x) <= self.domain_size) & (torch.abs(y) <= self.domain_size)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of the checkerboard distribution
        
        Args:
            z: input tensor of shape (batch_size, 2)
        Returns:
            log_prob: log probability of shape (batch_size,)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        x, y = z[:, 0], z[:, 1]
        
        # Check if points are in domain
        in_domain = self._is_in_domain(x, y)
        
        # Get square indices
        i, j = self._get_square_indices(x, y)
        
        # Check if in high density square
        is_high = self._is_high_density_square(i, j)
        
        # Assign densities
        density = torch.where(is_high, self.high_density, self.low_density)
        density = torch.where(in_domain, density, 1e-10)  # Very low density outside domain
        
        # Apply normalization and return log probability
        normalized_density = density * self.normalization
        log_prob = torch.log(normalized_density + 1e-10)  # Add small epsilon for numerical stability
        
        return log_prob
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the checkerboard distribution using rejection sampling
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
        Returns:
            samples: tensor of shape (num_samples, 2)
        """
        samples = []
        max_density = self.high_density * self.normalization
        
        # Generate more samples than needed for rejection sampling
        while len(samples) * (samples[0].shape[0] if samples else 1) < num_samples:
            # Proposal samples uniformly in domain
            n_proposals = num_samples * 3  # Generate extra for rejection
            
            x_proposals = torch.rand(n_proposals, device=device) * (2 * self.domain_size) - self.domain_size
            y_proposals = torch.rand(n_proposals, device=device) * (2 * self.domain_size) - self.domain_size
            proposals = torch.stack([x_proposals, y_proposals], dim=1)
            
            # Compute densities
            log_probs = self.log_prob(proposals)
            probs = torch.exp(log_probs)
            
            # Rejection sampling
            uniform_samples = torch.rand(n_proposals, device=device)
            accept_probs = probs / max_density
            accepted = uniform_samples < accept_probs
            
            if accepted.any():
                samples.append(proposals[accepted])
        
        # Concatenate and trim to exact number requested
        all_samples = torch.cat(samples, dim=0)
        return all_samples[:num_samples]
    
    def visualize_density(self, resolution: int = 200, save_path: str = None):
        """Visualize the checkerboard density"""
        x = torch.linspace(-self.domain_size, self.domain_size, resolution)
        y = torch.linspace(-self.domain_size, self.domain_size, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        log_probs = self.log_prob(grid_points)
        probs = torch.exp(log_probs).view(xx.shape)
        
        plt.figure(figsize=(10, 10))
        plt.pcolormesh(xx.numpy(), yy.numpy(), probs.numpy(), cmap='RdBu_r', shading='auto')
        plt.colorbar(label='Density')
        plt.title(f'{self.grid_size}×{self.grid_size} Checkerboard Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        
        # Add grid lines to show squares
        for i in range(self.grid_size + 1):
            line_pos = -self.domain_size + i * self.square_size
            plt.axhline(y=line_pos, color='black', linewidth=0.5, alpha=0.3)
            plt.axvline(x=line_pos, color='black', linewidth=0.5, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Test the checkerboard distribution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create checkerboard
    checkerboard = CheckerboardDistribution(grid_size=8, domain_size=3.0)
    
    # Test sampling
    print("Testing sampling...")
    samples = checkerboard.sample(1000, device)
    print(f"Samples shape: {samples.shape}")
    print(f"Sample range X: [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]")
    print(f"Sample range Y: [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
    
    # Test log probability
    print("\nTesting log probability...")
    log_probs = checkerboard.log_prob(samples)
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Mean log prob: {log_probs.mean():.4f}")
    print(f"Log prob range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
    
    # Visualize density
    print("\nVisualizing density...")
    checkerboard.visualize_density()
    
    # Test specific points
    print("\nTesting specific points...")
    test_points = torch.tensor([
        [0.0, 0.0],    # Center
        [1.0, 1.0],    # Should be same color as center
        [1.0, 0.0],    # Should be different color
        [3.0, 3.0],    # Edge
        [4.0, 4.0],    # Outside domain
    ], device=device)
    
    test_log_probs = checkerboard.log_prob(test_points)
    for i, (point, log_prob) in enumerate(zip(test_points, test_log_probs)):
        print(f"Point {point.tolist()}: log_prob = {log_prob:.4f}")
    
    print("Checkerboard distribution test completed!")
