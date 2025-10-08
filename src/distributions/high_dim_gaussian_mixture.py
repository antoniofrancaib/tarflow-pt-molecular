#!/usr/bin/env python3
"""
High-dimensional Gaussian mixture distributions for testing autoregressive flows
"""

import torch
import numpy as np
from typing import List, Optional


class HighDimGaussianMixture:
    """
    High-dimensional Gaussian mixture with configurable structure
    Useful for testing autoregressive flows in higher dimensions
    """
    
    def __init__(self, 
                 dim: int,
                 num_components: int = 4,
                 separation: float = 3.0,
                 component_std: float = 0.5,
                 correlation_strength: float = 0.0,
                 random_seed: int = 42):
        """
        Args:
            dim: Dimensionality of the space
            num_components: Number of mixture components
            separation: Distance between component centers
            component_std: Standard deviation of each component
            correlation_strength: [0,1] - how correlated dimensions are within components
            random_seed: For reproducible distributions
        """
        self.dim = dim
        self.num_components = num_components
        self.separation = separation
        self.component_std = component_std
        self.correlation_strength = correlation_strength
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Generate component centers with good separation
        self.centers = self._generate_centers()
        
        # Generate covariance matrices
        self.covariances = self._generate_covariances()
        
        # Uniform mixture weights
        self.weights = torch.ones(num_components) / num_components
        
        print(f"Created {num_components}-component Gaussian mixture in {dim}D")
        print(f"Center separation: {separation:.2f}, Component std: {component_std:.2f}")
        print(f"Correlation strength: {correlation_strength:.2f}")
    
    def _generate_centers(self) -> torch.Tensor:
        """Generate well-separated component centers"""
        if self.num_components == 1:
            return torch.zeros(1, self.dim)
        
        # Place centers on a hypersphere for good separation
        centers = torch.randn(self.num_components, self.dim)
        centers = centers / torch.norm(centers, dim=1, keepdim=True)
        centers = centers * self.separation
        
        # Add some random offset to break perfect symmetry
        centers += torch.randn_like(centers) * 0.1
        
        return centers
    
    def _generate_covariances(self) -> torch.Tensor:
        """Generate covariance matrices with controllable correlation"""
        covariances = torch.zeros(self.num_components, self.dim, self.dim)
        
        for i in range(self.num_components):
            if self.correlation_strength == 0.0:
                # Diagonal covariance (independent dimensions)
                cov = torch.eye(self.dim) * (self.component_std ** 2)
            else:
                # Generate random correlation structure
                # Start with random matrix
                A = torch.randn(self.dim, self.dim) * 0.5
                # Make it symmetric positive definite
                cov = A @ A.T
                # Scale to desired variance
                cov = cov / torch.diag(cov).mean() * (self.component_std ** 2)
                # Mix with diagonal based on correlation strength
                diag_cov = torch.eye(self.dim) * (self.component_std ** 2)
                cov = (1 - self.correlation_strength) * diag_cov + self.correlation_strength * cov
            
            covariances[i] = cov
        
        return covariances
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the mixture
        
        Args:
            x: input tensor of shape (batch_size, dim)
        Returns:
            log_prob: log probability of shape (batch_size,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        log_probs = torch.zeros(batch_size, self.num_components)
        
        for k in range(self.num_components):
            # Compute log probability for component k
            center = self.centers[k]
            cov = self.covariances[k]
            
            diff = x - center
            
            # Use Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(cov)
                log_det = 2 * torch.sum(torch.log(torch.diag(L)))
                
                # Solve L @ y = diff.T for y, then compute ||y||^2
                y = torch.linalg.solve_triangular(L, diff.T, upper=False)
                mahalanobis = torch.sum(y ** 2, dim=0)
                
            except torch.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                U, S, V = torch.svd(cov)
                log_det = torch.sum(torch.log(S))
                inv_cov = (U * (1.0 / S).unsqueeze(0)) @ U.T
                mahalanobis = torch.sum((diff @ inv_cov) * diff, dim=1)
            
            # Gaussian log probability
            log_prob_k = -0.5 * (mahalanobis + log_det + self.dim * np.log(2 * np.pi))
            log_probs[:, k] = log_prob_k + torch.log(self.weights[k])
        
        # Log-sum-exp for mixture
        return torch.logsumexp(log_probs, dim=1)
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the mixture distribution
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
        Returns:
            samples: tensor of shape (num_samples, dim)
        """
        # Choose components
        component_indices = torch.multinomial(
            self.weights, num_samples, replacement=True
        ).to(device)
        
        samples = torch.zeros(num_samples, self.dim, device=device)
        
        for k in range(self.num_components):
            mask = component_indices == k
            n_k = mask.sum().item()
            
            if n_k > 0:
                # Sample from component k
                center = self.centers[k].to(device)
                cov = self.covariances[k].to(device)
                
                # Use Cholesky decomposition for sampling
                try:
                    L = torch.linalg.cholesky(cov)
                    noise = torch.randn(n_k, self.dim, device=device)
                    component_samples = center + (L @ noise.T).T
                except torch.linalg.LinAlgError:
                    # Fallback to eigenvalue decomposition
                    eigenvals, eigenvecs = torch.linalg.eigh(cov)
                    eigenvals = torch.clamp(eigenvals, min=1e-8)  # Ensure positive
                    sqrt_cov = eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.T
                    noise = torch.randn(n_k, self.dim, device=device)
                    component_samples = center + (sqrt_cov @ noise.T).T
                
                samples[mask] = component_samples
        
        return samples
    
    def get_statistics(self) -> dict:
        """Get distribution statistics for analysis"""
        return {
            'dim': self.dim,
            'num_components': self.num_components,
            'centers': self.centers,
            'center_distances': torch.pdist(self.centers),
            'component_volumes': torch.det(self.covariances),
            'total_variance': torch.trace(self.covariances.mean(dim=0)),
            'condition_numbers': torch.linalg.cond(self.covariances)
        }


class HighDimSwissRoll:
    """
    High-dimensional Swiss roll manifold
    Tests flow's ability to learn curved manifolds in high-D
    """
    
    def __init__(self, 
                 dim: int,
                 intrinsic_dim: int = 2,
                 noise_level: float = 0.1,
                 roll_params: dict = None):
        """
        Args:
            dim: Ambient dimension
            intrinsic_dim: Intrinsic dimensionality of the manifold
            noise_level: Gaussian noise added to the manifold
            roll_params: Parameters for the Swiss roll (length, width, etc.)
        """
        self.dim = dim
        self.intrinsic_dim = intrinsic_dim
        self.noise_level = noise_level
        
        if roll_params is None:
            roll_params = {'length': 4.0, 'width': 2.0, 'turns': 1.5}
        self.roll_params = roll_params
        
        print(f"Created Swiss roll: {intrinsic_dim}D → {dim}D, noise: {noise_level:.3f}")
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from the Swiss roll manifold"""
        # Sample in intrinsic coordinates
        t = torch.rand(num_samples, device=device) * self.roll_params['length']
        s = (torch.rand(num_samples, device=device) - 0.5) * self.roll_params['width']
        
        # Swiss roll transformation
        x1 = t * torch.cos(t * self.roll_params['turns'])
        x2 = t * torch.sin(t * self.roll_params['turns'])
        x3 = s
        
        # Embed in higher dimensions
        samples = torch.zeros(num_samples, self.dim, device=device)
        samples[:, 0] = x1
        samples[:, 1] = x2
        if self.dim > 2:
            samples[:, 2] = x3
        
        # Add noise to all dimensions
        noise = torch.randn_like(samples) * self.noise_level
        samples = samples + noise
        
        return samples
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate log probability (not exact for manifolds)"""
        # For manifolds, exact density is complex. Use approximation.
        # Distance to the manifold as a proxy
        batch_size = x.shape[0]
        
        # Project back to intrinsic coordinates (approximate)
        x1, x2 = x[:, 0], x[:, 1]
        t_est = torch.sqrt(x1**2 + x2**2)
        
        # Compute distance from manifold
        x1_manifold = t_est * torch.cos(t_est * self.roll_params['turns'])
        x2_manifold = t_est * torch.sin(t_est * self.roll_params['turns'])
        
        manifold_dist = torch.sqrt((x1 - x1_manifold)**2 + (x2 - x2_manifold)**2)
        
        # Add other dimensions
        if self.dim > 3:
            off_manifold_norm = torch.norm(x[:, 3:], dim=1)
            manifold_dist = manifold_dist + off_manifold_norm
        
        # Gaussian density around manifold
        log_prob = -0.5 * (manifold_dist / self.noise_level)**2 - np.log(self.noise_level * np.sqrt(2 * np.pi))
        
        return log_prob


if __name__ == "__main__":
    # Test the high-dimensional distributions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing High-Dimensional Gaussian Mixture...")
    
    # Test 10D mixture
    mixture_10d = HighDimGaussianMixture(
        dim=10, 
        num_components=4, 
        separation=3.0,
        correlation_strength=0.3
    )
    
    samples = mixture_10d.sample(1000, device)
    log_probs = mixture_10d.log_prob(samples)
    
    print(f"10D Mixture - Samples shape: {samples.shape}")
    print(f"Log prob range: [{log_probs.min():.2f}, {log_probs.max():.2f}]")
    print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")
    
    stats = mixture_10d.get_statistics()
    print(f"Center distances: {stats['center_distances'].mean():.2f} ± {stats['center_distances'].std():.2f}")
    
    print("\nTesting Swiss Roll...")
    swiss_roll = HighDimSwissRoll(dim=20, intrinsic_dim=2, noise_level=0.1)
    roll_samples = swiss_roll.sample(1000, device)
    print(f"Swiss roll samples shape: {roll_samples.shape}")
    print(f"First 3 dims statistics: {roll_samples[:, :3].std(dim=0)}")
    
    print("\nHigh-dimensional distributions created successfully!")
