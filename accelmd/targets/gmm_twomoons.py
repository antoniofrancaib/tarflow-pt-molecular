"""Gaussian Mixture Model and Two-Moons distributions for testing TARFlow.

Simple 2D distributions for validating autoregressive flow architecture.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import math

from . import register_target

__all__ = ["GaussianMixtureModel", "TwoMoonsDistribution"]


@register_target("gmm")
class GaussianMixtureModel:
    """8-component Gaussian Mixture Model in 2D.
    
    Source distribution for testing TARFlow coordinate transformations.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        n_components: int = 8,
        radius: float = 2.0,
        **kwargs
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.beta = 1.0 / temperature  # For compatibility with Boltzmann interface
        self.n_components = n_components
        
        # Define n_components Gaussian components in 2D
        # Arranged evenly around a circle
        angles = torch.linspace(0, 2 * math.pi, n_components + 1)[:-1]  # Evenly spaced angles (exclude 2π)
        
        self.means = torch.zeros(n_components, 2, device=self.device, dtype=torch.float32)
        for i in range(n_components):
            self.means[i, 0] = radius * torch.cos(angles[i])  # x coordinate
            self.means[i, 1] = radius * torch.sin(angles[i])  # y coordinate
        
        # Equal weights for all components
        self.weights = torch.ones(n_components, device=self.device) / n_components
        
        # Smaller covariance matrices (reduced variance)
        self.cov_scale = 0.15  # Much smaller variance for tighter clusters
        self.log_det_cov = 2 * math.log(self.cov_scale)  # log|Σ| for 2D identity * scale²
        
        # Dimension
        self.dim = 2
    
    def log_prob(self, coords: Tensor) -> Tensor:
        """Compute log probability of coordinates under GMM.
        
        Parameters
        ----------
        coords : Tensor
            Coordinates of shape [B, 2] or [B, 1, 2] or [B, N*2]
            
        Returns
        -------
        Tensor
            Log probabilities of shape [B]
        """
        # Handle different input shapes
        original_shape = coords.shape
        if coords.ndim == 3 and coords.shape[1] == 1:
            # [B, 1, 2] -> [B, 2]
            coords_2d = coords.squeeze(1)
        elif coords.ndim == 2 and coords.shape[1] == 2:
            # [B, 2] - already correct
            coords_2d = coords
        elif coords.ndim == 2 and coords.shape[1] == 2:
            # [B, 2] - flat coordinates, assume 2D
            coords_2d = coords
        else:
            raise ValueError(f"Expected coordinates of shape [B, 2] or [B, 1, 2], got {coords.shape}")
        
        coords_2d = coords_2d.to(self.device)
        B = coords_2d.shape[0]
        
        # Compute log probabilities for each component
        # coords_2d: [B, 2], means: [n_components, 2]
        diff = coords_2d.unsqueeze(1) - self.means.unsqueeze(0)  # [B, n_components, 2]
        
        # Mahalanobis distance for scaled identity covariance
        mahal_dist = torch.sum(diff**2, dim=-1) / self.cov_scale  # [B, n_components]
        
        # Log probability for each component (without normalization constant)
        log_prob_components = -0.5 * mahal_dist - 0.5 * self.log_det_cov - math.log(2 * math.pi)  # [B, n_components]
        
        # Add component weights
        log_weights = torch.log(self.weights).unsqueeze(0)  # [1, n_components]
        log_prob_weighted = log_prob_components + log_weights  # [B, n_components]
        
        # Log-sum-exp to get mixture probability
        log_prob_total = torch.logsumexp(log_prob_weighted, dim=1)  # [B]
        
        # Apply temperature scaling (for compatibility with Boltzmann interface)
        return log_prob_total / self.temperature
    
    def sample(self, n_samples: int) -> Tensor:
        """Sample from the GMM.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        Tensor
            Samples of shape [n_samples, 2]
        """
        # Sample component indices
        component_indices = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # Sample from selected components
        selected_means = self.means[component_indices]  # [n_samples, 2]
        
        # Add Gaussian noise
        noise = torch.randn(n_samples, 2, device=self.device) * math.sqrt(self.cov_scale)
        samples = selected_means + noise
        
        return samples
    
    __call__ = log_prob


@register_target("twomoons") 
class TwoMoonsDistribution:
    """Two-moons distribution in 2D.
    
    Target distribution for testing TARFlow coordinate transformations.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        noise_scale: float = 0.3,
        **kwargs
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.beta = 1.0 / temperature  # For compatibility
        self.noise_scale = noise_scale
        
        # Two-moons parameters
        self.radius = 1.0
        self.separation = 1.0
        
        # Dimension
        self.dim = 2
    
    def log_prob(self, coords: Tensor) -> Tensor:
        """Compute log probability of coordinates under two-moons.
        
        This is approximate - we use distance to nearest moon manifold.
        
        Parameters
        ----------
        coords : Tensor
            Coordinates of shape [B, 2] or [B, 1, 2]
            
        Returns
        -------
        Tensor
            Log probabilities of shape [B]
        """
        # Handle input shapes
        if coords.ndim == 3 and coords.shape[1] == 1:
            coords_2d = coords.squeeze(1)
        elif coords.ndim == 2 and coords.shape[1] == 2:
            coords_2d = coords
        else:
            raise ValueError(f"Expected coordinates of shape [B, 2] or [B, 1, 2], got {coords.shape}")
        
        coords_2d = coords_2d.to(self.device)
        
        # Two moon centers and orientations
        # Upper moon: centered at (0, 0.5*separation)
        # Lower moon: centered at (separation, -0.5*separation), flipped
        
        x, y = coords_2d[:, 0], coords_2d[:, 1]
        
        # Distance to upper moon manifold (semicircle)
        upper_center_x = 0.0
        upper_center_y = 0.5 * self.separation
        dist_to_upper_center = torch.sqrt((x - upper_center_x)**2 + (y - upper_center_y)**2)
        # Use squared distance instead of abs for smooth gradients
        dist_to_upper_manifold = (dist_to_upper_center - self.radius)**2
        
        # Distance to lower moon manifold (flipped semicircle)
        lower_center_x = self.separation
        lower_center_y = -0.5 * self.separation
        dist_to_lower_center = torch.sqrt((x - lower_center_x)**2 + (y - lower_center_y)**2)
        # Use squared distance instead of abs for smooth gradients
        dist_to_lower_manifold = (dist_to_lower_center - self.radius)**2
        
        # Use distance to closest manifold (no hard restrictions on regions)
        # This allows the model to learn the transformation more easily
        dist_upper = dist_to_upper_manifold
        dist_lower = dist_to_lower_manifold
        
        # Compute log probability for each manifold separately
        log_prob_upper = -0.5 * (dist_upper / self.noise_scale**2) - 0.5 * math.log(2 * math.pi * self.noise_scale**2)
        log_prob_lower = -0.5 * (dist_lower / self.noise_scale**2) - 0.5 * math.log(2 * math.pi * self.noise_scale**2)
        
        # Use smooth maximum that avoids the ridge problem
        # Smooth approximation: max(a,b) ≈ (a + b + |a - b|) / 2, but |a-b| isn't differentiable
        # Better: use smooth absolute value |x| ≈ sqrt(x² + ε²)
        eps = 1e-6
        diff = log_prob_upper - log_prob_lower
        smooth_abs_diff = torch.sqrt(diff**2 + eps)
        log_prob = (log_prob_upper + log_prob_lower + smooth_abs_diff) / 2
        
        # Apply temperature scaling
        return log_prob / self.temperature
    
    def sample(self, n_samples: int) -> Tensor:
        """Sample from two-moons distribution.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        Tensor
            Samples of shape [n_samples, 2]
        """
        # Sample which moon (50/50)
        moon_choice = torch.randint(0, 2, (n_samples,), device=self.device)
        
        # Sample angles for semicircles
        angles = torch.rand(n_samples, device=self.device) * math.pi  # [0, π]
        
        # Generate base moon coordinates
        x_base = self.radius * torch.cos(angles)
        y_base = self.radius * torch.sin(angles)
        
        # Apply moon-specific transformations
        x = torch.where(moon_choice == 0, 
                       x_base,  # Upper moon: normal orientation
                       self.separation - x_base)  # Lower moon: flipped + shifted
        
        y = torch.where(moon_choice == 0,
                       y_base + 0.5 * self.separation,  # Upper moon: shifted up
                       -y_base - 0.5 * self.separation)  # Lower moon: flipped + shifted down
        
        # Add noise
        noise = torch.randn(n_samples, 2, device=self.device) * self.noise_scale
        
        samples = torch.stack([x, y], dim=1) + noise
        return samples
    
    __call__ = log_prob
