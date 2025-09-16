#!/usr/bin/env python3
"""
Two Moons distribution for 2D normalizing flow training
"""

import torch
import numpy as np


class TwoMoons:
    """
    Two moons target distribution for 2D normalizing flows
    """
    
    def __init__(self):
        self.n_dims = 2
        
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of the two moons distribution
        
        Mathematical form:
        log(p) = -1/2 * ((norm(z) - 2) / 0.2)²
                 + log(exp(-1/2 * ((z[0] - 2) / 0.3)²) + exp(-1/2 * ((z[0] + 2) / 0.3)²))
        
        Args:
            z: input tensor of shape (batch_size, 2)
        Returns:
            log_prob: log probability of shape (batch_size,)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        x, y = z[:, 0], z[:, 1]
        
        # Radial component: points should be roughly at radius 2
        radius = torch.norm(z, dim=1)
        radial_term = -0.5 * ((radius - 2) / 0.2) ** 2
        
        # Angular component: two modes at x = ±2
        left_mode = torch.exp(-0.5 * ((x + 2) / 0.3) ** 2)
        right_mode = torch.exp(-0.5 * ((x - 2) / 0.3) ** 2)
        angular_term = torch.log(left_mode + right_mode + 1e-8)  # Small epsilon for numerical stability
        
        return radial_term + angular_term
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the two moons distribution using rejection sampling
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
        Returns:
            samples: tensor of shape (num_samples, 2)
        """
        samples = []
        max_log_prob = 0.0  # Approximate maximum log probability
        
        while len(samples) < num_samples:
            # Proposal samples from a broader distribution
            candidates = torch.randn(num_samples * 2, 2, device=device) * 3
            
            # Compute log probabilities
            log_probs = self.log_prob(candidates)
            
            # Rejection sampling
            uniform_samples = torch.rand(candidates.shape[0], device=device)
            accept_probs = torch.exp(log_probs - max_log_prob)
            accepted = uniform_samples < accept_probs
            
            samples.append(candidates[accepted])
            
            if len(samples) > 0:
                samples_tensor = torch.cat(samples, dim=0)
                if len(samples_tensor) >= num_samples:
                    break
        
        samples_tensor = torch.cat(samples, dim=0)
        return samples_tensor[:num_samples]
