#!/usr/bin/env python3
"""
Base distributions for normalizing flows
"""

import torch
import torch.nn as nn
import numpy as np


class DiagonalGaussian:
    """
    Diagonal Gaussian base distribution
    """
    
    def __init__(self, dim: int, trainable: bool = False):
        self.dim = dim
        self.trainable = trainable
        
        if trainable:
            self.loc = nn.Parameter(torch.zeros(dim))
            self.log_scale = nn.Parameter(torch.zeros(dim))
        else:
            self.loc = torch.zeros(dim)
            self.log_scale = torch.zeros(dim)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under diagonal Gaussian
        
        Args:
            z: input tensor of shape (batch_size, dim)
        Returns:
            log_prob: log probability of shape (batch_size,)
        """
        if isinstance(self.loc, torch.Tensor):
            loc = self.loc.to(z.device)
            log_scale = self.log_scale.to(z.device)
        else:
            loc = self.loc
            log_scale = self.log_scale
            
        # Standard Gaussian case
        if torch.allclose(log_scale, torch.zeros_like(log_scale)) and torch.allclose(loc, torch.zeros_like(loc)):
            return -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        
        # General diagonal Gaussian
        scale = torch.exp(log_scale)
        normalized = (z - loc) / scale
        log_prob = -0.5 * normalized**2 - log_scale - 0.5 * np.log(2 * np.pi)
        return log_prob.sum(dim=1)
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from diagonal Gaussian
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
        Returns:
            samples: tensor of shape (num_samples, dim)
        """
        eps = torch.randn(num_samples, self.dim, device=device)
        
        if isinstance(self.loc, torch.Tensor):
            loc = self.loc.to(device)
            scale = torch.exp(self.log_scale.to(device))
        else:
            loc = torch.zeros(self.dim, device=device)
            scale = torch.ones(self.dim, device=device)
            
        return loc + scale * eps
