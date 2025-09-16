#!/usr/bin/env python3
"""
Simplified and more stable autoregressive flow implementation
Based on MAF (Masked Autoregressive Flow) for 2D problems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedLinear(nn.Module):
    """Linear layer with autoregressive masking"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Create mask for autoregressive property
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask: torch.Tensor):
        """Set the autoregressive mask"""
        self.mask.data.copy_(mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


class AutoregressiveTransformer(nn.Module):
    """Simple autoregressive transformer for 2D data"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # For 2D: x[0] doesn't depend on anything, x[1] depends on x[0]
        # We'll use simple MLPs with proper masking
        
        # Network for dimension 0 (unconditional)
        self.net0 = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Just takes a constant input
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # outputs log_scale, shift
        )
        
        # Network for dimension 1 (conditional on dimension 0)
        self.net1 = nn.Sequential(
            nn.Linear(1, hidden_dim),  # takes x[0] as input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # outputs log_scale, shift
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch_size, 2)
        Returns:
            log_scale, shift: parameters for each dimension
        """
        batch_size = x.shape[0]
        
        # For dimension 0: no dependencies (use constant input)
        const_input = torch.ones(batch_size, 1, device=x.device)
        params0 = self.net0(const_input)  # (batch_size, 2)
        
        # For dimension 1: depends on dimension 0
        x0_input = x[:, 0:1]  # (batch_size, 1)
        params1 = self.net1(x0_input)  # (batch_size, 2)
        
        # Stack parameters
        log_scale = torch.stack([params0[:, 0], params1[:, 0]], dim=1)  # (batch_size, 2)
        shift = torch.stack([params0[:, 1], params1[:, 1]], dim=1)      # (batch_size, 2)
        
        return log_scale, shift


class SimpleAutoregressiveFlow(nn.Module):
    """Simple autoregressive flow layer"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.transformer = AutoregressiveTransformer(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation"""
        log_scale, shift = self.transformer(x)
        
        # Apply autoregressive masking: first dimension unchanged
        log_scale = log_scale * torch.tensor([0.0, 1.0], device=x.device)
        shift = shift * torch.tensor([0.0, 1.0], device=x.device)
        
        # Transform: z = x * exp(log_scale) + shift
        scale = torch.exp(log_scale)
        z = x * scale + shift
        
        # Log determinant
        log_det = log_scale.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation"""
        batch_size = z.shape[0]
        x = torch.zeros_like(z)
        
        # First dimension: no transformation
        x[:, 0] = z[:, 0]
        
        # Second dimension: inverse transformation
        log_scale, shift = self.transformer(x)
        log_scale_1 = log_scale[:, 1]
        shift_1 = shift[:, 1]
        
        scale_1 = torch.exp(log_scale_1)
        x[:, 1] = (z[:, 1] - shift_1) / scale_1
        
        # Log determinant (negative for inverse)
        log_det = -log_scale_1
        
        return x, log_det


class SimpleAutoregressiveFlowModel(nn.Module):
    """Complete autoregressive flow model with multiple layers"""
    
    def __init__(self, num_layers: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_layers = num_layers
        
        # Create flow layers
        self.flows = nn.ModuleList([
            SimpleAutoregressiveFlow(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Permutation layers (simple dimension swap for 2D)
        self.permute = lambda x: torch.stack([x[:, 1], x[:, 0]], dim=1)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all flows"""
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for i, flow in enumerate(self.flows):
            z, log_det = flow(z)
            log_det_total += log_det
            
            # Permute dimensions (except for last layer)
            if i < len(self.flows) - 1:
                z = self.permute(z)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass through all flows"""
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        # Apply flows in reverse order
        for i in reversed(range(len(self.flows))):
            # Reverse permutation first (except for last layer)
            if i < len(self.flows) - 1:
                x = self.permute(x)
            
            x, log_det = self.flows[i].inverse(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability"""
        z, log_det = self.forward(x)
        
        # Standard Gaussian log probability
        log_prob_z = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        
        return log_prob_z + log_det
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the model"""
        # Sample from standard Gaussian
        z = torch.randn(num_samples, 2, device=device)
        
        # Transform through inverse flow
        x, _ = self.inverse(z)
        
        return x


if __name__ == "__main__":
    # Test the simple autoregressive flow
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SimpleAutoregressiveFlowModel(num_layers=3, hidden_dim=64).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(10, 2).to(device)
    z, log_det = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Log det shape: {log_det.shape}")
    print(f"Log det values: {log_det}")
    
    # Test inverse
    x_recon, log_det_inv = model.inverse(z)
    print(f"Reconstruction error: {torch.mean((x - x_recon)**2):.6f}")
    
    # Test log probability
    log_prob = model.log_prob(x)
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Log prob values: {log_prob}")
    
    # Test sampling
    samples = model.sample(5, device)
    print(f"Samples shape: {samples.shape}")
    print(f"Sample values:\n{samples}")
    
    print("Simple autoregressive flow test completed!")
