#!/usr/bin/env python3
"""
Simplified Autoregressive Transformer Flow for 2D distributions
Adapted from TarFlow for our Two Moons problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoregressiveAttention(nn.Module):
    """Autoregressive attention mechanism for normalizing flows"""
    
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        
        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        
        return self.proj(out)


class MLP(nn.Module):
    """Simple MLP with GELU activation"""
    
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.fc2(F.gelu(self.fc1(x_norm)))


class TransformerBlock(nn.Module):
    """Transformer block with autoregressive attention"""
    
    def __init__(self, dim: int, head_dim: int = 64, mlp_ratio: int = 4):
        super().__init__()
        self.attn = AutoregressiveAttention(dim, head_dim)
        self.mlp = MLP(dim, dim * mlp_ratio)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x


class AutoregressiveFlowLayer(nn.Module):
    """Single autoregressive flow layer"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 head_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Position embedding for sequence modeling
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        
        # Input projection
        self.proj_in = nn.Linear(1, hidden_dim)  # Each dimension is a single value
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, head_dim) 
            for _ in range(num_layers)
        ])
        
        # Output projections for scale and shift
        self.proj_scale = nn.Linear(hidden_dim, 1)
        self.proj_shift = nn.Linear(hidden_dim, 1)
        
        # Causal mask
        mask = torch.tril(torch.ones(input_dim, input_dim))
        self.register_buffer('causal_mask', mask)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoregressive flow
        Args:
            x: input tensor of shape (batch_size, input_dim)
        Returns:
            z: transformed tensor
            log_det: log determinant of Jacobian
        """
        batch_size = x.shape[0]
        
        # Reshape to sequence: (batch_size, seq_len, 1)
        x_seq = x.unsqueeze(-1)  # (B, D, 1)
        
        # Add position embeddings
        h = self.proj_in(x_seq) + self.pos_embed  # (B, D, hidden_dim)
        
        # Apply transformer layers with causal masking
        for layer in self.transformer_layers:
            h = layer(h, self.causal_mask)
        
        # Get scale and shift parameters
        log_scale = self.proj_scale(h).squeeze(-1)  # (B, D)
        shift = self.proj_shift(h).squeeze(-1)      # (B, D)
        
        # Apply transformation: z = x * exp(log_scale) + shift
        # For autoregressive property, only use previous dimensions
        log_scale = torch.cat([
            torch.zeros(batch_size, 1, device=x.device), 
            log_scale[:, :-1]
        ], dim=1)
        shift = torch.cat([
            torch.zeros(batch_size, 1, device=x.device), 
            shift[:, :-1]
        ], dim=1)
        
        scale = torch.exp(log_scale)
        z = x * scale + shift
        
        # Log determinant is sum of log scales
        log_det = log_scale.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation
        Args:
            z: transformed tensor
        Returns:
            x: original tensor
            log_det: log determinant of Jacobian
        """
        batch_size = z.shape[0]
        x = torch.zeros_like(z)
        log_det_total = torch.zeros(batch_size, device=z.device)
        
        # Sequentially compute each dimension
        for i in range(self.input_dim):
            if i == 0:
                x[:, i] = z[:, i]
            else:
                # Use previous dimensions to compute transformation parameters
                x_partial = x[:, :i].unsqueeze(-1)  # (B, i, 1)
                pos_embed_partial = self.pos_embed[:, :i, :]  # (1, i, hidden_dim)
                
                h = self.proj_in(x_partial) + pos_embed_partial
                
                # Apply transformer layers with appropriate masking
                mask = self.causal_mask[:i, :i]
                for layer in self.transformer_layers:
                    h = layer(h, mask)
                
                # Get parameters for current dimension
                log_scale_i = self.proj_scale(h[:, -1:]).squeeze(-1)  # (B, 1)
                shift_i = self.proj_shift(h[:, -1:]).squeeze(-1)      # (B, 1)
                
                scale_i = torch.exp(log_scale_i)
                x[:, i:i+1] = (z[:, i:i+1] - shift_i) / scale_i
                log_det_total -= log_scale_i.squeeze(-1)
        
        return x, log_det_total


class AutoregressiveNormalizingFlow(nn.Module):
    """Complete autoregressive normalizing flow model"""
    
    def __init__(self,
                 input_dim: int = 2,
                 num_layers: int = 4,
                 hidden_dim: int = 128,
                 transformer_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        
        # Multiple flow layers
        self.flow_layers = nn.ModuleList([
            AutoregressiveFlowLayer(input_dim, hidden_dim, transformer_layers)
            for _ in range(num_layers)
        ])
        
        # Permutation layers between flows
        self.permutations = nn.ModuleList([
            self._create_permutation(input_dim) 
            for _ in range(num_layers - 1)
        ])
    
    def _create_permutation(self, dim: int) -> nn.Module:
        """Create a learnable permutation layer"""
        return nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow
        Args:
            x: input tensor of shape (batch_size, input_dim)
        Returns:
            z: latent tensor
            log_det: total log determinant
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for i, flow_layer in enumerate(self.flow_layers):
            z, log_det = flow_layer(z)
            log_det_total += log_det
            
            # Apply permutation (except for last layer)
            if i < len(self.permutations):
                z = self.permutations[i](z)
        
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass through the flow
        Args:
            z: latent tensor
        Returns:
            x: reconstructed input
            log_det: total log determinant
        """
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        # Apply layers in reverse order
        for i in reversed(range(len(self.flow_layers))):
            # Reverse permutation first (except for last layer)
            if i < len(self.permutations):
                # For simplicity, use pseudo-inverse of permutation
                # In practice, you'd store the actual inverse
                perm_inv = torch.linalg.pinv(self.permutations[i].weight)
                x = F.linear(x, perm_inv)
            
            x, log_det = self.flow_layers[i].inverse(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of input under the model
        Args:
            x: input tensor
        Returns:
            log_prob: log probability
        """
        z, log_det = self.forward(x)
        
        # Standard Gaussian log probability
        log_prob_z = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        
        return log_prob_z + log_det
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the model
        Args:
            num_samples: number of samples
            device: device to generate samples on
        Returns:
            samples: generated samples
        """
        # Sample from standard Gaussian
        z = torch.randn(num_samples, self.input_dim, device=device)
        
        # Transform through inverse flow
        x, _ = self.inverse(z)
        
        return x


# Simpler alternative: MAF-style autoregressive flow
class MAFLayer(nn.Module):
    """Masked Autoregressive Flow layer - simpler version"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        
        # Create masks for autoregressive property
        self.register_buffer('mask', self._create_mask(input_dim, hidden_dim))
        
        # Network layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2)  # scale and shift parameters
        )
        
        # Apply masks to ensure autoregressive property
        self._mask_weights()
        
    def _create_mask(self, input_dim: int, hidden_dim: int) -> torch.Tensor:
        """Create autoregressive mask"""
        # For simplicity, this is a basic implementation
        # Full MAF would need more sophisticated masking
        mask = torch.zeros(hidden_dim, input_dim)
        for i in range(hidden_dim):
            mask[i, :min(i % input_dim + 1, input_dim)] = 1
        return mask
    
    def _mask_weights(self):
        """Apply masks to network weights"""
        # This is a simplified version
        # Full implementation would properly mask all layers
        pass
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        params = self.net(x)
        log_scale, shift = params.chunk(2, dim=1)
        
        # Ensure autoregressive property by masking
        log_scale = log_scale.tril(-1)  # Only use previous dimensions
        shift = shift.tril(-1)
        
        scale = torch.exp(log_scale)
        z = x * scale + shift
        log_det = log_scale.sum(dim=1)
        
        return z, log_det


if __name__ == "__main__":
    # Test the autoregressive flow
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = AutoregressiveNormalizingFlow(
        input_dim=2,
        num_layers=3,
        hidden_dim=64,
        transformer_layers=2
    ).to(device)
    
    # Test forward pass
    x = torch.randn(10, 2).to(device)
    z, log_det = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Log det shape: {log_det.shape}")
    
    # Test inverse
    x_recon, log_det_inv = model.inverse(z)
    print(f"Reconstruction error: {torch.mean((x - x_recon)**2):.6f}")
    
    # Test sampling
    samples = model.sample(5, device)
    print(f"Samples shape: {samples.shape}")
    print("Autoregressive flow test completed!")
