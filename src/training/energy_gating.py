"""Energy gating for stable molecular normalizing flow training.

Implements energy regularization to prevent gradient instability from extreme
energy values during training. Uses a two-stage approach:
1. Hard clamp at E_max to prevent numerical overflow
2. Soft logarithmic compression above E_cut to maintain gradient flow

References:
- Boltzmann Generators (Noé et al., 2019)
- Energy-based models stabilization techniques
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import warnings
import numpy as np


class EnergyGating:
    """Energy gating for stable training with physical constraints.
    
    Prevents gradient instability by filtering/regularizing samples with
    physically unreasonable energies.
    
    Attributes:
        E_cut: Soft regularization threshold (kJ/mol)
        E_max: Hard clamp threshold (kJ/mol)
        skip_threshold: Fraction of batch that can exceed E_max before skipping
    """
    
    def __init__(
        self,
        E_cut: float = 500.0,
        E_max: float = 100000.0,  # Increased from 10k to 100k kJ/mol
        skip_threshold: float = 0.9,
        log_warnings: bool = True,
        warmup_epochs: int = 300,  # Increased from 100 to 300 epochs
        min_learning_epochs: int = 50,  # Keep very lenient for first 50 epochs
    ):
        """Initialize energy gating with progressive thresholds.
        
        Args:
            E_cut: Soft regularization threshold in kJ/mol
                   (typical peptide energies: -600 to +200)
            E_max: Hard clamp threshold in kJ/mol after warmup
                   (prevents numerical overflow, default 100k)
            skip_threshold: If this fraction of batch exceeds E_max, skip batch
                            (0.9 = skip if >90% of samples are extreme)
            log_warnings: Whether to log when batches are skipped
            warmup_epochs: Number of epochs for progressive E_max decay
                           (uses exponential schedule to stay lenient longer)
            min_learning_epochs: Epochs to keep E_max extremely high
                                 (allows model to learn basics first)
        """
        self.E_cut = E_cut
        self.E_max_target = E_max  # Target E_max after warmup (100k)
        self.E_max_initial = 1e10  # Start EXTREMELY lenient (10 billion kJ/mol)
        self.E_max = self.E_max_initial
        self.skip_threshold = skip_threshold
        self.log_warnings = log_warnings
        self.warmup_epochs = warmup_epochs
        self.min_learning_epochs = min_learning_epochs
        self.current_epoch = 0
        
        # Statistics tracking
        self.total_batches = 0
        self.skipped_batches = 0
        self.regularized_samples = 0
        self.total_samples = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive gating.
        
        Uses a 3-stage schedule:
        1. Epochs 0-min_learning: E_max = initial (extremely lenient)
        2. Epochs min_learning-warmup: Exponential decay to target
        3. Epochs warmup+: E_max = target (strict)
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        if epoch < self.min_learning_epochs:
            # Stage 1: Keep E_max extremely high to allow initial learning
            self.E_max = self.E_max_initial
        elif epoch < self.warmup_epochs:
            # Stage 2: Exponential decay from initial to target
            # Exponential decay stays lenient longer than linear
            warmup_progress = (epoch - self.min_learning_epochs) / (self.warmup_epochs - self.min_learning_epochs)
            
            # Use exponential interpolation in log space
            # E_max = initial * (target/initial)^progress
            log_initial = np.log(self.E_max_initial)
            log_target = np.log(self.E_max_target)
            log_E_max = log_initial + warmup_progress * (log_target - log_initial)
            self.E_max = np.exp(log_E_max)
        else:
            # Stage 3: Use final target threshold
            self.E_max = self.E_max_target
    
    def apply_gating(
        self,
        energies: Tensor,
    ) -> Tuple[Optional[Tensor], dict]:
        """Apply energy gating to a batch of energies.
        
        Two-stage regularization:
        1. Hard clamp: U_clamped = min(U, E_max)
        2. Soft regularization:
           U_reg = U                           if U ≤ E_cut
                 = E_cut + log(1 + U - E_cut)  if E_cut < U ≤ E_max
        
        Args:
            energies: Tensor of energies [batch_size] in kJ/mol
            
        Returns:
            regularized_energies: Regularized energies, or None if batch should be skipped
            info: Dict with gating statistics
        """
        self.total_batches += 1
        batch_size = energies.shape[0]
        self.total_samples += batch_size
        
        # Check for NaN or Inf
        invalid_mask = torch.isnan(energies) | torch.isinf(energies)
        if invalid_mask.any():
            if self.log_warnings:
                warnings.warn(f"Found {invalid_mask.sum().item()} NaN/Inf energies in batch")
            # Replace with sentinel value
            energies = torch.where(invalid_mask, torch.tensor(self.E_max, device=energies.device), energies)
        
        # Count samples exceeding thresholds
        extreme_mask = energies > self.E_max
        high_mask = (energies > self.E_cut) & (energies <= self.E_max)
        normal_mask = energies <= self.E_cut
        
        num_extreme = extreme_mask.sum().item()
        num_high = high_mask.sum().item()
        num_normal = normal_mask.sum().item()
        
        # Skip batch if too many samples are extreme
        if num_extreme / batch_size > self.skip_threshold:
            self.skipped_batches += 1
            if self.log_warnings:
                warnings.warn(
                    f"Skipping batch: {num_extreme}/{batch_size} samples exceed E_max={self.E_max:.0f} kJ/mol. "
                    f"Max energy: {energies.max().item():.2e} kJ/mol"
                )
            return None, {
                'skipped': True,
                'num_extreme': num_extreme,
                'num_high': num_high,
                'num_normal': num_normal,
                'max_energy': energies.max().item(),
                'mean_energy': energies.mean().item(),
            }
        
        # Apply two-stage regularization
        regularized = torch.zeros_like(energies)
        
        # Stage 1: Normal energies (≤ E_cut) - no change
        regularized[normal_mask] = energies[normal_mask]
        
        # Stage 2: High energies (E_cut < U ≤ E_max) - soft regularization
        if num_high > 0:
            high_energies = energies[high_mask]
            # U_reg = E_cut + log(1 + U - E_cut)
            regularized[high_mask] = self.E_cut + torch.log(1.0 + high_energies - self.E_cut)
            self.regularized_samples += num_high
        
        # Stage 3: Extreme energies (> E_max) - hard clamp
        if num_extreme > 0:
            # Hard clamp to E_max, then apply soft regularization
            clamped = torch.clamp(energies[extreme_mask], max=self.E_max)
            regularized[extreme_mask] = self.E_cut + torch.log(1.0 + clamped - self.E_cut)
            self.regularized_samples += num_extreme
        
        info = {
            'skipped': False,
            'num_extreme': num_extreme,
            'num_high': num_high,
            'num_normal': num_normal,
            'max_energy': energies.max().item(),
            'mean_energy': energies.mean().item(),
            'regularized_mean': regularized.mean().item(),
        }
        
        return regularized, info
    
    def get_statistics(self) -> dict:
        """Get gating statistics.
        
        Returns:
            Dict with cumulative statistics
        """
        skip_rate = self.skipped_batches / max(self.total_batches, 1)
        reg_rate = self.regularized_samples / max(self.total_samples, 1)
        
        return {
            'total_batches': self.total_batches,
            'skipped_batches': self.skipped_batches,
            'skip_rate': skip_rate,
            'regularized_samples': self.regularized_samples,
            'total_samples': self.total_samples,
            'regularization_rate': reg_rate,
        }
    
    def reset_statistics(self):
        """Reset cumulative statistics."""
        self.total_batches = 0
        self.skipped_batches = 0
        self.regularized_samples = 0
        self.total_samples = 0


def apply_energy_regularization(
    energies: Tensor,
    E_cut: float = 500.0,
    E_max: float = 10000.0,
) -> Tensor:
    """Apply energy regularization without batch skipping (functional API).
    
    Args:
        energies: Energies in kJ/mol [batch_size]
        E_cut: Soft regularization threshold
        E_max: Hard clamp threshold
        
    Returns:
        Regularized energies [batch_size]
    """
    # Hard clamp first
    clamped = torch.clamp(energies, max=E_max)
    
    # Soft regularization for high energies
    mask = clamped > E_cut
    regularized = torch.zeros_like(clamped)
    regularized[~mask] = clamped[~mask]
    regularized[mask] = E_cut + torch.log(1.0 + clamped[mask] - E_cut)
    
    return regularized


if __name__ == "__main__":
    # Test energy gating
    print("Testing Energy Gating...")
    
    gating = EnergyGating(E_cut=500.0, E_max=10000.0)
    
    # Test 1: Normal energies
    print("\n1. Normal energies (should pass through):")
    normal_energies = torch.tensor([-200.0, -100.0, 50.0, 200.0, 400.0])
    reg, info = gating.apply_gating(normal_energies)
    print(f"   Input:  {normal_energies.tolist()}")
    print(f"   Output: {reg.tolist()}")
    print(f"   Info: {info}")
    
    # Test 2: High energies (soft regularization)
    print("\n2. High energies (should be compressed):")
    high_energies = torch.tensor([600.0, 1000.0, 5000.0])
    reg, info = gating.apply_gating(high_energies)
    print(f"   Input:  {high_energies.tolist()}")
    print(f"   Output: {[f'{x:.2f}' for x in reg.tolist()]}")
    print(f"   Info: {info}")
    
    # Test 3: Extreme energies (should skip batch)
    print("\n3. Extreme energies (should skip batch):")
    extreme_energies = torch.tensor([50000.0, 100000.0, 1000000.0])
    reg, info = gating.apply_gating(extreme_energies)
    print(f"   Input:  {extreme_energies.tolist()}")
    print(f"   Output: {reg}")
    print(f"   Info: {info}")
    
    # Test 4: Mixed energies
    print("\n4. Mixed energies:")
    mixed_energies = torch.tensor([-100.0, 200.0, 800.0, 5000.0, 15000.0])
    reg, info = gating.apply_gating(mixed_energies)
    print(f"   Input:  {mixed_energies.tolist()}")
    print(f"   Output: {[f'{x:.2f}' for x in reg.tolist()] if reg is not None else None}")
    print(f"   Info: {info}")
    
    # Statistics
    print("\n5. Cumulative statistics:")
    print(f"   {gating.get_statistics()}")

