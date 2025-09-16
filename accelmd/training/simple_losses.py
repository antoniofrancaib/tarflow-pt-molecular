"""Simplified loss functions for TARFlow training.

Simple bidirectional NLL loss for distribution mapping experiments.
"""

import torch
from torch import Tensor
from typing import Dict, Optional

from ..flows.pt_tarflow import PTTARFlow

__all__ = ["simple_bidirectional_nll"]


def simple_bidirectional_nll(
    model: PTTARFlow,
    source_coords: Tensor,
    target_coords: Tensor,
    source_temp: Tensor,
    target_temp: Tensor,
    **kwargs
) -> Dict[str, Tensor]:
    """Compute simple bidirectional negative log-likelihood loss.
    
    Args:
        model: PTTARFlow model
        source_coords: Source distribution coordinates [batch_size, num_atoms, coordinate_dim]
        target_coords: Target distribution coordinates [batch_size, num_atoms, coordinate_dim]
        source_temp: Source temperature [batch_size]
        target_temp: Target temperature [batch_size]
        
    Returns:
        Dictionary with loss components
    """
    batch_size = source_coords.shape[0]
    
    # Forward direction: source -> target
    try:
        target_pred, log_det_forward = model.transform(source_coords, reverse=False)
        # Compute log probability of predicted coordinates under target distribution
        log_prob_forward = model.target_target.log_prob(target_pred)
        # NLL forward: negative of (log_prob + log_det)
        nll_forward = -(log_prob_forward + log_det_forward).mean()
        

    except Exception as e:
        print(f"Forward transform failed: {e}")
        nll_forward = torch.tensor(float('inf'), device=source_coords.device, requires_grad=True)
    
    # Backward direction: target -> source  
    try:
        source_pred, log_det_backward = model.transform(target_coords, reverse=True)
        # Compute log probability of predicted coordinates under source distribution
        log_prob_backward = model.source_target.log_prob(source_pred)
        # NLL backward: negative of (log_prob + log_det)
        nll_backward = -(log_prob_backward + log_det_backward).mean()
        

    except Exception as e:
        print(f"Backward transform failed: {e}")
        nll_backward = torch.tensor(float('inf'), device=source_coords.device, requires_grad=True)
    
    # Total bidirectional loss
    total_loss = nll_forward + nll_backward
    
    return {
        "loss": total_loss,
        "nll_forward": nll_forward,
        "nll_backward": nll_backward,
        "log_det_forward": log_det_forward.mean() if not torch.isinf(nll_forward) else torch.tensor(0.0),
        "log_det_backward": log_det_backward.mean() if not torch.isinf(nll_backward) else torch.tensor(0.0),
    }
