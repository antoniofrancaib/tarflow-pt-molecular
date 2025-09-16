#!/usr/bin/env python3
"""
Evaluation utilities for normalizing flows
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ..utils.plotting import plot_results


def evaluate_model(model, target_dist, base_dist, config: dict, device: str = 'cpu'):
    """
    Comprehensive model evaluation
    
    Args:
        model: trained normalizing flow model
        target_dist: target distribution
        base_dist: base distribution  
        config: configuration dict
        device: compute device
    Returns:
        metrics: comprehensive evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Generate samples for evaluation
        num_samples = 2000
        model_samples = model.sample(num_samples, device)
        target_samples = target_dist.sample(num_samples, device)
        base_samples = base_dist.sample(num_samples, device)
        
        # Compute log likelihoods
        target_log_likelihood = model.log_prob(target_samples).mean().item()
        
        # Statistical metrics
        metrics = {
            'target_log_likelihood': target_log_likelihood,
            'model_samples_mean': model_samples.mean(dim=0).cpu().numpy(),
            'model_samples_std': model_samples.std(dim=0).cpu().numpy(),
            'target_samples_mean': target_samples.mean(dim=0).cpu().numpy(),
            'target_samples_std': target_samples.std(dim=0).cpu().numpy(),
            'samples': {
                'model': model_samples.cpu().numpy(),
                'target': target_samples.cpu().numpy(),
                'base': base_samples.cpu().numpy()
            }
        }
        
        # Compute additional metrics if possible
        try:
            # Wasserstein distance approximation
            from scipy.stats import wasserstein_distance
            model_np = model_samples.cpu().numpy()
            target_np = target_samples.cpu().numpy()
            
            wd_x = wasserstein_distance(model_np[:, 0], target_np[:, 0])
            wd_y = wasserstein_distance(model_np[:, 1], target_np[:, 1])
            metrics['wasserstein_distance'] = (wd_x + wd_y) / 2
        except ImportError:
            metrics['wasserstein_distance'] = None
    
    return metrics


def plot_training_summary(loss_history: list, config: dict):
    """
    Plot training loss summary
    
    Args:
        loss_history: list of loss values during training
        config: configuration dict
    """
    # Filter out NaN values
    valid_losses = [l for l in loss_history if not np.isnan(l)]
    
    if valid_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_losses)
        
        model_type = config.get('model_type', 'Model').title()
        target_name = config['target'].replace('_', ' ').title()
        plt.title(f'Training Loss - {model_type} on {target_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log Likelihood')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        target_name = config['target']
        model_type = config.get('model_type', 'simple')
        save_dir = f"plots_{model_type}_{target_name}"
        plt.savefig(f'{save_dir}/training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("Warning: No valid loss values to plot")
