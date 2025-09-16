#!/usr/bin/env python3
"""
Plotting utilities for normalizing flows
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def create_evaluation_grid(device: str, config: dict):
    """Create grid for density evaluation using config"""
    visualization_config = config.get('visualization', {})
    domain = visualization_config.get('domain', 3.5)
    xlim = [-domain, domain]
    ylim = [-domain, domain]
    resolution = visualization_config.get('resolution', 100)
    
    x = torch.linspace(xlim[0], xlim[1], resolution)
    y = torch.linspace(ylim[0], ylim[1], resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    return xx, yy, grid_points


def plot_results(model, target_dist, base_dist, device: str, epoch: int, config: dict, save_dir: str = None):
    """Plot comprehensive results for normalizing flow training"""
    
    if save_dir is None:
        target_name = config['target']
        model_type = config.get('model_type', 'simple')
        save_dir = f"plots_{model_type}_{target_name}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Create evaluation grid from config
    xx, yy, grid_points = create_evaluation_grid(device, config)
    
    # Get domain limits from config
    visualization_config = config.get('visualization', {})
    domain = visualization_config.get('domain', 3.5)
    xlim = [-domain, domain]
    ylim = [-domain, domain]
    
    with torch.no_grad():
        # Evaluate densities
        target_log_prob = target_dist.log_prob(grid_points.cpu()).view(xx.shape)
        target_prob = torch.exp(target_log_prob)
        
        # Safe model evaluation with better handling
        try:
            model_log_prob = model.log_prob(grid_points).cpu().view(xx.shape)
            
            # Debug: Print log prob statistics
            if epoch % 300 == 0:
                print(f"Model log_prob stats: min={model_log_prob.min():.2f}, max={model_log_prob.max():.2f}, mean={model_log_prob.mean():.2f}")
            
            # Check for NaN values in log probabilities
            if torch.isnan(model_log_prob).any() or torch.isinf(model_log_prob).any():
                print(f"Warning: Invalid values in model log_prob at epoch {epoch}")
                model_log_prob = torch.full_like(model_log_prob, -10.0)  # Set to reasonable default
            
            # Convert to probabilities - handle very negative log probs
            model_prob = torch.exp(model_log_prob)
            
            # If probabilities are too small to visualize, normalize them better
            if model_prob.max() < 1e-10:
                print(f"Warning: Model probabilities too small (max={model_prob.max():.2e}), using log-space visualization")
                # Use log probabilities shifted to positive range for visualization
                model_prob = torch.exp(model_log_prob - model_log_prob.max() + 5)
                
        except Exception as e:
            print(f"Error evaluating model probability at epoch {epoch}: {e}")
            model_prob = torch.ones_like(target_prob) * 1e-6
        
        base_log_prob = base_dist.log_prob(grid_points.cpu()).view(xx.shape)
        base_prob = torch.exp(base_log_prob)
        
        # Generate samples
        try:
            model_samples = model.sample(2000, device).cpu().numpy()
            # Check for NaN values in samples
            if np.isnan(model_samples).any():
                print(f"Warning: NaN values in model samples at epoch {epoch}")
                model_samples = np.random.randn(2000, 2)  # Fallback to random samples
        except:
            print(f"Error generating model samples at epoch {epoch}")
            model_samples = np.random.randn(2000, 2)
            
        target_samples = target_dist.sample(2000, device).numpy()
        base_samples = base_dist.sample(2000, device).numpy()
        
        # Test forward and inverse transforms
        try:
            test_points = target_dist.sample(500, device)
            z, _ = model.forward(test_points)
            x_recon, _ = model.inverse(z)
            
            base_test = base_dist.sample(500, device)
            x_from_base, _ = model.inverse(base_test)
            
            # Convert to numpy and check for NaN
            x_from_base = x_from_base.cpu().numpy()
            x_recon = x_recon.cpu().numpy()
            
            if np.isnan(x_from_base).any() or np.isnan(x_recon).any():
                print(f"Warning: NaN in transform tests at epoch {epoch}")
                x_from_base = np.random.randn(500, 2)
                x_recon = np.random.randn(500, 2)
        except:
            print(f"Error in transform tests at epoch {epoch}")
            x_from_base = np.random.randn(500, 2)
            x_recon = np.random.randn(500, 2)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    target_name = config['target'].replace('_', ' ').title()
    model_type = config.get('model_type', 'Simple').title()
    fig.suptitle(f'{model_type} Autoregressive Flow vs {target_name} - Epoch {epoch}', fontsize=16)
    
    # Row 1: Density plots
    im1 = axes[0, 0].pcolormesh(xx, yy, base_prob.numpy(), cmap='Blues', shading='auto')
    axes[0, 0].set_title('Base Distribution\n(Gaussian)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    target_cmap = 'RdBu_r' if 'checkerboard' in config['target'] else 'Reds'
    im2 = axes[0, 1].pcolormesh(xx, yy, target_prob.numpy(), cmap=target_cmap, shading='auto')
    axes[0, 1].set_title(f'Target Distribution\n({target_name})')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Better visualization of model distribution
    model_prob_np = model_prob.numpy()
    
    # Use log scale if probabilities are very small
    if model_prob_np.max() > 0 and model_prob_np.max() / (model_prob_np.min() + 1e-10) > 1000:
        # Use log scale for better visualization
        model_prob_viz = np.log(model_prob_np + 1e-10)
        im3 = axes[0, 2].pcolormesh(xx, yy, model_prob_viz, cmap='Greens', shading='auto')
        axes[0, 2].set_title(f'Learned Distribution\n({model_type} - Log Scale)')
    else:
        im3 = axes[0, 2].pcolormesh(xx, yy, model_prob_np, cmap='Greens', shading='auto')
        axes[0, 2].set_title(f'Learned Distribution\n({model_type})')
    
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Comparison with better contour handling
    try:
        target_prob_np = target_prob.numpy()
        model_prob_np = model_prob.numpy()
        
        # Target contours
        if target_prob_np.max() > target_prob_np.min():
            axes[0, 3].contour(xx, yy, target_prob_np, levels=8, colors='red', alpha=0.7, linewidths=2)
        
        # Model contours - handle small values
        if model_prob_np.max() > model_prob_np.min() + 1e-10:
            # Normalize model probabilities for better contour visualization
            model_norm = (model_prob_np - model_prob_np.min()) / (model_prob_np.max() - model_prob_np.min() + 1e-10)
            axes[0, 3].contour(xx, yy, model_norm, levels=8, colors='green', alpha=0.7, linewidths=2)
        else:
            print(f"Skipping model contours - insufficient variation in probabilities")
            
    except Exception as e:
        print(f"Warning: Could not plot contours: {e}")
    
    axes[0, 3].set_title('Target vs Learned\n(Red: Target, Green: Learned)')
    axes[0, 3].set_aspect('equal')
    axes[0, 3].set_xlim(xlim)
    axes[0, 3].set_ylim(ylim)
    
    # Row 2: Sample plots and transformations
    axes[1, 0].scatter(base_samples[:, 0], base_samples[:, 1], alpha=0.6, s=5, c='blue')
    axes[1, 0].set_title('Base Samples')
    axes[1, 0].set_xlim(xlim)
    axes[1, 0].set_ylim(ylim)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.6, s=5, c='red')
    axes[1, 1].set_title('Target Samples')
    axes[1, 1].set_xlim(xlim)
    axes[1, 1].set_ylim(ylim)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.6, s=5, c='green')
    axes[1, 2].set_title('Generated Samples')
    axes[1, 2].set_xlim(xlim)
    axes[1, 2].set_ylim(ylim)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Forward/Inverse test
    axes[1, 3].scatter(x_from_base[:, 0], x_from_base[:, 1], 
                      alpha=0.6, s=5, c='purple', label='Base→Flow')
    axes[1, 3].scatter(x_recon[:, 0], x_recon[:, 1], 
                      alpha=0.6, s=5, c='orange', label='Target→Base→Target')
    axes[1, 3].set_title('Transform Tests')
    axes[1, 3].set_xlim(xlim)
    axes[1, 3].set_ylim(ylim)
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    target_name = config['target']
    filename = f"{save_dir}/{target_name}_analysis_epoch_{epoch:04d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 3000)
    show_threshold = epochs - 300  # Show final results near end
    if epoch >= show_threshold:
        plt.show()
    else:
        plt.close()
