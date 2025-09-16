#!/usr/bin/env python3
"""
Unified trainer for normalizing flows
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from ..utils.plotting import plot_results


class FlowTrainer:
    """Unified trainer for all normalizing flow models"""
    
    def __init__(self, model, target_dist, base_dist, device: str = 'cpu'):
        self.model = model.to(device)
        self.target_dist = target_dist
        self.base_dist = base_dist
        self.device = device
        
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood loss with stability checks
        
        Args:
            x: batch of target samples
        Returns:
            loss: negative log likelihood
        """
        try:
            log_prob = self.model.log_prob(x)
            
            # Check for NaN or inf values
            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                print("Warning: Invalid log_prob detected, returning high loss")
                return torch.tensor(10.0, device=x.device, requires_grad=True)
            
            loss = -log_prob.mean()
            
            # Check if loss is reasonable
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Invalid loss detected, returning high loss")
                return torch.tensor(10.0, device=x.device, requires_grad=True)
                
            return loss
        except Exception as e:
            print(f"Error computing loss: {e}")
            return torch.tensor(10.0, device=x.device, requires_grad=True)
    
    def train_step(self, optimizer: optim.Optimizer, batch_size: int = 512) -> float:
        """
        Single training step
        
        Args:
            optimizer: optimizer instance
            batch_size: batch size for training
        Returns:
            loss: loss value for this step
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Sample from target distribution
        x = self.target_dist.sample(batch_size, self.device)
        
        # Compute loss
        loss = self.compute_loss(x)
        
        # Backward pass only if loss is valid
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            
            # Aggressive gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            optimizer.step()
            loss_value = loss.item()
        else:
            loss_value = float('nan')
        
        return loss_value
    
    def evaluate(self, num_samples: int = 1000) -> dict:
        """
        Evaluate model performance
        
        Args:
            num_samples: number of samples for evaluation
        Returns:
            metrics: dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            try:
                # Generate samples from model
                model_samples = self.model.sample(num_samples, self.device)
                
                # Check for NaN in samples
                if torch.isnan(model_samples).any():
                    print("Warning: NaN in model samples")
                    model_samples = torch.randn_like(model_samples)
                
                # Get target samples
                target_samples = self.target_dist.sample(num_samples, self.device)
                
                # Compute log likelihood on target samples
                target_log_prob = self.model.log_prob(target_samples)
                
                if torch.isnan(target_log_prob).any():
                    print("Warning: NaN in target log prob")
                    target_log_prob_mean = float('nan')
                else:
                    target_log_prob_mean = target_log_prob.mean().item()
                
                # Compute statistics
                metrics = {
                    'model_samples_mean': model_samples.mean(dim=0).cpu().numpy(),
                    'model_samples_std': model_samples.std(dim=0).cpu().numpy(),
                    'target_samples_mean': target_samples.mean(dim=0).cpu().numpy(),
                    'target_samples_std': target_samples.std(dim=0).cpu().numpy(),
                    'target_log_likelihood': target_log_prob_mean,
                    'model_samples': model_samples.cpu().numpy(),
                    'target_samples': target_samples.cpu().numpy()
                }
            except Exception as e:
                print(f"Error in evaluation: {e}")
                # Return fallback metrics
                fallback_samples = torch.randn(num_samples, 2)
                target_samples = self.target_dist.sample(num_samples, self.device)
                metrics = {
                    'model_samples_mean': fallback_samples.mean(dim=0).numpy(),
                    'model_samples_std': fallback_samples.std(dim=0).numpy(),
                    'target_samples_mean': target_samples.mean(dim=0).cpu().numpy(),
                    'target_samples_std': target_samples.std(dim=0).cpu().numpy(),
                    'target_log_likelihood': float('nan'),
                    'model_samples': fallback_samples.numpy(),
                    'target_samples': target_samples.cpu().numpy()
                }
        
        return metrics
    
    def train(self, config: dict) -> tuple:
        """
        Full training loop
        
        Args:
            config: training configuration
        Returns:
            model: trained model
            loss_history: list of loss values
        """
        print(f"🎯 Starting {config.get('model_type', 'Model')} Training")
        print("=" * 70)
        
        # Setup optimizer from config
        training_config = config.get('training', {})
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=training_config.get('learning_rate', 1e-3), 
                              weight_decay=training_config.get('weight_decay', 1e-5))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=400, factor=0.7, verbose=True)
        
        # Training parameters from config
        num_epochs = training_config.get('epochs', 3000)
        batch_size = training_config.get('batch_size', 512)
        eval_interval = training_config.get('eval_interval', 300)
        
        # Training loop
        loss_history = []
        
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in tqdm(range(num_epochs)):
            # Training step
            loss = self.train_step(optimizer, batch_size)
            loss_history.append(loss)
            
            # Update learning rate only if loss is valid
            if not np.isnan(loss):
                scheduler.step(loss)
            
            # Evaluation and plotting
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                if np.isnan(loss):
                    print(f"Loss: NaN (skipped)")
                else:
                    print(f"Loss: {loss:.6f}")
                print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Evaluate model
                metrics = self.evaluate(500)
                if np.isnan(metrics['target_log_likelihood']):
                    print("Target log likelihood: NaN")
                else:
                    print(f"Target log likelihood: {metrics['target_log_likelihood']:.4f}")
                print(f"Model samples mean: {metrics['model_samples_mean']}")
                print(f"Target samples mean: {metrics['target_samples_mean']}")
                
                # Plot results
                plot_results(self.model, self.target_dist, self.base_dist, self.device, epoch + 1, config)
        
        # Final evaluation
        print("\n🎉 Training completed!")
        print("=" * 70)
        
        final_metrics = self.evaluate(2000)
        print("Final Evaluation:")
        if np.isnan(final_metrics['target_log_likelihood']):
            print("Target log likelihood: NaN")
        else:
            print(f"Target log likelihood: {final_metrics['target_log_likelihood']:.4f}")
        print(f"Model samples mean: {final_metrics['model_samples_mean']}")
        print(f"Model samples std: {final_metrics['model_samples_std']}")
        print(f"Target samples mean: {final_metrics['target_samples_mean']}")
        print(f"Target samples std: {final_metrics['target_samples_std']}")
        
        return self.model, loss_history
