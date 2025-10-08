"""Molecular Parallel Tempering trainer for cross-temperature transport.

Implements the bidirectional NLL loss from theory.md that maximizes PT swap acceptance.
"""

import os
from typing import Dict, Tuple, List

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .openmm_energy import compute_reduced_energy


class MolecularPTTrainer:
    """Trainer for molecular cross-temperature normalizing flows.
    
    Implements bidirectional transport loss:
        L_NLL = E[β_cold·U(T^{-1}(x_hot))] - E[log|det J_inv|] 
              + E[β_hot·U(T(x_cold))] - E[log|det J_fwd|]
    """
    
    def __init__(
        self,
        model,
        dataset,
        device: str = 'cpu',
        use_energy: bool = True,
    ):
        """Initialize molecular PT trainer.
        
        Args:
            model: Normalizing flow model (must support forward/inverse)
            dataset: MolecularPTDataset instance
            device: Device for training
            use_energy: If True, use OpenMM energy; else use density matching
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.use_energy = use_energy
        
        # Get inverse temperatures
        self.beta_source, self.beta_target = dataset.get_betas()
        
        print(f"Molecular PT Trainer initialized:")
        print(f"  {dataset.source_temp}K → {dataset.target_temp}K")
        print(f"  β_source = {self.beta_source:.4f} mol/kJ")
        print(f"  β_target = {self.beta_target:.4f} mol/kJ")
        print(f"  Energy evaluation: {'OpenMM' if use_energy else 'density matching'}")
    
    def compute_forward_loss(
        self,
        x_source: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute forward transport loss: source → target.
        
        Loss = E[β_target·U(T(x_source))] - E[log|det J_fwd|]
        
        Args:
            x_source: Batch from source distribution [B, 69]
            
        Returns:
            loss: Scalar loss
            metrics: Dict with loss components
        """
        # Forward transform: x_source → x_target_pred
        x_target_pred, log_det_fwd = self.model.forward_with_logdet(x_source)
        
        if self.use_energy:
            # Denormalize for OpenMM energy evaluation
            x_target_denorm = self.dataset.denormalize(x_target_pred)
            
            # Compute reduced energy β·U(T(x_source))
            reduced_energy = compute_reduced_energy(x_target_denorm, self.beta_target)
            energy_term = reduced_energy.mean()
        else:
            # Density matching: minimize KL divergence (no explicit energy)
            energy_term = torch.tensor(0.0, device=x_source.device)
        
        # Jacobian term: -E[log|det J|] (negative because we want high Jacobian)
        jacobian_term = -log_det_fwd.mean()
        
        # Total forward loss
        loss = energy_term + jacobian_term
        
        metrics = {
            'fwd_energy': energy_term.item(),
            'fwd_log_det': -jacobian_term.item(),  # Store actual log det
            'fwd_loss': loss.item(),
        }
        
        return loss, metrics
    
    def compute_inverse_loss(
        self,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute inverse transport loss: target → source.
        
        Loss = E[β_source·U(T^{-1}(x_target))] - E[log|det J_inv|]
        
        Args:
            x_target: Batch from target distribution [B, 69]
            
        Returns:
            loss: Scalar loss
            metrics: Dict with loss components
        """
        # Inverse transform: x_target → x_source_pred
        x_source_pred, log_det_inv = self.model.inverse_with_logdet(x_target)
        
        if self.use_energy:
            # Denormalize for OpenMM energy evaluation
            x_source_denorm = self.dataset.denormalize(x_source_pred)
            
            # Compute reduced energy β·U(T^{-1}(x_target))
            reduced_energy = compute_reduced_energy(x_source_denorm, self.beta_source)
            energy_term = reduced_energy.mean()
        else:
            # Density matching: minimize KL divergence
            energy_term = torch.tensor(0.0, device=x_target.device)
        
        # Jacobian term: -E[log|det J|]
        jacobian_term = -log_det_inv.mean()
        
        # Total inverse loss
        loss = energy_term + jacobian_term
        
        metrics = {
            'inv_energy': energy_term.item(),
            'inv_log_det': -jacobian_term.item(),
            'inv_loss': loss.item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        optimizer: optim.Optimizer,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Single bidirectional training step.
        
        Args:
            optimizer: Optimizer instance
            batch_size: Batch size
            
        Returns:
            metrics: Dict with all loss components
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Sample batches from both distributions
        x_source, x_target = self.dataset.get_batch(batch_size)
        x_source = x_source.to(self.device)
        x_target = x_target.to(self.device)
        
        # Compute bidirectional losses
        loss_fwd, metrics_fwd = self.compute_forward_loss(x_source)
        loss_inv, metrics_inv = self.compute_inverse_loss(x_target)
        
        # Total loss (sum of forward and inverse)
        total_loss = loss_fwd + loss_inv
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Combine metrics
        metrics = {
            **metrics_fwd,
            **metrics_inv,
            'total_loss': total_loss.item(),
        }
        
        return metrics
    
    def train(
        self,
        config: Dict,
        save_dir: str = "checkpoints",
    ) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
        """Full training loop with loss curve tracking.
        
        Args:
            config: Training configuration dict
            save_dir: Directory to save checkpoints and plots
            
        Returns:
            model: Trained model
            history: List of metric dicts per epoch
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract training config
        training_config = config.get('training', {})
        num_epochs = training_config.get('epochs', 3000)
        batch_size = training_config.get('batch_size', 256)
        learning_rate = training_config.get('learning_rate', 5e-4)
        eval_interval = training_config.get('eval_interval', 100)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=200, factor=0.7, verbose=True
        )
        
        # Training history
        history = []
        
        print(f"\n{'='*70}")
        print(f"🧬 Molecular Cross-Temperature Transport Training")
        print(f"{'='*70}")
        print(f"Model: {config.get('model_type', 'ScalableTransformerFlow')}")
        print(f"Transport: {self.dataset.source_temp}K → {self.dataset.target_temp}K")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        print(f"{'='*70}\n")
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training"):
            metrics = self.train_step(optimizer, batch_size)
            history.append(metrics)
            
            # Learning rate scheduling
            scheduler.step(metrics['total_loss'])
            
            # Periodic logging
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Forward  - Energy: {metrics['fwd_energy']:.4f}, LogDet: {metrics['fwd_log_det']:.4f}")
                print(f"  Inverse  - Energy: {metrics['inv_energy']:.4f}, LogDet: {metrics['inv_log_det']:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save final model
        model_path = os.path.join(save_dir, f"molecular_pt_{int(self.dataset.source_temp)}_{int(self.dataset.target_temp)}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✅ Model saved: {model_path}")
        
        # Plot loss curves
        self.plot_loss_curves(history, save_dir)
        
        print(f"\n{'='*70}")
        print(f"🎉 Training completed!")
        print(f"{'='*70}\n")
        
        return self.model, history
    
    def plot_loss_curves(self, history: List[Dict[str, float]], save_dir: str):
        """Plot training loss curves.
        
        Args:
            history: List of metric dicts
            save_dir: Directory to save plots
        """
        epochs = np.arange(len(history))
        
        # Extract metrics
        total_loss = [h['total_loss'] for h in history]
        fwd_loss = [h['fwd_loss'] for h in history]
        inv_loss = [h['inv_loss'] for h in history]
        fwd_energy = [h['fwd_energy'] for h in history]
        inv_energy = [h['inv_energy'] for h in history]
        fwd_logdet = [h['fwd_log_det'] for h in history]
        inv_logdet = [h['inv_log_det'] for h in history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].plot(epochs, total_loss, linewidth=2, color='#2E86AB')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Total Loss', fontsize=12)
        axes[0, 0].set_title('Total Bidirectional Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        
        # Forward vs Inverse loss
        axes[0, 1].plot(epochs, fwd_loss, label='Forward', linewidth=2, color='#A23B72')
        axes[0, 1].plot(epochs, inv_loss, label='Inverse', linewidth=2, color='#F18F01')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Forward vs Inverse Loss', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(alpha=0.3)
        
        # Energy terms
        axes[1, 0].plot(epochs, fwd_energy, label='Fwd Energy', linewidth=2, color='#A23B72')
        axes[1, 0].plot(epochs, inv_energy, label='Inv Energy', linewidth=2, color='#F18F01')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Energy Term (β·U)', fontsize=12)
        axes[1, 0].set_title('Energy Components', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(alpha=0.3)
        
        # Jacobian log determinants
        axes[1, 1].plot(epochs, fwd_logdet, label='Fwd LogDet', linewidth=2, color='#A23B72')
        axes[1, 1].plot(epochs, inv_logdet, label='Inv LogDet', linewidth=2, color='#F18F01')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Log |det J|', fontsize=12)
        axes[1, 1].set_title('Jacobian Determinants', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(
            f'Molecular PT Training: {int(self.dataset.source_temp)}K → {int(self.dataset.target_temp)}K',
            fontsize=16,
            fontweight='bold',
            y=0.998
        )
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f"loss_curves_{int(self.dataset.source_temp)}_{int(self.dataset.target_temp)}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss curves saved: {plot_path}")

