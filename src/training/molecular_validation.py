"""Validation and visualization tools for molecular cross-temperature transport.

Provides Ramachandran plot comparisons, energy distribution analysis, and
coordinate RMSD metrics for evaluating flow quality.
"""

import os
from typing import Dict, Tuple

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from simtk.openmm.app import PDBFile

from .openmm_energy import compute_potential_energy
from ..utils.plot_utils import plot_Ramachandran


class MolecularValidator:
    """Validation tools for molecular transport flows."""
    
    def __init__(
        self,
        model,
        dataset,
        pdb_path: str = "datasets/AA/ref.pdb",
        device: str = 'cpu',
    ):
        """Initialize molecular validator.
        
        Args:
            model: Trained normalizing flow
            dataset: MolecularPTDataset instance
            pdb_path: Path to reference PDB for topology
            device: Device for computation
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        # Load topology for mdtraj
        pdb = PDBFile(pdb_path)
        self.topology = md.Topology.from_openmm(pdb.topology)
        
        print(f"Molecular Validator initialized for {dataset.source_temp}K → {dataset.target_temp}K")
    
    def compute_dihedrals(
        self,
        coords: Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute phi/psi dihedral angles from coordinates.
        
        Args:
            coords: Normalized coordinates [N, 69]
            
        Returns:
            (phi, psi) arrays in radians, shape [N, n_dihedrals]
        """
        # Denormalize to nm
        coords_nm = self.dataset.denormalize(coords).cpu().numpy()
        
        # Reshape to [N, 23, 3]
        coords_reshaped = coords_nm.reshape(-1, 23, 3)
        
        # Create mdtraj trajectory
        traj = md.Trajectory(coords_reshaped, self.topology)
        
        # Compute dihedrals
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)
        
        return phi, psi
    
    def plot_ramachandran_comparison(
        self,
        num_samples: int = 2000,
        save_path: str = "plots/ramachandran_comparison.png",
    ):
        """Create side-by-side Ramachandran plots for validation.
        
        Plots:
        1. Source distribution (ground truth)
        2. Transformed source → target (model prediction)
        3. Target distribution (ground truth)
        
        Args:
            num_samples: Number of samples for visualization
            save_path: Path to save figure
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get source and target samples
            x_source, x_target = self.dataset.get_batch(num_samples)
            x_source = x_source.to(self.device)
            x_target = x_target.to(self.device)
            
            # Transform source → target
            x_transformed, _ = self.model.forward_with_logdet(x_source)
            
            # Compute dihedrals
            phi_source, psi_source = self.compute_dihedrals(x_source)
            phi_transformed, psi_transformed = self.compute_dihedrals(x_transformed)
            phi_target, psi_target = self.compute_dihedrals(x_target)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot source
        plot_Ramachandran(axes[0], phi_source, psi_source)
        axes[0].set_title(
            f'Source {int(self.dataset.source_temp)}K\n(Ground Truth)',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].set_ylabel('ψ (Psi)', fontsize=14)
        
        # Plot transformed
        plot_Ramachandran(axes[1], phi_transformed, psi_transformed)
        axes[1].set_title(
            f'Transformed {int(self.dataset.source_temp)}K→{int(self.dataset.target_temp)}K\n(Model Prediction)',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].set_ylabel('')
        
        # Plot target
        plot_Ramachandran(axes[2], phi_target, psi_target)
        axes[2].set_title(
            f'Target {int(self.dataset.target_temp)}K\n(Ground Truth)',
            fontsize=14,
            fontweight='bold'
        )
        axes[2].set_ylabel('')
        
        plt.suptitle(
            f'Ramachandran Validation: {int(self.dataset.source_temp)}K → {int(self.dataset.target_temp)}K',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Ramachandran comparison saved: {save_path}")
    
    def plot_energy_distributions(
        self,
        num_samples: int = 2000,
        save_path: str = "plots/energy_distributions.png",
    ):
        """Plot energy distribution comparisons.
        
        Args:
            num_samples: Number of samples
            save_path: Path to save figure
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get samples
            x_source, x_target = self.dataset.get_batch(num_samples)
            x_source = x_source.to(self.device)
            x_target = x_target.to(self.device)
            
            # Transform
            x_transformed, _ = self.model.forward_with_logdet(x_source)
            x_reconstructed, _ = self.model.inverse_with_logdet(x_transformed)
            
            # Denormalize
            x_source_denorm = self.dataset.denormalize(x_source)
            x_target_denorm = self.dataset.denormalize(x_target)
            x_transformed_denorm = self.dataset.denormalize(x_transformed)
            
            # Compute energies
            U_source = compute_potential_energy(x_source_denorm).cpu().numpy()
            U_target = compute_potential_energy(x_target_denorm).cpu().numpy()
            U_transformed = compute_potential_energy(x_transformed_denorm).cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Energy distributions
        bins = 50
        axes[0].hist(U_source, bins=bins, alpha=0.6, label=f'Source {int(self.dataset.source_temp)}K', color='#2E86AB', density=True)
        axes[0].hist(U_transformed, bins=bins, alpha=0.6, label=f'Transformed', color='#A23B72', density=True)
        axes[0].hist(U_target, bins=bins, alpha=0.6, label=f'Target {int(self.dataset.target_temp)}K', color='#F18F01', density=True)
        axes[0].set_xlabel('Potential Energy (kJ/mol)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Energy Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)
        
        # Reconstruction error (RMSD)
        rmsd = torch.sqrt(((x_source - x_reconstructed) ** 2).sum(dim=-1)).cpu().numpy()
        axes[1].hist(rmsd, bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
        axes[1].axvline(rmsd.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rmsd.mean():.4f}')
        axes[1].set_xlabel('RMSD (normalized units)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Reconstruction Error (T → T⁻¹ → T)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(
            f'Energy & Reconstruction Validation: {int(self.dataset.source_temp)}K → {int(self.dataset.target_temp)}K',
            fontsize=16,
            fontweight='bold',
            y=1.00
        )
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Energy distributions saved: {save_path}")
        print(f"   Mean RMSD: {rmsd.mean():.6f} ± {rmsd.std():.6f}")
    
    def compute_metrics(self, num_samples: int = 2000) -> Dict[str, float]:
        """Compute quantitative validation metrics.
        
        Args:
            num_samples: Number of samples for evaluation
            
        Returns:
            metrics: Dict with validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get samples
            x_source, x_target = self.dataset.get_batch(num_samples)
            x_source = x_source.to(self.device)
            x_target = x_target.to(self.device)
            
            # Forward transform
            x_transformed, log_det_fwd = self.model.forward_with_logdet(x_source)
            
            # Reconstruction
            x_reconstructed, log_det_inv = self.model.inverse_with_logdet(x_transformed)
            
            # Denormalize
            x_target_denorm = self.dataset.denormalize(x_target)
            x_transformed_denorm = self.dataset.denormalize(x_transformed)
            
            # Energies
            U_target = compute_potential_energy(x_target_denorm)
            U_transformed = compute_potential_energy(x_transformed_denorm)
            
            # RMSD metrics
            reconstruction_rmsd = torch.sqrt(((x_source - x_reconstructed) ** 2).sum(dim=-1))
            
            # Energy differences
            energy_diff = torch.abs(U_target - U_transformed)
        
        metrics = {
            'mean_reconstruction_rmsd': reconstruction_rmsd.mean().item(),
            'std_reconstruction_rmsd': reconstruction_rmsd.std().item(),
            'mean_energy_diff': energy_diff.mean().item(),
            'std_energy_diff': energy_diff.std().item(),
            'mean_log_det_fwd': log_det_fwd.mean().item(),
            'mean_log_det_inv': log_det_inv.mean().item(),
            'mean_U_target': U_target.mean().item(),
            'mean_U_transformed': U_transformed.mean().item(),
        }
        
        return metrics
    
    def full_validation(
        self,
        num_samples: int = 2000,
        save_dir: str = "plots",
    ):
        """Run full validation suite.
        
        Args:
            num_samples: Number of samples
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🔬 Running Molecular Validation")
        print(f"{'='*70}")
        
        # Compute metrics
        metrics = self.compute_metrics(num_samples)
        
        print("\nQuantitative Metrics:")
        print(f"  Reconstruction RMSD: {metrics['mean_reconstruction_rmsd']:.6f} ± {metrics['std_reconstruction_rmsd']:.6f}")
        print(f"  Energy Difference: {metrics['mean_energy_diff']:.2f} ± {metrics['std_energy_diff']:.2f} kJ/mol")
        print(f"  Mean LogDet (fwd): {metrics['mean_log_det_fwd']:.4f}")
        print(f"  Mean LogDet (inv): {metrics['mean_log_det_inv']:.4f}")
        print(f"  Mean U(target): {metrics['mean_U_target']:.2f} kJ/mol")
        print(f"  Mean U(transformed): {metrics['mean_U_transformed']:.2f} kJ/mol")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        rama_path = os.path.join(
            save_dir,
            f"ramachandran_{int(self.dataset.source_temp)}_{int(self.dataset.target_temp)}.png"
        )
        self.plot_ramachandran_comparison(num_samples, rama_path)
        
        energy_path = os.path.join(
            save_dir,
            f"energy_validation_{int(self.dataset.source_temp)}_{int(self.dataset.target_temp)}.png"
        )
        self.plot_energy_distributions(num_samples, energy_path)
        
        print(f"\n{'='*70}")
        print(f"✅ Validation Complete!")
        print(f"{'='*70}\n")
        
        return metrics

