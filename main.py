#!/usr/bin/env python3
"""
Main driver script for normalizing flow experiments
"""

import argparse
import torch
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import SimpleAutoregressiveFlowModel, AutoregressiveNormalizingFlow
from src.models.scalable_transformer_flow import ScalableTransformerFlow
from src.distributions import DiagonalGaussian, TwoMoons, CheckerboardDistribution, HighDimGaussianMixture, HighDimSwissRoll
from src.training import FlowTrainer
from src.training.evaluation import evaluate_model, plot_training_summary
from src.utils.yaml_config import ConfigManager, load_config, get_distributions, create_target_distribution, create_base_distribution
from src.distributions.molecular_pt import MolecularPTDataset
from src.training.molecular_pt_trainer import MolecularPTTrainer
from src.training.molecular_validation import MolecularValidator


def create_model(model_type: str, config: dict, device: str, input_dim: int = 2):
    """Create model based on type and config"""
    if model_type == 'simple':
        model_config = config.get('model_params', {})
        return SimpleAutoregressiveFlowModel(
            num_layers=model_config.get('num_layers', 6),
            hidden_dim=model_config.get('hidden_dim', 128)
        ).to(device)
    elif model_type == 'transformer':
        model_config = config.get('transformer_params', {})
        return AutoregressiveNormalizingFlow(
            input_dim=input_dim,
            num_layers=model_config.get('num_layers', 3),
            hidden_dim=model_config.get('hidden_dim', 64),
            transformer_layers=model_config.get('transformer_layers', 2)
        ).to(device)
    elif model_type == 'scalable_transformer':
        model_config = config.get('model_params', {})
        return ScalableTransformerFlow(
            input_dim=input_dim,
            num_flow_layers=model_config.get('num_flow_layers', 6),
            embed_dim=model_config.get('embed_dim', 128),
            num_heads=model_config.get('num_heads', 8),
            num_transformer_layers=model_config.get('num_transformer_layers', 4),
            dropout=model_config.get('dropout', 0.1)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_high_dim_distributions(config: dict):
    """Create high-dimensional source and target distributions"""
    target_type = config['target']
    source_type = config.get('source', 'high_dim_gaussian')
    
    # Create target distribution
    if target_type == 'high_dim_gaussian_mixture':
        target_params = config['target_params']
        target_dist = HighDimGaussianMixture(**target_params)
    elif target_type == 'high_dim_swiss_roll':
        target_params = config['target_params']
        target_dist = HighDimSwissRoll(**target_params)
    else:
        raise ValueError(f"Unknown target distribution: {target_type}")
    
    # Create source distribution
    if source_type == 'high_dim_gaussian_mixture':
        source_params = config['source_params']
        source_dist = HighDimGaussianMixture(**source_params)
    elif source_type == 'high_dim_gaussian':
        source_params = config['source_params']
        # Convert to HighDimGaussianMixture with 1 component
        source_dist = HighDimGaussianMixture(
            dim=source_params['dim'],
            num_components=1,
            separation=0.0,
            component_std=source_params['component_std'],
            correlation_strength=source_params.get('correlation_strength', 0.0)
        )
    else:
        raise ValueError(f"Unknown source distribution: {source_type}")
    
    return source_dist, target_dist


class HighDimFlowTrainer:
    """Specialized trainer for high-dimensional flows"""
    
    def __init__(self, model, source_dist, target_dist, device: str):
        self.model = model.to(device)
        self.source_dist = source_dist
        self.target_dist = target_dist
        self.device = device
        
    def compute_transport_loss(self, batch_size: int = 256) -> torch.Tensor:
        """Compute transport loss: source samples should map to target distribution"""
        # Sample from source distribution
        x_source = self.source_dist.sample(batch_size, self.device)
        
        # Transform through flow
        z, log_det_forward = self.model.forward(x_source)
        
        # Compute loss: negative log likelihood under target
        try:
            # Method 1: Direct target log probability (if available)
            if hasattr(self.target_dist, 'log_prob'):
                target_log_prob = self.target_dist.log_prob(z)
                loss = -(target_log_prob + log_det_forward).mean()
            else:
                # Method 2: Use standard Gaussian + penalty for not matching target
                base_log_prob = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
                loss = -(base_log_prob + log_det_forward).mean()
                
        except Exception as e:
            # Fallback: minimize distance to target samples
            target_samples = self.target_dist.sample(batch_size, self.device)
            distance = torch.norm(z.unsqueeze(1) - target_samples.unsqueeze(0), dim=2).min(dim=1)[0]
            loss = distance.mean() - log_det_forward.mean() * 0.1
            
        return loss
    
    def train_step(self, optimizer, batch_size: int = 256) -> float:
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_transport_loss(batch_size)
        
        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            return float('nan')
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def evaluate_transport(self, num_samples: int = 1000) -> dict:
        """Evaluate transport quality"""
        self.model.eval()
        
        with torch.no_grad():
            # Sample from source
            source_samples = self.source_dist.sample(num_samples, self.device)
            
            # Transform through flow
            transformed_samples, _ = self.model.forward(source_samples)
            
            # Sample from target for comparison
            target_samples = self.target_dist.sample(num_samples, self.device)
            
            # Compute metrics
            transformed_np = transformed_samples.cpu().numpy()
            target_np = target_samples.cpu().numpy()
            source_np = source_samples.cpu().numpy()
            
            # Statistical distances (approximate)
            mean_diff = np.linalg.norm(transformed_np.mean(axis=0) - target_np.mean(axis=0))
            std_diff = np.abs(transformed_np.std() - target_np.std())
            
            metrics = {
                'source_samples': source_np,
                'transformed_samples': transformed_np,
                'target_samples': target_np,
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'source_mean': source_np.mean(axis=0),
                'transformed_mean': transformed_np.mean(axis=0),
                'target_mean': target_np.mean(axis=0),
                'source_std': source_np.std(axis=0),
                'transformed_std': transformed_np.std(axis=0),
                'target_std': target_np.std(axis=0)
            }
            
        return metrics
    
    def train(self, config: dict, save_dir: str = 'results_high_dim') -> tuple:
        """Full training loop"""
        import torch.optim as optim
        
        os.makedirs(save_dir, exist_ok=True)
        
        training_config = config['training']
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.8, verbose=True
        )
        
        # Training parameters
        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        eval_interval = training_config.get('eval_interval', 200)
        
        print(f"Training for {epochs} epochs, batch_size={batch_size}")
        
        loss_history = []
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training step
            loss = self.train_step(optimizer, batch_size)
            loss_history.append(loss)
            
            # Update learning rate
            if not np.isnan(loss):
                scheduler.step(loss)
            
            # Evaluation and plotting
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                if np.isnan(loss):
                    print(f"Loss: NaN (skipped)")
                else:
                    print(f"Loss: {loss:.6f}")
                    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Evaluate transport
                metrics = self.evaluate_transport(1000)
                print(f"Mean difference: {metrics['mean_difference']:.4f}")
                print(f"Std difference: {metrics['std_difference']:.4f}")
                
                # Plot results
                self.plot_transport_results(metrics, epoch + 1, save_dir)
        
        return self.model, loss_history, metrics
    
    def plot_transport_results(self, metrics: dict, epoch: int, save_dir: str):
        """Plot transport results"""
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        source_samples = metrics['source_samples']
        transformed_samples = metrics['transformed_samples']
        target_samples = metrics['target_samples']
        
        dim = source_samples.shape[1]
        
        # PCA projections for visualization
        combined = np.vstack([source_samples, transformed_samples, target_samples])
        pca = PCA(n_components=min(3, dim))
        combined_pca = pca.fit_transform(combined)
        
        n = len(source_samples)
        source_pca = combined_pca[:n]
        transformed_pca = combined_pca[n:2*n]
        target_pca = combined_pca[2*n:]
        
        # Plot 1: PCA projection
        axes[0, 0].scatter(source_pca[:, 0], source_pca[:, 1], alpha=0.6, s=2, c='blue', label='Source')
        axes[0, 0].scatter(transformed_pca[:, 0], transformed_pca[:, 1], alpha=0.6, s=2, c='green', label='Transformed')
        axes[0, 0].scatter(target_pca[:, 0], target_pca[:, 1], alpha=0.6, s=2, c='red', label='Target')
        axes[0, 0].set_title('PCA: PC1 vs PC2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: First two raw dimensions (if available)
        if dim >= 2:
            axes[0, 1].scatter(source_samples[:, 0], source_samples[:, 1], alpha=0.6, s=2, c='blue', label='Source')
            axes[0, 1].scatter(transformed_samples[:, 0], transformed_samples[:, 1], alpha=0.6, s=2, c='green', label='Transformed')
            axes[0, 1].scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.6, s=2, c='red', label='Target')
            axes[0, 1].set_title('Raw: X₁ vs X₂')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Per-dimension means
        dims_to_plot = min(20, dim)
        x_dims = range(dims_to_plot)
        
        axes[0, 2].plot(x_dims, metrics['source_mean'][:dims_to_plot], 'b-o', label='Source', markersize=3)
        axes[0, 2].plot(x_dims, metrics['transformed_mean'][:dims_to_plot], 'g-o', label='Transformed', markersize=3)
        axes[0, 2].plot(x_dims, metrics['target_mean'][:dims_to_plot], 'r-o', label='Target', markersize=3)
        axes[0, 2].set_title('Per-Dimension Means')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Per-dimension stds
        axes[1, 0].plot(x_dims, metrics['source_std'][:dims_to_plot], 'b-o', label='Source', markersize=3)
        axes[1, 0].plot(x_dims, metrics['transformed_std'][:dims_to_plot], 'g-o', label='Transformed', markersize=3)
        axes[1, 0].plot(x_dims, metrics['target_std'][:dims_to_plot], 'r-o', label='Target', markersize=3)
        axes[1, 0].set_title('Per-Dimension Standard Deviations')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Distance distributions
        source_dists = np.linalg.norm(source_samples[::10, None] - source_samples[None, ::10], axis=2)
        source_dists = source_dists[np.triu_indices_from(source_dists, k=1)]
        
        transformed_dists = np.linalg.norm(transformed_samples[::10, None] - transformed_samples[None, ::10], axis=2)
        transformed_dists = transformed_dists[np.triu_indices_from(transformed_dists, k=1)]
        
        target_dists = np.linalg.norm(target_samples[::10, None] - target_samples[None, ::10], axis=2)
        target_dists = target_dists[np.triu_indices_from(target_dists, k=1)]
        
        axes[1, 1].hist(source_dists, bins=30, alpha=0.7, density=True, label='Source')
        axes[1, 1].hist(transformed_dists, bins=30, alpha=0.7, density=True, label='Transformed')
        axes[1, 1].hist(target_dists, bins=30, alpha=0.7, density=True, label='Target')
        axes[1, 1].set_title('Pairwise Distance Distributions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Quality metrics
        metrics_vals = [
            metrics['mean_difference'],
            metrics['std_difference'],
            pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0,
            pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
        ]
        metrics_labels = ['Mean\nDiff', 'Std\nDiff', 'PC1\nVar', 'PC2\nVar']
        
        bars = axes[1, 2].bar(metrics_labels, metrics_vals, alpha=0.7)
        axes[1, 2].set_title('Transport Quality Metrics')
        axes[1, 2].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, val in zip(bars, metrics_vals):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'High-Dimensional Transport Results - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, f'transport_results_epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Show only at end of training
        if epoch % 1000 == 0 or epoch < 100:
            plt.show()
        else:
            plt.close()


def train_high_dim_model(args):
    """Train high-dimensional transport flow"""
    print("🚀 High-Dimensional Transport Training")
    print("=" * 60)
    
    # Load high-dim configuration
    config_file = 'configs/high_dim_experiments.yaml'
    if not os.path.exists(config_file):
        print(f"❌ High-dim config file not found: {config_file}")
        return
        
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if args.preset not in config_data['high_dim_presets']:
        available = list(config_data['high_dim_presets'].keys())
        print(f"❌ Unknown high-dim preset: {args.preset}")
        print(f"Available presets: {available}")
        return
    
    config = config_data['high_dim_presets'][args.preset]
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        
    print(f"📋 Using preset: {args.preset}")
    print(f"📊 Problem: {config['target_params']['dim']}D transport")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create distributions
    try:
        source_dist, target_dist = create_high_dim_distributions(config)
        input_dim = config['target_params']['dim']
        print(f"Created {input_dim}D source → target transport problem")
    except Exception as e:
        print(f"❌ Error creating distributions: {e}")
        return
    
    # Create model
    try:
        model = create_model('scalable_transformer', config, device, input_dim)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Use the high-dimensional trainer (defined below)
    
    trainer = HighDimFlowTrainer(model, source_dist, target_dist, device)
    
    # Train
    save_dir = f'results_high_dim/{args.preset}'
    try:
        trained_model, loss_history, final_metrics = trainer.train(config, save_dir)
        
        print(f"\n✅ Training completed!")
        print(f"Final mean difference: {final_metrics['mean_difference']:.4f}")
        print(f"Final std difference: {final_metrics['std_difference']:.4f}")
        
        # Save model
        if args.save_model:
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'config': config,
                'loss_history': loss_history,
                'final_metrics': final_metrics
            }, args.save_model)
            print(f"💾 Model saved to: {args.save_model}")
        else:
            # Default save location
            save_path = os.path.join(save_dir, 'final_model.pt')
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'config': config, 
                'loss_history': loss_history,
                'final_metrics': final_metrics
            }, save_path)
            print(f"💾 Model saved to: {save_path}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def train_molecular_pt(args):
    """Train molecular cross-temperature transport flow"""
    print("🧬 Molecular Cross-Temperature Transport Training")
    print("=" * 60)
    
    # Load molecular PT configuration
    config_file = 'configs/experiments.yaml'
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return
        
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Check if molecular_pt section exists
    if 'molecular_pt' not in config_data:
        print(f"❌ No molecular_pt configurations found in {config_file}")
        return
    
    # Check molecular_pt presets first, then ablation_studies
    if args.preset in config_data.get('molecular_pt', {}):
        config = config_data['molecular_pt'][args.preset]
    elif args.preset in config_data.get('ablation_studies', {}):
        config = config_data['ablation_studies'][args.preset]
        print("🔬 Using ABLATION STUDY preset")
    else:
        available_pt = list(config_data.get('molecular_pt', {}).keys())
        available_ablation = list(config_data.get('ablation_studies', {}).keys())
        print(f"❌ Unknown preset: {args.preset}")
        print(f"Available molecular_pt presets: {available_pt}")
        print(f"Available ablation_studies presets: {available_ablation}")
        return
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        
    print(f"📋 Using preset: {args.preset}")
    print(f"📊 Transport: {config.get('source_temp_idx', 0)} → {config.get('target_temp_idx', 1)}")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create molecular dataset
    try:
        dataset = MolecularPTDataset(
            data_path=config.get('data_path', 'datasets/AA/pt_AA.pt'),
            source_temp_idx=config.get('source_temp_idx', 0),
            target_temp_idx=config.get('target_temp_idx', 1),
            normalize=True,
            normalize_mode=config.get('normalization', {}).get('mode', 'per_atom'),
        )
        print(f"✓ Dataset loaded: {dataset.source_temp}K → {dataset.target_temp}K")
        print(f"  Samples: {len(dataset)}")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model (69D for alanine dipeptide)
    try:
        input_dim = 69
        model = create_model(
            config.get('model', 'scalable_transformer'),
            config,
            device,
            input_dim
        )
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    trainer = MolecularPTTrainer(
        model=model,
        dataset=dataset,
        device=device,
        use_energy=config.get('use_energy', True),
        use_energy_gating=config.get('use_energy_gating', True),
        E_cut=config.get('E_cut', 500.0),
        E_max=config.get('E_max', 10000.0),
    )
    
    # Train
    save_dir = f'checkpoints/molecular_pt_{args.preset}'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print("\n🚀 Starting training...")
        trained_model, history = trainer.train(config, save_dir)
        
        print(f"\n✅ Training completed!")
        print(f"   Final loss: {history[-1]['total_loss']:.4f}")
        
        # Run validation
        if args.validate:
            print("\n🔬 Running validation...")
            validator = MolecularValidator(
                model=trained_model,
                dataset=dataset,
                pdb_path=config.get('pdb_path', 'datasets/AA/ref.pdb'),
                device=device,
            )
            
            metrics = validator.full_validation(
                num_samples=2000,
                save_dir=f'plots/molecular_pt_{args.preset}'
            )
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def train_model(args):
    """Train a normalizing flow model"""
    print("🚀 Training Mode")
    print("=" * 60)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Get configuration
    if args.preset:
        available_presets = config_manager.list_available()['presets']
        if args.preset not in available_presets:
            print(f"❌ Unknown preset: {args.preset}")
            print(f"Available presets: {available_presets}")
            return
        config = config_manager.get_preset_config(args.preset)
        print(f"📋 Using preset: {args.preset}")
    else:
        config = config_manager.get_default_config()
    
    # Override config with command line arguments
    if args.target:
        config['target'] = args.target
    if args.epochs:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
    if args.lr:
        if 'training' not in config:
            config['training'] = {}
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        if 'training' not in config:
            config['training'] = {}
        config['training']['batch_size'] = args.batch_size
    
    # Add model type to config
    config['model_type'] = args.model
    
    config_manager.print_config(config)
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create distributions
    try:
        target_dist = create_target_distribution(config['target'], config_manager)
        base_dist = create_base_distribution(config.get('base', 'gaussian'), config_manager)
    except Exception as e:
        print(f"❌ Error creating distributions: {e}")
        return
    
    # Create model
    try:
        # For 2D distributions, input_dim is 2
        input_dim = 2
        model = create_model(args.model, config, device, input_dim)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Create trainer and train
    trainer = FlowTrainer(model, target_dist, base_dist, device)
    
    try:
        trained_model, loss_history = trainer.train(config)
        
        # Plot training summary
        plot_training_summary(loss_history, config)
        
        # Save model if requested
        if args.save_model:
            torch.save(trained_model.state_dict(), args.save_model)
            print(f"💾 Model saved to: {args.save_model}")
        
        # Final evaluation
        metrics = evaluate_model(trained_model, target_dist, base_dist, config, device)
        print(f"\n📊 Final Metrics:")
        print(f"   Target log likelihood: {metrics['target_log_likelihood']:.4f}")
        print(f"   Model samples mean: {metrics['model_samples_mean']}")
        print(f"   Target samples mean: {metrics['target_samples_mean']}")
        if metrics['wasserstein_distance'] is not None:
            print(f"   Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")


def evaluate_model_cmd(args):
    """Evaluate a trained model"""
    print("📊 Evaluation Mode")
    print("=" * 60)
    
    if not args.checkpoint:
        print("❌ Checkpoint path required for evaluation")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Get configuration
    if args.preset:
        config = config_manager.get_preset_config(args.preset)
    else:
        config = config_manager.get_default_config()
    
    # Override config with command line arguments
    if args.target:
        config['target'] = args.target
    
    config['model_type'] = args.model
    
    config_manager.print_config(config)
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create distributions and model
    try:
        target_dist = create_target_distribution(config['target'], config_manager)
        base_dist = create_base_distribution(config.get('base', 'gaussian'), config_manager)
        # For 2D distributions, input_dim is 2  
        input_dim = 2
        model = create_model(args.model, config, device, input_dim)
        
        # Load checkpoint
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"📁 Loaded model from: {args.checkpoint}")
        
        # Evaluate
        metrics = evaluate_model(model, target_dist, base_dist, config, device)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   Target log likelihood: {metrics['target_log_likelihood']:.4f}")
        print(f"   Model samples mean: {metrics['model_samples_mean']}")
        print(f"   Model samples std: {metrics['model_samples_std']}")
        print(f"   Target samples mean: {metrics['target_samples_mean']}")
        print(f"   Target samples std: {metrics['target_samples_std']}")
        if metrics['wasserstein_distance'] is not None:
            print(f"   Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
        
        # Plot results
        from src.utils.plotting import plot_results
        plot_results(model, target_dist, base_dist, device, 0, config)
        
        print("\n✅ Evaluation completed!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Normalizing Flow Training and Evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a normalizing flow')
    train_parser.add_argument('--model', choices=['simple', 'transformer'], default='simple',
                            help='Model architecture to use')
    train_parser.add_argument('--target', 
                            choices=['two_moons', 'checkerboard'],
                            help='Target distribution')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--preset', 
                            help='Use a preset configuration')
    train_parser.add_argument('--save-model', type=str, help='Path to save trained model')
    train_parser.add_argument('--output-dir', type=str, help='Output directory for plots')
    
    # High-dimensional training command
    highdim_parser = subparsers.add_parser('train-highdim', help='Train high-dimensional transport flows')
    highdim_parser.add_argument('--preset', type=str, required=True,
                               choices=['10d_gaussian_to_mixture', '25d_uncorr_to_corr', 
                                       '50d_gaussian_to_manifold', '100d_simple_to_complex'],
                               help='High-dimensional preset configuration')
    highdim_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    highdim_parser.add_argument('--lr', type=float, help='Learning rate')
    highdim_parser.add_argument('--batch-size', type=int, help='Batch size')
    highdim_parser.add_argument('--save-model', type=str, help='Path to save trained model')
    
    # Molecular PT training command
    molecular_parser = subparsers.add_parser('train-molecular', help='Train molecular cross-temperature transport')
    molecular_parser.add_argument('--preset', type=str, required=True,
                                 choices=['aa_300_450', 'aa_300_450_gpu', 'aa_300_670', 'aa_300_1000'],
                                 help='Molecular PT preset configuration')
    molecular_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    molecular_parser.add_argument('--lr', type=float, help='Learning rate')
    molecular_parser.add_argument('--batch-size', type=int, help='Batch size')
    molecular_parser.add_argument('--validate', action='store_true', 
                                 help='Run validation after training (Ramachandran plots, energy distributions)')
    molecular_parser.add_argument('--save-model', type=str, help='Path to save trained model')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--model', choices=['simple', 'transformer'], required=True,
                           help='Model architecture to use')
    eval_parser.add_argument('--target',
                           choices=['two_moons', 'checkerboard'],
                           help='Target distribution')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--preset',
                           help='Use a preset configuration')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'train-highdim':
        train_high_dim_model(args)
    elif args.command == 'train-molecular':
        train_molecular_pt(args)
    elif args.command == 'eval':
        evaluate_model_cmd(args)
    else:
        parser.print_help()
        print("\n💡 Example commands:")
        print("   # 2D experiments:")
        print("   python main.py train --model simple --target two_moons --epochs 2000")
        print("   python main.py train --preset two_moons")
        print("   python main.py train --model transformer --preset checkerboard")
        print("")
        print("   # High-dimensional experiments:")
        print("   python main.py train-highdim --preset 10d_gaussian_to_mixture")
        print("   python main.py train-highdim --preset 25d_uncorr_to_corr --epochs 3000")
        print("   python main.py train-highdim --preset 50d_gaussian_to_manifold")
        print("")
        print("   # Molecular cross-temperature transport:")
        print("   python main.py train-molecular --preset aa_300_450 --validate")
        print("   python main.py train-molecular --preset aa_300_450 --epochs 3000 --lr 5e-4")
        print("   python main.py train-molecular --preset aa_300_670 --validate")
        print("")
        print("   # Evaluation:")
        print("   python main.py eval --model simple --target two_moons --checkpoint model.pt")


if __name__ == "__main__":
    main()
