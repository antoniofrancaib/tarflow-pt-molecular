#!/usr/bin/env python3
"""
Main driver script for normalizing flow experiments
"""

import argparse
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import SimpleAutoregressiveFlowModel, AutoregressiveNormalizingFlow
from src.distributions import DiagonalGaussian, TwoMoons, CheckerboardDistribution
from src.training import FlowTrainer
from src.training.evaluation import evaluate_model, plot_training_summary
from src.utils.yaml_config import ConfigManager, load_config, get_distributions, create_target_distribution, create_base_distribution


def create_model(model_type: str, config: dict, device: str):
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
            input_dim=2,
            num_layers=model_config.get('num_layers', 3),
            hidden_dim=model_config.get('hidden_dim', 64),
            transformer_layers=model_config.get('transformer_layers', 2)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
        model = create_model(args.model, config, device)
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
        model = create_model(args.model, config, device)
        
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
    elif args.command == 'eval':
        evaluate_model_cmd(args)
    else:
        parser.print_help()
        print("\n💡 Example commands:")
        print("   python main.py train --model simple --target two_moons --epochs 2000")
        print("   python main.py train --preset two_moons_simple")
        print("   python main.py train --model transformer --preset checkerboard_challenge")
        print("   python main.py eval --model simple --target two_moons --checkpoint model.pt")


if __name__ == "__main__":
    main()
