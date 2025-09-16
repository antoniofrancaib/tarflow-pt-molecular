#!/usr/bin/env python3
"""
Minimalistic YAML-based configuration system for normalizing flow experiments
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

class ConfigManager:
    """Manages simple YAML-based experiment configurations"""
    
    def __init__(self, config_file: str = "experiments.yaml"):
        """Initialize with default config file"""
        self.config_file = CONFIG_DIR / config_file
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get complete preset configuration"""
        if preset_name not in self._config['presets']:
            available = list(self._config['presets'].keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
            
        # Start with default config
        config = deepcopy(self._config['default'])
        
        # Update with preset-specific settings
        preset = self._config['presets'][preset_name]
        
        # Merge configurations carefully
        for key, value in preset.items():
            if key in ['training', 'model_params', 'visualization']:
                if key not in config:
                    config[key] = {}
                config[key].update(value)
            else:
                config[key] = value
                
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return deepcopy(self._config['default'])
    
    def list_available(self) -> Dict[str, Any]:
        """List all available options"""
        return {
            'targets': ['two_moons', 'checkerboard'],
            'bases': ['gaussian', 'gaussian_trainable'],
            'models': ['simple', 'transformer'],
            'presets': list(self._config['presets'].keys())
        }
    
    def print_available(self):
        """Print all available options"""
        available = self.list_available()
        
        print("🎯 Available Targets: " + ", ".join(available['targets']))
        print("📊 Available Bases: " + ", ".join(available['bases']))
        print("🏗️ Available Models: " + ", ".join(available['models']))
        print("🎨 Available Presets: " + ", ".join(available['presets']))
    
    def print_config(self, config: Dict[str, Any]):
        """Pretty print a configuration"""
        print("🔧 Configuration:")
        print(f"   Target: {config.get('target')}")
        print(f"   Base: {config.get('base', 'gaussian')}")
        print(f"   Model: {config.get('model')}")
        
        if 'training' in config:
            training = config['training']
            print(f"   Training: {training.get('epochs')} epochs, "
                  f"batch_size={training.get('batch_size')}, "
                  f"lr={training.get('learning_rate')}")
        
        if 'visualization' in config:
            viz = config['visualization']
            domain = viz.get('domain', 3.5)
            print(f"   Domain: [-{domain}, {domain}] x [-{domain}, {domain}]")


# =============================================================================
# 🔧 DISTRIBUTION FACTORY FUNCTIONS
# =============================================================================

def create_target_distribution(name: str, config_manager: Optional[ConfigManager] = None):
    """Create target distribution from simple name"""
    # Import here to avoid circular imports
    try:
        from ..distributions import TwoMoons, CheckerboardDistribution
    except ImportError:
        # When running directly, adjust the path
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.distributions import TwoMoons, CheckerboardDistribution
    
    if name == 'two_moons':
        return TwoMoons()
    elif name == 'checkerboard':
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Get distribution parameters from config
        dist_config = config_manager._config.get('distributions', {}).get('checkerboard', {})
        return CheckerboardDistribution(
            grid_size=dist_config.get('grid_size', 8),
            domain_size=dist_config.get('domain_size', 3.0),
            high_density=dist_config.get('high_density', 1.0),
            low_density=dist_config.get('low_density', 0.1)
        )
    else:
        raise ValueError(f"Unknown target distribution: {name}. Available: two_moons, checkerboard")


def create_base_distribution(name: str, config_manager: Optional[ConfigManager] = None):
    """Create base distribution from simple name"""
    # Import here to avoid circular imports
    try:
        from ..distributions import DiagonalGaussian
    except ImportError:
        # When running directly, adjust the path
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.distributions import DiagonalGaussian
    
    if name == 'gaussian':
        return DiagonalGaussian(2, trainable=False)
    elif name == 'gaussian_trainable':
        return DiagonalGaussian(2, trainable=True)
    else:
        raise ValueError(f"Unknown base distribution: {name}. Available: gaussian, gaussian_trainable")


# =============================================================================
# 🔧 CONVENIENCE FUNCTIONS
# =============================================================================

def load_config(preset: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration (preset or default)"""
    config_manager = ConfigManager()
    
    if preset is not None:
        return config_manager.get_preset_config(preset)
    else:
        return config_manager.get_default_config()


def get_distributions(config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
    """Get target and base distributions from config"""
    if config_manager is None:
        config_manager = ConfigManager()
    
    target_dist = create_target_distribution(config['target'], config_manager)
    base_dist = create_base_distribution(config.get('base', 'gaussian'), config_manager)
    
    return target_dist, base_dist


# =============================================================================
# 🎯 MAIN DEMO FUNCTION
# =============================================================================

if __name__ == "__main__":
    # Demo the simplified YAML configuration system
    print("🎯 Minimalistic YAML Configuration System")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    # Show available options
    config_manager.print_available()
    
    print("\n" + "=" * 50)
    print("🔧 Default Configuration:")
    default_config = config_manager.get_default_config()
    config_manager.print_config(default_config)
    
    print("\n" + "=" * 50)
    print("🎨 Preset Configuration (two_moons):")
    preset_config = config_manager.get_preset_config('two_moons')
    config_manager.print_config(preset_config)
    
    print("\n" + "=" * 50)
    print("🧪 Testing Distribution Creation:")
    target_dist, base_dist = get_distributions(preset_config)
    print(f"   Target: {type(target_dist).__name__}")
    print(f"   Base: {type(base_dist).__name__}")
    
    print("\n✅ Minimalistic YAML Configuration System Working!")