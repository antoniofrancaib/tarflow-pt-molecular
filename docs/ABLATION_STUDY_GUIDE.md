# 🔬 Ablation Study Guide

## Overview

Ablation studies systematically remove features to understand their individual contribution to model performance. This guide shows how to run controlled experiments using the pre-configured ablation presets.

## Why Ablation Studies Matter

Your training improvements included:
1. ✅ **Energy gating** - prevents gradient explosions
2. ✅ **LR warmup** - stable initialization
3. ✅ **LR scheduler** - adapts to plateaus
4. ✅ **Best model checkpointing** - saves optimal state
5. ✅ **Gradient clipping** - prevents gradient explosions

**Question**: Which of these actually help? How much does each contribute?

**Answer**: Run ablation studies!

## Pre-Configured Ablation Presets

All presets are in `configs/experiments.yaml` under `ablation_studies`:

### 1. **baseline_gpu** (ALL features ON)
```yaml
use_energy_gating: true
use_warmup: true
use_scheduler: true
save_best_model: true
gradient_clip_norm: 1.0
```

**Expected**: Best performance, most stable training

### 2. **no_energy_gating_gpu**
```yaml
use_energy_gating: false  # ABLATED
# All other features ON
```

**Tests**: Does energy gating prevent divergence?

**Expected**: May diverge like your epoch 400→500 explosion

### 3. **no_warmup_gpu**
```yaml
use_warmup: false  # ABLATED
# All other features ON
```

**Tests**: Does warmup prevent bad initialization?

**Expected**: Higher initial loss (~486T vs ~5T)

### 4. **no_scheduler_gpu**
```yaml
use_scheduler: false  # ABLATED
# All other features ON
```

**Tests**: Does LR adaptation help convergence?

**Expected**: May get stuck at higher final loss

### 5. **no_grad_clip_gpu**
```yaml
gradient_clip_norm: 0.0  # ABLATED
# All other features ON
```

**Tests**: Does gradient clipping stabilize training?

**Expected**: May see occasional gradient spikes

### 6. **minimal_gpu** (ALL features OFF)
```yaml
use_energy_gating: false
use_warmup: false
use_scheduler: false
save_best_model: false
gradient_clip_norm: 0.0
```

**Tests**: Baseline without any improvements

**Expected**: Likely to diverge or converge poorly

## How to Run Ablation Studies

### Quick Example

```bash
# Baseline (all features)
python main.py train-molecular --preset baseline_gpu --validate

# No energy gating
python main.py train-molecular --preset no_energy_gating_gpu --validate

# No warmup
python main.py train-molecular --preset no_warmup_gpu --validate
```

### Complete Ablation Suite

```bash
# Create results directory
mkdir -p ablation_results

# 1. Baseline
python main.py train-molecular --preset baseline_gpu --validate
cp -r checkpoints/molecular_pt_baseline_gpu ablation_results/
cp -r plots/molecular_pt_baseline_gpu ablation_results/

# 2. No energy gating
python main.py train-molecular --preset no_energy_gating_gpu --validate
cp -r checkpoints/molecular_pt_no_energy_gating_gpu ablation_results/
cp -r plots/molecular_pt_no_energy_gating_gpu ablation_results/

# 3. No warmup
python main.py train-molecular --preset no_warmup_gpu --validate
cp -r checkpoints/molecular_pt_no_warmup_gpu ablation_results/
cp -r plots/molecular_pt_no_warmup_gpu ablation_results/

# 4. No scheduler
python main.py train-molecular --preset no_scheduler_gpu --validate
cp -r checkpoints/molecular_pt_no_scheduler_gpu ablation_results/
cp -r plots/molecular_pt_no_scheduler_gpu ablation_results/

# 5. No gradient clipping
python main.py train-molecular --preset no_grad_clip_gpu --validate
cp -r checkpoints/molecular_pt_no_grad_clip_gpu ablation_results/
cp -r plots/molecular_pt_no_grad_clip_gpu ablation_results/

# 6. Minimal (everything off)
python main.py train-molecular --preset minimal_gpu --validate
cp -r checkpoints/molecular_pt_minimal_gpu ablation_results/
cp -r plots/molecular_pt_minimal_gpu ablation_results/
```

## Metrics to Compare

### 1. **Final Loss**
```
Baseline:          15-50
No energy gating:  ???  (may diverge!)
No warmup:         20-60
No scheduler:      50-100
No grad clip:      15-60
Minimal:           ???  (likely diverges)
```

### 2. **Training Stability**
Check loss curves for:
- Smoothness (fewer spikes)
- Monotonic decrease (no explosions)
- Convergence speed

### 3. **Best Epoch**
When did best model occur?
```
Baseline:          Late (epoch 2500+)
No energy gating:  Early (before divergence)
No warmup:         Late (after recovering from bad start)
No scheduler:      Middle (stuck without LR reduction)
```

### 4. **Energy Validation**
From validation plots:
```
Baseline:          Small energy difference (< 50 kJ/mol)
No energy gating:  Large or NaN (invalid coordinates)
No warmup:         Similar to baseline (if converged)
```

### 5. **Ramachandran Quality**
Visual inspection:
- **Good**: Structured regions (α-helix, β-sheet)
- **Bad**: Uniform noise or invalid regions

## Creating Custom Ablations

To test your own combinations, add to `configs/experiments.yaml`:

```yaml
ablation_studies:
  custom_test:
    data_path: "datasets/AA/pt_AA.pt"
    source_temp_idx: 0
    target_temp_idx: 1
    pdb_path: "datasets/AA/ref.pdb"
    model: "scalable_transformer"
    use_energy: true
    
    # Customize these!
    use_energy_gating: true
    E_cut: 300.0  # More aggressive than default 500
    E_max: 5000.0  # Lower threshold
    
    training:
      epochs: 1500
      batch_size: 64
      learning_rate: 1.0e-3  # Higher LR
      use_warmup: true
      warmup_epochs: 100  # Longer warmup
      use_scheduler: true
      scheduler_patience: 30  # More aggressive
      scheduler_factor: 0.3  # Stronger reduction
      save_best_model: true
      gradient_clip_norm: 0.5  # Tighter clipping
      
    model_params:
      num_flow_layers: 6
      embed_dim: 128
      num_heads: 8
      num_transformer_layers: 4
      dropout: 0.1
      
    normalization:
      mode: "per_atom"
```

Then run:
```bash
python main.py train-molecular --preset custom_test --validate
```

## Analysis Template

Create a comparison table:

| Preset | Final Loss | Best Epoch | Converged? | Energy Diff | Notes |
|--------|-----------|------------|------------|-------------|-------|
| baseline_gpu | 23.5 | 2850 | ✅ | 12.3 kJ/mol | Stable, best performance |
| no_energy_gating_gpu | DIV | 450 | ❌ | NaN | Diverged at epoch 500 |
| no_warmup_gpu | 45.2 | 2900 | ✅ | 15.1 kJ/mol | Slower convergence |
| no_scheduler_gpu | 87.3 | 1200 | ⚠️ | 28.4 kJ/mol | Stuck, didn't reduce LR |
| no_grad_clip_gpu | 31.2 | 2700 | ✅ | 13.8 kJ/mol | Few gradient spikes |
| minimal_gpu | DIV | 350 | ❌ | NaN | Multiple issues |

## Expected Findings

Based on your observed failures:

### **Energy Gating** - CRITICAL
- **With**: Stable training, prevents epoch 400→500 explosion
- **Without**: Catastrophic divergence when energies explode
- **Importance**: ⭐⭐⭐⭐⭐

### **Warmup** - VERY IMPORTANT
- **With**: Initial loss ~5T, smooth start
- **Without**: Initial loss ~486T, wild early epochs
- **Importance**: ⭐⭐⭐⭐

### **LR Scheduler** - IMPORTANT
- **With**: Adapts to plateaus, reaches better minimum
- **Without**: May get stuck at higher loss
- **Importance**: ⭐⭐⭐

### **Gradient Clipping** - HELPFUL
- **With**: Prevents occasional gradient spikes
- **Without**: Usually fine with energy gating, occasional instability
- **Importance**: ⭐⭐

### **Best Model Checkpointing** - SAFETY NET
- **With**: Keeps best model even if divergence
- **Without**: Lost good epoch 400 model when epoch 500 diverged
- **Importance**: ⭐⭐⭐ (for safety)

## Publication-Ready Results

For papers/reports, generate:

### 1. Loss Curve Comparison
Overlay all loss curves on one plot

### 2. Feature Importance Ranking
Bar chart showing contribution of each feature

### 3. Statistical Significance
Run each ablation 3-5 times, report mean ± std

### 4. Compute Budget Analysis
Feature | Training Time | Final Loss | Cost-Benefit
--------|--------------|------------|-------------
Energy gating | +1% time | -95% loss | HIGH
Warmup | +0% time | -50% loss | HIGH
Scheduler | +0% time | -40% loss | MEDIUM

## Automation Script

Save this as `run_ablations.sh`:

```bash
#!/bin/bash

PRESETS=(
  "baseline_gpu"
  "no_energy_gating_gpu"
  "no_warmup_gpu"
  "no_scheduler_gpu"
  "no_grad_clip_gpu"
  "minimal_gpu"
)

for preset in "${PRESETS[@]}"; do
  echo "========================================="
  echo "Running ablation: $preset"
  echo "========================================="
  
  python main.py train-molecular --preset $preset --validate
  
  # Copy results
  mkdir -p ablation_results/$preset
  cp -r checkpoints/molecular_pt_$preset ablation_results/$preset/checkpoints
  cp -r plots/molecular_pt_$preset ablation_results/$preset/plots
  
  echo "Completed: $preset"
  echo ""
done

echo "All ablations complete!"
echo "Results in: ablation_results/"
```

Usage:
```bash
chmod +x run_ablations.sh
./run_ablations.sh
```

## Tips for Success

1. **Run baseline first** - establish performance target
2. **Check for divergence** - monitor training, stop if obvious failure
3. **Use consistent seeds** - for reproducibility (add to config if needed)
4. **Document everything** - note any anomalies during training
5. **Compare loss curves** - visual inspection often reveals issues
6. **Validate thoroughly** - energy plots and Ramachandran plots matter!

## Example Workflow

```bash
# Day 1: Core ablations
python main.py train-molecular --preset baseline_gpu --validate
python main.py train-molecular --preset no_energy_gating_gpu --validate
python main.py train-molecular --preset no_warmup_gpu --validate

# Day 2: Secondary ablations
python main.py train-molecular --preset no_scheduler_gpu --validate
python main.py train-molecular --preset no_grad_clip_gpu --validate

# Day 3: Minimal + analysis
python main.py train-molecular --preset minimal_gpu --validate
# Analyze results, create comparison plots

# Day 4: Paper writing
# Use results to justify design choices
```

## Summary

**Ablation studies answer**:
- ❓ Which features are essential?
- ❓ Which are nice-to-have?
- ❓ What's the minimum viable configuration?
- ❓ How do features interact?

**Your config system makes this easy**:
- ✅ All features configurable
- ✅ Pre-built ablation presets
- ✅ Easy to add custom tests
- ✅ Reproducible experiments

**Run the ablations, publish the results!** 🎉

