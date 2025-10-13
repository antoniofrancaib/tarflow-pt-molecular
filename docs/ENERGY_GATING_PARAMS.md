# Energy Gating Parameters Explained

## Current Configuration (After User Adjustments)

Based on observed training behavior where the model generates energies up to **4e10 kJ/mol** during early epochs, the energy gating parameters have been adjusted:

### Key Parameters

```python
E_cut = 500.0               # Soft regularization threshold
E_max_target = 1e8          # 100 million kJ/mol (final strict threshold)
E_max_initial = 1e12        # 1 trillion kJ/mol (ultra-lenient start)
warmup_epochs = 100         # Progressive decay period
min_learning_epochs = 50    # Keep max leniency for first 50 epochs
```

### Progressive Schedule (3 Stages)

**Stage 1: Free Learning (Epochs 0-50)**
- E_max = **1 trillion** kJ/mol (1e12)
- Purpose: Let model learn basic transformations without any restrictions
- Expected: Wild energies (1e10-1e17) are OK, model is exploring

**Stage 2: Exponential Decay (Epochs 50-100)**
- E_max decreases exponentially from 1e12 → 1e8
- Purpose: Gradually guide model toward physically reasonable energies
- Expected: Energies trend downward, occasional batch skips (< 10%)

**Schedule progression**:
```
Epoch 50:  E_max = 1,000,000,000,000 (1 trillion)
Epoch 60:  E_max = 398,107,171,000 (398 billion)
Epoch 70:  E_max = 158,489,319,000 (158 billion)
Epoch 80:  E_max = 63,095,734,000 (63 billion)
Epoch 90:  E_max = 25,118,864,000 (25 billion)
Epoch 100: E_max = 100,000,000 (100 million) ← target reached
```

**Stage 3: Strict Enforcement (Epochs 100+)**
- E_max = **100 million** kJ/mol (1e8)
- Purpose: Enforce physically reasonable conformations
- Expected: Energies stabilize, very few batch skips (< 5%)

## Why These Values?

### E_max_initial = 1e12 (1 trillion)
- Observation: Model generates 4e10 (40 billion) kJ/mol at epoch 45
- Need: Threshold must be higher than worst-case energies
- Safety margin: 1e12 is 25× higher than observed max
- Result: No batches skipped during free learning phase

### E_max_target = 1e8 (100 million)
- Typical peptide energies: -600 to +200 kJ/mol (equilibrated)
- Poor conformations: 1e3-1e5 kJ/mol (clashes/overlaps)
- Extreme unphysical: 1e6-1e8 kJ/mol (numerical issues)
- Threshold at 1e8: Strict enough to prevent overflow, lenient enough to allow learning
- Note: User-requested value (original was 1e5)

### warmup_epochs = 100
- Balances speed vs. stability
- Too fast (< 50): Model can't adapt, batches get skipped
- Too slow (> 300): Wastes time with overly lenient gating
- 100 epochs: Sweet spot for this model/dataset
- Note: User-requested value (original was 300)

### min_learning_epochs = 50
- First 50 epochs: Completely free exploration
- Model learns:
  - Basic coordinate transformations
  - Jacobian structure
  - Rough energy landscape
- No pressure from energy gating yet

## Expected Training Behavior

### Epochs 1-40: ✅ Normal
```
Loss: 346 → -134 (improving)
Energies: 138-207 kJ/mol (reasonable)
LogDet: Changing (learning)
Batches skipped: 0
```

### Epochs 40-50: ⚠️ Temporary Spike
```
Loss: May spike or show 0 occasionally
Energies: 1e10-1e11 kJ/mol (model exploring)
LogDet: Large values
Batches skipped: 0-2 per epoch (< 10%)
Status: NORMAL - model is still in free learning phase
```

### Epochs 50-100: ✅ Recovery
```
Loss: Should decrease steadily
Energies: Trending downward (1e10 → 1e8 → 1e6)
LogDet: Stabilizing
Batches skipped: < 10% per epoch
Status: Model learning to avoid extreme energies
```

### Epochs 100+: ✅ Convergence
```
Loss: Steady decrease or plateau
Energies: < 1e8 kJ/mol (within threshold)
LogDet: Stable, meaningful values
Batches skipped: < 5% per epoch
Status: Model trained, generating valid conformations
```

## Troubleshooting

### All batches still skipped after epoch 100
**Problem**: E_max_target too low for this dataset
**Fix**: Increase `E_max` in config to `1e9` or `1e10`
```yaml
E_max: 1000000000.0  # 1 billion kJ/mol
```

### Training very slow (many batch skips)
**Problem**: Exponential decay too fast
**Fix**: Increase `warmup_epochs` in `energy_gating.py` to 200
```python
warmup_epochs: int = 200
```

### Energies explode after epoch 100
**Problem**: E_max_target dropped too quickly
**Fix**: Use linear decay instead of exponential
- Modify `set_epoch()` in `energy_gating.py`
- Replace exponential with linear interpolation

### Loss stuck at 0 for 20+ epochs
**Problem**: All batches being skipped
**Quick fix**: 
1. Check warnings - are 32/32 samples exceeding E_max?
2. If yes, increase `E_max_initial` to `1e13` or `1e14`
3. Increase `min_learning_epochs` to 100

## Configuration in YAML

All energy gating parameters (except internal scheduling) are configurable:

```yaml
aa_300_450_gpu:
  use_energy_gating: true
  E_cut: 500.0
  E_max: 100000000.0  # This sets E_max_target
```

**Note**: `warmup_epochs`, `min_learning_epochs`, and `E_max_initial` are currently hardcoded in `src/training/energy_gating.py`. To make them configurable, you would need to:
1. Add them to the YAML config
2. Pass them to `EnergyGating()` constructor in `molecular_pt_trainer.py`

## References

- Progressive curriculum: Bengio et al., "Curriculum Learning" (2009)
- Energy-based stabilization: Noé et al., "Boltzmann Generators" (2019)
- Exponential schedules: Loshchilov & Hutter, "SGDR" (2017)

