# Progressive Energy Gating Fix

## Problem Diagnosis

### What Happened

Training showed promising initial progress (loss decreased 43% in 30 epochs), but then **completely collapsed**:

```
Epoch 1-30:  Loss 344 → 196 ✅ (learning normally)
Epoch 30-50: Batches start getting skipped ⚠️
Epoch 50+:   Loss = 0, all batches skipped ❌ (no learning)
```

**Root cause**: Progressive energy gating was **too aggressive**:
- E_max decreased from 10M → 10k kJ/mol over only 100 epochs (linear decay)
- By epoch 50, E_max had dropped to ~5M kJ/mol
- Model was still generating energies of 1e8-1e14 kJ/mol (normal for untrained model)
- Result: **ALL batches were skipped** → no gradient updates → stuck at zero loss

### Why This Matters

Energy gating is essential for stability, but if it's too strict too early:
1. **No learning signal**: All batches skipped → no gradients → model can't improve
2. **Catch-22**: Model needs to learn to generate better conformations, but gating prevents it from learning
3. **Training stalls**: Loss stays at 0, no progress for hundreds of epochs

## The Fix

### Three-Stage Progressive Schedule

**Old approach (too aggressive)**:
```
E_max = 10M - progress * (10M - 10k)  # Linear decay over 100 epochs
```

**New approach (lenient → strict)**:
```
Stage 1 (epochs 0-50):   E_max = 10 billion kJ/mol (extremely lenient)
Stage 2 (epochs 50-300): E_max decays exponentially from 10B → 100k
Stage 3 (epochs 300+):   E_max = 100k kJ/mol (final strict threshold)
```

### Key Improvements

1. **Longer warmup**: 300 epochs instead of 100
   - Gives model time to learn basic structure
   - Prevents premature batch skipping

2. **Higher final E_max**: 100k instead of 10k kJ/mol
   - Still prevents overflow (peptides rarely exceed 50k kJ/mol when trained)
   - Less likely to skip batches after warmup

3. **Exponential decay**: Stays lenient longer
   - Linear: halfway through → halfway between values
   - Exponential: stays high for most of warmup, then drops quickly
   - Example at epoch 150: Linear = 5M, Exponential = 3.2B

4. **Minimum learning period**: First 50 epochs at maximum leniency
   - Model can learn basic patterns without any gating pressure
   - Prevents immediate batch skipping at initialization

### Mathematical Details

Exponential decay in log-space:
```python
log_E_max = log(E_initial) + progress * (log(E_target) - log(E_initial))
E_max = exp(log_E_max)
```

This creates a smooth exponential curve:
- Progress 0%:   E_max = 10,000,000,000 kJ/mol
- Progress 25%:  E_max = 1,778,279,410 kJ/mol
- Progress 50%:  E_max = 316,227,766 kJ/mol
- Progress 75%:  E_max = 56,234,133 kJ/mol
- Progress 100%: E_max = 100,000 kJ/mol

## Expected Behavior

### Training Phases

**Phase 1 (epochs 0-50): Free exploration**
- E_max = 10B kJ/mol (essentially disabled)
- Model learns basic coordinate transformations
- Energies may be astronomical (1e8-1e15) but that's OK
- Few/no batches skipped

**Phase 2 (epochs 50-200): Gentle guidance**
- E_max exponentially decreases (still very high)
- Model learns to avoid extreme overlaps/clashes
- Occasional batch skips are normal (< 5%)
- Loss should steadily decrease

**Phase 3 (epochs 200-300): Progressive tightening**
- E_max approaches 100k kJ/mol
- Model refines to physically reasonable conformations
- Batch skip rate should be low (< 10%)
- Loss continues improving

**Phase 4 (epochs 300+): Strict enforcement**
- E_max = 100k kJ/mol (final threshold)
- Model should consistently generate valid structures
- Very few batches skipped (< 5%)
- Loss plateaus or slowly improves

### Monitoring Training

**Good signs** ✅:
- Loss decreases over time
- Energy values trend downward (e.g., 1e10 → 1e8 → 1e6)
- Batch skip rate < 10% after epoch 100
- LogDet values change (sign of learning)

**Warning signs** ⚠️:
- Loss stuck at 0 for > 20 epochs
- All batches being skipped (32/32)
- Energy values increasing over time
- No change in LogDet values

**Emergency fixes** 🚨:
If training still fails:
1. Increase `warmup_epochs` to 500 in config
2. Increase `E_max` to 500k or 1M in config
3. Check dataset quality (run validation script)
4. Reduce learning rate to 1e-4

## Configuration

All parameters are configurable in `configs/experiments.yaml`:

```yaml
aa_300_450_gpu:
  use_energy_gating: true
  E_cut: 500.0      # Soft regularization (typical peptide range)
  E_max: 100000.0   # Hard clamp after warmup (physically reasonable)
  
  training:
    # These affect progressive schedule:
    warmup_epochs: 50  # LR warmup (separate from energy gating)
    # Energy gating warmup is hardcoded to 300 epochs in energy_gating.py
```

**Note**: The energy gating `warmup_epochs` (300) is currently hardcoded in `src/training/energy_gating.py`. To make it configurable, you would need to pass it from the config to the `EnergyGating` constructor in `molecular_pt_trainer.py`.

## Results

With this fix, training should:
- ✅ Progress smoothly from epoch 1 onwards
- ✅ Show steady loss decrease (no sudden jumps to 0)
- ✅ Maintain reasonable batch skip rates (< 10%)
- ✅ Converge to physically valid conformations
- ✅ Complete full 1500 epochs without stalling

## References

This fix is inspired by:
- **Curriculum learning**: Start easy, gradually increase difficulty
- **Cosine annealing schedules**: Exponential decay for learning rates
- **Simulated annealing**: Gradually decrease temperature in optimization

Similar progressive strategies are used in:
- AlphaFold2: Progressive structure refinement
- Diffusion models: Progressive denoising schedules
- GAN training: Progressive growing of generator/discriminator

