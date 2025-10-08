# 🧬 Molecular Cross-Temperature Transport Guide

## Overview

This module implements **normalizing flows for cross-temperature molecular transport**, enabling enhanced parallel tempering (PT) through learned bijective maps between Boltzmann distributions at different temperatures.

### Key Features

- ✅ **69D molecular coordinates** (alanine dipeptide: 23 atoms × 3)
- ✅ **Differentiable OpenMM energy evaluation** during training
- ✅ **Per-atom normalization** for numerical stability
- ✅ **Bidirectional transport loss** (source ↔ target)
- ✅ **Automatic validation** with Ramachandran plots and energy distributions
- ✅ **Loss curve tracking** for convergence monitoring

---

## Quick Start

### 1. Train 300K → 450K Transport

```bash
python main.py train-molecular --preset aa_300_450 --validate
```

This trains a scalable transformer flow to learn the transport map between 300K and 450K Boltzmann distributions.

### 2. Customize Training

```bash
python main.py train-molecular --preset aa_300_450 \
    --epochs 5000 \
    --lr 3e-4 \
    --batch-size 512 \
    --validate
```

### 3. Available Presets

| Preset | Source | Target | Gap | Difficulty |
|--------|--------|--------|-----|------------|
| `aa_300_450` | 300K | 450K | Δβ=133 | **Easy** (recommended start) |
| `aa_300_670` | 300K | 670K | Δβ=267 | Medium |
| `aa_300_1000` | 300K | 1000K | Δβ=334 | Hard |

---

## Architecture

### Data Flow

```
PT Trajectory (pt_AA.pt)
    [4 temps, 1 replica, 10K steps, 69 coords]
           ↓
    MolecularPTDataset
    - Per-atom normalization: x_norm = (x - μ_per_coord) / σ_per_coord
    - Paired sampling: (x_cold, x_hot)
           ↓
    ScalableTransformerFlow (69D)
    - 8 autoregressive flow layers
    - 192-dim embeddings, 8 attention heads
    - 5 transformer layers per flow
           ↓
    MolecularPTTrainer
    - Bidirectional NLL loss with OpenMM energies
    - Gradient clipping (max_norm=1.0)
    - Loss curve plotting
           ↓
    MolecularValidator
    - Ramachandran plot comparison
    - Energy distribution analysis
    - Reconstruction RMSD
```

### Loss Function

Bidirectional transport loss from `theory.md`:

```
L_NLL = E[β_cold · U(T⁻¹(x_hot))] - E[log|det J_inv|]
      + E[β_hot · U(T(x_cold))] - E[log|det J_fwd|]
```

Where:
- `T`: Forward flow (cold → hot)
- `T⁻¹`: Inverse flow (hot → cold)
- `U(x)`: OpenMM potential energy (kJ/mol)
- `β = 1/(kB·T)`: Inverse temperature (mol/kJ)

**Minimizing this loss maximizes PT swap acceptance rates!**

---

## Data Validation

### Ramachandran Plot Analysis

We verified data adequacy by plotting φ/ψ dihedral distributions:

**300K (10K samples):**
- ✅ Three clear conformational basins
- ✅ Peak densities: 100+ samples in core regions
- ✅ Well-defined α-helix and β-sheet regions

**450K (10K samples):**
- ✅ Same basins as 300K (thermal expansion observed)
- ✅ More diffuse distributions (higher entropy)
- ✅ Sufficient coverage for transport learning

**Conclusion:** 10K samples provide adequate coverage despite 69D space, because the molecule lives on a low-dimensional φ/ψ manifold.

---

## Configuration

### Default Config (`aa_300_450`)

```yaml
molecular_pt:
  aa_300_450:
    data_path: "datasets/AA/pt_AA.pt"
    source_temp_idx: 0  # 300K
    target_temp_idx: 1  # 450K
    pdb_path: "datasets/AA/ref.pdb"
    
    model: "scalable_transformer"
    use_energy: true  # Use OpenMM energy evaluation
    
    training:
      epochs: 3000
      batch_size: 256
      learning_rate: 5.0e-4
      eval_interval: 100
      weight_decay: 1.0e-5
    
    model_params:
      num_flow_layers: 8
      embed_dim: 192
      num_heads: 8
      num_transformer_layers: 5
      dropout: 0.1
    
    normalization:
      mode: "per_atom"  # 69 independent normalizations
```

**Model Size:** ~598K parameters for 10D → ~6.35M for 69D

---

## Output Files

### Checkpoints

```
checkpoints/molecular_pt_aa_300_450/
├── molecular_pt_300_450.pt          # Trained model weights
└── loss_curves_300_450.png          # Training convergence plot
```

### Validation Plots

```
plots/molecular_pt_aa_300_450/
├── ramachandran_300_450.png         # 3-panel comparison:
│                                     #   [Source | Transformed | Target]
└── energy_validation_300_450.png    # Energy distributions + RMSD
```

### Loss Curve Plot

4-panel figure tracking:
1. **Total Loss** (sum of forward + inverse)
2. **Forward vs Inverse Loss** (transport quality)
3. **Energy Components** (β·U terms)
4. **Jacobian Log Determinants** (volume preservation)

---

## Validation Metrics

The validator computes:

### Ramachandran Metrics
- φ/ψ dihedral angle distributions
- Visual comparison: source vs transformed vs target
- Basin overlap assessment

### Energy Metrics
- **Mean energy difference:** |U(transformed) - U(target)|
- **Energy distribution KL divergence** (approximate)
- **Reduced energy:** β·U evaluation

### Reconstruction Metrics
- **RMSD:** ||x - T⁻¹(T(x))||₂ in normalized units
- Cycle consistency: T → T⁻¹ → T accuracy

### Example Output

```
Quantitative Metrics:
  Reconstruction RMSD: 0.003215 ± 0.001847
  Energy Difference: 12.34 ± 8.76 kJ/mol
  Mean LogDet (fwd): 42.1234
  Mean LogDet (inv): -42.0987
  Mean U(target): -145.23 kJ/mol
  Mean U(transformed): -142.89 kJ/mol
```

---

## Implementation Details

### 1. OpenMM Energy Bridge (`src/training/openmm_energy.py`)

- **Adapted from `accelmd/`** for 23-atom systems
- **Differentiable:** Backpropagates forces as gradients
- **Batch processing:** Handles `[B, 69]` coordinates
- **Units:** Nanometers (input) → kJ/mol (output)

**Key Function:**
```python
U = compute_potential_energy(coords_nm)  # [B] energies in kJ/mol
reduced_energy = compute_reduced_energy(coords_nm, beta)  # β·U
```

### 2. Molecular Dataset (`src/distributions/molecular_pt.py`)

- **Per-atom normalization:** Each of 69 coords normalized independently
- **Temperature ladder:** [300K, 450K, 670K, 1000K] from PT trajectory
- **Denormalization:** Converts back to nm for OpenMM evaluation
- **Sampling:** Provides paired batches `(x_source, x_target)`

### 3. Trainer (`src/training/molecular_pt_trainer.py`)

- **Bidirectional loss:** Computes forward + inverse transport
- **Energy evaluation:** Calls OpenMM on denormalized coordinates
- **Gradient clipping:** `max_norm=1.0` for stability
- **Learning rate scheduling:** ReduceLROnPlateau (patience=200)

### 4. Validator (`src/training/molecular_validation.py`)

- **MDTraj integration:** Computes φ/ψ dihedrals
- **3-panel Ramachandran:** Source | Transformed | Target
- **Energy histograms:** Distribution overlap visualization
- **RMSD histograms:** Reconstruction quality

---

## Troubleshooting

### Issue: Training Loss Explodes

**Solution:**
1. Reduce learning rate: `--lr 1e-4`
2. Increase gradient clipping in trainer
3. Check for NaN in data (shouldn't happen with per-atom norm)

### Issue: Poor Ramachandran Match

**Possible causes:**
- Underfitting: Increase `num_flow_layers` or `embed_dim`
- Undertraining: Increase `--epochs`
- Too high LR: Reduce `--lr`

**Check:**
```python
# Look at loss curve - should converge to negative values
# Forward/Inverse losses should be balanced (~similar magnitude)
```

### Issue: High Reconstruction RMSD

**Diagnosis:**
- RMSD > 0.1: Invertibility issues (check model capacity)
- RMSD > 1.0: Serious problem (check normalization)

**Solution:**
- Increase model capacity (`embed_dim`, `num_transformer_layers`)
- Ensure per-atom normalization is working

### Issue: OpenMM Import Error

If you see `ImportError: simtk.openmm`:

```bash
conda install -c conda-forge openmm openmmtools mdtraj
```

---

## Extending to New Systems

### 1. Generate PT Trajectory

```python
python run_pt.py --system MY_SYSTEM --steps 10000
# Saves: datasets/MY_SYSTEM/pt_MY_SYSTEM.pt
```

### 2. Add Config

Edit `configs/experiments.yaml`:

```yaml
molecular_pt:
  my_system_300_450:
    data_path: "datasets/MY_SYSTEM/pt_MY_SYSTEM.pt"
    source_temp_idx: 0
    target_temp_idx: 1
    pdb_path: "datasets/MY_SYSTEM/ref.pdb"
    # ... rest of config
```

### 3. Train

```bash
python main.py train-molecular --preset my_system_300_450 --validate
```

---

## Theory References

- **Loss function derivation:** `docs/theory.md`
- **Change of variables:** `docs/genai-taxonomy/02a-change-of-variables-elegant-derivation.md`
- **Autoregressive flows:** `docs/genai-taxonomy/02b-normalizing-flows-architectures.md`

---

## Next Steps

### Immediate
1. ✅ Train 300K→450K transport (easiest gap)
2. ✅ Validate with Ramachandran plots
3. ✅ Check energy distribution overlap

### Advanced
1. Train 300K→670K (larger gap)
2. Implement flow-enhanced PT swaps
3. Measure swap acceptance improvement: α_flow vs α_naive
4. Test transferability to other dipeptides (AK, AS, etc.)

### Research
1. Conditional architecture: Single model for all temperature pairs
2. Equivariant layers for molecular symmetries
3. Multi-scale transport (coarse-grain → all-atom)

---

## Citation

If you use this code, please cite:

```bibtex
@software{tarflow-pt,
  title = {Transformer Autoregressive Flows for Parallel Tempering},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/tarflow-pt}
}
```

---

**Happy training! 🎯🧬**

