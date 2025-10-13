# 🚀 Google Colab Setup - Molecular Cross-Temperature Transport

## ✅ WORKING SETUP (Copy-Paste Ready)

### Cell 1: GPU Check
```python
import torch
print(f"🎮 GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
```

### Cell 2: Clone Repository
```python
# Clone and get latest changes
!git clone https://github.com/antoniofrancaib/tarflow-pt-molecular.git
%cd tarflow-pt-molecular
!git pull origin main  # Ensure latest code with GPU preset

!ls -lh datasets/AA/pt_AA.pt
```

### Cell 3: Install OpenMM (Working Method)
```python
# ====================
# WORKING: Miniforge23 + OpenMM
# ====================
print("📦 Installing Miniforge...")

import os

# Use Miniforge23 (stable version, no 404 errors)
!wget -q https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-Linux-x86_64.sh
!bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Set environment
os.environ['PATH'] = f"{os.environ['HOME']}/miniforge3/bin:" + os.environ['PATH']

# Install via mamba (faster than conda)
print("📦 Installing OpenMM...")
!$HOME/miniforge3/bin/mamba install -c conda-forge openmm openmmtools mdtraj -y

# Verify
print("\n✅ Verification:")
!$HOME/miniforge3/bin/python -c "import openmm; print(f'OpenMM: {openmm.version.short_version}')"
!$HOME/miniforge3/bin/python -c "import mdtraj; print(f'MDTraj: {mdtraj.__version__}')"

# Fix import statements
!sed -i 's/from simtk import openmm as mm/import openmm as mm/g' src/training/openmm_energy.py
!sed -i 's/from simtk import unit/from openmm import unit/g' src/training/openmm_energy.py
!sed -i 's/from simtk.openmm import app/from openmm import app/g' src/training/openmm_energy.py

print("\n🎉 Setup complete!")
```

### Cell 4: Test OpenMM Energy
```python
print("🧪 Testing OpenMM...")

!$HOME/miniforge3/bin/python << 'EOF'
import sys
sys.path.insert(0, '.')
import torch
from src.training.openmm_energy import compute_potential_energy

data = torch.load('datasets/AA/pt_AA.pt', weights_only=False)
test_coords = data[0, 0, :5, :]
energies = compute_potential_energy(test_coords)

print(f"✅ Energy computation works!")
print(f"   Sample energies: {[f'{e:.2f}' for e in energies.tolist()]} kJ/mol")
print(f"   Mean: {energies.mean():.2f} kJ/mol")
EOF
```

### Cell 5: Quick Training Test (100 epochs)
```python
print("🚀 Quick training test (100 epochs, ~15 min)...")

# Clear GPU cache and optimize memory allocation
import torch
import os
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Use GPU-optimized preset (6.5M params, balanced for T4)
!$HOME/miniforge3/bin/python main.py train-molecular \
    --preset aa_300_450_gpu \
    --epochs 100 \
    --validate
```

### Cell 6: Full Training (3000 epochs)
```python
print("🚀 Full training (3000 epochs, ~4-5 hours)...")

# Clear GPU cache and optimize memory allocation
import torch
import os
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# GPU-optimized preset: 6.5M params, batch_size=64
!$HOME/miniforge3/bin/python main.py train-molecular \
    --preset aa_300_450_gpu \
    --epochs 3000 \
    --validate
```

### Cell 7: Display Results
```python
from IPython.display import Image, display

print("📊 Training Results:")
display(Image('checkpoints/molecular_pt_aa_300_450/loss_curves_300_450.png'))
display(Image('plots/molecular_pt_aa_300_450/ramachandran_300_450.png'))
display(Image('plots/molecular_pt_aa_300_450/energy_validation_300_450.png'))
```

### Cell 8: Download Results
```python
!zip -r results.zip checkpoints/molecular_pt_aa_300_450 plots/molecular_pt_aa_300_450 -q

from google.colab import files
files.download('results.zip')

print("✅ Download complete!")
```

---

## 🔑 Key Points

1. **Uses Miniforge23** (stable, no 404 errors)
2. **All commands use** `$HOME/miniforge3/bin/python` 
3. **Fixes imports** automatically (simtk → openmm)
4. **GPU accelerated** PyTorch (from Colab)
5. **GPU-optimized model** (aa_300_450_gpu preset: ~4M params vs ~18M for CPU version)
6. **Self-contained** - no external dependencies

---

## ⏱️ Time Estimates

- Setup (Cells 1-4): ~3 minutes
- Quick test (Cell 5): ~15 minutes
- Full training (Cell 6): ~3-4 hours
- Download: ~30 seconds

**Note**: Batch sizes are optimized for T4 GPU (15GB). For larger GPUs (A100/V100), you can increase batch size.

---

## 📋 Troubleshooting

**Issue: "No module named 'openmm'"**
- Solution: Use `$HOME/miniforge3/bin/python` not just `python`

**Issue: CUDA out of memory**
- Solution: The setup now uses `aa_300_450_gpu` preset (smaller model, batch_size=32)
- If still OOM, reduce batch further: `--batch-size 16` or `--batch-size 8`
- Or restart runtime: Runtime → Restart runtime and try again
- For larger model (aa_300_450): Use A100 GPU or CPU

**Issue: Training too slow**
- Solution: Use T4 GPU or higher (Runtime → Change runtime type)
- T4 (15GB): Use aa_300_450_gpu preset (recommended)
- V100/A100 (16GB+): Can use aa_300_450 with `--batch-size 64`

**Issue: Want better model quality**
- The aa_300_450_gpu model is smaller but faster
- For best quality, use aa_300_450 preset on local machine (CPU with large RAM)
- Or use A100 GPU: `--preset aa_300_450 --batch-size 64`

---

**Just copy each cell and run in order!** 🚀

