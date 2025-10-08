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
!git clone https://github.com/antoniofrancaib/tarflow-pt-molecular.git
%cd tarflow-pt-molecular
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
print("🚀 Quick training test (100 epochs, ~10 min)...")

!$HOME/miniforge3/bin/python main.py train-molecular \
    --preset aa_300_450 \
    --epochs 100 \
    --batch-size 256 \
    --validate
```

### Cell 6: Full Training (3000 epochs)
```python
print("🚀 Full training (3000 epochs, ~2-3 hours)...")

!$HOME/miniforge3/bin/python main.py train-molecular \
    --preset aa_300_450 \
    --epochs 3000 \
    --lr 5e-4 \
    --batch-size 512 \
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
5. **Self-contained** - no external dependencies

---

## ⏱️ Time Estimates

- Setup (Cells 1-4): ~3 minutes
- Quick test (Cell 5): ~10 minutes
- Full training (Cell 6): ~2-3 hours
- Download: ~30 seconds

---

## 📋 Troubleshooting

**Issue: "No module named 'openmm'"**
- Solution: Use `$HOME/miniforge3/bin/python` not just `python`

**Issue: CUDA out of memory**
- Solution: Reduce batch size to 128

**Issue: Training too slow**
- Solution: Use T4 GPU or higher (Runtime → Change runtime type)

---

**Just copy each cell and run in order!** 🚀

