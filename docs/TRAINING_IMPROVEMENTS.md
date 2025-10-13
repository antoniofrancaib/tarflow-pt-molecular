# 🚀 Training Improvements Guide

## What's New

### 1. 📊 **Live Progress Metrics in Progress Bar**

You'll now see **real-time metrics** in the progress bar:

```
Training:  10% 150/1500 [12:30<1:52:15, 5.01s/it, Loss=1.2e+12, Fwd_E=6.1e+11, Inv_E=6.1e+11, LR=5.0e-4]
```

**Metrics shown**:
- `Loss`: Total loss (forward + inverse)
- `Fwd_E`: Forward energy term
- `Inv_E`: Inverse energy term  
- `LR`: Current learning rate

### 2. 🔥 **Learning Rate Warmup**

**Problem**: Starting with full LR causes wild initial updates
**Solution**: Gradual warmup over first 50 epochs

```
Epoch 1:  LR = 5e-4 * (1/50)  = 1e-5  ← Gentle start
Epoch 25: LR = 5e-4 * (25/50) = 2.5e-4
Epoch 50: LR = 5e-4 * (50/50) = 5e-4  ← Full speed
```

This prevents the **catastrophic initial loss** you saw (486 trillion → should now be much lower).

### 3. 📈 **Adaptive Logging Frequency**

**Early epochs** (when things change fast):
- Epochs 1-10: Log EVERY epoch
- Epochs 11-100: Log every 10 epochs

**Later epochs** (steady progress):
- Epochs 100+: Log every 100 epochs (standard)

You'll see detailed progress when it matters most!

### 4. 📉 **Improvement Tracking**

Every log now shows **% improvement from start**:

```
Epoch 100/3000
  Total Loss: 1.2345e+12
  Forward  - Energy: 6.1e+11, LogDet: 12.34
  Inverse  - Energy: 6.2e+11, LogDet: -12.30
  LR: 5.00e-04
  Improvement from start: 99.75% ← You're making progress!
```

### 5. 📊 **Loss Curve Plots (Always Generated)**

After training completes, you'll get a **4-panel plot**:

**Top Left**: Total Bidirectional Loss
- Shows overall convergence
- Should decrease from trillions → tens

**Top Right**: Forward vs Inverse Loss
- Both should decrease together
- If one explodes, model is broken

**Bottom Left**: Energy Components
- Shows β·U(x) for both directions
- Should converge to similar values

**Bottom Right**: Jacobian Determinants  
- Shows log|det J| for forward/inverse
- Should be symmetric (±same value)

**Saved to**: `checkpoints/molecular_pt_aa_300_450_gpu/loss_curves_300_450.png`

## 🎯 What to Watch During Training

### ✅ **Good Signs**

1. **Initial loss < 10^15** (thanks to warmup!)
2. **Both Fwd_E and Inv_E decreasing**
3. **LR gradually increasing** epochs 1-50
4. **Steady improvement %** increasing

### ⚠️ **Warning Signs**

1. **Loss increasing** after epoch 100
   - LR might be too high
   - Model might be too small

2. **Inv_E exploding** while Fwd_E drops
   - Inverse direction failing
   - Need more model capacity

3. **Loss stuck** at high value (>1000) after 500 epochs
   - Model too small
   - Try full `aa_300_450` preset

4. **NaN or Inf** in metrics
   - Gradient explosion
   - Reduce learning rate

## 📖 Example Training Output

```bash
======================================================================
🧬 Molecular Cross-Temperature Transport Training
======================================================================
Model: ScalableTransformerFlow
Transport: 300.0K → 450.0K
Epochs: 3000, Batch size: 64, LR: 0.0005
======================================================================

Training:   0% 0/3000 [00:00<?, ?it/s]
Epoch 1/3000
  Total Loss: 5.1234e+13  ← Much better than 486 trillion!
  Forward  - Energy: 2.5e+13, LogDet: 0.0000
  Inverse  - Energy: 2.6e+13, LogDet: 0.0000
  LR: 1.00e-05  ← Warmup: starting gentle

Epoch 2/3000
  Total Loss: 3.2145e+13
  Forward  - Energy: 1.6e+13, LogDet: 0.0021
  Inverse  - Energy: 1.6e+13, LogDet: -0.0019
  LR: 2.00e-05
  Improvement from start: 37.2%  ← Good progress!

...

Training:   3% 100/3000 [08:30<4:12:15, 5.22s/it, Loss=1.2e+10, Fwd_E=6.1e+09, Inv_E=6.0e+09, LR=5.0e-4]

Epoch 100/3000
  Total Loss: 1.2345e+10
  Forward  - Energy: 6.1e+09, LogDet: 13.24
  Inverse  - Energy: 6.0e+09, LogDet: -13.18
  LR: 5.00e-04
  Improvement from start: 99.98%  ← Excellent!

...

Training: 100% 3000/3000 [4:15:32<00:00, 5.11s/it, Loss=2.3e+01, Fwd_E=1.1e+01, Inv_E=1.2e+01, LR=1.2e-4]

Epoch 3000/3000
  Total Loss: 2.3456e+01  ← Success!
  Forward  - Energy: 1.1e+01, LogDet: 14.52
  Inverse  - Energy: 1.2e+01, LogDet: -14.48
  LR: 1.23e-04
  Improvement from start: 99.999996%

✅ Model saved: checkpoints/.../molecular_pt_300_450.pt

📊 Generating loss curve plots...
✅ Loss curves saved: checkpoints/.../loss_curves_300_450.png

======================================================================
🎉 Training completed!
======================================================================

📁 Results saved to: checkpoints/molecular_pt_aa_300_450_gpu/
   - Model: molecular_pt_300_450.pt
   - Loss curves: loss_curves_300_450.png
```

## 🔍 Interpreting Loss Curves

### Total Loss Plot

**Good**:
```
^
│  *
│   *
│    *
│     *___
│         -----
└─────────────> epochs
```
Smooth decrease, plateaus at low value

**Bad**:
```
^
│  *
│   ***
│      **
│        *****
│            ******→
└─────────────────> epochs
```
Stuck at high value = model too small

### Forward vs Inverse

**Good**:
```
Both lines decrease together
Gap between them narrows
Both end at similar values
```

**Bad**:
```
Forward decreases, Inverse explodes
Large gap persists
One line goes to NaN/Inf
```

## 💡 Quick Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Initial loss > 10^15 | Warmup not working | Check code update applied |
| Loss stuck > 1000 | Model too small | Use standard preset |
| Inv_E exploding | Inverse failing | Increase model capacity |
| Loss = NaN | Gradient explosion | Reduce LR to 1e-4 |
| Training too slow | Batch size too small | Increase if memory allows |

## 📚 Related Docs

- `TRAINING_DIAGNOSIS.md`: Why your first run failed
- `MOLECULAR_PT_GUIDE.md`: Theory and setup
- `COLAB_SETUP.md`: Colab-specific instructions

---

**TL;DR**: Training now shows live metrics, uses warmup for stable start, logs frequently when needed, and always generates loss curve plots. Watch for both forward and inverse energies decreasing together!

