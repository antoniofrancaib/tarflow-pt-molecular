# 🎯 **Easy Configuration System - Switch Distributions with One Line!**

## ✅ **What's Been Implemented**

I've created a comprehensive configuration system that allows you to **easily switch between different base and target distributions** without editing code!

### **🔧 Key Features**

1. **🎯 Multiple Target Distributions**:
   - `two_moons` - Classic two moons dataset  
   - `checkerboard_8x8` - 8×8 checkerboard pattern
   - `checkerboard_6x6` - 6×6 checkerboard pattern  
   - `checkerboard_4x4` - 4×4 checkerboard (easier)
   - `checkerboard_dense` - High contrast checkerboard

2. **📊 Multiple Base Distributions**:
   - `gaussian_2d` - Standard 2D Gaussian
   - `gaussian_2d_trainable` - Learnable parameters

3. **🎨 Ready-Made Presets**:
   - `two_moons_simple` - Two moons with conservative settings
   - `checkerboard_challenge` - Full 8×8 checkerboard challenge
   - `checkerboard_easy` - 4×4 checkerboard for testing
   - `checkerboard_dense` - High contrast version

4. **⚙️ Automatic Configuration**:
   - **Plot domains** auto-adjust to distribution
   - **Training epochs** optimized per difficulty 
   - **Model parameters** tuned for each task
   - **Learning rates** set appropriately
   - **Save directories** auto-named

### **🚀 Super Easy Usage**

#### **Method 1: Edit One Line in config.py**

```python
# In config.py, change this line:
CURRENT_CONFIG = {
    'target_distribution': 'two_moons',        # ← Change this!
    'base_distribution': 'gaussian_2d',       # ← Or this!
    # ... rest stays the same
}
```

#### **Method 2: Use Presets in Training Scripts**

```python
# Simple autoregressive with different distributions:
from train_simple_autoregressive import train_model

model, history = train_model('two_moons_simple')      # Two moons
model, history = train_model('checkerboard_easy')     # 4×4 checkerboard  
model, history = train_model('checkerboard_challenge') # 8×8 checkerboard
model, history = train_model()                        # Current config
```

```python
# Transformer autoregressive with different distributions:
from train_autoregressive import main

model, history = main('two_moons_simple')      # Two moons
model, history = main('checkerboard_challenge') # 8×8 checkerboard
model, history = main()                        # Current config
```

#### **Method 3: Quick Terminal Commands**

```bash
# Run with current config (edit config.py first):
python src/train_simple_autoregressive.py

# Or modify the script to use presets:
# Add this line to train_simple_autoregressive.py:
# model, history = train_model('two_moons_simple')
```

### **🎯 Available Distributions**

| Target Distribution | Description | Difficulty | Grid Size | Epochs |
|-------------------|-------------|------------|-----------|---------|
| `two_moons` | Classic two moons | Easy | 2D continuous | 2000 |
| `checkerboard_4x4` | Small checkerboard | Medium | 4×4 | 2000 |
| `checkerboard_6x6` | Medium checkerboard | Hard | 6×6 | 2500 |
| `checkerboard_8x8` | Large checkerboard | Very Hard | 8×8 | 3000 |
| `checkerboard_dense` | High contrast board | Extreme | 8×8 | 3500 |

### **🎨 Preset Configurations**

| Preset | Target | Epochs | Model Size | Description |
|--------|--------|---------|------------|-------------|
| `two_moons_simple` | Two Moons | 2000 | Small | Conservative for testing |
| `checkerboard_easy` | 4×4 Board | 2000 | Medium | Good starting point |
| `checkerboard_challenge` | 8×8 Board | 3000 | Large | Full challenge |
| `checkerboard_dense` | Dense 8×8 | 3500 | XLarge | Maximum difficulty |

### **⚡ Quick Start Examples**

```python
# Train Two Moons (easiest):
from src.train_simple_autoregressive import train_model
model, history = train_model('two_moons_simple')

# Train 4×4 Checkerboard (medium):  
model, history = train_model('checkerboard_easy')

# Train 8×8 Checkerboard Challenge (hard):
model, history = train_model('checkerboard_challenge')

# Use transformer on checkerboard:
from src.train_autoregressive import main
model, history = main('checkerboard_challenge')
```

### **📁 Automatic File Organization**

The system automatically creates organized output directories:

```
plots_simple_two_moons/           # Simple autoregressive + Two moons
plots_simple_checkerboard_8x8/    # Simple autoregressive + 8×8 checkerboard  
plots_transformer_two_moons/      # Transformer + Two moons
plots_transformer_checkerboard_8x8/ # Transformer + 8×8 checkerboard
```

### **🔍 Explore Available Options**

```python
# See all available distributions and presets:
from src.config import list_available
list_available()

# Check current configuration:
from src.config import print_current_config  
print_current_config()

# Test the system:
python src/demo_config.py
```

### **✨ Key Benefits**

1. **🎯 One-Line Changes**: Switch distributions instantly
2. **🎨 Smart Presets**: Optimized settings for each task  
3. **📊 Auto-Plotting**: Correct domains and colormaps
4. **📁 Clean Organization**: Auto-named output directories
5. **⚙️ Tuned Parameters**: Each distribution gets optimal settings
6. **🔄 Easy Comparison**: Test multiple distributions quickly

### **🎯 Perfect For**

- **🧪 Rapid Experimentation**: Test distributions quickly
- **📊 Architecture Comparison**: Simple vs Transformer flows
- **🎯 Difficulty Progression**: Easy → Hard challenges  
- **📈 Performance Benchmarking**: Consistent evaluation
- **🎨 Visualization**: Clean, auto-labeled plots

### **🚀 Ready to Use!**

Just edit one line in `config.py` or use presets in your training calls!

**No more manual code changes - just pure research efficiency!** 🔥
