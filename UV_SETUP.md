# UV Environment Setup Guide

**Modern Python package management with `uv` for AlphAction**

---

## 🚀 Why UV?

[UV](https://github.com/astral-sh/uv) is a blazing-fast Python package installer and resolver written in Rust. Benefits:

- ⚡ **10-100x faster** than pip
- 🎯 **Better dependency resolution** than conda
- 📦 **Single tool** for environment and package management
- 🔒 **Reproducible builds** with lock files
- 💾 **Disk space efficient** with global cache

---

## 📋 Prerequisites

- **Python 3.9+** (recommended: 3.9.24)
- **NVIDIA GPU** with CUDA 11.8+ drivers
- **Git** for cloning the repository
- **GCC/G++** for compiling CUDA extensions

---

## ⚙️ Installation

### Step 1: Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
# Output: uv 0.9.7 (or later)
```

### Step 2: Clone AlphAction Repository

```bash
git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction
```

### Step 3: Create Virtual Environment

```bash
uv venv --python 3.9 .venv
```

This creates a `.venv` directory with Python 3.9.

### Step 4: Install PyTorch with CUDA Support

```bash
source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install AlphAction Dependencies

```bash
uv pip install ultralytics av opencv-python yacs easydict pyyaml \
  matplotlib tensorboardx tqdm scipy cython cython-bbox ninja gdown psutil
```

### Step 6: Activate Environment (Recommended Way)

```bash
source activate_uv_env.sh
```

This script:
- Activates the `.venv` virtual environment
- Sets `LD_LIBRARY_PATH` for CUDA extensions
- Adds AlphAction to `PYTHONPATH`
- Displays environment info

---

## ✅ Verification

Test the installation:

```bash
source activate_uv_env.sh

python -c "
import torch
import torchvision
from ultralytics import YOLO
import cv2
import av
from detector.yolo11_api import YOLO11Detector
import alphaction._custom_cuda_ext
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"
```

Expected output:
```
✅ All imports successful!
PyTorch: 2.7.1+cu118
CUDA: 11.8
CUDA Available: True
```

---

## 🎬 Running the Demo

### Quick Test

```bash
cd demo
source ../activate_uv_env.sh

python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --visualizer fast
```

### With Custom Task Filter

```bash
# Test custom task filter (fall-down, lay-down, sleep, sit, stand, walk, run)
python custom_task_filter.py
```

---

## 📦 Package Management

### List Installed Packages

```bash
source .venv/bin/activate
uv pip list
```

### Install New Package

```bash
uv pip install package_name
```

### Export Dependencies

```bash
uv pip freeze > requirements.txt
```

### Update Package

```bash
uv pip install --upgrade package_name
```

---

## 🔧 Project Structure

```
AlphAction/
├── .venv/                      # UV virtual environment
├── .python-version             # Python version specification (3.9.24)
├── pyproject.toml              # Project configuration and dependencies
├── activate_uv_env.sh          # Environment activation script
├── alphaction/                 # Main package
│   └── _custom_cuda_ext.so    # Pre-compiled CUDA extensions
├── demo/                       # Demo scripts
│   ├── demo.py
│   ├── fast_visualizer.py
│   ├── nvenc_visualizer.py
│   └── custom_task_filter.py
└── unittest/                   # Test scripts
```

---

## 🐛 Troubleshooting

### Issue: CUDA Extensions Not Loading

**Error**: `ImportError: libc10.so: cannot open shared object file`

**Solution**: Use `activate_uv_env.sh` which sets `LD_LIBRARY_PATH` correctly:
```bash
source activate_uv_env.sh
```

Or manually set it:
```bash
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH
```

### Issue: CUDA Version Mismatch

**Error**: `The detected CUDA version (12.8) mismatches PyTorch (11.8)`

**Cause**: System CUDA is newer than PyTorch CUDA

**Solution**: The pre-compiled CUDA extensions (`.so` files) already exist and work. No recompilation needed. Just use `activate_uv_env.sh`.

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**: Ensure virtual environment is activated:
```bash
source activate_uv_env.sh
```

---

## 🔄 Migration from Conda

If you're migrating from conda (recommended to keep both for now):

### Conda Environment (Old)
```bash
conda activate alphaction_yolo11
python demo.py ...
```

### UV Environment (New)
```bash
source activate_uv_env.sh
python demo.py ...
```

Both environments coexist and use the same CUDA extensions!

---

## 🌟 Advantages Over Conda

| Feature | UV | Conda |
|---------|-------|-------|
| **Installation Speed** | ⚡ 10-100x faster | ⏱️ Slow |
| **Disk Usage** | 💾 Efficient (global cache) | 💽 High (per-env) |
| **Dependency Resolution** | 🎯 Fast & accurate | 🐌 Slow |
| **Lock Files** | ✅ Built-in | ❌ Requires separate tool |
| **Python Only** | ✅ Yes | ❌ No (also system packages) |
| **Package Availability** | ✅ PyPI (largest) | ⚠️ conda-forge (smaller) |

---

## 📚 Additional Resources

- **UV Documentation**: https://github.com/astral-sh/uv
- **PyTorch CUDA Wheels**: https://download.pytorch.org/whl/cu118
- **AlphAction Original Repo**: https://github.com/MVIG-SJTU/AlphAction

---

## 🎯 Quick Reference

```bash
# Activate environment
source activate_uv_env.sh

# Run demo with fast visualizer
cd demo && python demo.py --video-path INPUT.mp4 --output-path OUTPUT.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --visualizer fast

# Test custom task filter
cd demo && python custom_task_filter.py

# Deactivate environment
deactivate
```

---

**✅ UV environment setup complete! Ready for blazing-fast Python package management.**

