# Installation Guide

## 🚀 Quick Start (Recommended: UV)

### Option 1: UV (Fast & Modern) ⚡

**UV is 10-100x faster than pip/conda and recommended for new installations.**

```bash
# 1. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repository
git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction

# 3. Create environment
uv venv --python 3.9 .venv

# 4. Install PyTorch with CUDA 11.8
source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118

# 5. Install dependencies
uv pip install ultralytics av opencv-python yacs easydict pyyaml \
  matplotlib tensorboardx tqdm scipy cython cython-bbox ninja gdown psutil

# 6. Activate environment (always use this to activate)
source activate_uv_env.sh

# 7. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

**📖 Full UV documentation**: See [UV_SETUP.md](UV_SETUP.md)

---

### Option 2: Conda (Traditional)

```bash
# 1. Create conda environment
conda create -n alphaction_yolo11 python=3.9
conda activate alphaction_yolo11

# 2. Install PyTorch with CUDA
conda install pytorch=2.7.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install dependencies
conda install av -c conda-forge
pip install ultralytics opencv-python yacs easydict pyyaml \
  matplotlib tensorboardx tqdm scipy cython cython-bbox ninja gdown psutil

# 4. Clone and install AlphAction
git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

---

## 📋 Requirements

**System Requirements:**
- Python >= 3.9
- NVIDIA GPU with CUDA 11.8+ drivers
- Linux (tested on Amazon Linux 2023)
- GCC/G++ compiler (for CUDA extensions)

**Key Dependencies:**
- [PyTorch](https://pytorch.org/) == 2.7.1 with CUDA 11.8
- [Ultralytics](https://github.com/ultralytics/ultralytics) >= 8.3.0 (YOLOv11)
- [PyAV](https://github.com/mikeboers/PyAV) >= 15.0
- [OpenCV](https://opencv.org/)
- [YACS](https://github.com/rbgirshick/yacs)
- [TensorboardX](https://github.com/lanpa/tensorboardX)

---

## 🎯 Which Installation Method to Choose?

| Feature | UV | Conda |
|---------|----|----|
| **Speed** | ⚡ 10-100x faster | ⏱️ Slow |
| **Disk Space** | 💾 ~2GB | 💽 ~5GB |
| **Package Resolution** | 🎯 Fast & accurate | 🐌 Can be slow |
| **Recommended For** | ✅ New installations | ⚠️ Existing conda users |

**Recommendation**: Use UV for new installations. Both methods produce identical working environments.

---

## ✅ Verification

Test your installation:

```bash
# For UV environment
source activate_uv_env.sh

# For Conda environment
conda activate alphaction_yolo11

# Run test
python -c "
import torch
import torchvision
from ultralytics import YOLO
import cv2
import av
print('✅ All dependencies installed successfully!')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda}')
print(f'   CUDA Available: {torch.cuda.is_available()}')
"
```

Expected output:
```
✅ All dependencies installed successfully!
   PyTorch: 2.7.1+cu118
   CUDA: 11.8
   CUDA Available: True
```

---

## 🐛 Troubleshooting

### CUDA Extensions Import Error

**Error**: `ImportError: libc10.so: cannot open shared object file`

**UV Solution**:
```bash
source activate_uv_env.sh  # This sets LD_LIBRARY_PATH automatically
```

**Conda Solution**:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
```

### CUDA Version Mismatch

**Error**: `RuntimeError: The detected CUDA version (12.8) mismatches PyTorch (11.8)`

**Solution**: This is expected. The CUDA extensions are pre-compiled and work fine. Just ensure you use the activation scripts.

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use a smaller model:
```bash
# Use ResNet50 instead of ResNet101
--cfg-path ../config_files/resnet50_4x16f_denseserial.yaml
--weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth
```

---

## 📦 Package Management

### UV

```bash
# List packages
uv pip list

# Install new package
uv pip install package_name

# Update package
uv pip install --upgrade package_name

# Export dependencies
uv pip freeze > requirements.txt
```

### Conda

```bash
# List packages
conda list

# Install new package
conda install package_name
# or
pip install package_name

# Update package
conda update package_name

# Export environment
conda env export > environment.yml
```

---

## 🔄 Switching Between Environments

You can have both UV and Conda environments installed:

```bash
# Use UV environment
source activate_uv_env.sh
python demo/demo.py ...

# Use Conda environment
conda activate alphaction_yolo11
python demo/demo.py ...
```

Both share the same CUDA extensions (`.so` files) in the AlphAction directory!

---

## 📚 Next Steps

After installation:

1. **Data Preparation**: See [DATA.md](DATA.md)
2. **Model Download**: See [MODEL_ZOO.md](MODEL_ZOO.md)
3. **Run Demo**: See [demo/README.md](demo/README.md)
4. **Training**: See [GETTING_STARTED.md](GETTING_STARTED.md)

---

## 💡 Additional Notes

- **UV is recommended** for new installations due to speed and efficiency
- **CUDA extensions** are pre-compiled and work with both environments
- **No recompilation needed** if you switch between environments
- **YOLOv11x** auto-downloads on first use (109.3 MB)
- **Action models** must be manually downloaded (see MODEL_ZOO.md)

---

**For detailed UV setup instructions, see [UV_SETUP.md](UV_SETUP.md)**

