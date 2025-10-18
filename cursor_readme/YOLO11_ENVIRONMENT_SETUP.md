# YOLOv11 Environment Setup - Complete Guide

## ✅ Installation Summary

A new conda environment `alphaction_yolo11` has been successfully created with full YOLOv11 support.

### Environment Details

| Component | Version | Status |
|-----------|---------|--------|
| **Environment Name** | `alphaction_yolo11` | ✅ Active |
| **Python** | 3.9.24 | ✅ Compatible (Required: ≥3.9) |
| **PyTorch** | 2.7.1+cu118 | ✅ Latest with CUDA 11.8 |
| **CUDA** | 11.8 (PyTorch) / 13.0 (Driver) | ✅ Compatible |
| **Ultralytics** | 8.3.217 | ✅ Supports YOLOv11 |
| **GPU** | Tesla T4 (14.56 GB) | ✅ Available |

### Key Features

- ✅ **YOLOv11x** model downloaded and tested (109.3 MB)
- ✅ **GPU acceleration** working on CUDA
- ✅ **BoT-SORT tracker** compatible (built-in to Ultralytics)
- ✅ All dependencies installed and verified

---

## 🚀 How to Use

### Activate the Environment

```bash
conda activate alphaction_yolo11
```

### Verify Installation

```bash
python -c "from ultralytics import YOLO; import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print('YOLOv11 Ready!')"
```

### Run Your AlphAction Code

The environment is ready for your existing code:

```bash
cd /home/ec2-user/AlphAction
conda activate alphaction_yolo11

# Your existing yolo11_api.py and yolo11_cfg.py should work directly
python demo/demo_run.py  # or your specific script
```

---

## 📝 Model Files Downloaded

The following YOLOv11 models are now cached in your project:

- `yolo11n.pt` - Nano model (5.4 MB) - for testing
- `yolo11x.pt` - Extra Large model (109.3 MB) - **your production model**

---

## 🔧 Troubleshooting

### If you need to switch between environments:

**Old environment** (Python 3.7, Ultralytics 8.0.145, **No YOLOv11**):
```bash
conda activate alphaction
```

**New environment** (Python 3.9, Ultralytics 8.3.217, **YOLOv11 supported**):
```bash
conda activate alphaction_yolo11
```

### Check current environment:
```bash
conda env list
python --version
python -c "import ultralytics; print(ultralytics.__version__)"
```

---

## 📊 PyTorch CUDA Compatibility

### Your System:
- **NVIDIA Driver**: CUDA 13.0
- **PyTorch CUDA**: 11.8
- **Compatibility**: ✅ **FULLY COMPATIBLE**

PyTorch with CUDA 11.8 works perfectly with CUDA 13.0 driver. The driver supports all CUDA versions ≤ 13.0.

### Test CUDA:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## 🎯 Recommended Workflow

1. **Always activate the new environment first:**
   ```bash
   conda activate alphaction_yolo11
   ```

2. **Run your AlphAction demos/scripts:**
   ```bash
   cd /home/ec2-user/AlphAction
   python your_script.py
   ```

3. **The YOLOv11 detector will automatically:**
   - Use the downloaded `yolo11x.pt` weights
   - Initialize BoT-SORT tracker
   - Run on GPU (CUDA)

---

## 📦 Installed Packages

Key packages in `alphaction_yolo11` environment:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
torchaudio==2.7.1+cu118
ultralytics==8.3.217
opencv-python==4.12.0.88
numpy==2.0.2
scipy==1.13.1
matplotlib==3.9.4
pyyaml==6.0.3
psutil==7.1.0
```

---

## 🔄 Comparison: Old vs New Environment

| Feature | alphaction (old) | alphaction_yolo11 (new) |
|---------|------------------|-------------------------|
| Python | 3.7.12 | 3.9.24 |
| PyTorch | 1.8.0+cu111 | 2.7.1+cu118 |
| CUDA | 11.1 | 11.8 |
| Ultralytics | 8.0.145 | 8.3.217 |
| YOLOv11 Support | ❌ NO | ✅ YES |
| YOLOv8 Support | ✅ YES | ✅ YES |

---

## ⚠️ Important Notes

1. **Use the new environment for YOLOv11**: The old `alphaction` environment cannot run YOLOv11 due to Python 3.7 limitations.

2. **Model naming**: YOLOv11 uses `yolo11x.pt` (not `yolov11x.pt`). Your code is already correct.

3. **Automatic downloads**: First run will download models from GitHub if not present.

4. **GPU Memory**: YOLOv11x is large. Ensure sufficient GPU memory (your Tesla T4 with 14.56 GB is adequate).

---

## ✅ Test Results

All tests passed successfully:

```
✓ Python version is compatible
✓ PyTorch 2.7.1 meets requirement (>=1.8)
✓ CUDA available and working
✓ Ultralytics is installed
✓ YOLOv11 model loaded successfully
✓ Inference completed successfully
✓ GPU acceleration available and working
```

---

## 📚 Additional Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv11 Release**: https://github.com/ultralytics/ultralytics
- **PyTorch CUDA Guide**: https://pytorch.org/get-started/locally/

---

**Environment created on**: 2025-10-18  
**Status**: ✅ Ready for production use

