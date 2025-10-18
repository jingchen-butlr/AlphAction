# YOLOv11 Quick Start Guide

## ✅ Setup Complete!

Your YOLOv11 environment is **ready to use**.

---

## 🚀 Quick Start (3 Steps)

### 1. Activate Environment
```bash
conda activate alphaction_yolo11
```

### 2. Navigate to Project
```bash
cd /home/ec2-user/AlphAction
```

### 3. Run Your Code
```bash
python demo/demo_run.py  # or your script
```

---

## ✅ What's Installed

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.9.24 | ✅ |
| PyTorch | 2.7.1+cu118 | ✅ |
| CUDA | 11.8 | ✅ |
| Ultralytics | 8.3.217 | ✅ |
| YOLOv11x Model | 110 MB | ✅ Downloaded |
| GPU | Tesla T4 (14.56 GB) | ✅ Working |

---

## 📝 Your Configuration

Your existing code at `/home/ec2-user/AlphAction/detector/yolo11_cfg.py`:
- ✅ Model: `yolo11x.pt`
- ✅ Confidence: 0.1
- ✅ NMS Threshold: 0.4
- ✅ BoT-SORT tracker enabled

**No code changes needed!** Everything is compatible.

---

## 🧪 Verify Installation

```bash
conda activate alphaction_yolo11
python test_yolo11x_full.py
```

Expected output: All 5 tests pass ✅

---

## 🔍 Check CUDA Status

```bash
conda activate alphaction_yolo11
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected:
```
CUDA: True
GPU: Tesla T4
```

---

## 📊 PyTorch CUDA Compatibility

✅ **Your setup is fully compatible:**

- **System CUDA Driver**: 13.0
- **PyTorch CUDA**: 11.8
- **Compatibility**: ✅ Perfect (driver 13.0 supports CUDA 11.8)

---

## ⚠️ Important Notes

1. **Always activate environment first**: `conda activate alphaction_yolo11`

2. **Old environment limitation**: The `alphaction` environment (Python 3.7) **cannot** run YOLOv11. Always use `alphaction_yolo11`.

3. **Model location**: `yolo11x.pt` is in `/home/ec2-user/AlphAction/`

4. **First run behavior**: Models auto-download if missing (already done for you)

---

## 🔄 Switch Between Environments

**For YOLOv11** (Python 3.9, new):
```bash
conda activate alphaction_yolo11
```

**For old code** (Python 3.7, YOLOv8 only):
```bash
conda activate alphaction
```

Check current environment:
```bash
conda info --envs
python --version
```

---

## 📁 Files Created

1. ✅ `YOLO11_ENVIRONMENT_SETUP.md` - Complete documentation
2. ✅ `QUICKSTART_YOLO11.md` - This quick reference
3. ✅ `activate_yolo11_env.sh` - Quick activation script
4. ✅ `test_yolo11x_full.py` - Comprehensive test suite
5. ✅ `yolo11x.pt` (110 MB) - Your production model

---

## 🎯 Test Your Detector API

```python
from detector.yolo11_api import YOLO11Detector
from detector.yolo11_cfg import cfg

class TestOpt:
    device = 'cuda:0'
    gpus = [0]
    tracker_box_thres = 0.1
    tracker_nms_thres = 0.4

opt = TestOpt()
detector = YOLO11Detector(cfg, opt)

# Now use detector.detect_one_img() or detector.images_detection()
```

---

## 🆘 Troubleshooting

### Issue: "No module named torch"
**Solution**: Activate environment first
```bash
conda activate alphaction_yolo11
```

### Issue: "CUDA not available"
**Solution**: Check GPU status
```bash
nvidia-smi  # Should show Tesla T4
```

### Issue: "Model not found"
**Solution**: The model should auto-download. If not:
```bash
cd /home/ec2-user/AlphAction
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
```

---

## 📚 Documentation

- Full setup guide: `YOLO11_ENVIRONMENT_SETUP.md`
- Ultralytics docs: https://docs.ultralytics.com/
- Your detector: `detector/yolo11_api.py`
- Your config: `detector/yolo11_cfg.py`

---

## ✅ Test Results

All tests passed:
```
✓ Python 3.9
✓ PyTorch 2.7.1+cu118
✓ CUDA 11.8
✓ GPU: Tesla T4
✓ YOLOv11x model loaded and tested
✓ Your custom detector API working
```

**You're ready to go! 🎉**

---

*Environment created: 2025-10-18*

