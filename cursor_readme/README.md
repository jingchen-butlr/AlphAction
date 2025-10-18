# Cursor AI Documentation

This folder contains comprehensive documentation generated during the AlphAction YOLOv11x integration and optimization project.

---

## 📚 Documentation Index

### Quick Start Guides

| File | Description |
|------|-------------|
| `QUICK_START.md` | Fast setup guide for new users |
| `QUICKSTART_YOLO11.md` | YOLOv11x-specific quick reference |
| `INSTALLATION_SUMMARY.md` | Step-by-step installation guide |

### Environment Setup

| File | Description |
|------|-------------|
| `YOLO11_ENVIRONMENT_SETUP.md` | Conda environment setup for YOLOv11x |
| `activate_yolo11_env.sh` | Shell script to activate environment |

### Integration & Migration

| File | Description |
|------|-------------|
| `YOLO11_INTEGRATION_SUMMARY.md` | Complete YOLOv11x + BoT-SORT integration details |
| `PYTORCH2_MIGRATION_SUMMARY.md` | PyTorch 1.x → 2.x migration guide |
| `CUDA_EXTENSIONS_FIXED.md` | CUDA extensions compatibility fixes |

### System Verification

| File | Description |
|------|-------------|
| `SYSTEM_VERIFICATION_COMPLETE.md` | Full end-to-end system test results |
| `RESNET101_TEST_RESULTS.md` | ResNet50 vs ResNet101 comparison |
| `TASK_COMPLETE.md` | Final task completion summary |

### Performance & Optimization

| File | Description |
|------|-------------|
| `PERFORMANCE_ANALYSIS.md` | Detailed bottleneck analysis |
| `FAST_VISUALIZER_SUCCESS.md` | 3.8x visualization speedup details |
| `GPU_VIDEO_ACCELERATION_GUIDE.md` | Advanced GPU acceleration (NVENC) |

### Feature Documentation

| File | Description |
|------|-------------|
| `VISUALIZER_CLI_GUIDE.md` | CLI visualizer selection guide |
| `VISUALIZER_OPTIONS_SUMMARY.md` | Quick visualizer reference |
| `CLI_VISUALIZER_SELECTION_COMPLETE.md` | Visualizer implementation details |
| `demo/VISUALIZER_GUIDE.md` | Detailed visualizer comparison |

### Debugging & Issues

| File | Description |
|------|-------------|
| `DEMO_DEBUG_SUMMARY.md` | Common demo issues and solutions |
| `DEMO_PREPARATION.md` | Demo preparation checklist |
| `PILLOW_FIX_SUMMARY.md` | Pillow 10.0+ compatibility fix |

---

## 🚀 Key Achievements

### 1. YOLOv11x Integration
- Replaced YOLOv3-SPP with YOLOv11x
- Integrated BoT-SORT tracking
- Auto-download support for model weights

### 2. PyTorch 2.x Compatibility
- Migrated all CUDA extensions
- Fixed deprecated API usage
- Full PyTorch 2.7.1 support

### 3. Performance Optimization
- Fast visualizer: 3.8x speedup (9 fps → 34 fps)
- CLI visualizer selection
- Reduced memory allocations

### 4. Documentation
- 17 comprehensive documentation files
- Step-by-step guides for all components
- Performance benchmarks and comparisons

---

## 📁 Helper Scripts

### `activate_yolo11_env.sh`
Quick activation of the YOLOv11 environment:
```bash
source cursor_readme/activate_yolo11_env.sh
```

### `download_models.sh`
Automated model download script:
```bash
bash cursor_readme/download_models.sh
```

### `run_demo_yolo11.sh`
One-command demo execution:
```bash
bash cursor_readme/run_demo_yolo11.sh
```

### `run_demo_test.sh`
Quick test on sample video:
```bash
bash cursor_readme/run_demo_test.sh
```

---

## 📊 Performance Metrics

### Detection & Tracking
- **YOLOv11x Inference:** ~6.8 fps (1080p, Tesla T4)
- **BoT-SORT Tracking:** Integrated, no separate overhead
- **Action Recognition:** 4 fps detection rate (configurable)

### Visualization
- **Original Visualizer:** ~9 fps (Pillow-based)
- **Fast Visualizer:** ~34 fps (OpenCV-based, 3.8x faster)

### Overall System
- **End-to-End:** ~6-7 fps (tracker is bottleneck, expected)
- **Video Writing:** No longer a bottleneck with fast visualizer

---

## 🛠️ Technical Stack

### Environment
- **Python:** 3.9.21
- **PyTorch:** 2.7.1 (CUDA 11.8)
- **Ultralytics:** 8.3.217+
- **OpenCV:** Latest (cv2)

### Models
- **Person Detection:** YOLOv11x (`yolo11x.pt`)
- **Tracking:** BoT-SORT (built-in)
- **Action Recognition:** ResNet50/101 (IA-structure)
- **Object Detection:** YOLOv3 (for object interactions)

### CUDA Extensions
- ROIAlign3d, ROIPool3d
- SigmoidFocalLoss, SoftmaxFocalLoss
- NMS (Non-Maximum Suppression)

---

## 📖 Reading Order

**For New Users:**
1. `QUICK_START.md` - Start here
2. `YOLO11_ENVIRONMENT_SETUP.md` - Setup environment
3. `QUICKSTART_YOLO11.md` - Run first demo
4. `VISUALIZER_CLI_GUIDE.md` - Choose visualizer

**For Developers:**
1. `YOLO11_INTEGRATION_SUMMARY.md` - Understand architecture
2. `PYTORCH2_MIGRATION_SUMMARY.md` - Learn migration details
3. `CUDA_EXTENSIONS_FIXED.md` - CUDA extension details
4. `PERFORMANCE_ANALYSIS.md` - Performance insights

**For Optimization:**
1. `PERFORMANCE_ANALYSIS.md` - Identify bottlenecks
2. `FAST_VISUALIZER_SUCCESS.md` - Visualizer optimization
3. `GPU_VIDEO_ACCELERATION_GUIDE.md` - Advanced GPU usage

**For Troubleshooting:**
1. `DEMO_DEBUG_SUMMARY.md` - Common issues
2. `PILLOW_FIX_SUMMARY.md` - Pillow compatibility
3. `SYSTEM_VERIFICATION_COMPLETE.md` - Expected results

---

## 🔄 Version History

### v2.0 - YOLOv11x + Fast Visualizer (Oct 18, 2025)
- ✅ YOLOv11x integration
- ✅ BoT-SORT tracking
- ✅ PyTorch 2.x compatibility
- ✅ Fast visualizer (3.8x speedup)
- ✅ CLI visualizer selection
- ✅ Comprehensive documentation

### v1.0 - Original AlphAction
- YOLOv3-SPP detection
- JDE tracking
- PyTorch 1.x
- Pillow-based visualizer

---

## 🎯 Usage Examples

### Basic Demo
```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction/demo

python demo.py \
  --video-path ../Data/test.mp4 \
  --output-path ../Data/output.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth
```

### With Fast Visualizer
```bash
python demo.py \
  --video-path ../Data/test.mp4 \
  --output-path ../Data/output.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --visualizer fast
```

### With ResNet101 Model
```bash
python demo.py \
  --video-path ../Data/test.mp4 \
  --output-path ../Data/output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --visualizer fast \
  --common-cate
```

---

## 📝 Contributing

When adding new documentation:
1. Use clear, descriptive filenames
2. Include creation date at bottom
3. Add to this README index
4. Use markdown formatting
5. Include code examples where relevant

---

## 🔗 Related Resources

- **Main README:** `/home/ec2-user/AlphAction/README.md`
- **Demo README:** `/home/ec2-user/AlphAction/demo/README.md`
- **Unit Tests:** `/home/ec2-user/AlphAction/unittest/`
- **Model Zoo:** `/home/ec2-user/AlphAction/MODEL_ZOO.md`

---

**Documentation Created:** October 2025  
**Project:** AlphAction with YOLOv11x Integration  
**Environment:** alphaction_yolo11  
**Hardware:** NVIDIA Tesla T4 GPU

