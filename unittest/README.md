# Unit Tests

This folder contains unit test scripts for AlphAction project components.

## Test Files

### 1. `test_yolo11x_full.py`
**Purpose:** Comprehensive test suite for YOLOv11x integration

**Tests:**
- PyTorch and CUDA installation
- Ultralytics library installation
- YOLOv11x model loading
- Inference on sample images
- Custom `YOLO11Detector` API integration
- BoT-SORT tracking functionality

**Usage:**
```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction
python unittest/test_yolo11x_full.py
```

**Expected Output:**
- ✅ All environment checks pass
- ✅ Model downloads automatically on first run
- ✅ Detection works on test images
- ✅ Custom API integration successful

---

### 2. `test_cuda_extensions.py`
**Purpose:** Verify custom CUDA extensions are properly compiled and functional

**Tests:**
- Import test for `_custom_cuda_ext` module
- ROIAlign3d functionality test
- ROIPool3d functionality test
- SigmoidFocalLoss forward/backward test
- SoftmaxFocalLoss forward/backward test
- NMS (Non-Maximum Suppression) test

**Usage:**
```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction
python unittest/test_cuda_extensions.py
```

**Expected Output:**
- ✅ All CUDA extensions load successfully
- ✅ All operations produce valid outputs
- ✅ No memory errors or crashes

**Note:** Requires CUDA-enabled GPU and proper PyTorch installation

---

### 3. `test_fast_visualizer.py`
**Purpose:** Test the fast OpenCV-based visualizer implementation

**Tests:**
- Visualizer initialization
- Action dictionary updates
- Frame visualization with bounding boxes
- Timestamp overlay rendering
- OpenCV drawing operations

**Usage:**
```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction
python unittest/test_fast_visualizer.py
```

**Expected Output:**
- ✅ Visualizer initializes successfully
- ✅ Frame rendering works correctly
- ✅ All drawing operations functional
- ✅ Output dimensions match input

**Note:** Requires test video file at `../Data/clip_7min_00s_to_7min_25s.mp4`

---

## Running All Tests

To run all tests sequentially:

```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction

echo "Testing YOLO11x integration..."
python unittest/test_yolo11x_full.py

echo "Testing CUDA extensions..."
python unittest/test_cuda_extensions.py

echo "Testing fast visualizer..."
python unittest/test_fast_visualizer.py
```

---

## Test Dependencies

All tests require the `alphaction_yolo11` conda environment with:
- Python 3.9+
- PyTorch 2.7.1 with CUDA 11.8
- Ultralytics 8.3.217+
- OpenCV (cv2)
- NumPy

---

## Troubleshooting

### CUDA Extension Import Errors
If you encounter import errors with `_custom_cuda_ext`, rebuild the extensions:
```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction
python setup.py build develop
```

### Model Download Failures
If YOLOv11x models fail to download, manually download from:
- https://github.com/ultralytics/assets/releases/

### Video File Not Found
Ensure test video exists at the expected path or update the path in test scripts.

---

## Adding New Tests

When adding new test files:
1. Name files with `test_` prefix
2. Include descriptive docstrings
3. Use clear success/failure indicators (✅/❌)
4. Update this README with test description
5. Add to the running sequence above

---

## Test Results Archive

Successful test runs are documented in `cursor_readme/` folder:
- `SYSTEM_VERIFICATION_COMPLETE.md` - Full system test results
- `CUDA_EXTENSIONS_FIXED.md` - CUDA extension test results
- `FAST_VISUALIZER_SUCCESS.md` - Visualizer test results

---

**Last Updated:** October 18, 2025
**Environment:** alphaction_yolo11 (Python 3.9.21, PyTorch 2.7.1+cu118)

