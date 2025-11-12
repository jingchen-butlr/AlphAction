# Unit Tests

This folder contains unit test scripts for AlphAction project components.

## Test Files

### 1. `test_thermal_dataset.py` ⭐ NEW
**Purpose:** Comprehensive test suite for ThermalAVADataset class

**Tests:**
- NpInfoDict and NpBoxDict helper classes
- Dataset initialization and HDF5 file loading
- Frame loading from HDF5 (64 consecutive frames)
- YOLO bbox format conversion (centerXYWH → XYXY)
- Video info retrieval
- Handling of empty annotations
- Transform integration

**Usage:**
```bash
cd /home/ec2-user/jingchen/AlphAction
python tests/test_thermal_dataset.py
```

**Expected Output:**
- ✅ All dataset operations functional
- ✅ HDF5 files loaded correctly
- ✅ Bbox conversions accurate
- ✅ Mock dataset creation successful

---

### 2. `test_thermal_integration.py` ⭐ NEW
**Purpose:** Integration tests for thermal dataset with AlphAction pipeline

**Tests:**
- Dataset catalog registration
- Dataloader creation with thermal config
- Configuration file loading
- Transform pipeline for thermal data
- End-to-end pipeline with actual data (if available)

**Usage:**
```bash
cd /home/ec2-user/jingchen/AlphAction
python tests/test_thermal_integration.py
```

**Expected Output:**
- ✅ Thermal datasets registered in catalog
- ✅ Config loads successfully
- ✅ Transforms produce correct shapes
- ✅ Integration with AlphAction complete

---

### 3. `run_thermal_tests.py` ⭐ NEW
**Purpose:** Comprehensive test runner for all thermal tests

**Features:**
- Runs all thermal-related tests
- Colored output for pass/fail/skip
- Detailed timing information
- Summary statistics

**Usage:**
```bash
cd /home/ec2-user/jingchen/AlphAction
python tests/run_thermal_tests.py
```

**Expected Output:**
- Comprehensive test report with colors
- Timing for each test
- Overall pass/fail summary

---

### 4. `test_yolo11x_full.py`
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

### Thermal Tests (Recommended)

Run comprehensive thermal integration tests:

```bash
cd /home/ec2-user/jingchen/AlphAction
python tests/run_thermal_tests.py
```

Or run individual thermal tests:

```bash
# Test dataset class
python tests/test_thermal_dataset.py

# Test integration with AlphAction
python tests/test_thermal_integration.py
```

### All Tests Sequentially

To run all tests (thermal + YOLO + CUDA + visualizer):

```bash
cd /home/ec2-user/AlphAction

echo "Testing thermal integration..."
python tests/run_thermal_tests.py

echo "Testing YOLO11x integration..."
python tests/test_yolo11x_full.py

echo "Testing CUDA extensions..."
python tests/test_cuda_extensions.py

echo "Testing fast visualizer..."
python tests/test_fast_visualizer.py
```

---

## Test Dependencies

### Thermal Tests
- Python 3.9+
- PyTorch with CUDA support
- h5py (for HDF5 file handling)
- NumPy
- AlphAction modules (dataset, config, structures)

### Other Tests
All other tests require the `alphaction_yolo11` conda environment with:
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

**Last Updated:** November 12, 2025
**Environment:** alphaction_yolo11 (Python 3.9.21, PyTorch 2.7.1+cu118)
**New Tests:** Thermal dataset integration tests added

