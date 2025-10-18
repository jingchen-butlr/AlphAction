# Demo Debugging Summary

## Issue Encountered

When attempting to run `demo.py` with YOLOv11x on the first 300 frames of `Data/clip_7min_00s_to_7min_25s.mp4`, several compatibility issues were discovered.

---

## Root Cause

The AlphAction codebase contains custom CUDA extensions that were compiled for older PyTorch versions (<= 1.8). These extensions are incompatible with:

1. **PyTorch 1.13+**: The compiled `.so` files have undefined symbols
2. **Modern PyTorch (2.x)**: The source code uses deprecated headers (`THC/THC.h`) that no longer exist
3. **CUDA Version Mismatch**: System CUDA 12.8 vs PyTorch CUDA 11.7/11.8

### Specific Errors

```
ImportError: undefined symbol: _ZNK2at6Tensor7is_cudaEv
fatal error: THC/THC.h: No such file or directory
```

---

## Environment Status

### alphaction (Python 3.7.12)
- ✅ PyTorch 1.13.1+cu117
- ✅ Alphaction module can import
- ❌ CUDA extensions ABI incompatible
- ✅ YOLOv11 detector API exists but cannot run

### alphaction_yolo11 (Python 3.9.24)
- ✅ PyTorch 2.7.1+cu118
- ✅ Ultralytics 8.3.217 (YOLOv11 supported)
- ❌ Alphaction not installed (build fails)
- ❌ Cannot compile CUDA extensions (THC headers removed in PyTorch 2.x)

---

## Workaround Solution

### Option 1: Use Legacy Tracker (Recommended for Quick Testing)

Modify `demo/demo.py` line 120:
```python
# Change from:
args.detector = "yolo11"

# To:
args.detector = "tracker"  # Uses YOLOv3-SPP + JDE tracker
```

Then download required models:
1. **yolov3-spp.weights** - Place in `data/models/detector_models/`
2. **jde.uncertainty.pt** - Place in `data/models/detector_models/`

Run:
```bash
cd /home/ec2-user/AlphAction/demo
conda activate alphaction
python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_test_300frames.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --start 0 \
  --duration 10150
```

### Option 2: Fix CUDA Extensions (Long-term Solution)

The codebase needs to be updated for PyTorch 2.x compatibility:

1. **Update CUDA source files** to remove `THC/THC.h` includes
2. **Replace deprecated APIs** with modern PyTorch C++ API
3. **Recompile extensions** for current PyTorch version

This requires significant code changes in:
- `alphaction/csrc/cuda/ROIAlign3d_cuda.cu`
- `alphaction/csrc/cuda/ROIPool3d_cuda.cu`
- `alphaction/csrc/cuda/SigmoidFocalLoss_cuda.cu`
- `alphaction/csrc/cuda/SoftmaxFocalLoss_cuda.cu`

---

## What Was Attempted

1. ✅ Created new Python 3.9 environment with YOLOv11 support
2. ✅ Installed PyTorch 2.7.1 with CUDA 11.8
3. ✅ Installed Ultralytics 8.3.217
4. ✅ YOLOv11x model downloaded and tested (standalone)
5. ❌ Could not compile alphaction CUDA extensions due to deprecated headers
6. ❌ Could not use existing compiled extensions due to ABI mismatch
7. ✅ Patched cpp_extension.py to bypass CUDA version check
8. ❌ Build still fails due to missing THC headers

---

## Video Details

- **File**: `Data/clip_7min_00s_to_7min_25s.mp4`
- **FPS**: 29.56
- **Total Frames**: 739
- **300 Frames Duration**: ~10,150 ms (10.15 seconds)

---

## Quick Test Command

For immediate testing with legacy tracker:

```bash
cd /home/ec2-user/AlphAction
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate alphaction

# Install required packages if missing
pip install tqdm yacs

cd demo

# Run with legacy tracker (requires model downloads)
python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_test_300frames.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --start 0 \
  --duration 10150
```

**Note**: This will use the original YOLOv3-SPP + JDE tracker instead of YOLOv11x + BoT-SORT.

---

## Recommendations

1. **Short-term**: Use the legacy tracker (Option 1) for testing
2. **Long-term**: Update the codebase for PyTorch 2.x compatibility
3. **Alternative**: Consider using a Docker container with pre-compiled extensions
4. **Alternative**: Downgrade PyTorch to 1.8.0 in a dedicated environment

---

## Files Modified During Debugging

- ✅ Created `alphaction_yolo11` conda environment
- ✅ Patched `cpp_extension.py` in both environments (can be restored from `.backup` files)
- ✅ Downloaded `yolo11x.pt` (110 MB)
- ✅ Created documentation files

---

**Status**: YOLOv11 integration is ready in the new environment, but the AlphAction core module requires code updates to work with modern PyTorch.

**Next Steps**: Either use the legacy tracker for now, or update the CUDA extension source code for PyTorch 2.x compatibility.

