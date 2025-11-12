# AlphAction Python Environment Setup Summary

## Environment Details

- **Location**: `/home/ec2-user/jingchen/AlphAction/.venv`
- **Python Version**: 3.9.24
- **Package Manager**: uv (v0.9.7)
- **PyTorch Version**: 2.7.1+cu118
- **CUDA Support**: Enabled (CUDA 11.8 compatible)

## Installation Summary

The environment was successfully set up using `uv` with the following steps:

1. **Installed uv package manager**
2. **Created virtual environment** in `.venv` folder within AlphAction directory
3. **Installed dependencies** from `pyproject.toml`:
   - PyTorch 2.7.1 with CUDA 11.8 support
   - YOLOv11 (ultralytics >=8.3.0)
   - Video processing libraries (av, opencv-python)
   - Scientific computing (numpy, scipy)
   - Configuration tools (yacs, easydict, pyyaml)
   - Visualization tools (matplotlib, tensorboardx)
   - Build tools (cython, ninja, gcc)

4. **Fixed build issues**:
   - Installed Python development headers (`python3.9-devel`)
   - Bypassed CUDA version mismatch (system CUDA 12.9 vs PyTorch CUDA 11.8)
   - Fixed absolute path issues in `setup.py` for editable installation

5. **Compiled C++/CUDA extensions**:
   - `alphaction._custom_cuda_ext` (ROI operations, focal loss)
   - `detector.nms.soft_nms_cpu` (Soft NMS CPU)
   - `detector.nms.nms_cpu` (NMS CPU)
   - `detector.nms.nms_cuda` (NMS CUDA)

## Activation

To activate the environment, use one of the following methods:

### Method 1: Using activation script (recommended)
```bash
cd /home/ec2-user/jingchen/AlphAction
source activate_env.sh
```

### Method 2: Direct activation
```bash
source /home/ec2-user/jingchen/AlphAction/.venv/bin/activate
```

## Verification

To verify the installation:

```bash
python -c "import alphaction; import torch; print('AlphAction:', alphaction.__name__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
AlphAction: alphaction
PyTorch: 2.7.1+cu118
CUDA available: True
```

## Installed Packages (Main)

- alphaction (0.1.0) - editable install
- torch (2.7.1+cu118)
- torchvision (0.22.1+cu118)
- torchaudio (2.7.1+cu118)
- ultralytics (8.3.225)
- av (15.1.0)
- opencv-python (4.11.0.86)
- numpy (1.26.3)
- scipy (1.13.1)
- matplotlib (3.9.4)
- yacs (0.1.8)
- tensorboardx (2.6.4)
- cython (3.2.0)
- cython-bbox (0.1.5)
- ninja (1.13.0)
- And more... (see `uv pip list` for complete list)

## Modified Files

The following files were modified during setup to resolve installation issues:

1. **`pyproject.toml`**: Added `torch==2.7.1` to `build-system.requires`
2. **`setup.py`**: 
   - Added monkeypatch to bypass CUDA version check
   - Changed to use relative paths instead of absolute paths for editable installation

## Notes

- The environment uses PyTorch with CUDA 11.8 support, which is compatible with the system's CUDA 12.9 installation
- All CUDA extensions were successfully compiled and installed
- The package is installed in editable mode (`-e`), so code changes will be immediately reflected
- Build isolation was disabled (`--no-build-isolation`) to work around dependency resolution issues

## Troubleshooting

If you encounter issues:

1. **CUDA-related errors**: Ensure CUDA 11.8 or 12.9 is available on the system
2. **Import errors**: Make sure the environment is activated before running Python
3. **Extension errors**: The CUDA extensions are compiled for compute capability 8.9 (sm_89)

## Maintenance

To update dependencies:
```bash
cd /home/ec2-user/jingchen/AlphAction
uv pip install <package-name> --upgrade
```

To add new dependencies:
1. Add to `pyproject.toml` under `[project.dependencies]`
2. Run: `uv sync`

