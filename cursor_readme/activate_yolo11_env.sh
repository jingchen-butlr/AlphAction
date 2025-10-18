#!/bin/bash
# Quick activation script for YOLOv11 environment

echo "=================================================="
echo "Activating YOLOv11 Environment"
echo "=================================================="

# Source conda
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate alphaction_yolo11

# Display environment info
echo ""
echo "✓ Environment: alphaction_yolo11"
echo "✓ Python: $(python --version 2>&1)"
echo "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "✓ Ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo 'Not found')"
echo "✓ CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""
echo "=================================================="
echo "Ready to run YOLOv11 code!"
echo "=================================================="
echo ""
echo "Your shell is now configured. You can run:"
echo "  python demo/demo_run.py"
echo "  python your_yolo11_script.py"
echo ""

