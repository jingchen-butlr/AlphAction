#!/bin/bash
# Activation script for AlphAction Python environment

# Activate the virtual environment
source /home/ec2-user/jingchen/AlphAction/.venv/bin/activate

echo "✓ AlphAction Python environment activated"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""
echo "To verify the environment:"
echo "  python -c 'import alphaction; import torch; print(\"AlphAction:\", alphaction.__name__); print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'"

