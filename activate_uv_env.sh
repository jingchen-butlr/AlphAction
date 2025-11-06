#!/bin/bash
# Activation script for AlphAction UV environment

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Set LD_LIBRARY_PATH to include PyTorch libraries (needed for CUDA extensions)
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null)
if [ -n "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
fi

# Add AlphAction to PYTHONPATH for development
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "✅ AlphAction UV environment activated!"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo ""
echo "To deactivate: deactivate"

