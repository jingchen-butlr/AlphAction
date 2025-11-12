#!/bin/bash
# Thermal SlowFast Finetuning Quick Start Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Thermal SlowFast Finetuning QuickStart${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check we're in the right directory
if [ ! -f "train_net.py" ]; then
    echo -e "${RED}Error: Must run from AlphAction root directory${NC}"
    exit 1
fi

# Parse command line arguments
COMMAND=${1:-test}

case $COMMAND in
    test)
        echo -e "${BLUE}Testing thermal dataset integration...${NC}\n"
        python test_thermal_dataset.py
        ;;
    
    train)
        echo -e "${BLUE}Starting thermal model training...${NC}\n"
        
        # Check if pretrained model exists
        if [ ! -f "data/models/aia_models/resnet101_8x8f_denseserial.pth" ]; then
            echo -e "${YELLOW}Warning: Pretrained model not found at:${NC}"
            echo "  data/models/aia_models/resnet101_8x8f_denseserial.pth"
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        # Check if thermal dataset exists
        if [ ! -d "ThermalDataGen/thermal_action_dataset/frames" ]; then
            echo -e "${RED}Error: Thermal dataset not found!${NC}"
            echo "Expected: ThermalDataGen/thermal_action_dataset/frames"
            exit 1
        fi
        
        # Start training
        python train_net.py \
            --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
            --transfer \
            --no-head \
            --use-tfboard \
            ${@:2}
        ;;
    
    train-small)
        echo -e "${BLUE}Starting training with reduced batch size...${NC}\n"
        python train_net.py \
            --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
            --transfer \
            --no-head \
            --use-tfboard \
            SOLVER.VIDEOS_PER_BATCH 2 \
            TEST.VIDEOS_PER_BATCH 2 \
            ${@:2}
        ;;
    
    eval)
        echo -e "${BLUE}Evaluating thermal model...${NC}\n"
        
        if [ -z "$2" ]; then
            # Use model_final if no checkpoint specified
            CHECKPOINT="data/output/thermal_resnet101_8x8f/model_final.pth"
        else
            CHECKPOINT=$2
        fi
        
        if [ ! -f "$CHECKPOINT" ]; then
            echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
            exit 1
        fi
        
        python test_net.py \
            --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
            MODEL.WEIGHT "$CHECKPOINT"
        ;;
    
    tensorboard)
        echo -e "${BLUE}Starting TensorBoard...${NC}\n"
        echo "Open browser at: http://localhost:6006"
        tensorboard --logdir=data/output/thermal_resnet101_8x8f
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning output directory...${NC}\n"
        read -p "Remove data/output/thermal_resnet101_8x8f? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf data/output/thermal_resnet101_8x8f
            echo -e "${GREEN}✓ Output directory cleaned${NC}"
        fi
        ;;
    
    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  test          - Test thermal dataset integration"
        echo "  train         - Start training (full config)"
        echo "  train-small   - Start training with batch size 2 (for limited GPU)"
        echo "  eval [ckpt]   - Evaluate model (default: model_final.pth)"
        echo "  tensorboard   - Start TensorBoard monitoring"
        echo "  clean         - Clean output directory"
        echo "  help          - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 test"
        echo "  $0 train"
        echo "  $0 train-small"
        echo "  $0 eval data/output/thermal_resnet101_8x8f/model_0005000.pth"
        echo "  $0 tensorboard"
        ;;
esac

