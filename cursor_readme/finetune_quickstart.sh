#!/bin/bash
# QuickStart Script for Finetuning SlowFast Model
# This script provides common finetuning scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SlowFast Model Finetuning QuickStart${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 <config_file> [options]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # 1. Finetune on 1 GPU with custom data (new classes)"
    echo "  $0 config_files/resnet50_4x16f_baseline.yaml --single-gpu --new-classes"
    echo ""
    echo "  # 2. Finetune on 4 GPUs with AVA data"
    echo "  $0 config_files/resnet50_4x16f_baseline.yaml --num-gpus 4"
    echo ""
    echo "  # 3. Resume training from checkpoint"
    echo "  $0 config_files/resnet50_4x16f_baseline.yaml --resume"
    echo ""
    echo "  # 4. Finetune with all 8 GPUs (default config)"
    echo "  $0 config_files/resnet50_4x16f_baseline.yaml"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --single-gpu          Train on single GPU (adjusts batch size and LR)"
    echo "  --num-gpus NUM        Train on NUM GPUs"
    echo "  --new-classes         Don't load head weights (for different num_classes)"
    echo "  --resume              Resume from checkpoint"
    echo "  --no-tensorboard      Disable TensorBoard logging"
    echo "  --skip-val            Skip validation during training"
    echo "  --seed SEED           Set random seed (default: 2)"
    exit 1
fi

CONFIG_FILE=$1
shift

# Default values
NUM_GPUS=8
SINGLE_GPU=false
NEW_CLASSES=false
RESUME=false
USE_TENSORBOARD=true
SKIP_VAL=false
SEED=2

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-gpu)
            SINGLE_GPU=true
            NUM_GPUS=1
            shift
            ;;
        --num-gpus)
            NUM_GPUS=$2
            shift 2
            ;;
        --new-classes)
            NEW_CLASSES=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --no-tensorboard)
            USE_TENSORBOARD=false
            shift
            ;;
        --skip-val)
            SKIP_VAL=true
            shift
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Config File: $CONFIG_FILE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  New Classes: $NEW_CLASSES"
echo "  Resume Training: $RESUME"
echo "  TensorBoard: $USE_TENSORBOARD"
echo "  Skip Validation: $SKIP_VAL"
echo "  Random Seed: $SEED"
echo ""

# Build command
CMD="python"
ARGS=""

# Add distributed launch for multi-GPU
if [ $NUM_GPUS -gt 1 ]; then
    CMD="python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
fi

# Add train_net.py
CMD="$CMD train_net.py"

# Add config file
CMD="$CMD --config-file \"$CONFIG_FILE\""

# Add transfer flag if not resuming
if [ "$RESUME" = false ]; then
    CMD="$CMD --transfer"
fi

# Add no-head flag for new classes
if [ "$NEW_CLASSES" = true ]; then
    CMD="$CMD --no-head"
fi

# Add tensorboard flag
if [ "$USE_TENSORBOARD" = true ]; then
    CMD="$CMD --use-tfboard"
fi

# Add skip-val flag
if [ "$SKIP_VAL" = true ]; then
    CMD="$CMD --skip-val-in-train"
fi

# Add seed
CMD="$CMD --seed $SEED"

# Adjust hyperparameters based on number of GPUs
# Linear scaling rule: LR and iterations scale with batch size
BASE_LR_8GPU=0.0004
BASE_BATCH_8GPU=16

if [ $NUM_GPUS -ne 8 ]; then
    # Calculate scaled learning rate
    SCALED_LR=$(python3 -c "print($BASE_LR_8GPU * $NUM_GPUS / 8)")
    SCALED_BATCH=$((BASE_BATCH_8GPU * NUM_GPUS / 8))
    
    # Calculate scaled iterations (inversely proportional to effective batch size)
    BASE_MAX_ITER=90000
    SCALED_MAX_ITER=$((BASE_MAX_ITER * 8 / NUM_GPUS))
    
    # Calculate scaled steps
    STEP1=$((50000 * 8 / NUM_GPUS))
    STEP2=$((70000 * 8 / NUM_GPUS))
    
    echo -e "${YELLOW}Applying linear scaling rule:${NC}"
    echo "  Base LR (8 GPU): $BASE_LR_8GPU → Scaled LR ($NUM_GPUS GPU): $SCALED_LR"
    echo "  Base Batch (8 GPU): $BASE_BATCH_8GPU → Scaled Batch ($NUM_GPUS GPU): $SCALED_BATCH"
    echo "  Base Max Iter (8 GPU): $BASE_MAX_ITER → Scaled Max Iter ($NUM_GPUS GPU): $SCALED_MAX_ITER"
    echo "  Scaled Steps: ($STEP1, $STEP2)"
    echo ""
    
    # Add hyperparameter overrides
    CMD="$CMD SOLVER.BASE_LR $SCALED_LR"
    CMD="$CMD SOLVER.STEPS '($STEP1, $STEP2)'"
    CMD="$CMD SOLVER.MAX_ITER $SCALED_MAX_ITER"
    CMD="$CMD SOLVER.VIDEOS_PER_BATCH $SCALED_BATCH"
    CMD="$CMD TEST.VIDEOS_PER_BATCH $SCALED_BATCH"
fi

# Special handling for single GPU with small batch
if [ "$SINGLE_GPU" = true ]; then
    echo -e "${YELLOW}Single GPU mode: Using minimal batch size${NC}"
    CMD="$CMD SOLVER.VIDEOS_PER_BATCH 2"
    CMD="$CMD TEST.VIDEOS_PER_BATCH 2"
fi

echo -e "${GREEN}Executing command:${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo ""

# Ask for confirmation
read -p "Proceed with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Training cancelled.${NC}"
    exit 1
fi

# Execute
eval $CMD

