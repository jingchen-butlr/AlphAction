#!/bin/bash
# Quick demo test script for AlphAction

set -e

echo "========================================="
echo "AlphAction Demo Test Runner"
echo "========================================="
echo ""

# Activate environment
echo "Activating alphaction environment..."
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate alphaction

# Check if models exist
echo "Checking for required models..."

DETECTOR_MODEL="data/models/detector_models/yolov3-spp.weights"
TRACKER_MODEL="data/models/detector_models/jde.uncertainty.pt"

if [ ! -f "$DETECTOR_MODEL" ]; then
    echo "❌ Error: $DETECTOR_MODEL not found!"
    echo "Please run: ./download_models.sh"
    exit 1
fi

if [ ! -f "$TRACKER_MODEL" ]; then
    echo "❌ Error: $TRACKER_MODEL not found!"
    echo "Please run: ./download_models.sh"
    exit 1
fi

echo "✓ Detector models found"

# Check for action models
ACTION_MODEL=""
CONFIG_FILE=""
COMMON_CATE=""

if [ -f "data/models/aia_models/resnet101_8x8f_denseserial.pth" ]; then
    ACTION_MODEL="data/models/aia_models/resnet101_8x8f_denseserial.pth"
    CONFIG_FILE="config_files/resnet101_8x8f_denseserial.yaml"
    echo "✓ Using ResNet101-8x8 Dense Serial model"
elif [ -f "data/models/aia_models/resnet50_4x16f_denseserial.pth" ]; then
    ACTION_MODEL="data/models/aia_models/resnet50_4x16f_denseserial.pth"
    CONFIG_FILE="config_files/resnet50_4x16f_denseserial.yaml"
    echo "✓ Using ResNet50-4x16 Dense Serial model"
elif [ -f "data/models/aia_models/common_cate_model.pth" ]; then
    ACTION_MODEL="data/models/aia_models/common_cate_model.pth"
    CONFIG_FILE="config_files/resnet101_8x8f_denseserial.yaml"
    COMMON_CATE="--common-cate"
    echo "✓ Using Common Categories model"
else
    echo "❌ Error: No action recognition model found!"
    echo "Please run: ./download_models.sh"
    exit 1
fi

# Check test video
TEST_VIDEO="Data/clip_9min_00s_to_9min_25s.mp4"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "❌ Error: Test video not found: $TEST_VIDEO"
    exit 1
fi

echo "✓ Test video found"

# Set output path
OUTPUT_VIDEO="Data/output_result_$(date +%Y%m%d_%H%M%S).mp4"

echo ""
echo "========================================="
echo "Running Demo..."
echo "========================================="
echo ""
echo "Input video: $TEST_VIDEO"
echo "Output video: $OUTPUT_VIDEO"
echo "Config: $CONFIG_FILE"
echo "Model: $ACTION_MODEL"
echo ""

# Navigate to demo directory
cd demo

# Run demo
python demo.py \
    --video-path ../$TEST_VIDEO \
    --output-path ../$OUTPUT_VIDEO \
    --cfg-path ../$CONFIG_FILE \
    --weight-path ../$ACTION_MODEL \
    $COMMON_CATE

echo ""
echo "========================================="
echo "✓ Demo completed successfully!"
echo "========================================="
echo ""
echo "Output video saved to: $OUTPUT_VIDEO"
echo ""
echo "You can view the annotated video at:"
echo "$(realpath ../$OUTPUT_VIDEO)"

