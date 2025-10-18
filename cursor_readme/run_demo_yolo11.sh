#!/bin/bash
# Demo script for AlphAction with YOLO11x and BoT-SORT tracker

set -e

echo "========================================="
echo "AlphAction Demo with YOLO11x + BoT-SORT"
echo "========================================="
echo ""

# Activate environment
echo "Activating alphaction environment..."
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate alphaction

echo "✓ Environment activated"
echo ""

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
    echo "Please download action models first."
    exit 1
fi

# Check test video
if [ -n "$1" ]; then
    TEST_VIDEO="Data/$1"
    if [ ! -f "$TEST_VIDEO" ]; then
        echo "❌ Error: Test video not found: $TEST_VIDEO"
        exit 1
    fi
else
    # Use default test video
    if [ -f "Data/clip_9min_00s_to_9min_25s.mp4" ]; then
        TEST_VIDEO="Data/clip_9min_00s_to_9min_25s.mp4"
    elif [ -f "Data/last_minute.mp4" ]; then
        TEST_VIDEO="Data/last_minute.mp4"
    else
        echo "❌ Error: No test video found in Data/ directory"
        exit 1
    fi
fi

echo "✓ Test video found: $(basename $TEST_VIDEO)"

# Set output path
VIDEO_NAME=$(basename "$TEST_VIDEO" .mp4)
OUTPUT_VIDEO="Data/output_yolo11_${VIDEO_NAME}.mp4"

echo ""
echo "========================================="
echo "Configuration Summary"
echo "========================================="
echo ""
echo "🔍 Detection Model: YOLO11x (auto-downloading on first run)"
echo "🎯 Tracker: BoT-SORT"
echo "🎬 Action Model: $(basename $ACTION_MODEL)"
echo "📹 Input Video: $(basename $TEST_VIDEO)"
echo "💾 Output Video: $(basename $OUTPUT_VIDEO)"
echo ""
echo "Note: YOLO11x model (~250MB) will download automatically"
echo "      on first run and be cached for future use."
echo ""

# Navigate to demo directory
cd demo

echo "========================================="
echo "Running Demo..."
echo "========================================="
echo ""

# Run demo with YOLO11
python demo.py \
    --video-path ../$TEST_VIDEO \
    --output-path ../$OUTPUT_VIDEO \
    --cfg-path ../$CONFIG_FILE \
    --weight-path ../$ACTION_MODEL \
    $COMMON_CATE

echo ""
echo "========================================="
echo "✅ Demo completed successfully!"
echo "========================================="
echo ""
echo "Output video saved to: $OUTPUT_VIDEO"
echo ""
echo "Full path:"
echo "$(cd .. && realpath $OUTPUT_VIDEO)"
echo ""
echo "========================================="
echo "Upgrade Summary:"
echo "  ✓ YOLOv3-SPP → YOLO11x"
echo "  ✓ JDE Tracker → BoT-SORT Tracker"
echo "  ✓ Improved detection accuracy"
echo "  ✓ Better tracking consistency"
echo "========================================="


