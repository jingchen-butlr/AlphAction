#!/bin/bash
# AlphAction Model Download Helper Script

set -e

echo "========================================="
echo "AlphAction Model Download Helper"
echo "========================================="
echo ""

# Activate environment
echo "Activating alphaction environment..."
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh
conda activate alphaction

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

# Create model directories
echo "Creating model directories..."
mkdir -p data/models/detector_models
mkdir -p data/models/aia_models

cd data/models

echo ""
echo "========================================="
echo "Downloading Detector Models..."
echo "========================================="

# Download detector models
cd detector_models

if [ ! -f "yolov3-spp.weights" ]; then
    echo "Downloading yolov3-spp.weights (~250MB)..."
    gdown 1260DRQM5XtSF7W213AWxk6RX2zfa3Zq6 -O yolov3-spp.weights
    echo "✓ yolov3-spp.weights downloaded"
else
    echo "✓ yolov3-spp.weights already exists"
fi

if [ ! -f "jde.uncertainty.pt" ]; then
    echo "Downloading jde.uncertainty.pt (~40MB)..."
    gdown 1nuCX5bR-1-HGZ0_WoH4xZzPiV_jgBphC -O jde.uncertainty.pt
    echo "✓ jde.uncertainty.pt downloaded"
else
    echo "✓ jde.uncertainty.pt already exists"
fi

cd ../aia_models

echo ""
echo "========================================="
echo "Downloading Action Recognition Model..."
echo "========================================="
echo ""
echo "Choose which model to download:"
echo "1) ResNet101-8x8 Dense Serial (Best Performance, 32.4 mAP, ~200MB)"
echo "2) ResNet50-4x16 Dense Serial (Faster, 30.0 mAP, ~100MB)"
echo "3) Common Categories Model (15 actions, 70 mAP, ~200MB)"
echo "4) Skip (download manually later)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        if [ ! -f "resnet101_8x8f_denseserial.pth" ]; then
            echo "Downloading ResNet101-8x8 Dense Serial model..."
            gdown 1yqqc2_X6Ywi165PIuq68NdTs2WwMygHh -O resnet101_8x8f_denseserial.pth
            echo "✓ resnet101_8x8f_denseserial.pth downloaded"
            echo ""
            echo "Use this command to run demo:"
            echo "cd demo"
            echo "python demo.py --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 --output-path ../Data/output_result.mp4 --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth"
        else
            echo "✓ resnet101_8x8f_denseserial.pth already exists"
        fi
        ;;
    2)
        if [ ! -f "resnet50_4x16f_denseserial.pth" ]; then
            echo "Downloading ResNet50-4x16 Dense Serial model..."
            gdown 1bYxGyf6kptfUBNAHtFcG7x4Ryp7mcWxH -O resnet50_4x16f_denseserial.pth
            echo "✓ resnet50_4x16f_denseserial.pth downloaded"
            echo ""
            echo "Use this command to run demo:"
            echo "cd demo"
            echo "python demo.py --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 --output-path ../Data/output_result.mp4 --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth"
        else
            echo "✓ resnet50_4x16f_denseserial.pth already exists"
        fi
        ;;
    3)
        if [ ! -f "common_cate_model.pth" ]; then
            echo "Downloading Common Categories model..."
            gdown 1gi6oKLj3wBGCOwwIiI9L4mS8pznFj7L1 -O common_cate_model.pth
            echo "✓ common_cate_model.pth downloaded"
            echo ""
            echo "Use this command to run demo:"
            echo "cd demo"
            echo "python demo.py --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 --output-path ../Data/output_result.mp4 --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml --weight-path ../data/models/aia_models/common_cate_model.pth --common-cate"
        else
            echo "✓ common_cate_model.pth already exists"
        fi
        ;;
    4)
        echo "Skipping action model download."
        echo "You can download manually from MODEL_ZOO.md"
        ;;
    *)
        echo "Invalid choice. Skipping action model download."
        ;;
esac

cd ../../..

echo ""
echo "========================================="
echo "Download Summary"
echo "========================================="
echo ""
echo "Downloaded files:"
ls -lh data/models/detector_models/
echo ""
ls -lh data/models/aia_models/
echo ""
echo "========================================="
echo "✓ All required models downloaded!"
echo "========================================="
echo ""
echo "To run the demo, navigate to the demo directory and use the appropriate command shown above."
echo ""
echo "For more information, see DEMO_PREPARATION.md"

