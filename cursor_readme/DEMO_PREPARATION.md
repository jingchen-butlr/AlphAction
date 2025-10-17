# AlphAction Demo Preparation Guide

## ✅ Current Status

### Installation
- ✅ Python 3.7.16 environment (alphaction)
- ✅ PyTorch 1.4.0 with CUDA support
- ✅ All dependencies installed
- ✅ AlphAction package installed
- ✅ Model directories created:
  - `data/models/detector_models/`
  - `data/models/aia_models/`

### Test Video Available
- ✅ Test video: `Data/clip_9min_00s_to_9min_25s.mp4` (23MB)

## 📥 Required Model Downloads

To run the demo, you need to download the following model files:

### 1. Object Detection Model (Required)
- **File**: `yolov3-spp.weights`
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1260DRQM5XtSF7W213AWxk6RX2zfa3Zq6/view?usp=sharing)
- **Place in**: `data/models/detector_models/yolov3-spp.weights`
- **Size**: ~250 MB

### 2. Person Tracking Model (Required)
- **File**: `jde.uncertainty.pt`
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1nuCX5bR-1-HGZ0_WoH4xZzPiV_jgBphC/view?usp=sharing)
- **Place in**: `data/models/detector_models/jde.uncertainty.pt`
- **Size**: ~40 MB

### 3. Action Recognition Model (Choose One)

#### Option A: Full AVA Model (Recommended for testing)
Choose one of these models from [MODEL_ZOO.md](MODEL_ZOO.md):

**Best Performance Model:**
- **Config**: `resnet101_8x8f_denseserial.yaml`
- **Model**: SlowFast-R101-8x8 with Dense Serial IA (32.4 mAP)
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1yqqc2_X6Ywi165PIuq68NdTs2WwMygHh/view?usp=sharing)
- **Place in**: `data/models/aia_models/resnet101_8x8f_denseserial.pth`
- **Size**: ~200 MB

**Faster Model (Good for quick testing):**
- **Config**: `resnet50_4x16f_denseserial.yaml`
- **Model**: SlowFast-R50-4x16 with Dense Serial IA (30.0 mAP)
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1bYxGyf6kptfUBNAHtFcG7x4Ryp7mcWxH/view?usp=sharing)
- **Place in**: `data/models/aia_models/resnet50_4x16f_denseserial.pth`
- **Size**: ~100 MB

#### Option B: Practical Common Categories Model (70 mAP on 15 actions)
- **Config**: `resnet101_8x8f_denseserial.yaml`
- **Model**: Trained on 15 common AVA action categories
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1gi6oKLj3wBGCOwwIiI9L4mS8pznFj7L1/view?usp=sharing)
- **Place in**: `data/models/aia_models/common_cate_model.pth`
- **Size**: ~200 MB
- **Note**: Use with `--common-cate` flag

## 📂 Expected Directory Structure After Downloads

```
AlphAction/
├── data/
│   └── models/
│       ├── detector_models/
│       │   ├── yolov3-spp.weights          # Required
│       │   └── jde.uncertainty.pt          # Required
│       └── aia_models/
│           └── [your_chosen_model].pth     # Required
├── Data/
│   └── clip_9min_00s_to_9min_25s.mp4      # Test video (already present)
├── config_files/
│   ├── resnet101_8x8f_denseserial.yaml
│   ├── resnet50_4x16f_denseserial.yaml
│   └── [other configs...]
└── demo/
    └── demo.py
```

## 🚀 How to Run the Demo

### Step 1: Activate Environment
```bash
source /home/ec2-user/activate_alphaction.sh
# OR
conda activate alphaction
```

### Step 2: Navigate to Demo Directory
```bash
cd /home/ec2-user/AlphAction/demo
```

### Step 3: Run Demo on Test Video

#### Using Best Performance Model (resnet101_8x8f_denseserial):
```bash
python demo.py \
  --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/output_result.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

#### Using Faster Model (resnet50_4x16f_denseserial):
```bash
python demo.py \
  --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/output_result.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth
```

#### Using Common Categories Model:
```bash
python demo.py \
  --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/output_result.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/common_cate_model.pth \
  --common-cate
```

### Step 4: Check Output
The processed video will be saved at: `Data/output_result.mp4`

## 🎥 Using Your Own Video

```bash
cd /home/ec2-user/AlphAction/demo

python demo.py \
  --video-path /path/to/your/video.mp4 \
  --output-path /path/to/output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

## 📹 Using Webcam

```bash
cd /home/ec2-user/AlphAction/demo

python demo.py \
  --webcam \
  --output-path ../Data/webcam_output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

## 🔧 Additional Options

- `--cpu`: Force CPU usage (slower, but works without CUDA issues)
- `--common-cate`: Use common categories model (15 action classes only)

## ⚠️ Troubleshooting

### If you get CUDA errors:
Add the `--cpu` flag to run on CPU:
```bash
python demo.py --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/output_result.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --cpu
```

### If models fail to load:
1. Check file paths are correct
2. Verify model files are completely downloaded
3. Check file permissions

## 📝 Download Helper Script

You can use this script to help download the models using `gdown` (if available):

```bash
# Install gdown if not available
pip install gdown

# Download detector models
cd /home/ec2-user/AlphAction/data/models/detector_models
gdown 1260DRQM5XtSF7W213AWxk6RX2zfa3Zq6  # yolov3-spp.weights
gdown 1nuCX5bR-1-HGZ0_WoH4xZzPiV_jgBphC  # jde.uncertainty.pt

# Download action model (choose one)
cd /home/ec2-user/AlphAction/data/models/aia_models
# Best performance model:
gdown 1yqqc2_X6Ywi165PIuq68NdTs2WwMygHh -O resnet101_8x8f_denseserial.pth
# OR faster model:
gdown 1bYxGyf6kptfUBNAHtFcG7x4Ryp7mcWxH -O resnet50_4x16f_denseserial.pth
# OR common categories model:
gdown 1gi6oKLj3wBGCOwwIiI9L4mS8pznFj7L1 -O common_cate_model.pth
```

## 📊 Expected Demo Output

The demo will:
1. Detect persons in the video
2. Track them across frames
3. Recognize their actions
4. Annotate the video with bounding boxes and action labels
5. Save the annotated video to the output path

Processing time depends on:
- Video length
- Model choice (ResNet50 is faster than ResNet101)
- Hardware (GPU vs CPU)

## 🎯 Quick Start Checklist

- [ ] Activate alphaction environment
- [ ] Download yolov3-spp.weights to detector_models/
- [ ] Download jde.uncertainty.pt to detector_models/
- [ ] Download an action model to aia_models/
- [ ] Run demo.py with test video
- [ ] Check output video

---

**Ready to test?** Once you've downloaded the required models, follow the commands in the "How to Run the Demo" section!

