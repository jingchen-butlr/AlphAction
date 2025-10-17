# AlphAction Quick Start Guide

## 🎯 Ready to Test in 3 Steps!

### Prerequisites
✅ Python 3.7.16 environment installed  
✅ PyTorch 1.4.0 with CUDA support  
✅ AlphAction package installed  
✅ Test video available: `Data/clip_9min_00s_to_9min_25s.mp4`

---

## 📥 Step 1: Download Models

Run the automated download script:

```bash
cd /home/ec2-user/AlphAction
./download_models.sh
```

This will download:
- Object detection model (yolov3-spp.weights, ~250MB)
- Person tracking model (jde.uncertainty.pt, ~40MB)  
- Action recognition model (your choice, ~100-200MB)

**Or download manually** from the links in [DEMO_PREPARATION.md](DEMO_PREPARATION.md)

---

## 🚀 Step 2: Run the Test

After downloading models, simply run:

```bash
cd /home/ec2-user/AlphAction
./run_demo_test.sh
```

This will:
- Activate the environment
- Check all models are present
- Run the demo on the test video
- Save output with timestamp

**Expected processing time:** 2-5 minutes (depending on GPU/CPU)

---

## 📹 Step 3: View Results

The annotated video will be saved in the `Data/` directory with a timestamp:
```
Data/output_result_YYYYMMDD_HHMMSS.mp4
```

The demo will show:
- Detected persons with bounding boxes
- Tracked person IDs
- Recognized actions as text labels

---

## 🔧 Manual Demo Execution

If you prefer to run the demo manually:

```bash
# Activate environment
conda activate alphaction

# Navigate to demo directory
cd /home/ec2-user/AlphAction/demo

# Run demo (example with ResNet101 model)
python demo.py \
  --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/my_output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

---

## 🎥 Test Your Own Videos

```bash
cd /home/ec2-user/AlphAction/demo

python demo.py \
  --video-path /path/to/your/video.mp4 \
  --output-path /path/to/output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

---

## ⚙️ Common Options

| Option | Description |
|--------|-------------|
| `--cpu` | Run on CPU (slower but works without GPU) |
| `--common-cate` | Use common categories model (15 actions) |
| `--webcam` | Use webcam as input instead of video file |

---

## 🐛 Troubleshooting

### CUDA Errors
If you get CUDA-related errors, try running on CPU:
```bash
./run_demo_test.sh --cpu
```
Or add `--cpu` flag when running demo.py manually.

### Out of Memory
If you run out of GPU memory, try:
1. Using CPU mode (`--cpu`)
2. Using the smaller ResNet50 model instead of ResNet101
3. Processing shorter video clips

### Model Not Found
Make sure you've downloaded all required models:
```bash
ls -lh data/models/detector_models/
ls -lh data/models/aia_models/
```

You should see:
- `yolov3-spp.weights`
- `jde.uncertainty.pt`
- At least one `.pth` file in aia_models/

---

## 📚 Additional Resources

- **Full preparation guide**: [DEMO_PREPARATION.md](DEMO_PREPARATION.md)
- **Installation details**: [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md)
- **Model options**: [MODEL_ZOO.md](MODEL_ZOO.md)
- **Original README**: [demo/README.md](demo/README.md)

---

## 🎓 What the Demo Does

1. **Person Detection**: Uses YOLO to detect people in each frame
2. **Person Tracking**: Tracks detected persons across frames with unique IDs
3. **Action Recognition**: Recognizes actions being performed by each person
4. **Visualization**: Draws bounding boxes and action labels on the video
5. **Output**: Saves the annotated video

The demo uses a 3D CNN (SlowFast network) to capture spatio-temporal features for action recognition, achieving state-of-the-art performance on the AVA dataset.

---

## 🎯 Quick Command Reference

```bash
# Download models
cd /home/ec2-user/AlphAction && ./download_models.sh

# Run test
cd /home/ec2-user/AlphAction && ./run_demo_test.sh

# Run on CPU
cd /home/ec2-user/AlphAction/demo
python demo.py --video-path ../Data/clip_9min_00s_to_9min_25s.mp4 \
  --output-path ../Data/output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --cpu

# Use webcam
cd /home/ec2-user/AlphAction/demo
python demo.py --webcam \
  --output-path ../Data/webcam_output.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
```

---

**Ready? Start with Step 1: Download the models!** 🚀

