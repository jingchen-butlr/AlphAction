# Demo

### Installation 

To run this demo, make sure that you install all requirements following [INSTALL.md](../INSTALL.md).

**For YOLOv11 support**, activate the new environment:
```bash
conda activate alphaction_yolo11
```

### Preparation

**🆕 YOLOv11x + BoT-SORT (Recommended)**

This demo now uses **YOLOv11x with BoT-SORT tracker** for improved person detection and tracking:

1. **Person Detection & Tracking**: YOLOv11x model will **auto-download** on first run (109.3 MB)
   - Model: `yolo11x.pt` 
   - Integrated BoT-SORT tracker for robust person tracking
   - No manual download needed!

2. **Action Recognition Models**: Please download our action models. Place them into ```data/models/aia_models```. All models are available in [MODEL_ZOO.md](../MODEL_ZOO.md).

3. **Practical Model (Optional)**: We also provide a practical model ([Google Drive](https://drive.google.com/file/d/1gi6oKLj3wBGCOwwIiI9L4mS8pznFj7L1/view?usp=sharing)) trained on 15 common action categories in AVA. This 
model achieves about 70 mAP on these categories. Please use [`resnet101_8x8f_denseserial.yaml`](../config_files/resnet101_8x8f_denseserial.yaml)
and enable `--common-cate` to apply this model.

---

**📦 Legacy Setup (YOLOv3-SPP + JDE)**

If you want to use the original detector/tracker instead:

1. Download **yolov3-spp.weights** ([Google Drive](https://drive.google.com/file/d/1260DRQM5XtSF7W213AWxk6RX2zfa3Zq6/view?usp=sharing)). Place it into `data/models/detector_models`.
2. Download **jde.uncertainty.pt** ([Google Drive](https://drive.google.com/file/d/1nuCX5bR-1-HGZ0_WoH4xZzPiV_jgBphC/view?usp=sharing)). Place it into `data/models/detector_models`.
3. Modify `demo.py` line 120: change `args.detector = "yolo11"` to `args.detector = "tracker"`

### Usage

**Step 1: Activate Environment**

```bash
conda activate alphaction_yolo11
cd /home/ec2-user/AlphAction/demo
```

**Step 2: Run Demo**

1. **Video Input**

    ```bash
    python demo.py --video-path path/to/your/video --output-path path/to/the/output \
    --cfg-path path/to/cfg/file --weight-path path/to/the/weight [--common-cate]
    ```

    Example:
    ```bash
    python demo.py --video-path ../data/videos/test.mp4 --output-path output.mp4 \
    --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
    --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth
    ```

2. **Webcam Input**

    ```bash
    python demo.py --webcam --output-path path/to/the/output \
    --cfg-path path/to/cfg/file --weight-path path/to/the/weight [--common-cate]
    ```

### Features

- ✅ **YOLOv11x**: State-of-the-art person detection
- ✅ **BoT-SORT Tracker**: Robust multi-person tracking
- ✅ **GPU Acceleration**: Automatic CUDA support if available
- ✅ **Dual Visualizers**: Choose between Original (9 fps) or Fast (34 fps) via `--visualizer` flag
- ✅ **Configurable Parameters**: 
  - `--visualizer {original,fast}`: Visualizer selection (default: original)
  - `--tracker-box-thres`: Detection confidence threshold (default: 0.1)
  - `--tracker-nms-thres`: NMS IoU threshold (default: 0.4)
  - `--visual-threshold`: Visualization confidence threshold (default: 0.5)
  - `--detect-rate`: Action detection rate in fps (default: 4)

### Technical Details

**Detector & Tracker Pipeline:**
- **Person Detection**: YOLOv11x (class 0: person) 
- **Person Tracking**: BoT-SORT (built-in to Ultralytics)
- **Object Detection**: YOLOv3 (for object-interaction actions)
- **Action Recognition**: ResNet101-based model with IA-structure

The YOLOv11x model automatically handles both detection and tracking, replacing the previous two-stage approach of YOLOv3-SPP + JDE tracker.
