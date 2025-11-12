# Custom Dataset Preparation for Finetuning

This guide explains how to prepare your custom dataset for finetuning the SlowFast model.

---

## Table of Contents
1. [Overview](#overview)
2. [Data Format Requirements](#data-format-requirements)
3. [Step-by-Step Preparation](#step-by-step-preparation)
4. [Dataset Registration](#dataset-registration)
5. [Validation](#validation)
6. [Example Scripts](#example-scripts)

---

## Overview

To finetune SlowFast on your custom dataset, you need:

1. **Video clips** (1-second clips at 25 FPS)
2. **Keyframes** (first frame of each clip as JPG)
3. **Action annotations** (COCO-style JSON format)
4. **Person bounding boxes** (JSON format)
5. **Dataset registration** (in `paths_catalog.py`)

---

## Data Format Requirements

### Directory Structure

```
data/YOUR_DATASET/
├── clips/
│   ├── train/
│   │   ├── video_001/
│   │   │   ├── 0001.mp4
│   │   │   ├── 0002.mp4
│   │   │   └── ...
│   │   └── video_002/
│   │       └── ...
│   └── val/
│       └── ...
├── keyframes/
│   ├── train/
│   │   ├── video_001/
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   └── ...
│   │   └── video_002/
│   │       └── ...
│   └── val/
│       └── ...
├── annotations/
│   ├── train.json
│   ├── train_min.json  (optional, minified)
│   ├── val.json
│   └── val_min.json    (optional, minified)
└── boxes/
    ├── train_person_bbox.json
    └── val_person_bbox.json
```

### Video Clip Requirements

- **Duration**: Exactly 1 second
- **FPS**: 25 frames per second (25 frames total)
- **Resolution**: Shortest side ≤ 360 pixels (maintains aspect ratio)
- **Format**: MP4 (H.264 codec recommended)
- **Naming**: Sequential numbering (0001.mp4, 0002.mp4, ...)

### Keyframe Requirements

- **Format**: JPEG
- **Resolution**: Same as video clips
- **Content**: First frame of corresponding video clip
- **Naming**: Same as video clips (0001.jpg for 0001.mp4)

---

## Step-by-Step Preparation

### Step 1: Prepare Raw Videos

Start with your raw videos in any format.

```bash
data/raw_videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### Step 2: Extract and Process Clips

Use the provided script to extract 1-second clips:

```python
# tools/process_custom_videos.py
import cv2
import os
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_clips(
    video_path: str,
    output_clip_dir: str,
    output_keyframe_dir: str,
    clip_length: float = 1.0,
    fps: int = 25,
    max_short_side: int = 360,
    start_time: float = 0.0,
    end_time: float = None
):
    """
    Extract 1-second clips and keyframes from a video
    
    Args:
        video_path: Path to input video
        output_clip_dir: Directory to save clips
        output_keyframe_dir: Directory to save keyframes
        clip_length: Length of each clip in seconds (default: 1.0)
        fps: Target FPS (default: 25)
        max_short_side: Maximum length of shortest side (default: 360)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None = entire video)
    """
    
    # Create output directories
    os.makedirs(output_clip_dir, exist_ok=True)
    os.makedirs(output_keyframe_dir, exist_ok=True)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    cap.release()
    
    if end_time is None:
        end_time = duration
    
    # Calculate scaling
    short_side = min(width, height)
    if short_side > max_short_side:
        scale = max_short_side / short_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Make dimensions divisible by 2
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        scale_filter = f"scale={new_width}:{new_height}"
    else:
        scale_filter = None
    
    logger.info(f"Processing {video_path}")
    logger.info(f"  Original: {width}x{height} @ {original_fps:.2f} FPS, {duration:.2f}s")
    logger.info(f"  Target: {fps} FPS, clips from {start_time:.2f}s to {end_time:.2f}s")
    
    # Extract clips
    clip_idx = 1
    current_time = start_time
    
    while current_time + clip_length <= end_time:
        clip_output = os.path.join(output_clip_dir, f"{clip_idx:04d}.mp4")
        keyframe_output = os.path.join(output_keyframe_dir, f"{clip_idx:04d}.jpg")
        
        # Build ffmpeg command for clip
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(current_time),
            "-i", video_path,
            "-t", str(clip_length),
            "-r", str(fps),
        ]
        
        if scale_filter:
            cmd.extend(["-vf", scale_filter])
        
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # No audio
            clip_output
        ])
        
        # Extract clip
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Extract keyframe (first frame)
        cmd_keyframe = [
            "ffmpeg", "-y",
            "-ss", str(current_time),
            "-i", video_path,
            "-frames:v", "1",
        ]
        
        if scale_filter:
            cmd_keyframe.extend(["-vf", scale_filter])
        
        cmd_keyframe.append(keyframe_output)
        
        subprocess.run(cmd_keyframe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        logger.info(f"  Extracted clip {clip_idx:04d} at {current_time:.2f}s")
        
        current_time += clip_length
        clip_idx += 1
    
    logger.info(f"Extracted {clip_idx - 1} clips")


def process_dataset(
    raw_video_dir: str,
    output_base_dir: str,
    split: str = "train",
    **kwargs
):
    """
    Process all videos in a directory
    
    Args:
        raw_video_dir: Directory containing raw videos
        output_base_dir: Base directory for output
        split: Dataset split (train/val)
        **kwargs: Additional arguments for extract_clips
    """
    
    raw_video_path = Path(raw_video_dir)
    
    for video_file in sorted(raw_video_path.glob("*.mp4")):
        video_id = video_file.stem
        
        clip_dir = os.path.join(output_base_dir, "clips", split, video_id)
        keyframe_dir = os.path.join(output_base_dir, "keyframes", split, video_id)
        
        extract_clips(
            video_path=str(video_file),
            output_clip_dir=clip_dir,
            output_keyframe_dir=keyframe_dir,
            **kwargs
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process videos for finetuning")
    parser.add_argument("--input", required=True, help="Input directory with raw videos")
    parser.add_argument("--output", required=True, help="Output base directory")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--fps", type=int, default=25, help="Target FPS")
    parser.add_argument("--clip-length", type=float, default=1.0, help="Clip length in seconds")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds")
    
    args = parser.parse_args()
    
    process_dataset(
        raw_video_dir=args.input,
        output_base_dir=args.output,
        split=args.split,
        fps=args.fps,
        clip_length=args.clip_length,
        start_time=args.start_time,
        end_time=args.end_time
    )
```

**Usage:**

```bash
# Process training videos
python tools/process_custom_videos.py \
  --input data/raw_videos/train \
  --output data/YOUR_DATASET \
  --split train

# Process validation videos
python tools/process_custom_videos.py \
  --input data/raw_videos/val \
  --output data/YOUR_DATASET \
  --split val
```

### Step 3: Create Annotations

Annotations should be in COCO JSON format:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "video_001/0001.jpg",
      "width": 640,
      "height": 360,
      "video_id": "video_001",
      "frame_id": 1
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 50, 200, 300],
      "area": 60000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "walking"},
    {"id": 2, "name": "running"},
    {"id": 3, "name": "sitting"}
  ]
}
```

**Key Points:**
- `images`: One entry per keyframe
- `annotations`: One entry per person-action pair in that frame
- `bbox`: [x, y, width, height] format
- `category_id`: Action class ID (1-indexed)
- Multiple annotations can have the same `image_id` (multiple persons)

**Example Script to Create Annotations:**

```python
# tools/create_annotations.py
import json
import os
from pathlib import Path
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_coco_annotations(
    keyframe_dir: str,
    output_file: str,
    action_labels: list,
    annotations_data: dict
):
    """
    Create COCO-style annotations
    
    Args:
        keyframe_dir: Directory containing keyframes
        output_file: Output JSON file path
        action_labels: List of action class names
        annotations_data: Dictionary with annotation data
                         Format: {
                             "video_001/0001.jpg": [
                                 {"bbox": [x, y, w, h], "action": "walking"},
                                 ...
                             ],
                             ...
                         }
    """
    
    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    for idx, label in enumerate(action_labels, start=1):
        coco_data["categories"].append({
            "id": idx,
            "name": label
        })
    
    # Create label to ID mapping
    label_to_id = {label: idx for idx, label in enumerate(action_labels, start=1)}
    
    # Process keyframes
    image_id = 0
    annotation_id = 0
    
    keyframe_path = Path(keyframe_dir)
    
    for video_dir in sorted(keyframe_path.iterdir()):
        if not video_dir.is_dir():
            continue
        
        video_id = video_dir.name
        
        for keyframe_file in sorted(video_dir.glob("*.jpg")):
            frame_id = keyframe_file.stem
            relative_path = f"{video_id}/{keyframe_file.name}"
            
            # Read image to get dimensions
            img = cv2.imread(str(keyframe_file))
            height, width = img.shape[:2]
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": relative_path,
                "width": width,
                "height": height,
                "video_id": video_id,
                "frame_id": int(frame_id)
            })
            
            # Add annotations for this image
            if relative_path in annotations_data:
                for ann in annotations_data[relative_path]:
                    bbox = ann["bbox"]
                    action = ann["action"]
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label_to_id[action],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
            
            image_id += 1
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"Created annotations: {output_file}")
    logger.info(f"  Images: {len(coco_data['images'])}")
    logger.info(f"  Annotations: {len(coco_data['annotations'])}")
    logger.info(f"  Categories: {len(coco_data['categories'])}")


# Example usage
if __name__ == "__main__":
    # Define your action labels
    action_labels = ["walking", "running", "sitting", "standing", "waving"]
    
    # Example annotation data (you would load this from your annotation tool)
    annotations_data = {
        "video_001/0001.jpg": [
            {"bbox": [100, 50, 200, 300], "action": "walking"},
            {"bbox": [350, 60, 180, 280], "action": "standing"}
        ],
        "video_001/0002.jpg": [
            {"bbox": [105, 52, 198, 298], "action": "walking"}
        ],
        # ... more annotations ...
    }
    
    create_coco_annotations(
        keyframe_dir="data/YOUR_DATASET/keyframes/train",
        output_file="data/YOUR_DATASET/annotations/train.json",
        action_labels=action_labels,
        annotations_data=annotations_data
    )
```

### Step 4: Detect Person Bounding Boxes

For validation/test set, you need person bounding boxes. You can use a pretrained person detector:

```python
# tools/detect_persons.py
import json
import logging
from pathlib import Path
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_persons_in_dataset(
    keyframe_dir: str,
    output_file: str,
    confidence_threshold: float = 0.7
):
    """
    Detect persons in keyframes using pretrained Faster R-CNN
    
    Args:
        keyframe_dir: Directory containing keyframes
        output_file: Output JSON file path
        confidence_threshold: Detection confidence threshold
    """
    
    # Load pretrained Faster R-CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    
    detections = {}
    keyframe_path = Path(keyframe_dir)
    
    for video_dir in sorted(keyframe_path.iterdir()):
        if not video_dir.is_dir():
            continue
        
        video_id = video_dir.name
        
        for keyframe_file in sorted(video_dir.glob("*.jpg")):
            relative_path = f"{video_id}/{keyframe_file.name}"
            
            # Load image
            img = Image.open(keyframe_file).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
            
            # Detect
            with torch.no_grad():
                predictions = model(img_tensor)[0]
            
            # Filter for persons (class 1 in COCO)
            person_boxes = []
            for i, (label, score, box) in enumerate(zip(
                predictions["labels"],
                predictions["scores"],
                predictions["boxes"]
            )):
                if label == 1 and score >= confidence_threshold:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    person_boxes.append({
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(score)
                    })
            
            if person_boxes:
                detections[relative_path] = person_boxes
            
            if len(detections) % 100 == 0:
                logger.info(f"Processed {len(detections)} images")
    
    # Save detections
    with open(output_file, 'w') as f:
        json.dump(detections, f, indent=2)
    
    logger.info(f"Saved person detections: {output_file}")
    logger.info(f"  Total images with detections: {len(detections)}")


if __name__ == "__main__":
    detect_persons_in_dataset(
        keyframe_dir="data/YOUR_DATASET/keyframes/val",
        output_file="data/YOUR_DATASET/boxes/val_person_bbox.json",
        confidence_threshold=0.7
    )
```

**Note:** For training, you typically use ground truth person boxes (from your annotations).

---

## Dataset Registration

Register your dataset in `alphaction/config/paths_catalog.py`:

```python
class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        # Your custom dataset
        "your_dataset_train": {
            "video_root": "data/YOUR_DATASET/clips/train",
            "ann_file": "data/YOUR_DATASET/annotations/train.json",
            "box_file": "",  # Empty for training (uses GT boxes from annotations)
            "eval_file_paths": {},
            "object_file": "",
        },
        "your_dataset_val": {
            "video_root": "data/YOUR_DATASET/clips/val",
            "ann_file": "data/YOUR_DATASET/annotations/val.json",
            "box_file": "data/YOUR_DATASET/boxes/val_person_bbox.json",
            "eval_file_paths": {
                "eval_file": "data/YOUR_DATASET/annotations/val.csv"  # Optional
            },
            "object_file": "",
        },
        # ... existing datasets ...
    }
```

---

## Validation

Validate your dataset before training:

```bash
# Check data structure
python tools/validate_dataset.py --dataset-name your_dataset_train

# Visualize samples
python tools/visualize_dataset.py \
  --dataset-name your_dataset_train \
  --num-samples 10 \
  --output-dir data/YOUR_DATASET/visualizations
```

---

## Complete Example

Here's a complete workflow:

```bash
# 1. Process videos
python tools/process_custom_videos.py \
  --input data/raw_videos/train \
  --output data/MY_DATASET \
  --split train

python tools/process_custom_videos.py \
  --input data/raw_videos/val \
  --output data/MY_DATASET \
  --split val

# 2. Create annotations (use your annotation tool)
# Save in COCO format to:
#   data/MY_DATASET/annotations/train.json
#   data/MY_DATASET/annotations/val.json

# 3. Detect persons for validation set
python tools/detect_persons.py \
  --keyframes data/MY_DATASET/keyframes/val \
  --output data/MY_DATASET/boxes/val_person_bbox.json

# 4. Register dataset
# Edit alphaction/config/paths_catalog.py

# 5. Generate config
python tools/generate_finetune_config.py \
  --output config_files/my_dataset.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --num-gpus 4

# 6. Start training!
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/my_dataset.yaml \
  --transfer --no-head --use-tfboard
```

---

## Tips

1. **Video Quality**: Higher quality videos lead to better results
2. **Annotation Quality**: Accurate bounding boxes are crucial
3. **Data Balance**: Balance action classes if possible
4. **Data Augmentation**: Enabled by default (jittering, color)
5. **Person Detection**: Use high-quality person detector for val/test sets
6. **Clip Extraction**: Ensure consistent FPS and resolution
7. **Storage**: Use fast storage (SSD) for better data loading speed

---

## Troubleshooting

### Issue: FFmpeg not found
```bash
# Install ffmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
# or
brew install ffmpeg  # macOS
```

### Issue: Out of disk space
```bash
# Check disk usage
du -sh data/YOUR_DATASET/*

# Consider using lower quality settings
--crf 28  # Instead of 23 in ffmpeg
```

### Issue: Video processing too slow
```bash
# Use faster preset
--preset ultrafast  # Instead of fast in ffmpeg

# Process in parallel
parallel -j 4 process_video ::: video_*.mp4
```

---

For more information, see:
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)
- [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md)

