# Finetuning SlowFast Model Guide

This guide explains how to finetune a pretrained SlowFast model on your own dataset or continue training on AVA dataset.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Understanding the Model Architecture](#understanding-the-model-architecture)
3. [Preparing Your Data](#preparing-your-data)
4. [Configuration Setup](#configuration-setup)
5. [Training Commands](#training-commands)
6. [Advanced Options](#advanced-options)
7. [Monitoring Training](#monitoring-training)
8. [Common Issues](#common-issues)

---

## Prerequisites

### 1. Download Pretrained Models

Download one of the SlowFast backbone models pretrained on Kinetics-700:

```bash
# Create directories
mkdir -p data/models/pretrained_models
cd data/models/pretrained_models

# Download SlowFast-ResNet50-4x16 (recommended for starting)
# Download from: https://drive.google.com/file/d/1hqFuhD1p0lMpl3Yi5paIGY-hlPTVYgyi/view?usp=sharing
# Save as: SlowFast-ResNet50-4x16.pth

# OR Download SlowFast-ResNet101-8x8 (more accurate but slower)
# Download from: https://drive.google.com/file/d/1JDQLyyL-GFd3qi0S31Mdt5oNmUXnyJza/view?usp=sharing
# Save as: SlowFast-ResNet101-8x8.pth
```

**Model Performance:**
- **SlowFast-R50-4x16**: Kinetics-700 Top-1: 66.34%, Top-5: 86.66%
- **SlowFast-R101-8x8**: Kinetics-700 Top-1: 69.32%, Top-5: 88.84%

### 2. Setup Output Directory

```bash
# Create output directory for checkpoints and logs
mkdir -p /path/to/output
ln -s /path/to/output data/output
```

---

## Understanding the Model Architecture

### SlowFast Architecture
- **Slow Pathway**: Processes frames at low frame rate (captures spatial semantics)
- **Fast Pathway**: Processes frames at high frame rate (captures motion)
- **Lateral Connections**: Transfer information between pathways

### Key Components
1. **Backbone**: SlowFast-ResNet50 or ResNet101 (pretrained on Kinetics)
2. **ROI Head**: Region-of-Interest head for action classification
3. **IA Structure** (Optional): Interaction Aggregation module for multi-person reasoning

---

## Preparing Your Data

### Option 1: Use AVA Dataset

If you want to finetune on AVA dataset, follow the instructions in [DATA.md](../DATA.md).

**Quick AVA Setup:**
```bash
# Download preprocessed AVA data
# Link: https://pan.baidu.com/s/1UrflK4IgiVbVBOP5fDHdKA (code: q5v5)

# Extract and link
tar zxvf AVA_compress.tar.gz -C /some/path/
ln -s /some/path/AVA data/AVA
```

### Option 2: Use Custom Dataset

To finetune on your own dataset, you need to:

1. **Prepare Videos**: Organize your videos and extract clips
2. **Create Annotations**: Convert annotations to COCO format
3. **Detect Person Boxes**: Generate person bounding boxes
4. **Update Configuration**: Modify config file to point to your data

**Required Annotation Format (COCO-style JSON):**
```json
{
  "images": [
    {
      "id": 0,
      "file_name": "movie_id/frame_id.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": area,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "action_name"}
  ]
}
```

**Data Directory Structure:**
```
data/
├── YOUR_DATASET/
│   ├── clips/
│   │   └── [video_id]/
│   │       └── [timestamp].mp4
│   ├── keyframes/
│   │   └── [video_id]/
│   │       └── [timestamp].jpg
│   ├── annotations/
│   │   ├── train.json
│   │   └── val.json
│   └── boxes/
│       ├── train_person_bbox.json
│       └── val_person_bbox.json
```

---

## Configuration Setup

### 1. Create Your Config File

Copy an existing config and modify it:

```bash
cp config_files/resnet50_4x16f_baseline.yaml config_files/my_custom_config.yaml
```

### 2. Key Configuration Parameters

Edit `config_files/my_custom_config.yaml`:

```yaml
MODEL:
  # Path to pretrained model
  WEIGHT: "data/models/pretrained_models/SlowFast-ResNet50-4x16.pth"
  
  BACKBONE:
    CONV_BODY: "Slowfast-Resnet50"  # or "Slowfast-Resnet101"
    FROZEN_BN: True  # Keep BatchNorm frozen for stability
    
  ROI_ACTION_HEAD:
    NUM_CLASSES: 80  # CHANGE THIS to your number of action classes
    DROPOUT_RATE: 0.2  # Dropout for regularization
    PROPOSAL_PER_CLIP: 10  # Max person proposals per clip
    
  IA_STRUCTURE:
    ACTIVE: False  # Set to True to use Interaction Aggregation

INPUT:
  FRAME_NUM: 64  # Total frames loaded
  FRAME_SAMPLE_RATE: 1  # Sample every N frames
  TAU: 16  # Temporal sampling stride
  ALPHA: 8  # Fast/Slow pathway ratio
  SLOW_JITTER: True  # Temporal augmentation
  COLOR_JITTER: True  # Color augmentation

DATASETS:
  TRAIN: ("your_dataset_train",)  # Training dataset name
  TEST: ("your_dataset_val",)  # Validation dataset name

SOLVER:
  BASE_LR: 0.0004  # Base learning rate (for 8 GPUs)
  WARMUP_FACTOR: 0.25
  WARMUP_ITERS: 2000
  STEPS: (50000, 70000)  # Learning rate decay steps
  MAX_ITER: 90000  # Total training iterations
  CHECKPOINT_PERIOD: 10000  # Save checkpoint every N iters
  EVAL_PERIOD: 10000  # Evaluate every N iters
  VIDEOS_PER_BATCH: 16  # Batch size (for 8 GPUs)
  WEIGHT_DECAY: 1e-7

TEST:
  BOX_THRESH: 0.8  # Person detection threshold
  ACTION_THRESH: 0.0  # Action prediction threshold
  VIDEOS_PER_BATCH: 16

OUTPUT_DIR: "data/output/my_experiment"  # Output directory
```

### 3. Register Your Dataset

If using custom data, edit `alphaction/config/paths_catalog.py` to register your dataset:

```python
class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        # Add your custom dataset
        "your_dataset_train": {
            "video_root": "data/YOUR_DATASET/clips/train",
            "ann_file": "data/YOUR_DATASET/annotations/train.json",
            "box_file": "data/YOUR_DATASET/boxes/train_person_bbox.json",
            "eval_file_paths": {},
            "object_file": "",
        },
        "your_dataset_val": {
            "video_root": "data/YOUR_DATASET/clips/val",
            "ann_file": "data/YOUR_DATASET/annotations/val.json",
            "box_file": "data/YOUR_DATASET/boxes/val_person_bbox.json",
            "eval_file_paths": {
                "eval_file": "data/YOUR_DATASET/annotations/val.csv"
            },
            "object_file": "",
        },
        # ... existing AVA datasets ...
    }
```

---

## Training Commands

### Single GPU Training

For training on 1 GPU (adjust batch size and learning rate):

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --no-head \
  --use-tfboard \
  SOLVER.BASE_LR 0.000125 \
  SOLVER.STEPS '(560000, 720000)' \
  SOLVER.MAX_ITER 880000 \
  SOLVER.VIDEOS_PER_BATCH 2 \
  TEST.VIDEOS_PER_BATCH 2
```

**Important Flags:**
- `--transfer`: Enable transfer learning mode
- `--no-head`: Don't load the final classification head (use when num_classes differs)
- `--use-tfboard`: Enable TensorBoard logging

**Linear Scaling Rule:**
When changing number of GPUs, adjust learning rate and iterations proportionally:
- LR: `BASE_LR = 0.0004 * (num_gpus / 8)`
- Iterations: `MAX_ITER = 90000 * (8 / num_gpus)`
- Steps: Scale accordingly

### Multi-GPU Training

For training on multiple GPUs (e.g., 4 GPUs):

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --no-head \
  --use-tfboard \
  SOLVER.BASE_LR 0.0002 \
  SOLVER.STEPS '(100000, 140000)' \
  SOLVER.MAX_ITER 180000 \
  SOLVER.VIDEOS_PER_BATCH 8 \
  TEST.VIDEOS_PER_BATCH 8
```

### 8 GPU Training (Default)

For training on 8 GPUs (use config defaults):

```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --no-head \
  --use-tfboard
```

---

## Advanced Options

### Resume Training from Checkpoint

To resume training from a checkpoint:

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --use-tfboard \
  MODEL.WEIGHT "data/output/my_experiment/model_0050000.pth"
```

**Note:** The code automatically looks for `last_checkpoint` in `OUTPUT_DIR` first.

### Adjust Learning Rate Schedule

When resuming from checkpoint with different schedule:

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --adjust-lr \
  --use-tfboard \
  MODEL.WEIGHT "data/output/my_experiment/model_0050000.pth"
```

### Skip Validation During Training

To speed up training (skip periodic validation):

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --no-head \
  --skip-val-in-train \
  --use-tfboard
```

### Load Head Weights (Same Number of Classes)

If your dataset has the same number of classes as pretrained model:

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --use-tfboard
```
(Remove `--no-head` flag)

### Custom Random Seed

For reproducibility:

```bash
python train_net.py \
  --config-file "config_files/my_custom_config.yaml" \
  --transfer \
  --no-head \
  --seed 42 \
  --use-tfboard
```

---

## Monitoring Training

### TensorBoard

If you used `--use-tfboard`, monitor training with TensorBoard:

```bash
tensorboard --logdir=data/output/my_experiment
```

**Metrics to Monitor:**
- `loss_action`: Action classification loss
- `loss_objectness`: Person detection loss
- `accuracy`: Action classification accuracy
- `learning_rate`: Current learning rate

### Log Files

Training logs are saved in:
```
data/output/my_experiment/
├── log.txt           # Training log
├── last_checkpoint   # Path to last checkpoint
├── model_*.pth       # Saved checkpoints
└── events.out.tfevents.*  # TensorBoard logs
```

### Checkpoints

Checkpoints include:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Current iteration
- Best validation results

---

## Common Issues

### 1. Out of Memory (OOM)

**Solution:** Reduce batch size and adjust learning rate accordingly:
```bash
SOLVER.VIDEOS_PER_BATCH 1 \
TEST.VIDEOS_PER_BATCH 1 \
SOLVER.BASE_LR 0.00005
```

### 2. BatchNorm Issues

**Solution:** Keep BatchNorm frozen (especially with small batch sizes):
```yaml
MODEL:
  BACKBONE:
    FROZEN_BN: True
  NONLOCAL:
    FROZEN_BN: True
```

### 3. Wrong Number of Classes Error

**Error:** Size mismatch when loading head weights

**Solution:** Use `--no-head` flag to skip loading head layer:
```bash
python train_net.py --no-head ...
```

### 4. Dataset Not Found

**Error:** Dataset name not in catalog

**Solution:** Register your dataset in `alphaction/config/paths_catalog.py`

### 5. Slow Training

**Solutions:**
- Increase `DATALOADER.NUM_WORKERS` (default: 4)
- Use NVMe SSD for data storage
- Enable mixed precision training (requires code modification)
- Reduce `INPUT.FRAME_NUM` or increase `FRAME_SAMPLE_RATE`

### 6. Low Accuracy

**Solutions:**
- Train longer (increase `MAX_ITER`)
- Use data augmentation (`SLOW_JITTER`, `COLOR_JITTER`)
- Reduce learning rate
- Try different backbone (ResNet101 vs ResNet50)
- Enable IA structure for multi-person scenarios

---

## Example: Complete Finetuning Workflow

### Step 1: Setup
```bash
# Activate environment
source activate_uv_env.sh

# Create directories
mkdir -p data/models/pretrained_models
mkdir -p data/output
```

### Step 2: Download Pretrained Model
```bash
# Download SlowFast-ResNet50-4x16.pth to data/models/pretrained_models/
```

### Step 3: Prepare Data
```bash
# If using AVA
ln -s /path/to/AVA data/AVA

# If using custom data
# Organize data according to structure above
# Register dataset in paths_catalog.py
```

### Step 4: Create Config
```bash
cp config_files/resnet50_4x16f_baseline.yaml config_files/my_config.yaml
# Edit my_config.yaml:
# - Set MODEL.ROI_ACTION_HEAD.NUM_CLASSES to your number of classes
# - Set DATASETS.TRAIN and DATASETS.TEST
# - Set OUTPUT_DIR
```

### Step 5: Start Training
```bash
# Single GPU
python train_net.py \
  --config-file "config_files/my_config.yaml" \
  --transfer \
  --no-head \
  --use-tfboard \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.VIDEOS_PER_BATCH 2 \
  TEST.VIDEOS_PER_BATCH 2

# OR Multi-GPU (4 GPUs)
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_net.py \
  --config-file "config_files/my_config.yaml" \
  --transfer \
  --no-head \
  --use-tfboard
```

### Step 6: Monitor Training
```bash
# In another terminal
tensorboard --logdir=data/output/my_experiment

# Watch logs
tail -f data/output/my_experiment/log.txt
```

### Step 7: Evaluate
```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  test_net.py \
  --config-file "config_files/my_config.yaml" \
  MODEL.WEIGHT "data/output/my_experiment/model_final.pth"
```

---

## Performance Benchmarks

### Expected Training Time (8x V100 GPUs)
- **ResNet50-4x16**: ~24 hours for 90K iterations
- **ResNet101-8x8**: ~36 hours for 90K iterations

### Expected mAP on AVA v2.2
- **ResNet50 Baseline**: ~26.7 mAP
- **ResNet50 + Dense Serial IA**: ~30.0 mAP
- **ResNet101 Baseline**: ~29.3 mAP
- **ResNet101 + Dense Serial IA**: ~32.4 mAP

---

## Tips for Best Results

1. **Start with Baseline**: Begin with baseline config (no IA structure)
2. **Use Pretrained Weights**: Always start from Kinetics pretrained model
3. **Gradual Learning**: Use warmup (`WARMUP_ITERS: 2000`)
4. **Data Augmentation**: Enable `SLOW_JITTER` and `COLOR_JITTER`
5. **Validation**: Monitor validation performance regularly
6. **Learning Rate**: Use linear scaling rule when changing batch size
7. **Checkpointing**: Save checkpoints frequently
8. **IA Structure**: Add IA structure after baseline converges for multi-person scenes

---

## References

- Paper: [Asynchronous Interaction Aggregation for Action Detection](https://arxiv.org/abs/2004.07485) (ECCV 2020)
- AVA Dataset: [https://research.google.com/ava/](https://research.google.com/ava/)
- Kinetics Dataset: [https://deepmind.com/research/open-source/kinetics](https://deepmind.com/research/open-source/kinetics)

---

## Need Help?

- Check [GETTING_STARTED.md](../GETTING_STARTED.md) for basic usage
- Check [DATA.md](../DATA.md) for data preparation
- Check [MODEL_ZOO.md](../MODEL_ZOO.md) for pretrained models
- Open an issue on GitHub for bugs or questions

