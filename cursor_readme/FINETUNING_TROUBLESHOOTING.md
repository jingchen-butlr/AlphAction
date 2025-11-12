# Finetuning Troubleshooting Guide

Common issues and solutions when finetuning SlowFast models.

---

## Table of Contents
1. [Training Issues](#training-issues)
2. [Memory Issues](#memory-issues)
3. [Data Loading Issues](#data-loading-issues)
4. [Performance Issues](#performance-issues)
5. [Configuration Issues](#configuration-issues)
6. [Environment Issues](#environment-issues)

---

## Training Issues

### 1. Training Loss Not Decreasing

**Symptoms:**
- Loss stays flat or decreases very slowly
- Validation accuracy remains low

**Possible Causes & Solutions:**

**A. Learning Rate Too Low**
```bash
# Increase learning rate
SOLVER.BASE_LR 0.001  # Try 2-5x higher
```

**B. Learning Rate Too High**
```bash
# Decrease learning rate
SOLVER.BASE_LR 0.0001  # Try 2-5x lower
```

**C. Not Loading Pretrained Weights**
```bash
# Verify pretrained model path
ls -lh data/models/pretrained_models/SlowFast-ResNet50-4x16.pth

# Make sure --transfer flag is used
python train_net.py --transfer ...
```

**D. BatchNorm Statistics Issue**
```yaml
# In config file, ensure:
MODEL:
  BACKBONE:
    FROZEN_BN: True
  NONLOCAL:
    FROZEN_BN: True
```

**E. Wrong Number of Classes**
```yaml
# Check NUM_CLASSES matches your dataset
MODEL:
  ROI_ACTION_HEAD:
    NUM_CLASSES: 15  # Should match your action count
```

---

### 2. Loss Becomes NaN or Inf

**Symptoms:**
- Loss shows `nan` or `inf` in logs
- Training crashes

**Solutions:**

**A. Gradient Clipping**
```python
# Add to solver configuration (requires code modification)
SOLVER:
  GRAD_CLIP: 1.0
```

**B. Lower Learning Rate**
```bash
SOLVER.BASE_LR 0.0001  # Reduce by 2-4x
SOLVER.WARMUP_ITERS 5000  # Increase warmup
```

**C. Check Data**
```python
# Verify your annotations don't have invalid values
import json
with open("data/YOUR_DATASET/annotations/train.json") as f:
    data = json.load(f)
    
# Check for invalid bboxes
for ann in data["annotations"]:
    bbox = ann["bbox"]
    assert all(v >= 0 for v in bbox), f"Negative bbox: {bbox}"
    assert bbox[2] > 0 and bbox[3] > 0, f"Invalid bbox size: {bbox}"
```

**D. Mixed Precision Issues**
```bash
# If using mixed precision, disable it temporarily
# (requires code modification to disable AMP)
```

---

### 3. Validation Accuracy Much Lower Than Training

**Symptoms:**
- Training accuracy high, validation accuracy low
- Overfitting

**Solutions:**

**A. Increase Regularization**
```yaml
MODEL:
  ROI_ACTION_HEAD:
    DROPOUT_RATE: 0.5  # Increase from 0.2

SOLVER:
  WEIGHT_DECAY: 1e-6  # Increase from 1e-7
```

**B. Enable Data Augmentation**
```yaml
INPUT:
  SLOW_JITTER: True
  COLOR_JITTER: True
```

**C. More Training Data**
- Collect more diverse training examples
- Use data augmentation techniques

**D. Early Stopping**
```bash
# Stop training when validation plateaus
# Monitor tensorboard and manually stop
```

---

### 4. Training Too Slow

**Symptoms:**
- Very low iterations per second
- Takes too long to complete

**Solutions:**

**A. Increase Data Workers**
```yaml
DATALOADER:
  NUM_WORKERS: 8  # Increase from 4
```

**B. Use Faster Storage**
```bash
# Move data to SSD/NVMe
# Check I/O wait time
iostat -x 1
```

**C. Skip Frequent Validation**
```yaml
SOLVER:
  EVAL_PERIOD: 20000  # Increase from 10000
  
# Or skip validation entirely during training
--skip-val-in-train
```

**D. Reduce Checkpoint Frequency**
```yaml
SOLVER:
  CHECKPOINT_PERIOD: 20000  # Increase from 10000
```

**E. Optimize Video Decoding**
```bash
# Make sure videos are properly encoded
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4
```

---

## Memory Issues

### 1. Out of Memory (OOM) During Training

**Symptoms:**
- CUDA out of memory error
- Training crashes

**Solutions (Apply in order):**

**A. Reduce Batch Size**
```bash
SOLVER.VIDEOS_PER_BATCH 1  # Minimum
TEST.VIDEOS_PER_BATCH 1
```

**B. Reduce Number of Frames**
```yaml
INPUT:
  FRAME_NUM: 32  # Reduce from 64
```

**C. Reduce Proposals Per Clip**
```yaml
MODEL:
  ROI_ACTION_HEAD:
    PROPOSAL_PER_CLIP: 5  # Reduce from 10
```

**D. Use Gradient Checkpointing** (requires code modification)

**E. Use Smaller Backbone**
```yaml
MODEL:
  BACKBONE:
    CONV_BODY: "Slowfast-Resnet50"  # Instead of Resnet101
```

**F. Disable IA Structure**
```yaml
MODEL:
  IA_STRUCTURE:
    ACTIVE: False
```

**G. Clear GPU Cache**
```python
# Add to training script
import torch
torch.cuda.empty_cache()
```

---

### 2. GPU Memory Fragmentation

**Symptoms:**
- OOM error despite having enough memory
- Memory usage gradually increases

**Solutions:**

**A. Restart Training Periodically**
```bash
# Save checkpoint and restart
# Clears memory fragmentation
```

**B. Use Consistent Batch Sizes**
```yaml
# Don't change batch size between train/test
SOLVER.VIDEOS_PER_BATCH: 8
TEST.VIDEOS_PER_BATCH: 8
```

**C. Reduce Memory Pool Size** (IA structure)
```yaml
MODEL:
  IA_STRUCTURE:
    MAX_PERSON: 15  # Reduce from 25
    MAX_PER_SEC: 3  # Reduce from 5
```

---

## Data Loading Issues

### 1. Dataset Not Found Error

**Symptoms:**
```
KeyError: 'your_dataset_train'
```

**Solutions:**

**A. Check Dataset Registration**
```python
# In alphaction/config/paths_catalog.py
DATASETS = {
    "your_dataset_train": {  # Name must match config
        "video_root": "data/YOUR_DATASET/clips/train",
        ...
    }
}
```

**B. Verify Config File**
```yaml
DATASETS:
  TRAIN: ("your_dataset_train",)  # Must match registered name
```

---

### 2. Video Loading Error

**Symptoms:**
```
Error loading video: [video_path]
Failed to read video
```

**Solutions:**

**A. Check Video Files**
```bash
# Verify videos exist and are readable
ffprobe data/YOUR_DATASET/clips/train/video_001/0001.mp4

# Check video format
ffmpeg -i input.mp4
```

**B. Re-encode Videos**
```bash
# Convert to compatible format
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4
```

**C. Check Permissions**
```bash
# Ensure files are readable
chmod -R 755 data/YOUR_DATASET/
```

---

### 3. Slow Data Loading

**Symptoms:**
- GPU utilization low
- CPU bottleneck

**Solutions:**

**A. Increase Workers**
```yaml
DATALOADER:
  NUM_WORKERS: 8  # Or more
```

**B. Use Faster Storage**
```bash
# Move data to SSD
# Check disk speed
dd if=/dev/zero of=test bs=1G count=1 oflag=direct
```

**C. Preload Data to RAM** (if enough RAM)
```bash
# Copy data to /dev/shm (RAM disk)
mkdir -p /dev/shm/data
cp -r data/YOUR_DATASET /dev/shm/data/
```

**D. Optimize Video Encoding**
```bash
# Use fast-decode preset
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast -tune fastdecode output.mp4
```

---

### 4. Annotation Format Error

**Symptoms:**
```
KeyError: 'bbox'
IndexError: list index out of range
```

**Solutions:**

**A. Validate JSON Structure**
```python
import json

with open("data/YOUR_DATASET/annotations/train.json") as f:
    data = json.load(f)
    
# Check required fields
assert "images" in data
assert "annotations" in data
assert "categories" in data

# Check annotation structure
for ann in data["annotations"]:
    assert "bbox" in ann
    assert "category_id" in ann
    assert "image_id" in ann
    assert len(ann["bbox"]) == 4
```

**B. Check Category IDs**
```python
# Category IDs should start from 1
category_ids = [cat["id"] for cat in data["categories"]]
assert min(category_ids) == 1
assert len(category_ids) == max(category_ids)  # Consecutive
```

---

## Performance Issues

### 1. Low mAP on Validation Set

**Symptoms:**
- Training seems fine but validation mAP is low

**Solutions:**

**A. Train Longer**
```yaml
SOLVER:
  MAX_ITER: 180000  # Double the iterations
```

**B. Adjust Learning Rate Schedule**
```yaml
SOLVER:
  STEPS: (120000, 160000)  # Later decay
```

**C. Use Better Backbone**
```yaml
MODEL:
  BACKBONE:
    CONV_BODY: "Slowfast-Resnet101"
  WEIGHT: "data/models/pretrained_models/SlowFast-ResNet101-8x8.pth"
```

**D. Enable IA Structure**
```yaml
MODEL:
  IA_STRUCTURE:
    ACTIVE: True
    STRUCTURE: "denseserial"
```

**E. Check Person Detector Quality**
```bash
# Visualize person detections on validation set
# Make sure bounding boxes are accurate
```

**F. Adjust Detection Thresholds**
```yaml
TEST:
  BOX_THRESH: 0.5  # Lower if missing detections
  ACTION_THRESH: 0.01  # Lower for more predictions
```

---

### 2. Some Actions Never Predicted

**Symptoms:**
- Certain action classes have 0 AP
- Class imbalance

**Solutions:**

**A. Check Class Distribution**
```python
# Count samples per class
from collections import Counter
with open("data/YOUR_DATASET/annotations/train.json") as f:
    data = json.load(f)
category_counts = Counter(ann["category_id"] for ann in data["annotations"])
print(category_counts)
```

**B. Oversample Rare Classes** (requires code modification)

**C. Adjust Class Weights** (requires code modification)

**D. Collect More Data for Rare Classes**

---

## Configuration Issues

### 1. Config File Not Found

**Symptoms:**
```
FileNotFoundError: config_files/my_config.yaml
```

**Solutions:**

**A. Check File Path**
```bash
# Use absolute or correct relative path
ls -lh config_files/my_config.yaml
```

**B. Generate Config**
```bash
python tools/generate_finetune_config.py \
  --output config_files/my_config.yaml \
  ...
```

---

### 2. Parameter Override Not Working

**Symptoms:**
- Command-line parameters not taking effect

**Solutions:**

**A. Check Override Syntax**
```bash
# Correct syntax
SOLVER.BASE_LR 0.0001

# Wrong syntax (missing quotes for tuples)
SOLVER.STEPS (50000, 70000)  # WRONG
SOLVER.STEPS '(50000, 70000)'  # CORRECT
```

**B. Check Parameter Path**
```yaml
# Parameter must exist in config structure
SOLVER.BASE_LR  # Correct path
SOLVER.LEARNING_RATE  # Wrong (doesn't exist)
```

---

### 3. Model Weight Not Loading

**Symptoms:**
```
FileNotFoundError: data/models/pretrained_models/SlowFast-ResNet50-4x16.pth
```

**Solutions:**

**A. Download Pretrained Model**
```bash
# Download from MODEL_ZOO.md
# Save to correct location
mkdir -p data/models/pretrained_models
```

**B. Check Model Path**
```yaml
MODEL:
  WEIGHT: "data/models/pretrained_models/SlowFast-ResNet50-4x16.pth"
```

**C. Use Absolute Path**
```bash
MODEL.WEIGHT "/absolute/path/to/SlowFast-ResNet50-4x16.pth"
```

---

## Environment Issues

### 1. CUDA Out of Memory on Import

**Symptoms:**
- Error before training even starts
- Import failures

**Solutions:**

**A. Clear GPU Memory**
```bash
# Kill other GPU processes
nvidia-smi
kill -9 <PID>

# Or reboot if necessary
```

**B. Reduce Number of Workers**
```yaml
DATALOADER:
  NUM_WORKERS: 0  # Temporarily disable multi-process loading
```

---

### 2. Distributed Training Issues

**Symptoms:**
```
RuntimeError: Address already in use
RuntimeError: NCCL error
```

**Solutions:**

**A. Check Port Availability**
```bash
# Use different port
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29501 \
  train_net.py ...
```

**B. Kill Zombie Processes**
```bash
# Find and kill old processes
ps aux | grep train_net.py
kill -9 <PID>
```

**C. Check GPU Visibility**
```bash
# Ensure all GPUs are visible
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**D. Verify NCCL Installation**
```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

---

### 3. Package Version Conflicts

**Symptoms:**
```
ImportError: cannot import name ...
AttributeError: module has no attribute ...
```

**Solutions:**

**A. Check Package Versions**
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

**B. Reinstall Environment**
```bash
# Using UV (recommended)
source activate_uv_env.sh

# Or recreate conda environment
conda env remove -n alphaction
conda create -n alphaction python=3.9
# Follow INSTALL.md
```

**C. Check CUDA Compatibility**
```bash
# Ensure PyTorch CUDA version matches system CUDA
nvidia-smi  # Check system CUDA version
python -c "import torch; print(torch.version.cuda)"
```

---

## Debugging Tips

### Enable Detailed Logging

```python
# Add to train_net.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Usage

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Log GPU usage
nvidia-smi dmon -s pucvmet -d 1 > gpu_log.txt
```

### Profile Training

```python
# Add to training loop
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Training code
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Visualize Data

```bash
# Visualize training samples
python tools/visualize_dataset.py \
  --dataset-name your_dataset_train \
  --num-samples 10
```

### Test Data Loading

```python
# Test data loader
from alphaction.config import cfg
from alphaction.dataset import make_data_loader

cfg.merge_from_file("config_files/my_config.yaml")
data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)

for batch in data_loader:
    print(f"Batch shapes: {batch[0].shape}")
    break  # Test first batch
```

---

## Getting Help

If you're still stuck after trying these solutions:

1. **Check Logs Carefully**: Error messages often contain the solution
2. **Search Issues**: Check GitHub issues for similar problems
3. **Simplify**: Try minimal config first, then add complexity
4. **Verify Data**: Most issues are data-related
5. **Ask for Help**: Post detailed error logs and configuration

### Information to Include When Asking for Help

```bash
# 1. System info
nvidia-smi
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 2. Config file
cat config_files/my_config.yaml

# 3. Command used
# (paste your training command)

# 4. Error log
# (paste relevant error messages)

# 5. Dataset info
ls -lh data/YOUR_DATASET/
wc -l data/YOUR_DATASET/annotations/train.json
```

---

## Preventing Issues

### Best Practices

1. **Start Small**: Test with small dataset first
2. **Validate Data**: Check data format before training
3. **Monitor Training**: Use TensorBoard
4. **Save Checkpoints**: Frequent checkpointing
5. **Version Control**: Track config files
6. **Document Changes**: Note what works and what doesn't
7. **Test Incrementally**: Add one feature at a time

### Pre-Training Checklist

- [ ] Pretrained model downloaded
- [ ] Data directory structure correct
- [ ] Annotations in COCO format
- [ ] Dataset registered in paths_catalog.py
- [ ] Config file NUM_CLASSES matches data
- [ ] Output directory created
- [ ] GPU memory sufficient
- [ ] Data loader test passed
- [ ] Visualization checked

---

## Related Documentation

- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Complete finetuning guide
- [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Quick commands
- [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) - Dataset preparation
- [GETTING_STARTED.md](../GETTING_STARTED.md) - Basic usage

