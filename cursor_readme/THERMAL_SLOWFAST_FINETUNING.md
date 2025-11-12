# Thermal SlowFast Model Finetuning Guide

**Version**: 1.0  
**Date**: November 12, 2025  
**Status**: ✅ Ready for Training

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Integration](#dataset-integration)
4. [Training Configuration](#training-configuration)
5. [Training Execution](#training-execution)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Expected Results](#expected-results)

---

## Overview

This guide explains how to finetune the pretrained ResNet101-8x8 SlowFast model with Dense Serial IA structure on the thermal action detection dataset. The thermal dataset contains 314 training samples and 73 validation samples across 14 action classes.

### Key Features

- **Pretrained Model**: ResNet101-8x8 with Dense Serial IA (trained on AVA dataset)
- **Thermal Dataset**: 40×60 resolution → resized to 256×384
- **14 Action Classes**: Person action categories from thermal sensors
- **Small Dataset Handling**: Increased dropout, lower LR, frequent evaluation
- **HDF5 Integration**: Direct loading from thermal HDF5 files

---

## Prerequisites

### 1. Environment Setup

Activate the AlphAction environment:

```bash
cd /home/ec2-user/jingchen/AlphAction
source activate_uv_env.sh
```

### 2. Pretrained Model

Ensure the pretrained model is available:

```bash
ls -lh data/models/aia_models/resnet101_8x8f_denseserial.pth
```

If not present, download from the model zoo.

### 3. Thermal Dataset

Verify thermal dataset is generated:

```bash
ls -lh ThermalDataGen/thermal_action_dataset/frames/*.h5
ls -lh ThermalDataGen/thermal_action_dataset/annotations/train.json
ls -lh ThermalDataGen/thermal_action_dataset/annotations/val.json
```

Expected output:
- 8 HDF5 files (one per sensor)
- train.json with 314 images
- val.json with 73 images

---

## Dataset Integration

### Architecture Overview

The thermal dataset is integrated via the `ThermalAVADataset` class, which adapts the HDF5 data format to AlphAction's AVA-style interface.

```
Thermal HDF5 Files
       ↓
ThermalAVADataset
       ↓
AlphAction DataLoader
       ↓
SlowFast Model
```

### Data Flow

1. **HDF5 Loading**: Load 64 consecutive frames from HDF5 (±32 frames around keyframe)
2. **Format Conversion**: YOLO bbox (centerXYWH) → XYXY absolute coordinates
3. **Resolution Scaling**: 40×60 → 256×384 via bilinear interpolation
4. **Channel Replication**: Single thermal channel → 3 channels (R=G=B)
5. **Action Encoding**: 14 thermal classes → 81-dim binary vector (AVA compatible)

### Key Components

**ThermalAVADataset** ([alphaction/dataset/datasets/thermal_ava.py](../alphaction/dataset/datasets/thermal_ava.py)):
- Loads HDF5 frames chronologically
- Converts thermal annotations to packed format
- Returns same structure as AVAVideoDataset

**Dataset Registration** ([alphaction/config/paths_catalog.py](../alphaction/config/paths_catalog.py)):
- `thermal_action_train`: Training dataset
- `thermal_action_val`: Validation dataset

---

## Training Configuration

### Configuration File

The training configuration is in [`config_files/thermal_resnet101_8x8f_denseserial.yaml`](../config_files/thermal_resnet101_8x8f_denseserial.yaml).

### Key Parameters

#### Model Configuration

```yaml
MODEL:
  WEIGHT: "data/models/aia_models/resnet101_8x8f_denseserial.pth"
  BACKBONE:
    CONV_BODY: "Slowfast-Resnet101"
    FROZEN_BN: True  # Keep BatchNorm frozen
  ROI_ACTION_HEAD:
    NUM_CLASSES: 14  # Thermal action classes (0-13)
    DROPOUT_RATE: 0.3  # Increased from 0.2 for regularization
  IA_STRUCTURE:
    ACTIVE: True
    STRUCTURE: "denseserial"  # Dense Serial IA for multi-person reasoning
```

#### Input Configuration

```yaml
INPUT:
  FRAME_NUM: 64  # 64 consecutive frames
  FRAME_SAMPLE_RATE: 1  # No temporal gaps
  TAU: 8  # Slow pathway stride
  ALPHA: 4  # Fast pathway 4x more frames
  MIN_SIZE_TRAIN: 256  # Resize to 256×384
  MAX_SIZE_TRAIN: 384
  SLOW_JITTER: True  # Temporal augmentation
  COLOR_JITTER: False  # Skip for thermal
```

#### Solver Configuration

```yaml
SOLVER:
  BASE_LR: 0.0001  # Lower LR for small dataset
  WARMUP_ITERS: 500  # Shorter warmup
  MAX_ITER: 10000  # ~32 epochs for 314 samples
  STEPS: (7000, 9000)  # LR decay at 70% and 90%
  VIDEOS_PER_BATCH: 4  # Small batch size
  CHECKPOINT_PERIOD: 1000  # Save every 1K iterations
  EVAL_PERIOD: 1000  # Evaluate every 1K iterations
```

---

## Training Execution

### Training Command

#### Single GPU Training

```bash
python train_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  --transfer \
  --no-head \
  --use-tfboard
```

**Flags Explanation:**
- `--transfer`: Load pretrained backbone and IA weights
- `--no-head`: Don't load classification head (80 classes → 14 classes)
- `--use-tfboard`: Enable TensorBoard logging

#### Adjust Batch Size (if OOM)

```bash
python train_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  --transfer \
  --no-head \
  --use-tfboard \
  SOLVER.VIDEOS_PER_BATCH 2 \
  TEST.VIDEOS_PER_BATCH 2
```

#### Multi-GPU Training (2 GPUs)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  train_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  --transfer \
  --no-head \
  --use-tfboard \
  SOLVER.BASE_LR 0.0002 \
  SOLVER.VIDEOS_PER_BATCH 8
```

**Note**: Scale learning rate proportionally with batch size.

---

## Monitoring and Evaluation

### TensorBoard Monitoring

Start TensorBoard to monitor training:

```bash
tensorboard --logdir=data/output/thermal_resnet101_8x8f
```

Open browser at: `http://localhost:6006`

**Key Metrics to Watch:**
- `loss_action`: Action classification loss (should decrease)
- `loss_objectness`: Person detection loss
- `accuracy`: Training accuracy
- `learning_rate`: Current learning rate

### Training Logs

Monitor training progress via logs:

```bash
tail -f data/output/thermal_resnet101_8x8f/log.txt
```

### Checkpoints

Checkpoints are saved every 1000 iterations:

```
data/output/thermal_resnet101_8x8f/
├── log.txt
├── last_checkpoint
├── model_0001000.pth
├── model_0002000.pth
├── ...
└── model_final.pth
```

### Validation During Training

The model is evaluated every 1000 iterations on the validation set. Check logs for:
- mAP (mean Average Precision)
- Per-class AP (Average Precision)

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

A. Reduce batch size:
```bash
SOLVER.VIDEOS_PER_BATCH 2
TEST.VIDEOS_PER_BATCH 2
```

B. Reduce frame number:
```bash
INPUT.FRAME_NUM 32
```

C. Disable IA structure temporarily:
```bash
MODEL.IA_STRUCTURE.ACTIVE False
```

### Issue 2: HDF5 File Not Found

**Symptoms:**
```
ValueError: HDF5 file not found for sensor: SL14_R1
```

**Solutions:**

A. Verify HDF5 files exist:
```bash
ls -lh ThermalDataGen/thermal_action_dataset/frames/
```

B. Check paths in config:
```bash
grep hdf5_root alphaction/config/paths_catalog.py
```

C. Use absolute paths:
```python
"hdf5_root": "/home/ec2-user/jingchen/AlphAction/ThermalDataGen/thermal_action_dataset/frames"
```

### Issue 3: Loss is NaN

**Symptoms:**
```
loss: nan
```

**Solutions:**

A. Lower learning rate:
```bash
SOLVER.BASE_LR 0.00005
```

B. Increase warmup:
```bash
SOLVER.WARMUP_ITERS 1000
```

C. Check data normalization (thermal should be [0, 1] after transforms)

### Issue 4: Low Validation mAP

**Symptoms:**
- Training loss decreases but validation mAP stays low
- Overfitting to training set

**Solutions:**

A. Increase regularization:
```bash
MODEL.ROI_ACTION_HEAD.DROPOUT_RATE 0.5
SOLVER.WEIGHT_DECAY 1e-6
```

B. More augmentation:
```bash
INPUT.SLOW_JITTER True
```

C. Train longer:
```bash
SOLVER.MAX_ITER 20000
SOLVER.STEPS '(14000, 18000)'
```

### Issue 5: Slow Training

**Symptoms:**
- Very low iterations per second
- Training takes too long

**Solutions:**

A. Increase dataloader workers:
```bash
DATALOADER.NUM_WORKERS 8
```

B. Check HDF5 compression (lower compression = faster loading)

C. Use faster storage (SSD/NVMe) for HDF5 files

---

## Expected Results

### Training Timeline

| Iteration | Expected mAP | Notes |
|-----------|--------------|-------|
| 0 (baseline) | 5-10% | Random predictions |
| 1000 | 15-25% | Initial learning |
| 3000 | 25-35% | Steady improvement |
| 7000 | 30-40% | Before LR decay |
| 10000 (final) | 35-50% | Target performance |

### Performance Metrics

With 314 training samples and pretrained weights:

- **Training Time**: ~2-3 hours on single GPU (V100/A100)
- **Memory Usage**: ~6-8GB VRAM per batch
- **Expected Final mAP**: 35-50% on 14 classes
- **Iterations per Second**: ~0.5-1.0 (depending on hardware)

### Per-Class Performance

Expected AP by class frequency:

| Action Class | Expected AP | Notes |
|--------------|-------------|-------|
| sitting | 50-70% | High frequency, easy |
| standing | 50-70% | High frequency, easy |
| walking | 40-60% | Medium frequency |
| lying down (bed) | 30-50% | Medium frequency |
| transition | 20-40% | Lower frequency, harder |
| leaning | 20-40% | Lower frequency |
| lower position | 20-40% | Lower frequency, ambiguous |
| other | 10-30% | Catch-all, low quality |

---

## Training Best Practices

### 1. Start with Small Batch

Begin with batch size 2 to avoid OOM, then gradually increase:

```bash
# First run
SOLVER.VIDEOS_PER_BATCH 2

# If no OOM, try 4
SOLVER.VIDEOS_PER_BATCH 4
```

### 2. Monitor Early Iterations

Check first 100 iterations carefully:
- Loss should decrease steadily
- No NaN or Inf values
- GPU memory stable

### 3. Validate Frequently

With small dataset (314 samples), evaluate often:
- Every 1000 iterations (default)
- Watch for overfitting

### 4. Save Checkpoints

Keep multiple checkpoints to rollback if needed:
- Best validation mAP checkpoint
- Regular interval checkpoints (every 1K)

### 5. Experiment with Hyperparameters

If initial results are poor, try:
- Different learning rates (0.00005, 0.0001, 0.0002)
- Different dropout rates (0.2, 0.3, 0.5)
- Different batch sizes (2, 4, 8)

---

## Post-Training Analysis

### Evaluate Final Model

```bash
python test_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  MODEL.WEIGHT data/output/thermal_resnet101_8x8f/model_final.pth
```

### Generate Confusion Matrix

Create a script to visualize per-class performance:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load predictions and ground truth
# (implement based on model outputs)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Thermal Action Detection Confusion Matrix')
plt.colorbar()
plt.tight_layout()
plt.savefig('confusion_matrix.png')
```

### Visualize Predictions

Visualize model predictions on thermal frames:

```python
# Load thermal frame and predictions
# Overlay bounding boxes and action labels
# Compare with ground truth
```

---

## Next Steps

### 1. Improve Dataset

- Collect more thermal data (target: 1000+ samples)
- Balance class distribution
- Add more diverse scenarios

### 2. Model Improvements

- Try different backbones (ResNet50, ResNet152)
- Experiment with IA structures (Serial, Parallel)
- Tune hyperparameters systematically

### 3. Deployment

- Export model for inference
- Optimize for real-time processing
- Integrate with thermal sensor pipeline

---

## References

### Documentation
- [Main Finetuning Guide](FINETUNING_GUIDE.md)
- [Thermal Dataset Implementation](../ThermalDataGen/cursor_readme/THERMAL_ACTION_IMPLEMENTATION_SUMMARY.md)
- [AlphAction Paper](https://arxiv.org/abs/2004.07485)

### Code Locations
- Thermal Dataset: [`alphaction/dataset/datasets/thermal_ava.py`](../alphaction/dataset/datasets/thermal_ava.py)
- Training Config: [`config_files/thermal_resnet101_8x8f_denseserial.yaml`](../config_files/thermal_resnet101_8x8f_denseserial.yaml)
- Dataset Catalog: [`alphaction/config/paths_catalog.py`](../alphaction/config/paths_catalog.py)

---

## Summary

This guide provides a complete workflow for finetuning the SlowFast model on thermal action detection data:

1. ✅ **Dataset Integration**: HDF5 thermal data → AlphAction format
2. ✅ **Model Setup**: ResNet101-8x8 with Dense Serial IA
3. ✅ **Training**: Transfer learning with small dataset optimizations
4. ✅ **Evaluation**: Frequent validation and monitoring
5. ✅ **Troubleshooting**: Solutions to common issues

**Ready to train?** Run the training command and monitor progress with TensorBoard!

```bash
python train_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  --transfer \
  --no-head \
  --use-tfboard
```

Good luck with your thermal action detection finetuning! 🚀

