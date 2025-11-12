# SlowFast Finetuning Quick Reference

Quick reference card for common finetuning scenarios.

---

## Quick Commands

### 1. Single GPU Training (New Classes)
```bash
python train_net.py \
  --config-file config_files/resnet50_4x16f_baseline.yaml \
  --transfer --no-head --use-tfboard \
  SOLVER.BASE_LR 0.00005 \
  SOLVER.VIDEOS_PER_BATCH 2 \
  TEST.VIDEOS_PER_BATCH 2 \
  SOLVER.MAX_ITER 720000 \
  SOLVER.STEPS '(400000, 560000)'
```

### 2. 4 GPU Training (Same Classes)
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/resnet50_4x16f_baseline.yaml \
  --transfer --use-tfboard \
  SOLVER.BASE_LR 0.0002 \
  SOLVER.VIDEOS_PER_BATCH 8 \
  TEST.VIDEOS_PER_BATCH 8 \
  SOLVER.MAX_ITER 180000 \
  SOLVER.STEPS '(100000, 140000)'
```

### 3. 8 GPU Training (Default)
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  train_net.py \
  --config-file config_files/resnet50_4x16f_baseline.yaml \
  --transfer --no-head --use-tfboard
```

### 4. Resume from Checkpoint
```bash
python train_net.py \
  --config-file config_files/resnet50_4x16f_baseline.yaml \
  --use-tfboard \
  MODEL.WEIGHT data/output/my_exp/model_0050000.pth
```

---

## Generate Config File

### Interactive Mode
```bash
python tools/generate_finetune_config.py --interactive
```

### Command Line
```bash
# For custom dataset with 15 classes
python tools/generate_finetune_config.py \
  --output config_files/my_config.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --backbone resnet50 \
  --num-gpus 4

# With Interaction Aggregation
python tools/generate_finetune_config.py \
  --output config_files/my_config.yaml \
  --dataset-name my_dataset \
  --num-classes 20 \
  --backbone resnet101 \
  --use-ia \
  --ia-structure denseserial \
  --num-gpus 8
```

---

## Linear Scaling Rule

When changing number of GPUs, adjust hyperparameters:

| GPUs | Batch Size | Base LR | Max Iter | Steps |
|------|------------|---------|----------|-------|
| 1 | 2 | 0.00005 | 720000 | (400000, 560000) |
| 2 | 4 | 0.0001 | 360000 | (200000, 280000) |
| 4 | 8 | 0.0002 | 180000 | (100000, 140000) |
| 8 | 16 | 0.0004 | 90000 | (50000, 70000) |

**Formula:**
- `new_lr = base_lr * (new_gpus / 8)`
- `new_iter = base_iter * (8 / new_gpus)`
- `new_batch = base_batch * (new_gpus / 8)`

---

## Common Flags

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `--transfer` | Enable transfer learning | Starting from pretrained model |
| `--no-head` | Don't load head weights | Different number of classes |
| `--use-tfboard` | Enable TensorBoard | Monitoring training |
| `--skip-val-in-train` | Skip validation | Speed up training |
| `--adjust-lr` | Adjust LR scheduler | Resuming with new schedule |
| `--seed SEED` | Set random seed | Reproducibility |

---

## Config File Key Parameters

### Model Configuration
```yaml
MODEL:
  WEIGHT: "data/models/pretrained_models/SlowFast-ResNet50-4x16.pth"
  BACKBONE:
    CONV_BODY: "Slowfast-Resnet50"  # or "Slowfast-Resnet101"
  ROI_ACTION_HEAD:
    NUM_CLASSES: 80  # CHANGE THIS!
    DROPOUT_RATE: 0.2
```

### Training Configuration
```yaml
SOLVER:
  BASE_LR: 0.0004
  MAX_ITER: 90000
  STEPS: (50000, 70000)
  VIDEOS_PER_BATCH: 16
  CHECKPOINT_PERIOD: 10000
  EVAL_PERIOD: 10000
```

### Dataset Configuration
```yaml
DATASETS:
  TRAIN: ("your_dataset_train",)
  TEST: ("your_dataset_val",)
```

### Input Configuration
```yaml
INPUT:
  FRAME_NUM: 64        # Total frames to load
  FRAME_SAMPLE_RATE: 1 # Sample every N frames
  TAU: 16              # Slow pathway stride
  ALPHA: 8             # Fast/Slow ratio
```

---

## Register Custom Dataset

Edit `alphaction/config/paths_catalog.py`:

```python
DATASETS = {
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
}
```

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=data/output/my_experiment
```

### Watch Logs
```bash
tail -f data/output/my_experiment/log.txt
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
SOLVER.VIDEOS_PER_BATCH 1
TEST.VIDEOS_PER_BATCH 1

# Reduce frames
INPUT.FRAME_NUM 32

# Reduce proposals
MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP 5
```

### Slow Training
```bash
# Increase data workers
DATALOADER.NUM_WORKERS 8

# Skip validation during training
--skip-val-in-train

# Reduce eval frequency
SOLVER.EVAL_PERIOD 20000
```

### Low Accuracy
```bash
# Train longer
SOLVER.MAX_ITER 120000

# Lower learning rate
SOLVER.BASE_LR 0.0002

# Enable augmentation
INPUT.SLOW_JITTER True
INPUT.COLOR_JITTER True
```

---

## File Locations

### Pretrained Models
```
data/models/pretrained_models/
├── SlowFast-ResNet50-4x16.pth
└── SlowFast-ResNet101-8x8.pth
```

### Checkpoints
```
data/output/my_experiment/
├── log.txt
├── last_checkpoint
├── model_0010000.pth
├── model_0020000.pth
└── ...
```

### TensorBoard Logs
```
data/output/my_experiment/
└── events.out.tfevents.*
```

---

## Performance Benchmarks

### Training Time (8x V100)
- ResNet50: ~24 hours (90K iter)
- ResNet101: ~36 hours (90K iter)

### Expected mAP (AVA v2.2)
- ResNet50 Baseline: 26.7
- ResNet50 + IA: 30.0
- ResNet101 Baseline: 29.3
- ResNet101 + IA: 32.4

---

## Useful Links

- **Detailed Guide**: [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)
- **Getting Started**: [../GETTING_STARTED.md](../GETTING_STARTED.md)
- **Data Preparation**: [../DATA.md](../DATA.md)
- **Model Zoo**: [../MODEL_ZOO.md](../MODEL_ZOO.md)

---

## Example Workflows

### Workflow 1: Quick Test (1 GPU, Small Dataset)
```bash
# 1. Generate config
python tools/generate_finetune_config.py \
  -o config_files/test.yaml \
  --dataset-name test \
  --num-classes 10 \
  --num-gpus 1 \
  --max-iter 10000

# 2. Train
python train_net.py \
  --config-file config_files/test.yaml \
  --transfer --no-head --use-tfboard \
  SOLVER.VIDEOS_PER_BATCH 2
```

### Workflow 2: Production Training (8 GPUs)
```bash
# 1. Generate config
python tools/generate_finetune_config.py \
  -o config_files/production.yaml \
  --dataset-name production \
  --num-classes 50 \
  --backbone resnet101 \
  --use-ia \
  --num-gpus 8

# 2. Train
python -m torch.distributed.launch --nproc_per_node=8 \
  train_net.py \
  --config-file config_files/production.yaml \
  --transfer --no-head --use-tfboard

# 3. Monitor
tensorboard --logdir=data/output/production_resnet101_64f_denseserial
```

### Workflow 3: Resume Training
```bash
# Check last checkpoint
cat data/output/my_experiment/last_checkpoint

# Resume
python -m torch.distributed.launch --nproc_per_node=8 \
  train_net.py \
  --config-file config_files/my_config.yaml \
  --use-tfboard
```

---

## Quick Tips

1. **Always use `--transfer`** when starting from pretrained model
2. **Use `--no-head`** if your `NUM_CLASSES` differs from 80
3. **Enable TensorBoard** with `--use-tfboard` for monitoring
4. **Follow linear scaling rule** when changing GPU count
5. **Keep BatchNorm frozen** (`FROZEN_BN: True`) for stability
6. **Start with baseline** before adding IA structure
7. **Save checkpoints frequently** (`CHECKPOINT_PERIOD: 10000`)
8. **Validate regularly** to catch overfitting early
9. **Use warmup** to stabilize initial training
10. **Monitor GPU memory** and adjust batch size accordingly

