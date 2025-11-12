# SlowFast Finetuning Workflow Diagram

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINETUNING WORKFLOW                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ 1. PREPARATION  │
└────────┬────────┘
         │
         ├──► Download Pretrained Model
         │    └─► data/models/pretrained_models/SlowFast-ResNet50-4x16.pth
         │
         ├──► Prepare Dataset
         │    ├─► Option A: Use AVA Dataset (DATA.md)
         │    └─► Option B: Custom Dataset (CUSTOM_DATASET_PREPARATION.md)
         │
         └──► Setup Environment
              └─► source activate_uv_env.sh

                         ↓

┌─────────────────┐
│ 2. REGISTRATION │
└────────┬────────┘
         │
         └──► Edit alphaction/config/paths_catalog.py
              └─► Register your dataset:
                  - video_root
                  - ann_file
                  - box_file

                         ↓

┌─────────────────┐
│ 3. CONFIGURATION│
└────────┬────────┘
         │
         ├──► Option A: Generate Config
         │    └─► python tools/generate_finetune_config.py --interactive
         │
         └──► Option B: Copy & Edit Config
              └─► cp config_files/resnet50_4x16f_baseline.yaml config_files/my_config.yaml
              └─► Edit: NUM_CLASSES, DATASETS, OUTPUT_DIR

                         ↓

┌─────────────────┐
│ 4. TRAINING     │
└────────┬────────┘
         │
         ├──► Option A: Quick Launch
         │    └─► bash cursor_readme/finetune_quickstart.sh config_files/my_config.yaml --num-gpus 4
         │
         └──► Option B: Manual Launch
              ├─► Single GPU:
              │   └─► python train_net.py --config-file config_files/my_config.yaml --transfer --no-head
              │
              └─► Multi-GPU:
                  └─► python -m torch.distributed.launch --nproc_per_node=4 train_net.py --config-file config_files/my_config.yaml --transfer --no-head

                         ↓

┌─────────────────┐
│ 5. MONITORING   │
└────────┬────────┘
         │
         ├──► TensorBoard
         │    └─► tensorboard --logdir=data/output/my_experiment
         │
         ├──► Watch Logs
         │    └─► tail -f data/output/my_experiment/log.txt
         │
         └──► Check GPU Usage
              └─► watch -n 1 nvidia-smi

                         ↓

┌─────────────────┐
│ 6. EVALUATION   │
└────────┬────────┘
         │
         └──► Test Model
              └─► python test_net.py --config-file config_files/my_config.yaml \
                  MODEL.WEIGHT data/output/my_experiment/model_final.pth

                         ↓

┌─────────────────┐
│ 7. ITERATION    │
└────────┬────────┘
         │
         ├──► Analyze Results
         │    └─► Check per-class performance
         │
         ├──► Adjust Hyperparameters
         │    ├─► Learning rate
         │    ├─► Training iterations
         │    └─► Data augmentation
         │
         └──► Resume Training or Start New Experiment
              └─► bash cursor_readme/finetune_quickstart.sh config_files/my_config.yaml --resume
```

---

## Decision Tree

```
                    START
                      │
                      ▼
         ┌────────────────────────┐
         │  Do you have dataset?  │
         └───────┬────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
       YES               NO
        │                 │
        │                 └──► Collect & Annotate Data
        │                      └──► Follow CUSTOM_DATASET_PREPARATION.md
        │
        ▼
┌───────────────────┐
│ Same 80 classes?  │
└────┬──────────────┘
     │
     ├──► YES ──► Use config without modifications
     │            └──► Don't use --no-head flag
     │
     └──► NO  ──► Update NUM_CLASSES in config
                  └──► Use --no-head flag

                      │
                      ▼
         ┌────────────────────┐
         │  GPU availability?  │
         └────┬───────────────┘
              │
              ├──► 1 GPU  ──► VIDEOS_PER_BATCH=2, BASE_LR=0.00005
              ├──► 4 GPUs ──► VIDEOS_PER_BATCH=8, BASE_LR=0.0002
              └──► 8 GPUs ──► Use default config values

                      │
                      ▼
         ┌────────────────────┐
         │ Backbone selection? │
         └────┬───────────────┘
              │
              ├──► Fast/Light ──► ResNet50-4x16
              └──► Accurate   ──► ResNet101-8x8

                      │
                      ▼
         ┌────────────────────┐
         │ Multi-person focus? │
         └────┬───────────────┘
              │
              ├──► YES ──► Enable IA Structure (Dense Serial)
              └──► NO  ──► Baseline (IA_STRUCTURE.ACTIVE=False)

                      │
                      ▼
                START TRAINING!
```

---

## File Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       INPUT FILES                            │
└─────────────────────────────────────────────────────────────┘

data/models/pretrained_models/
├── SlowFast-ResNet50-4x16.pth ─────┐
└── SlowFast-ResNet101-8x8.pth      │
                                    │
data/YOUR_DATASET/                  │  ┌────────────────┐
├── clips/                          ├─►│                │
│   ├── train/                      │  │                │
│   └── val/                        │  │                │
├── keyframes/                      │  │  TRAINING      │
│   ├── train/                      ├─►│   PROCESS      │
│   └── val/                        │  │                │
└── annotations/                    │  │                │
    ├── train.json ─────────────────┤  │                │
    └── val.json ───────────────────┘  └───────┬────────┘
                                               │
config_files/                                  │
└── my_config.yaml ────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT FILES                            │
└─────────────────────────────────────────────────────────────┘

data/output/my_experiment/
├── log.txt                    # Training log
├── last_checkpoint            # Path to last checkpoint
├── model_0010000.pth         # Checkpoint at 10K iters
├── model_0020000.pth         # Checkpoint at 20K iters
├── model_final.pth           # Final model
└── events.out.tfevents.*     # TensorBoard logs
```

---

## Time Estimation

```
TYPICAL TIMELINE FOR FINETUNING PROJECT

Day 1: Setup (2-4 hours)
├── Environment setup              [30 min]
├── Download pretrained models     [20 min]
├── Data preparation              [1-2 hours]
└── Initial testing               [30 min]

Day 2-3: Training (24-48 hours)
├── Baseline training            [24 hours on 8x V100]
├── Monitoring & validation       [periodic checks]
└── Checkpoint saving            [automatic]

Day 4: Analysis & Iteration (4-8 hours)
├── Evaluate results             [1 hour]
├── Analyze per-class performance [2 hours]
├── Adjust hyperparameters       [1 hour]
└── Resume training              [4+ hours]

TOTAL: ~4-5 days for complete cycle
```

---

## Resource Requirements

```
MINIMUM REQUIREMENTS
├── GPU: 1x NVIDIA GPU with 11GB+ VRAM
├── CPU: 8+ cores
├── RAM: 32GB
├── Storage: 500GB SSD
└── Time: 48+ hours for training

RECOMMENDED SETUP
├── GPU: 4-8x NVIDIA V100/A100
├── CPU: 32+ cores
├── RAM: 128GB
├── Storage: 1TB+ NVMe SSD
└── Time: 24-36 hours for training

STORAGE BREAKDOWN
├── Pretrained models: ~500MB
├── AVA dataset: ~50GB
├── Checkpoints: ~10GB per experiment
├── Logs & results: ~1GB
└── Custom dataset: varies (typically 20-100GB)
```

---

## Common Paths Through Documentation

```
PATH 1: EXPERIENCED USER (Quick Start)
1. FINETUNING_INDEX.md          [2 min]
2. FINETUNING_QUICK_REFERENCE.md [5 min]
3. generate_finetune_config.py   [2 min]
4. finetune_quickstart.sh        [1 min]
   Total: ~10 minutes to start training

PATH 2: BEGINNER (Comprehensive)
1. FINETUNING_INDEX.md           [5 min]
2. FINETUNING_GUIDE.md           [30 min]
3. CUSTOM_DATASET_PREPARATION.md [20 min]
4. Practice with AVA dataset     [60 min]
5. Try custom dataset            [120 min]
   Total: ~3.5 hours to full understanding

PATH 3: TROUBLESHOOTING (Problem Solving)
1. Identify problem
2. FINETUNING_TROUBLESHOOTING.md [variable]
3. Apply solution
4. Resume training
   Total: varies by issue

PATH 4: CUSTOM DATASET (From Scratch)
1. FINETUNING_INDEX.md                   [5 min]
2. CUSTOM_DATASET_PREPARATION.md         [20 min]
3. Prepare data                          [2-4 hours]
4. FINETUNING_GUIDE.md (Dataset section) [15 min]
5. generate_finetune_config.py           [5 min]
6. Start training                        [5 min]
   Total: ~3-5 hours before training
```

---

## Success Metrics

```
TRAINING PROGRESS INDICATORS

Good Training:
├── Loss decreases steadily
├── Validation mAP increases
├── GPU utilization > 80%
├── No NaN/Inf in losses
└── Checkpoints saving successfully

Warning Signs:
├── Loss plateaus early
├── Validation mAP drops
├── GPU utilization < 50%
├── Frequent OOM errors
└── Loss becomes NaN

FINAL EVALUATION TARGETS

For AVA v2.2:
├── ResNet50 Baseline: 26-27 mAP
├── ResNet50 + IA: 29-30 mAP
├── ResNet101 Baseline: 28-29 mAP
└── ResNet101 + IA: 31-32 mAP

For Custom Dataset:
├── Depends on dataset difficulty
├── Compare to baseline (no pretrain)
├── Should see 2-3x improvement with pretrain
└── Monitor per-class performance
```

---

## Command Cheatsheet

```bash
# GENERATE CONFIG
python tools/generate_finetune_config.py --interactive

# TRAINING
# Single GPU, new classes
python train_net.py \
  --config-file config_files/my_config.yaml \
  --transfer --no-head --use-tfboard \
  SOLVER.VIDEOS_PER_BATCH 2

# Multi-GPU (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/my_config.yaml \
  --transfer --no-head --use-tfboard

# Using quickstart script
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 --new-classes

# MONITORING
tensorboard --logdir=data/output/my_experiment
tail -f data/output/my_experiment/log.txt
watch -n 1 nvidia-smi

# EVALUATION
python test_net.py \
  --config-file config_files/my_config.yaml \
  MODEL.WEIGHT data/output/my_experiment/model_final.pth

# RESUME TRAINING
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 --resume
```

---

## Quick Troubleshooting

```
PROBLEM                         QUICK FIX
─────────────────────────────────────────────────────────────
Out of Memory                   SOLVER.VIDEOS_PER_BATCH 1
Loss is NaN                     SOLVER.BASE_LR 0.0001 (reduce)
Training too slow               DATALOADER.NUM_WORKERS 8
Dataset not found               Check paths_catalog.py
Low accuracy                    Train longer (MAX_ITER)
Wrong number of classes         Update NUM_CLASSES, use --no-head
Model not loading               Check MODEL.WEIGHT path
```

---

## Next Steps

After reviewing this workflow:

1. **Go to** [FINETUNING_INDEX.md](FINETUNING_INDEX.md) for navigation
2. **Choose** your path based on experience level
3. **Follow** the appropriate guide
4. **Use** the tools to simplify setup
5. **Start** training!

---

**Remember**: The key to successful finetuning is:
- ✅ Good quality data
- ✅ Proper configuration
- ✅ Patient monitoring
- ✅ Iterative improvement

Happy Finetuning! 🚀

