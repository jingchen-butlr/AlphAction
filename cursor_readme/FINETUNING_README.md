# SlowFast Model Finetuning Documentation

## Overview

This directory contains comprehensive documentation and tools for finetuning pretrained SlowFast models on AVA dataset or your custom action detection datasets.

---

## 📚 What's Available

### Documentation (5 Guides)

1. **[FINETUNING_INDEX.md](FINETUNING_INDEX.md)** - Start here! Navigation hub for all finetuning resources
2. **[FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)** - Complete step-by-step finetuning guide (30 min read)
3. **[FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md)** - Quick commands and common scenarios (5 min read)
4. **[CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md)** - Prepare your own dataset (20 min read)
5. **[FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md)** - Solutions to common problems (reference)

### Tools (2 Scripts)

1. **[finetune_quickstart.sh](finetune_quickstart.sh)** - Bash script for easy training launch
2. **[../tools/generate_finetune_config.py](../tools/generate_finetune_config.py)** - Python tool to generate config files

---

## 🚀 Quick Start (60 seconds)

### For AVA Dataset:

```bash
# 1. Download pretrained model
# Get SlowFast-ResNet50-4x16.pth from MODEL_ZOO.md
# Save to: data/models/pretrained_models/

# 2. Launch training on 4 GPUs
bash cursor_readme/finetune_quickstart.sh \
  config_files/resnet50_4x16f_baseline.yaml \
  --num-gpus 4

# 3. Monitor
tensorboard --logdir=data/output/resnet50_4x16f_baseline
```

### For Custom Dataset:

```bash
# 1. Prepare your data (see CUSTOM_DATASET_PREPARATION.md)

# 2. Generate config
python tools/generate_finetune_config.py --interactive

# 3. Register dataset in alphaction/config/paths_catalog.py

# 4. Launch training
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 \
  --new-classes
```

---

## 📖 Documentation Structure

```
cursor_readme/
├── FINETUNING_INDEX.md              # 👈 START HERE - Navigation hub
├── FINETUNING_README.md             # This file - Overview
├── FINETUNING_GUIDE.md              # Complete detailed guide
├── FINETUNING_QUICK_REFERENCE.md    # Quick commands
├── CUSTOM_DATASET_PREPARATION.md    # Dataset preparation
├── FINETUNING_TROUBLESHOOTING.md    # Problem solving
└── finetune_quickstart.sh           # Training launcher script

tools/
└── generate_finetune_config.py      # Config file generator
```

---

## 🎯 Use Case Matrix

| Your Situation | Recommended Starting Point |
|----------------|---------------------------|
| First time finetuning | [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) |
| Experienced user | [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) |
| Custom dataset | [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) |
| Have a problem | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) |
| Looking for specific info | [FINETUNING_INDEX.md](FINETUNING_INDEX.md) |

---

## 🔑 Key Concepts

### What is Finetuning?

Finetuning means taking a model pretrained on one dataset (Kinetics-700) and adapting it to your specific task or dataset. This is much faster and more accurate than training from scratch.

### Pretrained Models

Two SlowFast backbones available:
- **ResNet50-4x16**: Faster, lighter (recommended for starting)
- **ResNet101-8x8**: Slower, more accurate (for production)

### When to Use `--no-head`

- **Use `--no-head`**: When your dataset has different number of action classes than 80
- **Don't use `--no-head`**: When using AVA's 80 action classes

### GPU Scaling

The config files assume 8 GPUs. When using different GPU counts:
- Adjust learning rate proportionally
- Adjust training iterations inversely
- Use the quickstart script to handle this automatically

---

## 💻 Tools Usage

### Config Generator

**Interactive Mode** (Recommended for beginners):
```bash
python tools/generate_finetune_config.py --interactive
```

**Command Line Mode** (For scripting):
```bash
python tools/generate_finetune_config.py \
  --output config_files/custom.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --backbone resnet50 \
  --num-gpus 4
```

### Training Launcher

**Single GPU**:
```bash
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --single-gpu \
  --new-classes
```

**Multiple GPUs**:
```bash
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 \
  --new-classes
```

**Resume Training**:
```bash
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 \
  --resume
```

---

## 📊 What to Expect

### Training Time

On 8x V100 GPUs:
- ResNet50: ~24 hours (90K iterations)
- ResNet101: ~36 hours (90K iterations)

Scale proportionally for different GPU counts.

### Performance

Expected mAP on AVA v2.2:
- ResNet50 Baseline: ~26.7 mAP
- ResNet50 + IA: ~30.0 mAP
- ResNet101 Baseline: ~29.3 mAP
- ResNet101 + IA: ~32.4 mAP

For custom datasets, performance varies based on:
- Dataset quality and size
- Number of action classes
- Video quality
- Annotation accuracy

---

## 🔍 Finding Information

The documentation is designed for quick navigation:

**"How do I..."**
- Start training quickly? → [Quick Reference](FINETUNING_QUICK_REFERENCE.md)
- Understand everything? → [Complete Guide](FINETUNING_GUIDE.md)
- Prepare my data? → [Dataset Preparation](CUSTOM_DATASET_PREPARATION.md)
- Fix a problem? → [Troubleshooting](FINETUNING_TROUBLESHOOTING.md)
- Find a specific topic? → [Index](FINETUNING_INDEX.md)

**"I'm having issues with..."**
- Memory problems → [Troubleshooting - Memory Issues](FINETUNING_TROUBLESHOOTING.md#memory-issues)
- Slow training → [Troubleshooting - Performance Issues](FINETUNING_TROUBLESHOOTING.md#performance-issues)
- Data loading → [Troubleshooting - Data Loading Issues](FINETUNING_TROUBLESHOOTING.md#data-loading-issues)

---

## 🎓 Learning Path

### Beginner Path (3-4 hours total)

1. **Read**: [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Introduction & Prerequisites (15 min)
2. **Download**: Pretrained model and AVA dataset (30 min)
3. **Practice**: Run first training on single GPU (30 min)
4. **Monitor**: Learn to use TensorBoard (15 min)
5. **Scale**: Try multi-GPU training (30 min)
6. **Custom**: Prepare custom dataset (60 min)

### Advanced Path (1 hour)

1. **Review**: [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) (10 min)
2. **Generate**: Create config for your use case (5 min)
3. **Launch**: Start training with quickstart script (5 min)
4. **Optimize**: Tune hyperparameters (40 min)

---

## 🛠️ Common Workflows

### Workflow 1: Test Finetuning on AVA

```bash
# Download pretrained model to data/models/pretrained_models/

# Train on 1 GPU for quick test
python train_net.py \
  --config-file config_files/resnet50_4x16f_baseline.yaml \
  --transfer --use-tfboard \
  SOLVER.BASE_LR 0.00005 \
  SOLVER.VIDEOS_PER_BATCH 2 \
  SOLVER.MAX_ITER 1000  # Short test run
```

### Workflow 2: Production Training

```bash
# 1. Generate production config
python tools/generate_finetune_config.py \
  --output config_files/production.yaml \
  --dataset-name my_dataset \
  --num-classes 20 \
  --backbone resnet101 \
  --use-ia \
  --num-gpus 8

# 2. Launch training
python -m torch.distributed.launch --nproc_per_node=8 \
  train_net.py \
  --config-file config_files/production.yaml \
  --transfer --no-head --use-tfboard

# 3. Monitor (in another terminal)
tensorboard --logdir=data/output/production_resnet101_64f_denseserial
```

### Workflow 3: Iterative Development

```bash
# Start with baseline
python tools/generate_finetune_config.py \
  -o config_files/baseline.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --backbone resnet50 \
  --num-gpus 4

# Train baseline
bash cursor_readme/finetune_quickstart.sh \
  config_files/baseline.yaml \
  --num-gpus 4 --new-classes

# After baseline converges, add IA structure
python tools/generate_finetune_config.py \
  -o config_files/with_ia.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --backbone resnet50 \
  --use-ia \
  --num-gpus 4

# Train with IA (loading baseline checkpoint)
bash cursor_readme/finetune_quickstart.sh \
  config_files/with_ia.yaml \
  --num-gpus 4 --resume
```

---

## ⚠️ Important Notes

### Before Training

1. **Check GPU Memory**: Ensure sufficient VRAM (≥11GB recommended)
2. **Verify Data**: Validate dataset format before training
3. **Storage Space**: Ensure enough disk space for checkpoints
4. **Pretrained Models**: Download to correct location

### During Training

1. **Monitor**: Use TensorBoard to watch training progress
2. **Checkpoints**: Saved every N iterations (configurable)
3. **Validation**: Runs periodically during training
4. **GPU Usage**: Monitor with `nvidia-smi`

### After Training

1. **Evaluate**: Test on validation set
2. **Analyze**: Check per-class performance
3. **Iterate**: Adjust hyperparameters based on results
4. **Deploy**: Use trained model for inference

---

## 🐛 Common Pitfalls

1. **Forgetting `--transfer`**: Results in random initialization
2. **Wrong `NUM_CLASSES`**: Causes shape mismatch errors
3. **Not using `--no-head`**: Error when class count differs
4. **Incorrect dataset registration**: Dataset not found error
5. **Insufficient GPU memory**: Reduce batch size
6. **Poor data quality**: Low accuracy despite training

See [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) for solutions.

---

## 📞 Getting Help

### Documentation
1. Check [FINETUNING_INDEX.md](FINETUNING_INDEX.md) for navigation
2. Read relevant guide section
3. Review [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md)

### Community
- GitHub Issues: For bugs and feature requests
- Discussions: For questions and sharing experiences

### Information to Provide
When asking for help, include:
- Command used
- Config file contents
- Error message (full traceback)
- System info (GPU, CUDA version, etc.)
- Dataset statistics

---

## 🎯 Success Checklist

Before starting your finetuning project:

- [ ] Read [FINETUNING_INDEX.md](FINETUNING_INDEX.md)
- [ ] Choose appropriate guide based on use case
- [ ] Download pretrained SlowFast model
- [ ] Prepare dataset (AVA or custom)
- [ ] Register dataset in `paths_catalog.py`
- [ ] Generate or modify config file
- [ ] Verify data format and paths
- [ ] Check GPU memory availability
- [ ] Set up TensorBoard for monitoring
- [ ] Run small test first
- [ ] Scale up to full training

---

## 📈 Recommended Settings

### For Quick Testing
- GPU: 1
- Batch Size: 2
- Max Iter: 10000
- Backbone: ResNet50

### For Production
- GPU: 8
- Batch Size: 16
- Max Iter: 90000+
- Backbone: ResNet101 + IA

### For Limited Resources
- GPU: 1-2
- Batch Size: 1-2
- Frame Num: 32 (reduced)
- Max Iter: 180000+ (compensate for small batch)

---

## 🔗 Related Resources

### In This Repository
- [Main README](../README.md)
- [Installation](../INSTALL_UV.md)
- [Getting Started](../GETTING_STARTED.md)
- [Data Preparation](../DATA.md)
- [Model Zoo](../MODEL_ZOO.md)

### External Resources
- [Paper: Asynchronous Interaction Aggregation](https://arxiv.org/abs/2004.07485)
- [AVA Dataset](https://research.google.com/ava/)
- [Kinetics Dataset](https://deepmind.com/research/open-source/kinetics)

---

## 📝 Summary

This documentation provides everything you need to finetune SlowFast models:

✅ **5 Comprehensive Guides** covering all aspects  
✅ **2 Automated Tools** for easy configuration and training  
✅ **Multiple Use Cases** with step-by-step examples  
✅ **Troubleshooting Guide** for common issues  
✅ **Quick Reference** for experienced users  

**Start here**: [FINETUNING_INDEX.md](FINETUNING_INDEX.md)

---

**Happy Finetuning!** 🚀

If you find these guides helpful, please consider:
- ⭐ Starring the repository
- 📢 Sharing with others
- 💬 Providing feedback for improvements

