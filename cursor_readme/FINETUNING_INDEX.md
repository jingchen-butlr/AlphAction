# SlowFast Model Finetuning - Complete Documentation Index

This document provides an overview and navigation guide for all finetuning-related documentation.

---

## 📚 Documentation Overview

### 🚀 Quick Start
Start here if you want to get up and running quickly:
- **[FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md)** - Quick commands and common scenarios

### 📖 Comprehensive Guides
Detailed documentation for thorough understanding:
- **[FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)** - Complete step-by-step finetuning guide
- **[CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md)** - How to prepare your own dataset

### 🛠️ Tools
Helper tools to streamline the finetuning process:
- **[finetune_quickstart.sh](finetune_quickstart.sh)** - Bash script for easy training launch
- **[generate_finetune_config.py](../tools/generate_finetune_config.py)** - Python tool to generate config files

### 🔧 Troubleshooting
Solutions to common problems:
- **[FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md)** - Comprehensive troubleshooting guide

### 📝 Related Documentation
Additional resources:
- **[../GETTING_STARTED.md](../GETTING_STARTED.md)** - Basic training and inference
- **[../DATA.md](../DATA.md)** - AVA dataset preparation
- **[../MODEL_ZOO.md](../MODEL_ZOO.md)** - Pretrained model downloads

---

## 🎯 Common Use Cases

### Use Case 1: Finetune on AVA Dataset
**Goal**: Continue training on AVA with different hyperparameters

**Steps**:
1. Download AVA dataset (see [../DATA.md](../DATA.md))
2. Download pretrained model (see [../MODEL_ZOO.md](../MODEL_ZOO.md))
3. Choose a config file from `config_files/`
4. Run training:
```bash
bash cursor_readme/finetune_quickstart.sh \
  config_files/resnet50_4x16f_baseline.yaml \
  --num-gpus 4
```

**Documentation**:
- [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Quick commands
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Detailed guide

---

### Use Case 2: Finetune on Custom Dataset (Same 80 Classes)
**Goal**: Finetune on your own videos with AVA action classes

**Steps**:
1. Prepare your dataset (see [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md))
2. Register dataset in `alphaction/config/paths_catalog.py`
3. Generate config file:
```bash
python tools/generate_finetune_config.py \
  --output config_files/my_dataset.yaml \
  --dataset-name my_dataset \
  --num-classes 80 \
  --num-gpus 4
```
4. Start training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/my_dataset.yaml \
  --transfer --use-tfboard
```

**Documentation**:
- [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) - Dataset preparation
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Training guide

---

### Use Case 3: Finetune on Custom Dataset (Different Classes)
**Goal**: Finetune on your own videos with custom action classes

**Steps**:
1. Prepare your dataset with custom annotations
2. Register dataset in `alphaction/config/paths_catalog.py`
3. Generate config file with your class count:
```bash
python tools/generate_finetune_config.py \
  --output config_files/custom_classes.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --num-gpus 4
```
4. Start training with `--no-head` flag:
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/custom_classes.yaml \
  --transfer --no-head --use-tfboard
```

**Documentation**:
- [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) - Dataset preparation
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - See "No-Head" section
- [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Quick commands

---

### Use Case 4: Resume Training from Checkpoint
**Goal**: Continue training from a saved checkpoint

**Steps**:
1. Locate your checkpoint:
```bash
cat data/output/my_experiment/last_checkpoint
```
2. Resume training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  train_net.py \
  --config-file config_files/my_config.yaml \
  --use-tfboard
```

**Documentation**:
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - See "Resume Training" section
- [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Quick commands

---

### Use Case 5: Single GPU Training (Limited Resources)
**Goal**: Train on single GPU with minimal memory

**Steps**:
1. Generate config or use existing
2. Run with adjusted parameters:
```bash
bash cursor_readme/finetune_quickstart.sh \
  config_files/resnet50_4x16f_baseline.yaml \
  --single-gpu --new-classes
```

**Documentation**:
- [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - See "Single GPU" section
- [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) - Memory issues

---

## 🔍 Finding What You Need

### "I want to..."

#### ...quickly start training on AVA
→ [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Section: Quick Commands

#### ...understand the complete finetuning process
→ [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)

#### ...prepare my own dataset
→ [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md)

#### ...generate a config file
→ Use `tools/generate_finetune_config.py` or see [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Configuration Setup

#### ...adjust hyperparameters for different GPU count
→ [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) - Linear Scaling Rule

#### ...understand what each config parameter does
→ [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Configuration Setup section

#### ...troubleshoot training issues
→ [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md)

#### ...monitor my training
→ [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Monitoring Training section

#### ...use the Interaction Aggregation structure
→ [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - See IA_STRUCTURE in config

#### ...improve my model's accuracy
→ [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) - Performance Issues

---

## 📊 Comparison: Which Guide Should I Read?

| Guide | Best For | Time to Read |
|-------|----------|--------------|
| [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md) | Quick lookup, experienced users | 5 minutes |
| [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) | First-time users, comprehensive understanding | 30 minutes |
| [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) | Preparing custom datasets | 20 minutes |
| [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) | Solving specific problems | As needed |

---

## 🛤️ Recommended Learning Path

### For Beginners

1. **Read**: [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) - Introduction & Model Architecture
2. **Try**: Use existing AVA dataset for first training
3. **Run**: Follow "Example: Complete Finetuning Workflow" in the guide
4. **Monitor**: Learn to use TensorBoard
5. **Troubleshoot**: Refer to [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md) if needed

### For Experienced Users

1. **Quick Start**: [FINETUNING_QUICK_REFERENCE.md](FINETUNING_QUICK_REFERENCE.md)
2. **Generate Config**: Use `tools/generate_finetune_config.py`
3. **Launch Training**: Use `cursor_readme/finetune_quickstart.sh`
4. **Reference**: Check quick reference for specific commands

### For Custom Dataset Users

1. **Prepare Data**: Follow [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md)
2. **Validate**: Check data format and structure
3. **Register**: Add dataset to `paths_catalog.py`
4. **Configure**: Generate config with correct `NUM_CLASSES`
5. **Train**: Start with small test run first
6. **Scale Up**: Once working, train full model

---

## 🔑 Key Concepts

### Pretrained Models
SlowFast models pretrained on Kinetics-700:
- **ResNet50-4x16**: Faster, lighter (66.34% Top-1)
- **ResNet101-8x8**: More accurate, heavier (69.32% Top-1)

Download from: [../MODEL_ZOO.md](../MODEL_ZOO.md)

### Transfer Learning
- Use `--transfer` flag to load pretrained backbone
- Use `--no-head` if your NUM_CLASSES differs from 80
- Backbone provides strong feature extraction

### Interaction Aggregation (IA)
- Models relationships between multiple persons
- Types: Parallel, Serial, Dense Serial
- Improves multi-person action detection
- Best mAP: 32.4 (ResNet101 + Dense Serial)

### Linear Scaling Rule
When changing GPU count, scale:
- Learning rate: `new_lr = base_lr * (new_gpus / 8)`
- Iterations: `new_iter = base_iter * (8 / new_gpus)`
- Batch size: `new_batch = base_batch * (new_gpus / 8)`

---

## 📦 Tools and Scripts Summary

### Interactive Tools

**Config Generator** (`tools/generate_finetune_config.py`):
```bash
# Interactive mode
python tools/generate_finetune_config.py --interactive

# Command-line mode
python tools/generate_finetune_config.py \
  --output config_files/my_config.yaml \
  --dataset-name my_dataset \
  --num-classes 15 \
  --backbone resnet50 \
  --num-gpus 4
```

**Training Launcher** (`cursor_readme/finetune_quickstart.sh`):
```bash
# Single GPU with new classes
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --single-gpu --new-classes

# Multi-GPU
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4
```

### Data Preparation Tools

See [CUSTOM_DATASET_PREPARATION.md](CUSTOM_DATASET_PREPARATION.md) for:
- Video clip extraction scripts
- Annotation format conversion
- Person detection scripts
- Dataset validation tools

---

## 📈 Expected Results

### Training Time (8x V100 GPUs)
- ResNet50: ~24 hours (90K iterations)
- ResNet101: ~36 hours (90K iterations)

### Performance on AVA v2.2
| Model | IA Structure | mAP |
|-------|--------------|-----|
| ResNet50 | Baseline | 26.7 |
| ResNet50 | Dense Serial | 30.0 |
| ResNet101 | Baseline | 29.3 |
| ResNet101 | Dense Serial | 32.4 |

---

## 🐛 Common Issues Quick Links

| Issue | Solution |
|-------|----------|
| Out of Memory | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md#1-out-of-memory-oom-during-training) |
| Loss is NaN | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md#2-loss-becomes-nan-or-inf) |
| Slow Training | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md#4-training-too-slow) |
| Dataset Not Found | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md#1-dataset-not-found-error) |
| Low Accuracy | [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md#1-low-map-on-validation-set) |

---

## 🎓 Additional Resources

### Official Documentation
- [Main README](../README.md)
- [Installation Guide](../INSTALL.md) or [UV Setup](../INSTALL_UV.md)
- [Getting Started](../GETTING_STARTED.md)
- [Data Preparation](../DATA.md)
- [Model Zoo](../MODEL_ZOO.md)

### Research Paper
- [Asynchronous Interaction Aggregation for Action Detection (ECCV 2020)](https://arxiv.org/abs/2004.07485)

### Datasets
- [AVA Dataset](https://research.google.com/ava/)
- [Kinetics Dataset](https://deepmind.com/research/open-source/kinetics)

---

## 💡 Tips for Success

1. **Start Small**: Test with small dataset before full training
2. **Validate Data**: Check annotations and video format
3. **Monitor Closely**: Use TensorBoard from the start
4. **Save Often**: Frequent checkpointing prevents data loss
5. **Be Patient**: Training takes time (24-36 hours typical)
6. **Read Errors**: Error messages usually indicate the problem
7. **Ask Questions**: Don't hesitate to seek help

---

## 📞 Getting Help

If you're stuck:
1. Check [FINETUNING_TROUBLESHOOTING.md](FINETUNING_TROUBLESHOOTING.md)
2. Review relevant guide section
3. Verify your data and configuration
4. Check GitHub issues for similar problems
5. Post detailed question with error logs

---

## 📝 Quick Reference Card

```bash
# 1. Generate Config
python tools/generate_finetune_config.py -i

# 2. Download Pretrained Model
# (from MODEL_ZOO.md to data/models/pretrained_models/)

# 3. Prepare Data
# (see CUSTOM_DATASET_PREPARATION.md)

# 4. Register Dataset
# (edit alphaction/config/paths_catalog.py)

# 5. Start Training
bash cursor_readme/finetune_quickstart.sh \
  config_files/my_config.yaml \
  --num-gpus 4 \
  --new-classes

# 6. Monitor
tensorboard --logdir=data/output/my_experiment

# 7. Evaluate
python test_net.py \
  --config-file config_files/my_config.yaml \
  MODEL.WEIGHT data/output/my_experiment/model_final.pth
```

---

## 🔄 Document Updates

This documentation is maintained in the `cursor_readme/` directory. All finetuning guides are regularly updated to reflect best practices and new features.

**Last Updated**: November 2025

---

Happy Finetuning! 🚀

