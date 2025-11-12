# Thermal SlowFast Training - Quick Start

**Status**: ✅ Ready to Train  
**Tests**: 19/19 Passing ✅  
**Date**: November 12, 2025

---

## 🚀 Quick Commands

### 1. Test Everything
```bash
cd /home/ec2-user/jingchen/AlphAction
python tests/run_thermal_tests.py
```

### 2. Start Training
```bash
./thermal_quickstart.sh train
```

### 3. Monitor Training
```bash
./thermal_quickstart.sh tensorboard
# Open browser: http://localhost:6006
```

---

## 📊 Dataset Info

- **Training samples**: 314
- **Validation samples**: 73
- **Action classes**: 14
- **Resolution**: 40×60 → 256×384
- **Frames per sample**: 64 consecutive

---

## 🎯 Expected Results

| Training Stage | Time | Val mAP |
|---------------|------|---------|
| 1K iterations | 20 min | 15-25% |
| 3K iterations | 1 hour | 25-35% |
| 7K iterations | 2 hours | 30-40% |
| 10K iterations | 2-3 hours | 35-50% |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `config_files/thermal_resnet101_8x8f_denseserial.yaml` | Training config |
| `alphaction/dataset/datasets/thermal_ava.py` | Dataset adapter |
| `tests/run_thermal_tests.py` | Test suite |
| `thermal_quickstart.sh` | Quick start commands |

---

## 📚 Documentation

1. **[THERMAL_SLOWFAST_FINETUNING.md](cursor_readme/THERMAL_SLOWFAST_FINETUNING.md)** - Complete training guide
2. **[THERMAL_INTEGRATION_COMPLETE.md](cursor_readme/THERMAL_INTEGRATION_COMPLETE.md)** - Technical details
3. **[THERMAL_FINETUNING_IMPLEMENTATION_COMPLETE.md](cursor_readme/THERMAL_FINETUNING_IMPLEMENTATION_COMPLETE.md)** - Implementation summary

---

## ⚡ Training Command

```bash
python train_net.py \
  --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \
  --transfer \
  --no-head \
  --use-tfboard
```

**Flags**:
- `--transfer`: Load pretrained weights
- `--no-head`: Reinitialize classification head (14 classes)
- `--use-tfboard`: Enable TensorBoard logging

---

## 🔧 If GPU Memory Issue

```bash
./thermal_quickstart.sh train-small
```

This uses batch size 2 instead of 4.

---

## ✅ Pre-Flight Checklist

- [x] Environment activated (`source activate_uv_env.sh`)
- [x] Thermal dataset available (`ThermalDataGen/thermal_action_dataset/`)
- [x] Pretrained model available (`data/models/aia_models/resnet101_8x8f_denseserial.pth`)
- [x] All tests passing (`python tests/run_thermal_tests.py`)
- [x] Output directory created (`data/output/`)

---

## 🎓 Training Tips

1. **Monitor Early**: Watch first 100 iterations for any issues
2. **Check Logs**: `tail -f data/output/thermal_resnet101_8x8f/log.txt`
3. **Save Checkpoints**: Saved every 1K iterations automatically
4. **GPU Usage**: Monitor with `watch -n 1 nvidia-smi`
5. **Patience**: Training takes 2-3 hours on V100/A100

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Out of Memory | Use `--train-small` |
| Dataset not found | Check paths in `paths_catalog.py` |
| Model not found | Download pretrained model |
| Tests failing | Check environment activation |

---

## 📞 Need Help?

1. Check [THERMAL_SLOWFAST_FINETUNING.md](cursor_readme/THERMAL_SLOWFAST_FINETUNING.md) troubleshooting section
2. Review test output for specific errors
3. Verify dataset paths and structure

---

**Ready to train?** Run:
```bash
./thermal_quickstart.sh test && ./thermal_quickstart.sh train
```

Good luck! 🚀

