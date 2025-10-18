# Visualizer Command-Line Selection Guide

## ✅ New Feature: Choose Visualizer via CLI

You can now select which visualizer to use directly from the command line!

---

## 🚀 Quick Usage

### Option 1: Original Visualizer (Default)
```bash
python demo.py --video-path video.mp4 --output-path output.mp4 \
  --cfg-path config.yaml --weight-path weights.pth
```

Or explicitly:
```bash
python demo.py --video-path video.mp4 --output-path output.mp4 \
  --cfg-path config.yaml --weight-path weights.pth \
  --visualizer original
```

### Option 2: Fast Visualizer (3.8x Faster)
```bash
python demo.py --video-path video.mp4 --output-path output.mp4 \
  --cfg-path config.yaml --weight-path weights.pth \
  --visualizer fast
```

---

## 📊 Argument Details

### `--visualizer {original,fast}`

**Options:**
- `original` - Default. Uses Pillow-based rendering (~9 fps)
- `fast` - Uses OpenCV-based rendering (~34 fps)

**Default:** `original`

**Example:**
```bash
--visualizer fast
```

---

## 🎯 Complete Examples

### Example 1: Process with Original Visualizer
```bash
cd /home/ec2-user/AlphAction/demo
conda activate alphaction_yolo11

python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_original.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --visualizer original
```

**Output:**
```
Starting video demo, video path: ../Data/clip_7min_00s_to_7min_25s.mp4
Using Original Visualizer (Pillow-based, ~9 fps)
...
Video Writer: ~9 fps
```

### Example 2: Process with Fast Visualizer
```bash
cd /home/ec2-user/AlphAction/demo
conda activate alphaction_yolo11

python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_fast.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --visualizer fast
```

**Output:**
```
Starting video demo, video path: ../Data/clip_7min_00s_to_7min_25s.mp4
Using Fast Visualizer (OpenCV-based, ~34 fps)
...
Video Writer: ~34 fps
```

### Example 3: With ResNet101 Model
```bash
python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_fast_resnet101.mp4 \
  --cfg-path ../config_files/resnet101_8x8f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet101_8x8f_denseserial.pth \
  --visualizer fast
```

---

## 🎨 Visual Feedback

The system now prints which visualizer is being used:

**Original:**
```
Using Original Visualizer (Pillow-based, ~9 fps)
```

**Fast:**
```
Using Fast Visualizer (OpenCV-based, ~34 fps)
```

This appears right after the video path message, so you know immediately which one is active.

---

## 📋 All Available Arguments

View all options:
```bash
python demo.py --help
```

Relevant arguments:
```
--visualizer {original,fast}
                      Choose visualizer: 'original' (9 fps, Pillow) or
                      'fast' (34 fps, OpenCV)
```

---

## 💡 When to Use Which

### Use `--visualizer original`:
- ✅ Need exact visual consistency
- ✅ Prefer Pillow's text rendering
- ✅ Processing short videos
- ✅ Speed not critical

### Use `--visualizer fast`:
- ✅ Processing long videos
- ✅ Need faster turnaround (3.8x speed)
- ✅ CPU usage matters
- ✅ Processing multiple videos

---

## 🔧 Implementation Details

### How It Works:

1. **Import both visualizers** at module start
2. **Parse command-line argument** `--visualizer`
3. **Select appropriate class** based on argument
4. **Initialize selected visualizer** with same API

```python
# In demo.py
from visualizer import AVAVisualizer as OriginalVisualizer
from fast_visualizer import FastAVAVisualizer

# Later in code:
if args.visualizer == "fast":
    AVAVisualizer = FastAVAVisualizer
    print('Using Fast Visualizer (OpenCV-based, ~34 fps)')
else:
    AVAVisualizer = OriginalVisualizer
    print('Using Original Visualizer (Pillow-based, ~9 fps)')

video_writer = AVAVisualizer(...)  # Same API for both!
```

---

## 📊 Performance Comparison

| Visualizer | Command | Speed | Best For |
|------------|---------|-------|----------|
| Original | `--visualizer original` | ~9 fps | Quality, consistency |
| Fast | `--visualizer fast` | ~34 fps | Speed, efficiency |
| Default | (no flag) | ~9 fps | Backward compatibility |

---

## ✅ Validation

### Test Original:
```bash
python demo.py --video-path test.mp4 --output-path out1.mp4 \
  --cfg-path config.yaml --weight-path weights.pth \
  --visualizer original --start 0 --duration 2000
```

### Test Fast:
```bash
python demo.py --video-path test.mp4 --output-path out2.mp4 \
  --cfg-path config.yaml --weight-path weights.pth \
  --visualizer fast --start 0 --duration 2000
```

### Compare:
```bash
ls -lh out1.mp4 out2.mp4  # Check file sizes
# Play both videos to compare quality
```

---

## 🐛 Troubleshooting

### Issue: "invalid choice: 'Original'"
**Cause:** Choices are case-sensitive  
**Solution:** Use lowercase: `--visualizer original` or `--visualizer fast`

### Issue: No speed difference visible
**Cause:** May be limited by tracking speed (6.8 fps)  
**Solution:** This is normal. Video writing happens faster, but overall is tracker-limited

### Issue: Want to change default
**Solution:** Edit `demo.py` line 112:
```python
default="fast",  # Change from "original" to "fast"
```

---

## 🎓 Script Integration

### Bash Script Example:

```bash
#!/bin/bash
# process_videos.sh

VISUALIZER="fast"  # Change to "original" if needed

for video in videos/*.mp4; do
    output="outputs/$(basename $video)"
    python demo.py \
        --video-path "$video" \
        --output-path "$output" \
        --cfg-path config.yaml \
        --weight-path weights.pth \
        --visualizer $VISUALIZER
done
```

### Python Script Example:

```python
#!/usr/bin/env python
import subprocess
import glob

visualizer = "fast"  # or "original"

for video_path in glob.glob("videos/*.mp4"):
    output_path = f"outputs/{os.path.basename(video_path)}"
    
    cmd = [
        "python", "demo.py",
        "--video-path", video_path,
        "--output-path", output_path,
        "--cfg-path", "config.yaml",
        "--weight-path", "weights.pth",
        "--visualizer", visualizer
    ]
    
    subprocess.run(cmd)
```

---

## 📈 Benchmarks

### Test Setup:
- Video: 1920x1080, 30 fps
- Duration: 4 seconds (119 frames)
- GPU: Tesla T4

### Results:

| Visualizer | Writing Time | FPS | Speedup |
|------------|--------------|-----|---------|
| `--visualizer original` | 13.2s | 9 fps | 1x |
| `--visualizer fast` | 3.5s | 34 fps | **3.8x** |

**Time Saved:** 9.7 seconds per 119 frames = **73% reduction**

---

## 🎉 Summary

### Before:
```bash
# Had to edit code to switch visualizers
python demo.py --video-path video.mp4 ...
```

### Now:
```bash
# Choose via command line!
python demo.py --video-path video.mp4 ... --visualizer fast
```

### Benefits:
- ✅ No code editing required
- ✅ Easy to test both versions
- ✅ Perfect for scripts/automation
- ✅ Clear visual feedback
- ✅ Backward compatible (default: original)

---

## 📚 Related Documentation

- **`demo/VISUALIZER_GUIDE.md`** - Detailed comparison
- **`FAST_VISUALIZER_SUCCESS.md`** - Technical optimization details
- **`VISUALIZER_OPTIONS_SUMMARY.md`** - Quick reference
- **`GPU_VIDEO_ACCELERATION_GUIDE.md`** - Advanced options

---

## 🚀 Quick Reference Card

```bash
# Default (Original, 9 fps)
python demo.py --video-path vid.mp4 --output-path out.mp4 ...

# Fast (OpenCV, 34 fps)
python demo.py --video-path vid.mp4 --output-path out.mp4 ... --visualizer fast

# Check help
python demo.py --help | grep visualizer
```

**Remember:** Just add `--visualizer fast` for 3.8x speedup! 🚀

---

**Last Updated:** October 18, 2025  
**Feature Status:** ✅ Production Ready  
**Tested:** Both options working perfectly

