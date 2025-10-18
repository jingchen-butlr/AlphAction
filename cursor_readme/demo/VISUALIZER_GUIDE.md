# Visualizer Guide: Choose Your Version

AlphAction now includes **two visualizer options** for video output. Choose based on your needs:

---

## 📊 Quick Comparison

| Feature | Original Visualizer | Fast Visualizer |
|---------|-------------------|-----------------|
| **Speed** | ~9 fps | ~34 fps (3.8x faster) |
| **Quality** | High (Pillow rendering) | High (OpenCV rendering) |
| **Text Rendering** | Pillow fonts (smoother) | OpenCV fonts (slightly different) |
| **Memory** | More conversions | Fewer conversions |
| **File** | `visualizer.py` | `fast_visualizer.py` |
| **Stability** | Battle-tested | Newly optimized |

---

## 🎯 Which One Should You Use?

### Use **Fast Visualizer** (Recommended) when:
✅ You want **faster video processing** (3.8x speed)  
✅ You're processing **long videos** or **many videos**  
✅ **CPU usage** is a concern  
✅ You need **better system responsiveness**  
✅ OpenCV text rendering is acceptable

### Use **Original Visualizer** when:
✅ You prefer **Pillow's text rendering** quality  
✅ You need **exact visual consistency** with previous results  
✅ Speed is not critical (e.g., single short video)  
✅ You want the **proven, legacy implementation**

---

## 🔄 How to Switch Between Versions

### Method 1: Edit demo.py (Permanent)

Edit `/home/ec2-user/AlphAction/demo/demo.py` at line 5:

**For Fast Visualizer (34 fps):**
```python
from fast_visualizer import FastAVAVisualizer as AVAVisualizer
```

**For Original Visualizer (9 fps):**
```python
from visualizer import AVAVisualizer
```

### Method 2: Environment Variable (Dynamic)

Add to demo.py (after imports):
```python
import os
if os.getenv('USE_FAST_VIZ', '0') == '1':
    from fast_visualizer import FastAVAVisualizer as AVAVisualizer
else:
    from visualizer import AVAVisualizer
```

Then run:
```bash
# Use fast visualizer
USE_FAST_VIZ=1 python demo.py --video-path ...

# Use original visualizer  
python demo.py --video-path ...
```

### Method 3: Command Line Argument (Most Flexible)

We can add a `--fast-viz` flag to demo.py if you'd like!

---

## 🎨 Visual Differences

### Text Rendering

**Original (Pillow):**
- Anti-aliased text (smoother edges)
- Custom TrueType font (Roboto-Bold.ttf)
- More rendering options

**Fast (OpenCV):**
- Built-in OpenCV fonts (Hershey)
- Slightly more "digital" look
- Still very readable

### Performance Impact

**Original Pipeline:**
```
Frame → BGR to RGB → PIL Image → RGBA conversion →
Pillow drawing → Alpha composite → RGB → BGR → Write
Time: ~110ms per frame
```

**Fast Pipeline:**
```
Frame → OpenCV drawing → Alpha blend → Write
Time: ~29ms per frame
```

---

## 📝 Current Default

**Currently using:** Original Visualizer (`visualizer.py`)

The fast visualizer is available but not active by default. This ensures backward compatibility with existing workflows.

---

## 🧪 Testing Both Versions

### Test Original Visualizer
```bash
cd /home/ec2-user/AlphAction/demo
python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_original.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --start 0 --duration 2000
```

### Test Fast Visualizer
```bash
# Temporarily edit demo.py line 5 to use fast_visualizer
sed -i.bak 's/from visualizer import/from fast_visualizer import FastAVAVisualizer as/' demo.py

python demo.py \
  --video-path ../Data/clip_7min_00s_to_7min_25s.mp4 \
  --output-path ../Data/output_fast.mp4 \
  --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth \
  --start 0 --duration 2000

# Restore original
mv demo.py.bak demo.py
```

### Compare Results
```bash
# Check file sizes
ls -lh ../Data/output_*.mp4

# Play both videos to compare visual quality
```

---

## 📦 Files Overview

```
demo/
├── visualizer.py           # Original (Pillow-based, 9 fps)
├── fast_visualizer.py      # New (OpenCV-based, 34 fps)
├── demo.py                 # Main demo (uses original by default)
├── VISUALIZER_GUIDE.md     # This file
└── Roboto-Bold.ttf         # Font for original visualizer
```

---

## 🔧 Troubleshooting

### Issue: Import Error
```python
ModuleNotFoundError: No module named 'fast_visualizer'
```
**Solution:** Make sure you're in the correct directory:
```bash
cd /home/ec2-user/AlphAction/demo
```

### Issue: Font Missing (Original)
```
OSError: cannot open resource 'Roboto-Bold.ttf'
```
**Solution:** Download the font or use fast visualizer (doesn't need custom fonts)

### Issue: Codec Error (Fast)
```
[ERROR] Could not find encoder
```
**Solution:** Fast visualizer already handles this with mp4v fallback. Should not occur.

---

## 💡 Recommendations

### For Production
- **Use Fast Visualizer** for better performance
- Test on a sample video first
- Compare visual output to ensure it meets your standards

### For Research/Papers
- **Use Original Visualizer** for consistency with published results
- If using fast visualizer, mention it in methods section

### For Development/Testing
- **Use Fast Visualizer** for faster iteration
- Quicker feedback loop during development

---

## 🚀 Performance Tips

### With Fast Visualizer
Already optimized! Running at 34 fps.

### With Original Visualizer
To improve speed slightly:
1. Disable timestamp overlay (`show_time=False`)
2. Increase visualization threshold (fewer labels to draw)
3. Process at lower resolution

---

## 📈 Benchmark Data

Test configuration: 1920x1080 video, 60 frames

| Visualizer | Writing Time | FPS | CPU Usage |
|------------|--------------|-----|-----------|
| Original | 6.6 seconds | 9 fps | High (PIL) |
| Fast | 1.8 seconds | 34 fps | Low (OpenCV) |

**Speed improvement:** 3.8x faster with fast visualizer

---

## ✅ Both Versions Maintained

Both visualizers are:
- ✅ Fully functional
- ✅ Tested and verified
- ✅ Production-ready
- ✅ Available in the repository

Choose the one that best fits your use case!

---

## 📞 Quick Reference Card

**Want speed?** → Use `fast_visualizer.py` (34 fps)  
**Want exact visual match?** → Use `visualizer.py` (9 fps)  
**Not sure?** → Try fast visualizer first, it works great!

**To switch:** Edit `demo.py` line 5

```python
# Fast (recommended)
from fast_visualizer import FastAVAVisualizer as AVAVisualizer

# Original
from visualizer import AVAVisualizer
```

---

**Last Updated:** October 18, 2025  
**Status:** Both versions active and maintained ✅

