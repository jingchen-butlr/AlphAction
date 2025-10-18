# GPU Video Writing Acceleration Guide

## Current Performance Analysis

**Current Video Writing Speed:** ~9 fps  
**Current Bottlenecks:**
1. ❌ OpenCV without CUDA support (CPU-only)
2. ❌ CPU-based image processing (Pillow for text/boxes)
3. ❌ Multiple BGR↔RGB conversions
4. ❌ Unoptimized codec (mp4v)
5. ❌ Sequential frame writing

---

## 🚀 Optimization Strategies

### Strategy 1: Use GPU-Accelerated Video Encoding (NVENC) ⭐ BEST

NVIDIA GPUs have hardware video encoders (NVENC) that can encode H.264/H.265 at 100+ fps!

#### Benefits:
- **10-20x faster encoding** (from 9 fps → 90+ fps)
- No CPU overhead
- Better video quality
- Smaller file sizes

#### Implementation Options:

##### Option A: PyAV with Hardware Acceleration (Recommended)

```python
# Install PyAV with hardware support
pip install av

# Modified visualizer with NVENC
import av

class FastVideoWriter:
    def __init__(self, output_path, width, height, fps):
        self.container = av.open(output_path, mode='w')
        
        # Use NVENC H.264 encoder (GPU)
        self.stream = self.container.add_stream('h264_nvenc', rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'
        
        # NVENC options for best performance
        self.stream.options = {
            'preset': 'fast',      # or 'hp' for high performance
            'gpu': '0',            # GPU device ID
            'delay': '0',          # Low latency
            'zerolatency': '1',    # Minimal delay
        }
    
    def write_frame(self, frame):
        # Convert BGR to RGB (numpy operation, fast)
        frame_rgb = frame[:, :, ::-1]
        
        # Create AVFrame
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        
        # Encode (GPU accelerated)
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
    
    def close(self):
        # Flush encoder
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
```

**Expected Speed:** 80-150 fps (limited by data transfer, not encoding)

##### Option B: OpenCV with FFMPEG Backend + NVENC

```python
import cv2

# Use H.264 with hardware acceleration hint
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
out_vid = cv2.VideoWriter(
    output_path, 
    fourcc, 
    fps, 
    (width, height),
    isColor=True
)

# Or use FFMPEG directly via VideoWriter_fourcc
fourcc = cv2.VideoWriter_fourcc(*'H264')
```

**Note:** This still uses CPU unless OpenCV is compiled with FFMPEG+NVENC support.

##### Option C: Direct FFMPEG with NVENC (Most Control)

```python
import subprocess
import numpy as np

class FFmpegNVENCWriter:
    def __init__(self, output_path, width, height, fps):
        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',  # Input from pipe
            '-c:v', 'h264_nvenc',  # NVENC encoder
            '-preset', 'fast',
            '-gpu', '0',
            '-b:v', '5M',  # Bitrate
            output_path
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    def write_frame(self, frame):
        self.process.stdin.write(frame.tobytes())
    
    def close(self):
        self.process.stdin.close()
        self.process.wait()
```

**Expected Speed:** 100+ fps

---

### Strategy 2: GPU-Accelerated Image Processing 🎨

Use GPU for drawing operations instead of CPU-based Pillow.

#### Option A: Use OpenCV Drawing (Faster than Pillow)

```python
import cv2

def draw_boxes_opencv(frame, boxes, labels, scores):
    """GPU-friendly drawing using OpenCV (faster than Pillow)"""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle (OpenCV is optimized)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (191, 40, 41), 2)
        
        # Draw text background
        text = f"{label} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - text_h - 5), 
                     (x1 + text_w, y1), (176, 85, 234), -1)
        
        # Draw text
        cv2.putText(frame, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
```

**Benefit:** 2-3x faster than Pillow for drawing

#### Option B: PyTorch GPU Operations

```python
import torch
import torchvision

def draw_boxes_torch(frame_tensor, boxes, labels):
    """Use PyTorch for GPU-accelerated drawing"""
    # Convert to GPU tensor if not already
    if not frame_tensor.is_cuda:
        frame_tensor = frame_tensor.cuda()
    
    # Use torchvision's draw_bounding_boxes (GPU)
    annotated = torchvision.utils.draw_bounding_boxes(
        frame_tensor,
        boxes,
        labels=labels,
        colors='red',
        width=2,
    )
    
    return annotated
```

**Benefit:** Stay on GPU, avoid CPU↔GPU transfers

---

### Strategy 3: Optimize Current Implementation 🔧

#### A. Reduce Color Space Conversions

```python
# Current (multiple conversions):
frame[BGR] → frame[RGB] → PIL Image → RGBA → processing → RGB → frame[BGR]

# Optimized (minimal conversions):
frame[BGR] → cv2 drawing → frame[BGR]
```

#### B. Avoid Pillow for Simple Operations

```python
# Instead of Pillow alpha compositing, use OpenCV's addWeighted
overlay = np.zeros_like(frame)
cv2.rectangle(overlay, (x1, y1), (x2, y2), (176, 85, 234), -1)
cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)  # Blend
```

#### C. Pre-compute Font Sizes

```python
# Cache font measurements instead of computing every frame
self.font_cache = {}

def get_text_size(self, text):
    if text not in self.font_cache:
        self.font_cache[text] = cv2.getTextSize(text, ...)
    return self.font_cache[text]
```

---

### Strategy 4: Parallel Video Writing 🔀

Use multiple threads/processes for video encoding:

```python
import queue
import threading

class ParallelVideoWriter:
    def __init__(self, output_path, width, height, fps, num_workers=4):
        self.frame_queue = queue.Queue(maxsize=100)
        self.writer = FFmpegNVENCWriter(output_path, width, height, fps)
        
        # Start encoder threads
        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._encode_worker)
            t.start()
            self.workers.append(t)
    
    def _encode_worker(self):
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            # Encode on separate thread
            self.writer.write_frame(frame)
    
    def write(self, frame):
        self.frame_queue.put(frame.copy())
    
    def close(self):
        for _ in self.workers:
            self.frame_queue.put(None)
        for w in self.workers:
            w.join()
        self.writer.close()
```

---

## 🎯 Recommended Implementation

### Quick Win: Replace Visualizer Drawing (30min work)

Replace Pillow-based drawing with OpenCV:

```python
# In demo/visualizer.py

def visual_result_fast(self, boxes, ids):
    """Fast OpenCV-based visualization (no Pillow)"""
    # Create blank overlay
    overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    for box, id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        
        # Get actions for this person
        caption_data = self.action_dictionary.get(int(id), None)
        if caption_data is None:
            continue
        
        captions = caption_data['captions']
        colors = caption_data['bg_colors']
        
        # Draw each action label
        y_offset = y1
        for caption, color_idx in zip(captions, colors):
            color = self.category_colors[color_idx]
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw background
            cv2.rectangle(overlay, 
                         (x1, y_offset - text_h - 4), 
                         (x1 + text_w + 8, y_offset),
                         color, -1)
            
            # Draw text
            cv2.putText(overlay, caption, (x1 + 4, y_offset - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset -= (text_h + 6)
        
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.box_color, 2)
    
    return overlay

def visual_frame_fast(self, frame, overlay):
    """Fast alpha blending"""
    # Simple addWeighted is much faster than Pillow alpha_composite
    cv2.addWeighted(frame, 1.0, overlay, 0.6, 0, frame)
    return frame
```

**Expected Improvement:** 9 fps → 25-30 fps

---

### Advanced: Add NVENC Support (2hr work)

1. Install PyAV with hardware acceleration:
```bash
conda activate alphaction_yolo11
pip install av
```

2. Replace cv2.VideoWriter with PyAV NVENC writer

3. Test encoding speed:
```python
# Test script
import av
import numpy as np
import time

# Test NVENC speed
container = av.open('test.mp4', mode='w')
stream = container.add_stream('h264_nvenc', rate=30)
stream.width = 1920
stream.height = 1080
stream.pix_fmt = 'yuv420p'

start = time.time()
for i in range(1000):
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
    for packet in stream.encode(av_frame):
        container.mux(packet)

# Flush
for packet in stream.encode():
    container.mux(packet)
container.close()

elapsed = time.time() - start
print(f"Encoded 1000 frames in {elapsed:.2f}s = {1000/elapsed:.1f} fps")
```

Expected output: `Encoded 1000 frames in 8.5s = 117.6 fps`

---

## 📊 Performance Comparison

| Method | FPS | Complexity | Setup Time |
|--------|-----|------------|------------|
| **Current (Pillow + mp4v)** | **9 fps** | Easy | 0 min |
| OpenCV Drawing | 25-30 fps | Easy | 30 min |
| OpenCV + H.264 | 35-40 fps | Medium | 1 hr |
| PyAV + NVENC | 80-120 fps | Medium | 2 hr |
| FFMPEG + NVENC | 100-150 fps | Hard | 3 hr |

---

## 🚀 Quick Start: Fastest Path

### Step 1: Test NVENC Availability

```bash
# Check if your system supports NVENC
nvidia-smi --query-gpu=encoder.stats.sessionCount --format=csv

# Check ffmpeg NVENC support
ffmpeg -encoders | grep nvenc
```

### Step 2: Install PyAV

```bash
conda activate alphaction_yolo11
pip install av
```

### Step 3: Create Fast Visualizer

I can create an optimized visualizer for you using:
- OpenCV drawing (no Pillow)
- NVENC encoding (if available)
- Minimal color conversions

Would you like me to implement this?

---

## 💡 Bottom Line

**Current bottleneck in video writing:**
1. Pillow image processing (CPU, slow)
2. Multiple BGR↔RGB conversions
3. CPU-based mp4v codec

**Best improvements:**
1. ✅ **Quick:** Replace Pillow with OpenCV → 25-30 fps (3x faster)
2. ✅ **Better:** Add NVENC → 80-120 fps (10x faster)
3. ✅ **Best:** NVENC + optimized pipeline → 120-150 fps (15x faster)

**Important:** Even with GPU video encoding, the overall system speed is still limited by tracker (6.8 fps). But faster video writing means:
- Better responsiveness
- Can handle multiple video streams
- Reduced CPU usage

---

## Next Steps

Let me know if you want me to:
1. ✅ Implement OpenCV-based visualizer (quick, 3x faster)
2. ✅ Add NVENC support (requires testing your GPU)
3. ✅ Create complete optimized video writer
4. ✅ Test current NVENC capabilities on your system

I can create optimized code for any of these options!

