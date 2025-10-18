#!/usr/bin/env python
"""Test script for fast visualizer"""

import sys
sys.path.insert(0, 'demo')

from fast_visualizer import FastAVAVisualizer
import torch
import numpy as np

print("=" * 60)
print("Testing Fast Visualizer")
print("=" * 60)

# Initialize visualizer
print("\n1. Initializing visualizer...")
try:
    vis = FastAVAVisualizer(
        video_path='Data/clip_7min_00s_to_7min_25s.mp4',
        output_path='Data/test_fast_viz.mp4',
        realtime=False,
        start=0,
        duration=2000,  # Just 2 seconds for quick test
        show_time=False,
        confidence_threshold=0.5,
    )
    print("✅ Visualizer initialized successfully!")
    print(f"   Video: {vis.width}x{vis.height} @ {vis.vid_info['fps']:.2f} fps")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

# Test update_action_dictionary
print("\n2. Testing action dictionary update...")
try:
    # Create fake scores and ids (one score tensor per person)
    scores = [torch.tensor([0.9, 0.1, 0.8, 0.2] + [0.1] * 76)]  # 80 classes total
    ids = [1]
    vis.update_action_dictionary(scores, ids)
    print(f"✅ Action dictionary updated: {len(vis.action_dictionary)} persons")
except Exception as e:
    print(f"❌ Failed to update dictionary: {e}")
    sys.exit(1)

# Test visualization
print("\n3. Testing frame visualization...")
try:
    # Create fake frame
    frame = np.random.randint(0, 255, (vis.height, vis.width, 3), dtype=np.uint8)
    
    # Create fake boxes
    boxes = torch.tensor([
        [100, 100, 300, 500],  # x1, y1, x2, y2
    ])
    ids = [1]
    
    # Visualize
    result_frame = vis.visual_frame_fast(frame, boxes, ids)
    print(f"✅ Frame visualization successful!")
    print(f"   Input shape: {frame.shape}")
    print(f"   Output shape: {result_frame.shape}")
except Exception as e:
    print(f"❌ Failed visualization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test timestamp overlay
print("\n4. Testing timestamp overlay...")
try:
    frame = np.random.randint(0, 255, (vis.height, vis.width, 3), dtype=np.uint8)
    result_frame = vis.visual_timestamp_fast(frame, 12345)  # 12.345 seconds
    print(f"✅ Timestamp overlay successful!")
except Exception as e:
    print(f"❌ Failed timestamp: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\n📊 Fast visualizer is ready to use!")
print("\nTo use in demo.py, change line 5 to:")
print("  from fast_visualizer import FastAVAVisualizer as AVAVisualizer")

