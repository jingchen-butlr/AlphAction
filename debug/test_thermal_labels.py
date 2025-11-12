#!/usr/bin/env python3
"""
Debug script to test thermal label packing and unpacking.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, '/home/ec2-user/jingchen/AlphAction')

from alphaction.dataset.datasets.thermal_ava import ThermalAVADataset

# Create dataset
print("Loading thermal dataset...")
dataset = ThermalAVADataset(
    hdf5_root="ThermalDataGen/thermal_action_dataset/frames",
    ann_file="ThermalDataGen/thermal_action_dataset/annotations/train.json",
    remove_clips_without_annotations=True,
    frame_span=64,
    transforms=None
)

print(f"Dataset loaded: {len(dataset)} samples\n")

# Load first sample
print("Loading first sample (without transforms)...")
video_data, boxes, idx, sensor_id, frame_idx = dataset[0]

print(f"Video data shape: {video_data.shape}")
print(f"Boxes: {boxes}")
print(f"Boxes size: {boxes.size}")
print(f"Boxes mode: {boxes.mode}")

# Check labels
if boxes.has_field("labels"):
    labels = boxes.get_field("labels")
    print(f"\nLabels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Labels:\n{labels}")
    print(f"Active classes per person: {labels.sum(dim=1)}")
    print(f"Total active classes: {labels.sum()}")
else:
    print("No labels field!")

print("\nTest unpacking logic:")
# Simulate what happens in the dataset
category_id = 0  # sitting
one_hot = np.zeros(14, dtype=np.bool_)
one_hot[category_id] = True
packed_act = np.packbits(one_hot, bitorder='little')

print(f"One-hot vector (14): {one_hot.astype(int)}")
print(f"Packed bytes: {packed_act} (shape: {packed_act.shape})")

# Unpack
unpacked = np.unpackbits(packed_act)
print(f"Unpacked (8*{len(packed_act)}={len(unpacked)}): {unpacked}")

# Try with big-endian (default)
packed_act_bigendian = np.packbits(one_hot)
unpacked_bigendian = np.unpackbits(packed_act_bigendian)
print(f"\nBig-endian packed: {packed_act_bigendian}")
print(f"Big-endian unpacked ({len(unpacked_bigendian)}): {unpacked_bigendian}")

dataset.close()

