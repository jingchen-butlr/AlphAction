#!/usr/bin/env python3
"""
Debug script to print shapes during forward pass.
"""

import sys
import os
sys.path.insert(0, '/home/ec2-user/jingchen/AlphAction')

import torch
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
from alphaction.modeling.detector import build_detection_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
cfg.merge_from_file("config_files/thermal_resnet101_8x8f_baseline.yaml")
cfg.merge_from_list(["SOLVER.VIDEOS_PER_BATCH", "2", "TEST.VIDEOS_PER_BATCH", "2"])
cfg.freeze()

print(f"\nConfig loaded:")
print(f"  NUM_CLASSES: {cfg.MODEL.ROI_ACTION_HEAD.NUM_CLASSES}")
print(f"  NUM_PERSON_MOVEMENT_CLASSES: {cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES}")
print(f"  NUM_OBJECT_MANIPULATION_CLASSES: {cfg.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES}")
print(f"  NUM_PERSON_INTERACTION_CLASSES: {cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES}")

# Create dataloader
print("\nCreating dataloader...")
train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
print(f"Dataloader created with {len(train_loader.dataset)} samples")

# Load one batch
print("\nLoading first batch...")
batch = next(iter(train_loader))
slow_video, fast_video, boxes, objects, extras, idx = batch

print(f"\nBatch shapes:")
print(f"  Slow video: {slow_video.shape}")
print(f"  Fast video: {fast_video.shape}")
print(f"  Boxes: {len(boxes)} samples")

for i, box in enumerate(boxes):
    print(f"    Sample {i}: {len(box)} boxes")
    if box.has_field("labels"):
        labels = box.get_field("labels")
        print(f"      Labels shape: {labels.shape}")
        print(f"      Labels: {labels}")

# Build model
print("\nBuilding model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_detection_model(cfg)
model.to(device)
model.train()

print(f"Model built successfully")
print(f"Model device: {device}")

# Try forward pass
print("\nTrying forward pass...")
slow_video = slow_video.to(device)
fast_video = fast_video.to(device)
boxes = [b.to(device) for b in boxes]

try:
    loss_dict, weight_dict, metric_dict, pooled_feature = model(slow_video, fast_video, boxes, objects, {})
    print("\n✅ Forward pass successful!")
    print(f"Losses: {loss_dict}")
except Exception as e:
    print(f"\n❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

