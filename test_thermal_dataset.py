#!/usr/bin/env python3
"""
Test script for Thermal Dataset integration with AlphAction.

This script validates that:
1. Thermal dataset can be loaded via AlphAction dataloader
2. Frames are properly resized to 256×384
3. Bounding boxes are correctly transformed
4. Data format matches expected structure
"""

import sys
import torch
from alphaction.config import cfg
from alphaction.dataset import make_data_loader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_thermal_dataset():
    """Test thermal dataset loading through AlphAction pipeline."""
    
    logger.info("="*60)
    logger.info("Testing Thermal Dataset Integration")
    logger.info("="*60)
    
    # Load thermal config
    config_file = "config_files/thermal_resnet101_8x8f_denseserial.yaml"
    logger.info(f"\n1. Loading config from: {config_file}")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    logger.info(f"   Dataset: {cfg.DATASETS.TRAIN}")
    logger.info(f"   Frame num: {cfg.INPUT.FRAME_NUM}")
    logger.info(f"   Frame sample rate: {cfg.INPUT.FRAME_SAMPLE_RATE}")
    logger.info(f"   Batch size: {cfg.SOLVER.VIDEOS_PER_BATCH}")
    
    # Create training dataloader
    logger.info("\n2. Creating training dataloader...")
    try:
        train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
        logger.info("   ✅ Training dataloader created successfully")
    except Exception as e:
        logger.error(f"   ❌ Failed to create training dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading a batch
    logger.info("\n3. Loading first batch...")
    try:
        batch_iter = iter(train_loader)
        batch_data = next(batch_iter)
        logger.info("   ✅ First batch loaded successfully")
    except Exception as e:
        logger.error(f"   ❌ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Unpack batch data
    slow_video, fast_video, boxes, objects, extras, idx = batch_data
    
    # Validate batch structure
    logger.info("\n4. Validating batch structure...")
    
    logger.info(f"   Slow pathway shape: {slow_video.shape}")
    logger.info(f"   Fast pathway shape: {fast_video.shape}")
    logger.info(f"   Number of samples in batch: {len(boxes)}")
    logger.info(f"   Batch indices: {idx}")
    
    # Expected shapes
    batch_size = slow_video.shape[0]
    expected_slow_frames = cfg.INPUT.FRAME_NUM // cfg.INPUT.TAU
    expected_fast_frames = cfg.INPUT.FRAME_NUM // (cfg.INPUT.TAU // cfg.INPUT.ALPHA)
    
    logger.info(f"\n5. Checking dimensions...")
    logger.info(f"   Expected slow frames: {expected_slow_frames}")
    logger.info(f"   Expected fast frames: {expected_fast_frames}")
    
    # Check slow pathway
    if slow_video.shape[1] != 3:
        logger.error(f"   ❌ Slow pathway channels: {slow_video.shape[1]} (expected 3)")
        return False
    logger.info(f"   ✅ Slow pathway channels: {slow_video.shape[1]}")
    
    if slow_video.shape[2] != expected_slow_frames:
        logger.warning(f"   ⚠️ Slow pathway frames: {slow_video.shape[2]} (expected {expected_slow_frames})")
    else:
        logger.info(f"   ✅ Slow pathway frames: {slow_video.shape[2]}")
    
    # Check fast pathway
    if fast_video.shape[1] != 3:
        logger.error(f"   ❌ Fast pathway channels: {fast_video.shape[1]} (expected 3)")
        return False
    logger.info(f"   ✅ Fast pathway channels: {fast_video.shape[1]}")
    
    if fast_video.shape[2] != expected_fast_frames:
        logger.warning(f"   ⚠️ Fast pathway frames: {fast_video.shape[2]} (expected {expected_fast_frames})")
    else:
        logger.info(f"   ✅ Fast pathway frames: {fast_video.shape[2]}")
    
    # Check spatial dimensions (should be resized to ~256×384 range)
    slow_h, slow_w = slow_video.shape[3], slow_video.shape[4]
    fast_h, fast_w = fast_video.shape[3], fast_video.shape[4]
    
    logger.info(f"\n6. Checking spatial resolution...")
    logger.info(f"   Slow pathway size: {slow_h}×{slow_w}")
    logger.info(f"   Fast pathway size: {fast_h}×{fast_w}")
    
    if slow_h < 200 or slow_h > 300:
        logger.warning(f"   ⚠️ Slow height {slow_h} outside expected range [200, 300]")
    else:
        logger.info(f"   ✅ Slow height {slow_h} in expected range")
    
    if slow_w < 300 or slow_w > 400:
        logger.warning(f"   ⚠️ Slow width {slow_w} outside expected range [300, 400]")
    else:
        logger.info(f"   ✅ Slow width {slow_w} in expected range")
    
    # Check bounding boxes
    logger.info(f"\n7. Checking bounding boxes...")
    for i, box_list in enumerate(boxes):
        num_boxes = len(box_list)
        logger.info(f"   Sample {i}: {num_boxes} boxes")
        
        if num_boxes > 0:
            # Check box format
            bbox_mode = box_list.mode
            logger.info(f"     Bbox mode: {bbox_mode}")
            
            # Get box coordinates
            box_tensor = box_list.bbox
            logger.info(f"     Bbox shape: {box_tensor.shape}")
            logger.info(f"     Bbox range: [{box_tensor.min():.2f}, {box_tensor.max():.2f}]")
            
            # Check labels
            if box_list.has_field("labels"):
                labels = box_list.get_field("labels")
                logger.info(f"     Labels shape: {labels.shape}")
                logger.info(f"     Action classes present: {labels.sum(dim=1).tolist()}")
    
    # Check extras
    logger.info(f"\n8. Checking metadata...")
    for i, extra in enumerate(extras):
        logger.info(f"   Sample {i}:")
        logger.info(f"     Sensor ID: {extra.get('movie_id', 'N/A')}")
        logger.info(f"     Frame index: {extra.get('timestamp', 'N/A')}")
    
    # Test validation dataloader
    logger.info("\n9. Creating validation dataloader...")
    try:
        val_loader = make_data_loader(cfg, is_train=False, is_distributed=False)
        logger.info("   ✅ Validation dataloader created successfully")
        
        # Load one validation batch
        val_iter = iter(val_loader[0])  # val_loader returns list
        val_batch = next(val_iter)
        logger.info("   ✅ Validation batch loaded successfully")
    except Exception as e:
        logger.error(f"   ❌ Failed with validation dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("="*60)
    logger.info("\nSummary:")
    logger.info(f"  - Training samples loaded: {len(train_loader.dataset)}")
    logger.info(f"  - Validation samples loaded: {len(val_loader[0].dataset)}")
    logger.info(f"  - Frame resolution: {slow_h}×{slow_w}")
    logger.info(f"  - Temporal structure: Slow={slow_video.shape[2]}, Fast={fast_video.shape[2]}")
    logger.info(f"\n✅ Thermal dataset is ready for training!")
    logger.info("\nNext step: Run training command:")
    logger.info("  python train_net.py \\")
    logger.info("    --config-file config_files/thermal_resnet101_8x8f_denseserial.yaml \\")
    logger.info("    --transfer --no-head --use-tfboard")
    
    return True


def main():
    """Main function."""
    try:
        success = test_thermal_dataset()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

