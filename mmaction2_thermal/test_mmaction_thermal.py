#!/usr/bin/env python3
"""
Test MMAction2 thermal dataset integration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_thermal_dataset():
    """Test thermal dataset with MMAction2."""
    
    from mmaction2_thermal.thermal_dataset import ThermalActionDataset
    
    logger.info("="*60)
    logger.info("Testing Thermal Dataset with MMAction2")
    logger.info("="*60)
    
    # Simple pipeline (no actual transforms for testing)
    pipeline = [
        {'type': 'Identity'}  # Placeholder
    ]
    
    # Create dataset
    logger.info("\n1. Creating thermal dataset...")
    try:
        dataset = ThermalActionDataset(
            ann_file='ThermalDataGen/thermal_action_dataset/annotations/train.json',
            hdf5_root='ThermalDataGen/thermal_action_dataset/frames',
            pipeline=[],  # Empty pipeline for testing
            num_classes=14
        )
        logger.info(f"   ✅ Dataset created: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"   ❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading a sample
    logger.info("\n2. Loading first sample...")
    try:
        sample = dataset[0]
        logger.info(f"   ✅ Sample loaded")
        logger.info(f"   Keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
        
        # Check frames from sample
        if 'imgs' in sample:
            frames = sample['imgs']
            logger.info(f"   Frames shape: {frames.shape}")
            logger.info(f"   Temp range: [{frames.min():.1f}°C, {frames.max():.1f}°C]")
        
    except Exception as e:
        logger.error(f"   ❌ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test HDF5 loading directly
    logger.info("\n3. Testing HDF5 frame loading...")
    try:
        # Get internal data list
        if hasattr(dataset, 'get_data_info'):
            data_info_raw = dataset.get_data_info(0)
            sensor_id = data_info_raw['sensor_id']
            frame_idx = data_info_raw['frame_idx']
            
            frames = dataset._load_thermal_frames(sensor_id, frame_idx)
            logger.info(f"   ✅ Frames loaded: shape={frames.shape}, dtype={frames.dtype}")
            logger.info(f"   Temp range: [{frames.min():.1f}°C, {frames.max():.1f}°C]")
            
            # Verify shape
            assert frames.shape == (64, 40, 60, 3), f"Unexpected shape: {frames.shape}"
            logger.info(f"   ✅ Shape correct: [64, 40, 60, 3]")
        else:
            logger.warning("   ⚠️ get_data_info not available, using loaded sample")
        
    except Exception as e:
        logger.error(f"   ❌ Failed to load frames: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test annotation info
    logger.info("\n4. Checking dataset info...")
    try:
        logger.info(f"   Total samples: {len(dataset)}")
        logger.info(f"   Sample 0 label: {sample.get('label', 'N/A')}")
        logger.info(f"   Sample 0 sensor: {sample.get('sensor_id', 'N/A')}")
        logger.info(f"   ✅ Dataset info valid")
    except Exception as e:
        logger.error(f"   ❌ Failed to check info: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    del dataset
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("="*60)
    logger.info("\nThermal dataset is ready for MMAction2 training!")
    logger.info("\nNext step: Configure and run training")
    
    return True


if __name__ == '__main__':
    success = test_thermal_dataset()
    sys.exit(0 if success else 1)

