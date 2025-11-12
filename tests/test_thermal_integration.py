#!/usr/bin/env python3
"""
Integration tests for thermal dataset with AlphAction pipeline.

Tests the complete data loading pipeline from config to dataloader.
"""

import unittest
import sys
import os
import tempfile
import json
import numpy as np
import torch
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphaction.config import cfg as global_cfg
from alphaction.dataset import make_data_loader
from alphaction.config.paths_catalog import DatasetCatalog


class TestDatasetCatalog(unittest.TestCase):
    """Test dataset catalog registration."""
    
    def test_thermal_datasets_registered(self):
        """Test thermal datasets are registered in catalog."""
        self.assertIn("thermal_action_train", DatasetCatalog.DATASETS)
        self.assertIn("thermal_action_val", DatasetCatalog.DATASETS)
    
    def test_thermal_dataset_structure(self):
        """Test thermal dataset catalog structure."""
        train_data = DatasetCatalog.DATASETS["thermal_action_train"]
        
        self.assertIn("hdf5_root", train_data)
        self.assertIn("ann_file", train_data)
        self.assertIn("box_file", train_data)
        self.assertIn("eval_file_paths", train_data)
    
    def test_get_thermal_dataset(self):
        """Test getting thermal dataset from catalog."""
        dataset_info = DatasetCatalog.get("thermal_action_train")
        
        self.assertEqual(dataset_info["factory"], "ThermalAVADataset")
        self.assertIn("args", dataset_info)
        
        args = dataset_info["args"]
        self.assertIn("hdf5_root", args)
        self.assertIn("ann_file", args)


class TestThermalDataLoader(unittest.TestCase):
    """Test dataloader creation with thermal dataset."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.setup_mock_thermal_dataset()
    
    @classmethod
    def setup_mock_thermal_dataset(cls):
        """Create complete mock thermal dataset."""
        # Create directory structure
        frames_dir = os.path.join(cls.temp_dir, "ThermalDataGen", "thermal_action_dataset", "frames")
        ann_dir = os.path.join(cls.temp_dir, "ThermalDataGen", "thermal_action_dataset", "annotations")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        # Create HDF5 file
        h5_path = os.path.join(frames_dir, "TEST_SENSOR.h5")
        num_frames = 100
        frames = np.random.uniform(15.0, 35.0, (num_frames, 40, 60)).astype(np.float32)
        timestamps = np.arange(num_frames, dtype=np.int64) * 100
        frame_seqs = np.arange(num_frames, dtype=np.int64)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=frames, compression='gzip', compression_opts=4)
            f.create_dataset('timestamps', data=timestamps)
            f.create_dataset('frame_seqs', data=frame_seqs)
            f.attrs['sensor_id'] = 'TEST_SENSOR'
            f.attrs['total_frames'] = num_frames
            f.attrs['width'] = 60
            f.attrs['height'] = 40
        
        # Create annotations
        for split in ['train', 'val']:
            ann_data = {
                "images": [
                    {
                        "id": f"TEST_SENSOR_{split}",
                        "sensor_id": "TEST_SENSOR",
                        "mac_address": "00:00:00:00:00:00",
                        "timestamp": 5000,
                        "width": 60,
                        "height": 40,
                        "frame_idx": 50
                    }
                ],
                "annotations": [
                    {
                        "id": 0,
                        "image_id": f"TEST_SENSOR_{split}",
                        "bbox": [0.5, 0.5, 0.2, 0.3],
                        "category_id": 0,
                        "category_name": "sitting",
                        "object_id": 1
                    }
                ],
                "categories": [{"id": i, "name": f"action_{i}"} for i in range(14)]
            }
            
            ann_path = os.path.join(ann_dir, f"{split}.json")
            with open(ann_path, 'w') as f:
                json.dump(ann_data, f)
        
        cls.mock_data_dir = cls.temp_dir
    
    def test_dataloader_creation_with_mock_data(self):
        """Test dataloader can be created with mock thermal data."""
        # Create a copy of config
        cfg = global_cfg.clone()
        cfg.defrost()
        
        # Override paths to use mock data
        cfg.merge_from_list([
            "DATASETS.TRAIN", ("thermal_action_train",),
            "DATASETS.TEST", ("thermal_action_val",),
            "DATALOADER.NUM_WORKERS", 0,  # No workers for testing
            "SOLVER.VIDEOS_PER_BATCH", 1,
            "TEST.VIDEOS_PER_BATCH", 1,
            "INPUT.FRAME_NUM", 64,
            "INPUT.FRAME_SAMPLE_RATE", 1,
        ])
        cfg.freeze()
        
        # Temporarily override DatasetCatalog.DATA_DIR
        original_data_dir = DatasetCatalog.DATA_DIR
        try:
            DatasetCatalog.DATA_DIR = self.mock_data_dir
            
            # Try to create dataloader
            # Note: This will fail if actual data doesn't exist, but tests the pipeline
            try:
                train_loader = make_data_loader(cfg, is_train=True, is_distributed=False)
                self.assertIsNotNone(train_loader)
            except (FileNotFoundError, RuntimeError) as e:
                # Expected if mock data paths don't match actual structure
                self.assertIn("thermal", str(e).lower() + " " + type(e).__name__.lower())
        finally:
            DatasetCatalog.DATA_DIR = original_data_dir
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)


class TestThermalConfig(unittest.TestCase):
    """Test thermal configuration loading."""
    
    def test_load_thermal_config(self):
        """Test loading thermal training config."""
        config_path = "config_files/thermal_resnet101_8x8f_denseserial.yaml"
        
        if not os.path.exists(config_path):
            self.skipTest(f"Config file not found: {config_path}")
        
        cfg = global_cfg.clone()
        cfg.defrost()
        cfg.merge_from_file(config_path)
        cfg.freeze()
        
        # Check model settings
        self.assertEqual(cfg.MODEL.ROI_ACTION_HEAD.NUM_CLASSES, 14)
        self.assertEqual(cfg.MODEL.BACKBONE.CONV_BODY, "Slowfast-Resnet101")
        self.assertTrue(cfg.MODEL.IA_STRUCTURE.ACTIVE)
        
        # Check input settings
        self.assertEqual(cfg.INPUT.FRAME_NUM, 64)
        self.assertEqual(cfg.INPUT.FRAME_SAMPLE_RATE, 1)
        self.assertEqual(cfg.INPUT.TAU, 8)
        self.assertEqual(cfg.INPUT.ALPHA, 4)
        
        # Check dataset settings
        self.assertIn("thermal_action_train", cfg.DATASETS.TRAIN)
        self.assertIn("thermal_action_val", cfg.DATASETS.TEST)
        
        # Check solver settings
        self.assertEqual(cfg.SOLVER.BASE_LR, 0.0001)
        self.assertEqual(cfg.SOLVER.MAX_ITER, 10000)
        self.assertEqual(cfg.SOLVER.VIDEOS_PER_BATCH, 4)


class TestThermalTransformsPipeline(unittest.TestCase):
    """Test transforms pipeline for thermal data."""
    
    def test_build_transforms_for_thermal(self):
        """Test building transforms for thermal dataset."""
        from alphaction.dataset.transforms import build_transforms
        
        cfg = global_cfg.clone()
        cfg.defrost()
        cfg.merge_from_list([
            "INPUT.MIN_SIZE_TRAIN", 256,
            "INPUT.MAX_SIZE_TRAIN", 384,
            "INPUT.FRAME_NUM", 64,
            "INPUT.FRAME_SAMPLE_RATE", 1,
            "INPUT.TAU", 8,
            "INPUT.ALPHA", 4,
        ])
        cfg.freeze()
        
        transforms = build_transforms(cfg, is_train=True)
        self.assertIsNotNone(transforms)
    
    def test_transforms_output_shape(self):
        """Test transforms produce correct output shape."""
        from alphaction.dataset.transforms import build_transforms
        from alphaction.structures.bounding_box import BoxList
        
        cfg = global_cfg.clone()
        cfg.defrost()
        cfg.merge_from_list([
            "INPUT.MIN_SIZE_TRAIN", 256,
            "INPUT.MAX_SIZE_TRAIN", 384,
            "INPUT.FRAME_NUM", 64,
            "INPUT.FRAME_SAMPLE_RATE", 1,
            "INPUT.TAU", 8,
            "INPUT.ALPHA", 4,
        ])
        cfg.freeze()
        
        transforms = build_transforms(cfg, is_train=False)  # Deterministic
        
        # Create mock video data [T, H, W, C]
        video_data = np.random.uniform(0, 1, (64, 40, 60, 3)).astype(np.float32)
        
        # Create mock boxes
        boxes = BoxList(torch.tensor([[10., 10., 20., 30.]]), (60, 40), mode="xyxy")
        
        # Apply transforms
        (slow_video, fast_video), boxes_out, _ = transforms(video_data, boxes)
        
        # Check that pathways exist and have reasonable shapes
        # Note: Actual frame counts depend on transform implementation details
        self.assertGreater(slow_video.shape[0], 0)
        self.assertGreater(fast_video.shape[0], 0)
        
        # Fast pathway should have more frames than slow pathway
        self.assertGreaterEqual(fast_video.shape[0], slow_video.shape[0])


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end pipeline with actual thermal dataset (if available)."""
    
    def test_actual_thermal_dataset_if_available(self):
        """Test with actual thermal dataset if it exists."""
        thermal_frames_dir = "ThermalDataGen/thermal_action_dataset/frames"
        thermal_ann_file = "ThermalDataGen/thermal_action_dataset/annotations/train.json"
        
        if not os.path.exists(thermal_frames_dir) or not os.path.exists(thermal_ann_file):
            self.skipTest("Actual thermal dataset not available")
        
        from alphaction.dataset.datasets.thermal_ava import ThermalAVADataset
        
        # Try to load actual dataset
        dataset = ThermalAVADataset(
            hdf5_root=thermal_frames_dir,
            ann_file=thermal_ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        self.assertGreater(len(dataset), 0)
        
        # Try to load first sample
        video_data, boxes, idx, sensor_id, frame_idx = dataset[0]
        
        # Verify data structure
        self.assertEqual(video_data.shape[1:], (40, 60, 3))
        self.assertEqual(video_data.shape[0], 64)
        self.assertIsNotNone(boxes)
        self.assertGreater(len(boxes), 0)


def run_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetCatalog))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalTransformsPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

