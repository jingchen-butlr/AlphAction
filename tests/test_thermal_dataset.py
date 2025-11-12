#!/usr/bin/env python3
"""
Unit tests for ThermalAVADataset class.

Tests the thermal dataset adapter that bridges HDF5 thermal data to AlphAction format.
"""

import unittest
import sys
import os
import tempfile
import json
import numpy as np
import torch
import h5py
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphaction.dataset.datasets.thermal_ava import ThermalAVADataset, NpInfoDict, NpBoxDict


class TestNpInfoDict(unittest.TestCase):
    """Test NpInfoDict helper class."""
    
    def test_basic_functionality(self):
        """Test basic dictionary operations."""
        info_dict = {"sensor1": [640, 480], "sensor2": [1920, 1080]}
        np_dict = NpInfoDict(info_dict, value_type=np.int32)
        
        self.assertEqual(len(np_dict), 2)
        key, val = np_dict[0]
        self.assertIn(key, ["sensor1", "sensor2"])
        self.assertEqual(len(val), 2)
    
    def test_convert_key(self):
        """Test key conversion."""
        info_dict = {"sensor1": [640, 480], "sensor2": [1920, 1080]}
        np_dict = NpInfoDict(info_dict, value_type=np.int32)
        
        idx = np_dict.convert_key("sensor1")
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(np_dict))


class TestNpBoxDict(unittest.TestCase):
    """Test NpBoxDict helper class."""
    
    def test_basic_functionality(self):
        """Test basic box dictionary operations."""
        box_dict = {
            "img1": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "packed_act": np.array([1, 2, 3], dtype=np.uint8)}
            ],
            "img2": [
                {"bbox": [0.5, 0.6, 0.7, 0.8], "packed_act": np.array([4, 5, 6], dtype=np.uint8)},
                {"bbox": [0.2, 0.3, 0.4, 0.5], "packed_act": np.array([7, 8, 9], dtype=np.uint8)}
            ]
        }
        
        np_box_dict = NpBoxDict(
            box_dict,
            value_types=[("bbox", np.float32), ("packed_act", np.uint8)]
        )
        
        self.assertEqual(len(np_box_dict), 2)
        
        # Get boxes for first image
        boxes, packed = np_box_dict[0]
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes.shape, (1, 4))
        
        # Get boxes for second image
        boxes, packed = np_box_dict[1]
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes.shape, (2, 4))


class TestThermalAVADataset(unittest.TestCase):
    """Test ThermalAVADataset class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.hdf5_dir = os.path.join(cls.temp_dir, "frames")
        cls.ann_dir = os.path.join(cls.temp_dir, "annotations")
        os.makedirs(cls.hdf5_dir, exist_ok=True)
        os.makedirs(cls.ann_dir, exist_ok=True)
        
        # Create mock HDF5 file
        cls._create_mock_hdf5()
        
        # Create mock annotations
        cls._create_mock_annotations()
    
    @classmethod
    def _create_mock_hdf5(cls):
        """Create mock HDF5 file with thermal frames."""
        h5_path = os.path.join(cls.hdf5_dir, "TEST_SENSOR.h5")
        
        # Create synthetic thermal data (100 frames, 40x60)
        num_frames = 100
        frames = np.random.uniform(15.0, 35.0, (num_frames, 40, 60)).astype(np.float32)
        timestamps = np.arange(num_frames, dtype=np.int64) * 100  # 100ms apart
        frame_seqs = np.arange(num_frames, dtype=np.int64)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('frames', data=frames, compression='gzip', compression_opts=4)
            f.create_dataset('timestamps', data=timestamps)
            f.create_dataset('frame_seqs', data=frame_seqs)
            f.attrs['sensor_id'] = 'TEST_SENSOR'
            f.attrs['mac_address'] = '00:00:00:00:00:00'
            f.attrs['total_frames'] = num_frames
            f.attrs['width'] = 60
            f.attrs['height'] = 40
    
    @classmethod
    def _create_mock_annotations(cls):
        """Create mock COCO-style annotations."""
        # Create training annotations
        train_data = {
            "images": [
                {
                    "id": "TEST_SENSOR_0",
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
                    "image_id": "TEST_SENSOR_0",
                    "bbox": [0.5, 0.5, 0.2, 0.3],  # centerX, centerY, w, h
                    "category_id": 0,
                    "category_name": "sitting",
                    "object_id": 1
                }
            ],
            "categories": [
                {"id": 0, "name": "sitting"},
                {"id": 1, "name": "standing"}
            ]
        }
        
        train_path = os.path.join(cls.ann_dir, "train.json")
        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        
        # Create validation annotations
        val_data = {
            "images": [
                {
                    "id": "TEST_SENSOR_1",
                    "sensor_id": "TEST_SENSOR",
                    "mac_address": "00:00:00:00:00:00",
                    "timestamp": 7000,
                    "width": 60,
                    "height": 40,
                    "frame_idx": 70
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": "TEST_SENSOR_1",
                    "bbox": [0.3, 0.4, 0.25, 0.35],
                    "category_id": 1,
                    "category_name": "standing",
                    "object_id": 1
                }
            ],
            "categories": [
                {"id": 0, "name": "sitting"},
                {"id": 1, "name": "standing"}
            ]
        }
        
        val_path = os.path.join(cls.ann_dir, "val.json")
        with open(val_path, 'w') as f:
            json.dump(val_data, f)
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 1)  # One training sample
    
    def test_hdf5_files_opened(self):
        """Test HDF5 files are opened correctly."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        self.assertIn("TEST_SENSOR", dataset.hdf5_files)
        self.assertIsInstance(dataset.hdf5_files["TEST_SENSOR"], h5py.File)
    
    def test_get_item_structure(self):
        """Test __getitem__ returns correct structure."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        # Get first sample (no transforms)
        video_data, boxes, idx, sensor_id, frame_idx = dataset[0]
        
        # Check video data shape: [64, 40, 60, 3]
        self.assertEqual(video_data.shape, (64, 40, 60, 3))
        self.assertEqual(video_data.dtype, np.float32)
        
        # Check boxes
        self.assertIsNotNone(boxes)
        self.assertGreater(len(boxes), 0)
        
        # Check metadata
        self.assertEqual(sensor_id, "TEST_SENSOR")
        self.assertEqual(frame_idx, 50)
    
    def test_frame_loading(self):
        """Test thermal frames are loaded correctly."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        # Load frames using internal method
        frames = dataset._load_thermal_frames("TEST_SENSOR", 50)
        
        # Check shape: [64, 40, 60, 3]
        self.assertEqual(frames.shape, (64, 40, 60, 3))
        
        # Check frames are replicated (all 3 channels should be equal)
        np.testing.assert_array_equal(frames[:, :, :, 0], frames[:, :, :, 1])
        np.testing.assert_array_equal(frames[:, :, :, 1], frames[:, :, :, 2])
        
        # Check temperature range (15-35°C)
        self.assertGreaterEqual(frames.min(), 15.0)
        self.assertLessEqual(frames.max(), 35.0)
    
    def test_yolo_to_xywh_conversion(self):
        """Test YOLO bbox format conversion."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        # Test bbox conversion
        boxes_yolo = torch.tensor([[0.5, 0.5, 0.2, 0.3]])  # centerX, centerY, w, h
        im_w, im_h = 60, 40
        
        boxes_xywh = dataset._yolo_to_xywh(boxes_yolo, im_w, im_h)
        
        # Expected: centerX=30, centerY=20, w=12, h=12
        # Top-left: x1=30-6=24, y1=20-6=14
        expected = torch.tensor([[24.0, 14.0, 12.0, 12.0]])
        
        torch.testing.assert_close(boxes_xywh, expected)
    
    def test_get_video_info(self):
        """Test get_video_info returns correct metadata."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=None
        )
        
        info = dataset.get_video_info(0)
        
        self.assertEqual(info['width'], 60)
        self.assertEqual(info['height'], 40)
        self.assertEqual(info['movie'], 'TEST_SENSOR')
        self.assertEqual(info['timestamp'], 50)
    
    def test_empty_annotations(self):
        """Test handling of clips without annotations."""
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        # Test with remove_clips_without_annotations=False
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=False,
            frame_span=64,
            transforms=None
        )
        
        # Should include the clip even if no annotations
        self.assertGreaterEqual(len(dataset), 1)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)


class TestThermalDatasetWithTransforms(unittest.TestCase):
    """Test ThermalAVADataset with transforms."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with transforms."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.hdf5_dir = os.path.join(cls.temp_dir, "frames")
        cls.ann_dir = os.path.join(cls.temp_dir, "annotations")
        os.makedirs(cls.hdf5_dir, exist_ok=True)
        os.makedirs(cls.ann_dir, exist_ok=True)
        
        # Reuse creation methods from TestThermalAVADataset
        TestThermalAVADataset._create_mock_hdf5.__func__(cls)
        TestThermalAVADataset._create_mock_annotations.__func__(cls)
    
    def test_with_mock_transforms(self):
        """Test dataset with mock transforms."""
        from alphaction.structures.bounding_box import BoxList
        
        class MockTransform:
            def __call__(self, video_data, boxes):
                # Split into slow and fast pathways
                slow = video_data[::8]  # Every 8th frame
                fast = video_data[::2]  # Every 2nd frame
                return (slow, fast), boxes, {}
        
        ann_file = os.path.join(self.ann_dir, "train.json")
        
        dataset = ThermalAVADataset(
            hdf5_root=self.hdf5_dir,
            ann_file=ann_file,
            remove_clips_without_annotations=True,
            frame_span=64,
            transforms=MockTransform()
        )
        
        # Get sample with transforms
        slow_video, fast_video, boxes, objects, extras, idx = dataset[0]
        
        # Check slow pathway (64/8 = 8 frames)
        self.assertEqual(slow_video.shape[0], 8)
        
        # Check fast pathway (64/2 = 32 frames)
        self.assertEqual(fast_video.shape[0], 32)
        
        # Check extras
        self.assertIn('movie_id', extras)
        self.assertIn('timestamp', extras)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNpInfoDict))
    suite.addTests(loader.loadTestsFromTestCase(TestNpBoxDict))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalAVADataset))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalDatasetWithTransforms))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

