"""
Thermal Action Detection Dataset for MMAction2.

Adapts thermal HDF5 data to MMAction2's video dataset interface.
"""

import os
import os.path as osp
import json
import logging
import numpy as np
import h5py
from typing import List, Dict, Optional, Tuple
import copy

from mmengine.registry import DATASETS
from mmaction.datasets.base import BaseActionDataset

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class ThermalActionDataset(BaseActionDataset):
    """
    Thermal action detection dataset for MMAction2.
    
    Args:
        ann_file: Path to annotation file
        hdf5_root: Path to HDF5 frames directory
        pipeline: List of data processing pipeline steps
        data_prefix: Data prefix (not used, for compatibility)
        test_mode: Whether in test mode
        num_classes: Number of action classes (default: 14)
        start_index: Start index for frame loading (default: 0)
    """
    
    def __init__(
        self,
        ann_file: str,
        hdf5_root: str,
        pipeline: List[Dict],
        data_prefix: Optional[str] = None,
        test_mode: bool = False,
        num_classes: int = 14,
        start_index: int = 0,
        **kwargs
    ):
        self.hdf5_root = hdf5_root
        self.num_classes = num_classes
        self.hdf5_files = {}
        
        # Set default data_prefix if not provided
        if data_prefix is None:
            data_prefix = dict(video='')
        
        # Initialize parent class
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            start_index=start_index,
            **kwargs
        )
    
    def load_data_list(self):
        """Load data list from COCO-style JSON file.
        
        Returns:
            list[dict]: List of data information dictionaries
        """
        logger.info(f"[Thermal-MM] Loading annotations from: {self.ann_file}")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to annotations mapping
        image_id_to_anns = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_id_to_anns:
                image_id_to_anns[image_id] = []
            image_id_to_anns[image_id].append(ann)
        
        # Create data list
        data_list = []
        for img_info in coco_data['images']:
            image_id = img_info['id']
            sensor_id = img_info['sensor_id']
            frame_idx = img_info['frame_idx']
            
            # Get annotations for this image
            anns = image_id_to_anns.get(image_id, [])
            
            # Build gt_bboxes and gt_labels
            gt_bboxes = []
            gt_labels = []
            for ann in anns:
                # Convert YOLO format (centerX, centerY, w, h) to (x1, y1, x2, y2)
                bbox = ann['bbox']  # normalized [0, 1]
                cx, cy, w, h = bbox
                
                # Denormalize to pixel coordinates
                width, height = img_info['width'], img_info['height']
                cx_px = cx * width
                cy_px = cy * height
                w_px = w * width
                h_px = h * height
                
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                
                gt_bboxes.append([x1, y1, x2, y2])
                gt_labels.append(ann['category_id'])
            
            # Single label (use first person's action)
            label = gt_labels[0] if gt_labels else 0
            
            data_info = dict(
                frame_dir=None,  # Not used for HDF5
                sensor_id=sensor_id,
                frame_idx=frame_idx,
                timestamp=img_info.get('timestamp', frame_idx),
                total_frames=64,  # Fixed 64-frame windows
                img_shape=(img_info['height'], img_info['width']),
                original_shape=(img_info['height'], img_info['width']),
                filename_tmpl='thermal_frame_{:04d}.jpg',  # Dummy template
                modality='Thermal',
                start_index=self.start_index,
                gt_bboxes=np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 4), dtype=np.float32),
                gt_labels=np.array(gt_labels, dtype=np.int64) if gt_labels else np.zeros((0,), dtype=np.int64),
                label=label,  # Single label for classification
            )
            
            data_list.append(data_info)
        
        logger.info(f"[Thermal-MM] Loaded {len(data_list)} samples")
        return data_list
    
    def get_data_info(self, idx: int) -> Dict:
        """Get data info for a single sample."""
        data_info = super().get_data_info(idx)
        
        # Load thermal frames from HDF5
        sensor_id = data_info['sensor_id']
        frame_idx = data_info['frame_idx']
        
        frames = self._load_thermal_frames(sensor_id, frame_idx)
        data_info['imgs'] = frames  # [T, H, W, C] format
        data_info['num_clips'] = 1
        data_info['clip_len'] = 64
        
        return data_info
    
    def _load_thermal_frames(self, sensor_id: str, frame_idx: int, num_frames: int = 64) -> np.ndarray:
        """
        Load thermal frames from HDF5.
        
        Args:
            sensor_id: Sensor ID string
            frame_idx: Center frame index
            num_frames: Number of frames to load (default: 64)
        
        Returns:
            frames: [T, H, W, C] numpy array
        """
        # Open HDF5 file if not already open
        if sensor_id not in self.hdf5_files:
            h5_path = osp.join(self.hdf5_root, f"{sensor_id}.h5")
            if not osp.exists(h5_path):
                raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
            self.hdf5_files[sensor_id] = h5py.File(h5_path, 'r')
        
        h5_file = self.hdf5_files[sensor_id]
        
        # Load frames: [frame_idx-32 : frame_idx+32] = 64 frames
        half_window = num_frames // 2
        start_idx = frame_idx - half_window
        end_idx = frame_idx + half_window
        
        # Load thermal frames [T, H, W] in Celsius
        frames = h5_file['frames'][start_idx:end_idx]
        
        # Replicate to 3 channels [T, H, W, 3]
        frames = np.stack([frames, frames, frames], axis=-1)
        
        return frames.astype(np.float32)
    
    def __del__(self):
        """Cleanup: close HDF5 files."""
        if hasattr(self, 'hdf5_files'):
            for h5_file in self.hdf5_files.values():
                h5_file.close()
    
    def evaluate(
        self,
        results: List,
        metrics: Optional[List[str]] = None,
        metric_options: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ) -> Dict:
        """Evaluate the dataset.
        
        Args:
            results: List of prediction results
            metrics: Evaluation metrics (e.g., 'top_k_accuracy', 'mean_class_accuracy')
            metric_options: Options for metrics
            logger: Logger for output
        
        Returns:
            eval_results: Dictionary with evaluation metrics
        """
        if metrics is None:
            metrics = ['top_k_accuracy', 'mean_class_accuracy']
        
        # Extract ground truth and predictions
        gt_labels = []
        pred_scores = []
        
        for i, result in enumerate(results):
            gt_label = self.data_list[i].get('label', 0)
            gt_labels.append(gt_label)
            pred_scores.append(result)
        
        gt_labels = np.array(gt_labels)
        pred_scores = np.array(pred_scores)
        
        eval_results = {}
        
        # Top-1 accuracy
        if 'top_k_accuracy' in metrics:
            pred_labels = pred_scores.argmax(axis=1)
            top1_acc = (pred_labels == gt_labels).mean()
            eval_results['top1_acc'] = float(top1_acc)
            
            if logger:
                logger.info(f"Top-1 Accuracy: {top1_acc:.4f}")
        
        # Mean class accuracy
        if 'mean_class_accuracy' in metrics:
            class_accs = []
            for class_id in range(self.num_classes):
                mask = gt_labels == class_id
                if mask.sum() > 0:
                    pred_labels_class = pred_scores[mask].argmax(axis=1)
                    class_acc = (pred_labels_class == class_id).mean()
                    class_accs.append(class_acc)
            
            mean_acc = np.mean(class_accs) if class_accs else 0.0
            eval_results['mean_class_accuracy'] = float(mean_acc)
            
            if logger:
                logger.info(f"Mean Class Accuracy: {mean_acc:.4f}")
        
        return eval_results

