"""
Thermal Action Detection Dataset for AlphAction

This module adapts the thermal HDF5 dataset to AlphAction's AVA-style interface.
"""

import os
import torch.utils.data as data
import time
import torch
import numpy as np
from alphaction.structures.bounding_box import BoxList
from collections import defaultdict
import json
import h5py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reuse helper classes from AVA dataset
class NpInfoDict(object):
    """Numpy-based dictionary to avoid PyTorch issue #13246"""
    def __init__(self, info_dict, key_type=None, value_type=None):
        keys = sorted(list(info_dict.keys()))
        self.key_arr = np.array(keys, dtype=key_type)
        self.val_arr = np.array([info_dict[k] for k in keys], dtype=value_type)
        self._key_idx_map = {k: i for i, k in enumerate(keys)}
    
    def __getitem__(self, idx):
        return self.key_arr[idx], self.val_arr[idx]
    
    def __len__(self):
        return len(self.key_arr)
    
    def convert_key(self, org_key):
        return self._key_idx_map[org_key]


class NpBoxDict(object):
    """Numpy-based box dictionary to avoid PyTorch issue #13246"""
    def __init__(self, id_to_box_dict, key_list=None, value_types=[]):
        value_fields, value_types = list(zip(*value_types))
        assert "bbox" in value_fields

        if key_list is None:
            key_list = sorted(list(id_to_box_dict.keys()))
        self.length = len(key_list)

        pointer_list = []
        value_lists = {field: [] for field in value_fields}
        cur = 0
        pointer_list.append(cur)
        for k in key_list:
            box_infos = id_to_box_dict[k]
            cur += len(box_infos)
            pointer_list.append(cur)
            for box_info in box_infos:
                for field in value_fields:
                    value_lists[field].append(box_info[field])
        self.pointer_arr = np.array(pointer_list, dtype=np.int32)
        self.attr_names = np.array(["vfield_" + field for field in value_fields])
        for field_name, value_type, attr_name in zip(value_fields, value_types, self.attr_names):
            setattr(self, attr_name, np.array(value_lists[field_name], dtype=value_type))

    def __getitem__(self, idx):
        l_pointer = self.pointer_arr[idx]
        r_pointer = self.pointer_arr[idx + 1]
        ret_val = [getattr(self, attr_name)[l_pointer:r_pointer] for attr_name in self.attr_names]
        return ret_val

    def __len__(self):
        return self.length


class ThermalAVADataset(data.Dataset):
    """
    Thermal Action Detection Dataset adapted for AlphAction.
    
    Loads thermal frames from HDF5 files and converts annotations to AVA format.
    """
    
    def __init__(
        self,
        hdf5_root,
        ann_file,
        remove_clips_without_annotations,
        frame_span,
        box_file=None,
        eval_file_paths={},
        box_thresh=0.0,
        action_thresh=0.0,
        transforms=None,
        object_file=None,
        object_transforms=None,
    ):
        """
        Initialize Thermal AVA Dataset.
        
        Args:
            hdf5_root: Path to HDF5 frames directory
            ann_file: COCO-style annotation file
            remove_clips_without_annotations: Whether to remove clips without annotations
            frame_span: Number of frames to load (should be 64)
            box_file: Optional person detection file (for validation)
            eval_file_paths: Evaluation file paths
            box_thresh: Box score threshold
            action_thresh: Action score threshold
            transforms: Video transforms
            object_file: Object detection file (not used for thermal)
            object_transforms: Object transforms (not used for thermal)
        """
        
        print('[Thermal] Loading annotations into memory...')
        tic = time.time()
        with open(ann_file, 'r') as f:
            json_dict = json.load(f)
        assert type(json_dict) == dict, f'annotation file format {type(json_dict)} not supported'
        print(f'[Thermal] Done (t={time.time() - tic:.2f}s)')

        self.hdf5_root = hdf5_root
        self.transforms = transforms
        self.frame_span = frame_span
        self.half_window = frame_span // 2
        
        self.eval_file_paths = eval_file_paths
        self.action_thresh = action_thresh
        
        # Initialize HDF5 files dict early to prevent __del__ errors
        self.hdf5_files = {}
        
        # Open HDF5 files (keep handles open for performance)
        self._open_hdf5_files()
        
        # Build clip to annotation mapping
        clip2ann = defaultdict(list)
        if "annotations" in json_dict:
            for ann in json_dict["annotations"]:
                image_id = ann["image_id"]
                category_id = ann["category_id"]
                
                # Convert thermal action class (0-13) to packed format
                # For thermal, we have 14 classes (all pose/movement classes)
                # Pad to 16 elements for packing (2 bytes = 16 bits)
                # np.unpackbits always unpacks to 8-bit boundaries
                one_hot = np.zeros(16, dtype=np.bool_)
                one_hot[category_id] = True
                # Pack to 2 bytes
                packed_act = np.packbits(one_hot)
                
                # Convert YOLO format (centerX, centerY, w, h) to XYWH format
                bbox = ann["bbox"]  # [centerX, centerY, width, height] normalized
                clip2ann[image_id].append(dict(bbox=bbox, packed_act=packed_act))
        
        # Build sensor size info (maps sensor_id to [width, height])
        sensor_sizes = {}
        clips_info = {}
        for img in json_dict["images"]:
            sensor_id = img["sensor_id"]
            if sensor_id not in sensor_sizes:
                # Thermal dimensions: width=60, height=40
                sensor_sizes[sensor_id] = [img["width"], img["height"]]
            
            # Store sensor_id and frame_idx for each clip
            clips_info[img["id"]] = [sensor_id, img["frame_idx"]]
        
        self.sensor_info = NpInfoDict(sensor_sizes, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))
        
        if remove_clips_without_annotations:
            clip_ids = [clip_id for clip_id in clip_ids if clip_id in clip2ann]
        
        # Handle detected person boxes (for validation/testing)
        if box_file:
            imgToBoxes = self.load_box_file(box_file, box_thresh)
            clip_ids = [
                img_id for img_id in clip_ids
                if len(imgToBoxes[img_id]) > 0
            ]
            self.det_persons = NpBoxDict(
                imgToBoxes, clip_ids,
                value_types=[("bbox", np.float32), ("score", np.float32)]
            )
        else:
            self.det_persons = None
        
        # Objects not used for thermal
        self.det_objects = None
        self.object_transforms = None
        
        # Store annotations
        self.anns = NpBoxDict(
            clip2ann, clip_ids,
            value_types=[("bbox", np.float32), ("packed_act", np.uint8)]
        )
        
        # Convert sensor_id strings to indices
        clips_info = {
            clip_id: [
                self.sensor_info.convert_key(clips_info[clip_id][0]),
                clips_info[clip_id][1]
            ] for clip_id in clip_ids
        }
        self.clips_info = NpInfoDict(clips_info, value_type=np.int32)
        
        logger.info(f"[Thermal] Loaded {len(self)} clips")
        logger.info(f"[Thermal] Frame span: {self.frame_span}")
    
    def _open_hdf5_files(self):
        """Open all HDF5 files in hdf5_root directory."""
        logger.info(f"[Thermal] Opening HDF5 files from: {self.hdf5_root}")
        
        for h5_file in os.listdir(self.hdf5_root):
            if h5_file.endswith('.h5'):
                sensor_id = h5_file.replace('.h5', '')
                h5_path = os.path.join(self.hdf5_root, h5_file)
                
                try:
                    self.hdf5_files[sensor_id] = h5py.File(h5_path, 'r')
                    total_frames = len(self.hdf5_files[sensor_id]['frames'])
                    logger.info(f"[Thermal]   {sensor_id}: {total_frames} frames")
                except Exception as e:
                    logger.error(f"[Thermal]   Failed to open {h5_path}: {e}")
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            slow_video: Slow pathway frames [T_slow, H, W, C]
            fast_video: Fast pathway frames [T_fast, H, W, C]
            boxes: BoxList with bounding boxes and labels
            objects: None (not used for thermal)
            extras: Dictionary with metadata
            idx: Sample index
        """
        
        _, clip_info = self.clips_info[idx]
        
        # sensor_id_idx is the id in self.sensor_info
        sensor_id_idx, frame_idx = clip_info
        # sensor_id is the human-readable sensor name
        sensor_id, sensor_size = self.sensor_info[sensor_id_idx]
        
        # Load thermal frames from HDF5
        video_data = self._load_thermal_frames(sensor_id, frame_idx)
        
        im_w, im_h = sensor_size
        
        if self.det_persons is None:
            # Training: use ground truth boxes
            boxes, packed_act = self.anns[idx]
            
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            # Convert from YOLO format (centerX, centerY, w, h) to XYWH
            boxes_xywh = self._yolo_to_xywh(boxes_tensor, im_w, im_h)
            boxes = BoxList(boxes_xywh, (im_w, im_h), mode="xywh").convert("xyxy")
            
            # Decode packed bits to one-hot labels
            one_hot_label = np.unpackbits(packed_act, axis=1)
            one_hot_label = torch.as_tensor(one_hot_label, dtype=torch.uint8)
            
            boxes.add_field("labels", one_hot_label)
        
        else:
            # Validation: use detected boxes
            boxes, box_score = self.det_persons[idx]
            boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xywh").convert("xyxy")
        
        boxes = boxes.clip_to_image(remove_empty=True)
        
        # Extra fields
        extras = {}
        
        if self.transforms is not None:
            video_data, boxes, transform_randoms = self.transforms(video_data, boxes)
            slow_video, fast_video = video_data
            
            objects = None  # No objects for thermal
            
            # Add metadata for memory features
            extras["movie_id"] = sensor_id
            extras["timestamp"] = frame_idx
            
            return slow_video, fast_video, boxes, objects, extras, idx
        
        return video_data, boxes, idx, sensor_id, frame_idx
    
    def _yolo_to_xywh(self, boxes_yolo, im_w, im_h):
        """
        Convert YOLO format (centerX, centerY, w, h) normalized to XYWH absolute.
        
        Args:
            boxes_yolo: [N, 4] tensor with [centerX, centerY, w, h] normalized [0, 1]
            im_w: Image width
            im_h: Image height
        
        Returns:
            boxes_xywh: [N, 4] tensor with [x1, y1, w, h] in absolute pixels
        """
        if len(boxes_yolo) == 0:
            return boxes_yolo
        
        # Denormalize
        centerX = boxes_yolo[:, 0] * im_w
        centerY = boxes_yolo[:, 1] * im_h
        w = boxes_yolo[:, 2] * im_w
        h = boxes_yolo[:, 3] * im_h
        
        # Convert center to top-left
        x1 = centerX - w / 2
        y1 = centerY - h / 2
        
        boxes_xywh = torch.stack([x1, y1, w, h], dim=1)
        return boxes_xywh
    
    def _load_thermal_frames(self, sensor_id, frame_idx):
        """
        Load thermal frames from HDF5 file.
        
        Args:
            sensor_id: Sensor ID string
            frame_idx: Center frame index
        
        Returns:
            video_data: [T, H, W, C] numpy array with thermal frames
        """
        if sensor_id not in self.hdf5_files:
            raise ValueError(f"HDF5 file not found for sensor: {sensor_id}")
        
        h5_file = self.hdf5_files[sensor_id]
        
        # Load frames [frame_idx-32 : frame_idx+32] = 64 frames
        start_idx = frame_idx - self.half_window
        end_idx = frame_idx + self.half_window
        
        # Load thermal frames [T, H, W] in Celsius
        frames = h5_file['frames'][start_idx:end_idx]
        
        # Replicate to 3 channels [T, H, W, 3]
        frames = np.stack([frames, frames, frames], axis=-1)
        
        return frames
    
    def get_video_info(self, index):
        """Get video info for aspect ratio grouping."""
        _, clip_info = self.clips_info[index]
        sensor_id_idx, frame_idx = clip_info
        sensor_id, sensor_size = self.sensor_info[sensor_id_idx]
        w, h = sensor_size
        return dict(width=w, height=h, movie=sensor_id, timestamp=frame_idx)
    
    def load_box_file(self, box_file, score_thresh=0.0):
        """Load person detection boxes from file."""
        print('[Thermal] Loading box file into memory...')
        tic = time.time()
        with open(box_file, "r") as f:
            box_results = json.load(f)
        print(f'[Thermal] Done (t={time.time() - tic:.2f}s)')
        
        boxImgIds = [box['image_id'] for box in box_results]
        
        imgToBoxes = defaultdict(list)
        for img_id, box in zip(boxImgIds, box_results):
            if box['score'] >= score_thresh:
                imgToBoxes[img_id].append(box)
        return imgToBoxes
    
    def return_null_box(self, im_w, im_h):
        """Return empty BoxList."""
        return BoxList(torch.zeros((0, 4)), (im_w, im_h), mode="xyxy")
    
    def __len__(self):
        return len(self.clips_info)
    
    def __del__(self):
        """Cleanup: close HDF5 files."""
        if hasattr(self, 'hdf5_files'):
            for h5_file in self.hdf5_files.values():
                h5_file.close()
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += f'    Number of datapoints: {self.__len__()}\n'
        fmt_str += f'    HDF5 Root Location: {self.hdf5_root}\n'
        tmp = '    Transforms (if any): '
        fmt_str += f'{tmp}{self.transforms.__repr__().replace(chr(10), chr(10) + " " * len(tmp))}\n'
        return fmt_str

