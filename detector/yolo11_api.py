# -----------------------------------------------------
# YOLO11 Detector with BoT-SORT Tracker
# Integrates Ultralytics YOLO11x model with BoT-SORT tracking
# -----------------------------------------------------

"""API of YOLO11 detector with BoT-SORT tracker"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import logging

import torch
import numpy as np
from ultralytics import YOLO
from detector.apis import BaseDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLO11Detector(BaseDetector):
    """YOLO11 detector with BoT-SORT tracking for person detection and tracking"""
    
    def __init__(self, cfg, opt=None):
        super(YOLO11Detector, self).__init__()
        
        self.detector_cfg = cfg
        self.detector_opt = opt
        
        # Configuration
        self.model_weights = cfg.get('WEIGHTS', 'yolo11x.pt')
        self.confidence = opt.tracker_box_thres if opt else cfg.get('CONFIDENCE', 0.1)
        self.iou_threshold = opt.tracker_nms_thres if opt else cfg.get('NMS_THRES', 0.4)
        self.tracker_config = cfg.get('TRACKER_CONFIG', None)  # Path to BoT-SORT config
        self.device_id = opt.device if opt else 'cuda:0'
        
        # Initialize model
        self.model = None
        self.frame_id = 0
        self.load_model()
        
        logger.info(f"YOLO11 Detector initialized with BoT-SORT tracker")
        logger.info(f"Model: {self.model_weights}")
        logger.info(f"Confidence threshold: {self.confidence}")
        logger.info(f"IoU threshold: {self.iou_threshold}")
    
    def load_model(self):
        """Load YOLO11 model with BoT-SORT tracker"""
        logger.info(f'Loading YOLO11 model from {self.model_weights}...')
        
        # Initialize YOLO11 model
        self.model = YOLO(self.model_weights)
        
        # Move model to appropriate device
        if self.detector_opt:
            if self.detector_opt.gpus[0] >= 0:
                device = f'cuda:{self.detector_opt.gpus[0]}'
            else:
                device = 'cpu'
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model.to(device)
        
        logger.info(f"YOLO11 model loaded successfully on {device}")
        logger.info(f"Using BoT-SORT tracker for person tracking")
    
    def image_preprocess(self, img_source):
        """
        Pre-process the img before fed to the object detection network
        Input: image name(str) or raw image data(ndarray or torch.Tensor, channel BGR)
        Output: raw image data (ultralytics handles preprocessing internally)
        """
        # Ultralytics YOLO handles preprocessing internally
        # Just return the raw image
        if isinstance(img_source, str):
            # If string path, ultralytics will load it
            return img_source
        elif isinstance(img_source, (torch.Tensor, np.ndarray)):
            # If already loaded, return as numpy array (BGR format)
            if isinstance(img_source, torch.Tensor):
                img_source = img_source.cpu().numpy()
            return img_source
        else:
            raise IOError(f'Unknown image source type: {type(img_source)}')
    
    def images_detection(self, imgs, orig_dim_list):
        """
        Feed the img data into YOLO11 network with BoT-SORT tracking
        Input: imgs: pre-processed image input (can be numpy array or path)
               orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original image size
        Output: dets(torch.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,track_id))): 
                person detection and tracking results
        """
        if not self.model:
            self.load_model()
        
        # Convert torch.Tensor to numpy array if needed (Ultralytics expects numpy HWC or list of numpy)
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
            # If batch dimension exists and batch size is 1, squeeze it out
            if imgs.ndim == 4 and imgs.shape[0] == 1:
                imgs = imgs[0]  # Remove batch dimension for single image
            # Ultralytics expects HWC format, convert from CHW if needed
            if imgs.ndim == 3 and imgs.shape[0] == 3:
                imgs = np.transpose(imgs, (1, 2, 0))  # CHW -> HWC
        
        # Ultralytics YOLO11 with tracking
        # Track method automatically handles detection + tracking
        results = self.model.track(
            imgs,
            persist=True,  # Persist tracker between frames
            tracker='botsort.yaml',  # Use BoT-SORT tracker
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[0],  # Only detect person class (class 0 in COCO)
            device=self.device,
            verbose=False,
        )
        
        output_dets = []
        
        for batch_idx, result in enumerate(results):
            self.frame_id += 1
            
            # Check if there are any detections
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # Extract boxes, scores, and track IDs
            boxes = result.boxes.xyxy.cpu()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu()  # confidence scores
            
            # Get track IDs (if available from tracker)
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu()
            else:
                # If no tracking, use sequential IDs
                track_ids = torch.arange(len(boxes)) + 1
            
            # Format: (batch_idx, x1, y1, x2, y2, confidence, track_id)
            for box, score, tid in zip(boxes, scores, track_ids):
                det = torch.tensor([
                    batch_idx,  # batch index
                    box[0].item(),  # x1
                    box[1].item(),  # y1
                    box[2].item(),  # x2
                    box[3].item(),  # y2
                    score.item(),  # confidence
                    tid.item()  # track_id
                ])
                output_dets.append(det)
        
        if len(output_dets) == 0:
            return 0
        
        return torch.stack(output_dets)
    
    def detect_one_img(self, img_name):
        """
        Detect bboxs in one image
        Input: 'str', full path of image
        Output: '[{"category_id":1,"score":float,"bbox":[x,y,w,h],"image_id":str},...]'
        """
        if not self.model:
            self.load_model()
        
        # Run detection (no tracking for single image)
        results = self.model.predict(
            img_name,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[0],  # Only detect person class
            device=self.device,
            verbose=False,
        )
        
        dets_results = []
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                return None
            
            boxes = result.boxes.xyxy.cpu()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu()
            
            for box, score in zip(boxes, scores):
                det_dict = {
                    "category_id": 1,  # person
                    "score": float(score),
                    "bbox": [
                        float(box[0]),  # x
                        float(box[1]),  # y
                        float(box[2] - box[0]),  # width
                        float(box[3] - box[1])   # height
                    ],
                    "image_id": int(os.path.basename(img_name).split('.')[0]) if isinstance(img_name, str) else 0
                }
                dets_results.append(det_dict)
        
        return dets_results if dets_results else None
    
    def reset_tracker(self):
        """Reset the tracker state (useful when starting a new video)"""
        self.frame_id = 0
        logger.info("Tracker reset")


