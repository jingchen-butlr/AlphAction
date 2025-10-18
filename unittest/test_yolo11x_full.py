#!/usr/bin/env python
"""
Complete test of YOLOv11x with your AlphAction configuration
Tests both the Ultralytics installation and your custom detector code
"""
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("COMPLETE YOLOv11x TEST - AlphAction Integration")
    logger.info("=" * 70)
    
    # Test 1: Basic imports
    logger.info("\n[1/5] Testing Basic Imports...")
    try:
        import torch
        from ultralytics import YOLO
        logger.info(f"  ✓ PyTorch {torch.__version__}")
        logger.info(f"  ✓ Ultralytics imported")
        logger.info(f"  ✓ CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.error(f"  ✗ Import failed: {e}")
        return False
    
    # Test 2: Your custom detector config
    logger.info("\n[2/5] Testing Your YOLO11 Configuration...")
    try:
        from detector.yolo11_cfg import cfg
        logger.info(f"  ✓ Config loaded")
        logger.info(f"  ✓ Model weights: {cfg.WEIGHTS}")
        logger.info(f"  ✓ Confidence threshold: {cfg.CONFIDENCE}")
        logger.info(f"  ✓ NMS threshold: {cfg.NMS_THRES}")
        logger.info(f"  ✓ Image size: {cfg.IMG_SIZE}")
    except Exception as e:
        logger.error(f"  ✗ Config load failed: {e}")
        return False
    
    # Test 3: Load YOLOv11x model
    logger.info("\n[3/5] Loading YOLOv11x Model...")
    try:
        model = YOLO(cfg.WEIGHTS)
        logger.info(f"  ✓ YOLOv11x loaded successfully")
        
        # Move to GPU if available
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        logger.info(f"  ✓ Model moved to {device}")
        
        # Check model parameters
        param_device = next(model.model.parameters()).device
        logger.info(f"  ✓ Model device verified: {param_device}")
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test inference
    logger.info("\n[4/5] Testing Inference...")
    try:
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run prediction
        results = model.predict(
            dummy_img,
            conf=cfg.CONFIDENCE,
            iou=cfg.NMS_THRES,
            classes=[0],  # person only
            device=device,
            verbose=False
        )
        
        logger.info(f"  ✓ Inference completed")
        logger.info(f"  ✓ Results obtained: {len(results)} frame(s)")
        if results[0].boxes is not None:
            logger.info(f"  ✓ Detections: {len(results[0].boxes)} objects")
        else:
            logger.info(f"  ✓ No detections (expected for random image)")
    except Exception as e:
        logger.error(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test your custom detector API
    logger.info("\n[5/5] Testing Your Custom YOLO11Detector API...")
    try:
        from detector.yolo11_api import YOLO11Detector
        from detector.yolo11_cfg import cfg
        
        # Create simple options object for testing
        class TestOpt:
            def __init__(self):
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.gpus = [0 if torch.cuda.is_available() else -1]
                self.tracker_box_thres = cfg.CONFIDENCE
                self.tracker_nms_thres = cfg.NMS_THRES
        
        opt = TestOpt()
        detector = YOLO11Detector(cfg, opt)
        
        logger.info(f"  ✓ YOLO11Detector initialized")
        logger.info(f"  ✓ Using device: {detector.device}")
        logger.info(f"  ✓ Model: {detector.model_weights}")
        
        # Test detection on dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.detect_one_img(dummy_img)
        
        if result is None:
            logger.info(f"  ✓ detect_one_img() works (no detections on random image)")
        else:
            logger.info(f"  ✓ detect_one_img() works ({len(result)} detections)")
        
    except Exception as e:
        logger.error(f"  ✗ Detector API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✅ All tests passed successfully!")
    logger.info("")
    logger.info("Your YOLOv11x setup is ready:")
    logger.info(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    logger.info(f"  ✓ PyTorch {torch.__version__}")
    logger.info(f"  ✓ CUDA {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"  ✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    logger.info(f"  ✓ YOLOv11x model loaded and tested")
    logger.info(f"  ✓ Your custom detector API working")
    logger.info("")
    logger.info("You can now run your AlphAction demos with YOLOv11x!")
    logger.info("=" * 70)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

