#!/usr/bin/env python3
"""
AlphAction Installation Verification Script
Tests all critical components without requiring video input
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test basic imports"""
    logger.info("=== Testing Basic Imports ===")
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} imported")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        
        import torchvision
        logger.info(f"✓ TorchVision {torchvision.__version__} imported")
        
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__} imported")
        
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__} imported")
        
        import alphaction
        logger.info(f"✓ AlphAction imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_cuda_extensions():
    """Test CUDA extensions"""
    logger.info("\n=== Testing CUDA Extensions ===")
    try:
        from alphaction import _custom_cuda_ext
        logger.info("✓ Custom CUDA extensions loaded")
        
        from detector.nms import nms_cuda, nms_cpu, soft_nms_cpu
        logger.info("✓ NMS CUDA extension loaded")
        logger.info("✓ NMS CPU extension loaded")
        logger.info("✓ Soft NMS CPU extension loaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ CUDA extensions failed: {e}")
        return False

def test_alphaction_modules():
    """Test AlphAction modules"""
    logger.info("\n=== Testing AlphAction Modules ===")
    try:
        from alphaction.config import cfg
        logger.info("✓ AlphAction config module loaded")
        
        from alphaction.dataset import make_data_loader
        logger.info("✓ AlphAction dataset module loaded")
        
        from alphaction.structures.bounding_box import BoxList
        logger.info("✓ AlphAction structures module loaded")
        
        from alphaction.layers import ROIAlign3D
        logger.info("✓ AlphAction layers module loaded")
        
        return True
    except Exception as e:
        logger.error(f"✗ AlphAction modules failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo11_detector():
    """Test YOLOv11 detector initialization"""
    logger.info("\n=== Testing YOLOv11 Detector ===")
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics library imported")
        
        # Initialize YOLOv11x (will download if not present)
        logger.info("Initializing YOLOv11x model (may download ~109 MB)...")
        model = YOLO('yolo11x.pt')
        logger.info(f"✓ YOLOv11x model loaded")
        logger.info(f"✓ Model device: {model.device}")
        
        return True
    except Exception as e:
        logger.error(f"✗ YOLOv11 detector failed: {e}")
        return False

def test_model_loading():
    """Test loading action recognition models"""
    logger.info("\n=== Testing Action Recognition Model ===")
    try:
        import torch
        from alphaction.config import cfg
        from alphaction.modeling.detector import build_detection_model
        
        # Use the ResNet101 config
        config_file = Path("config_files/resnet101_8x8f_denseserial.yaml")
        model_file = Path("data/models/aia_models/resnet101_8x8f_denseserial.pth")
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return False
            
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_file}")
            return False
        
        # Load config
        cfg.merge_from_file(str(config_file))
        cfg.freeze()
        logger.info(f"✓ Config loaded from {config_file}")
        
        # Build model
        model = build_detection_model(cfg)
        logger.info("✓ Action recognition model built")
        
        # Load weights
        checkpoint = torch.load(str(model_file), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logger.info(f"✓ Model weights loaded from {model_file}")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        logger.info(f"✓ Model moved to {device} and set to eval mode")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model has {total_params:,} parameters")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo_modules():
    """Test demo script modules"""
    logger.info("\n=== Testing Demo Modules ===")
    try:
        import sys
        import os
        
        # Add demo directory to path temporarily
        demo_dir = Path("demo")
        sys.path.insert(0, str(demo_dir))
        
        try:
            # Check if demo files exist
            demo_files = [
                "action_predictor.py",
                "video_detection_loader.py", 
                "fast_visualizer.py",
                "nvenc_visualizer.py",
                "demo.py"
            ]
            
            for file in demo_files:
                file_path = demo_dir / file
                if not file_path.exists():
                    logger.warning(f"Demo file not found: {file}")
                    return False
            
            logger.info(f"✓ All demo script files found")
            
            # Test importing detector API
            from detector.apis import get_detector
            logger.info("✓ Detector API module loaded")
            
            return True
        finally:
            # Remove demo dir from path
            sys.path.remove(str(demo_dir))
            
    except Exception as e:
        logger.error(f"✗ Demo modules failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    logger.info("╔═══════════════════════════════════════════════════════╗")
    logger.info("║   AlphAction Installation Verification               ║")
    logger.info("╚═══════════════════════════════════════════════════════╝\n")
    
    results = []
    
    # Run tests
    results.append(("Basic Imports", test_imports()))
    results.append(("CUDA Extensions", test_cuda_extensions()))
    results.append(("AlphAction Modules", test_alphaction_modules()))
    results.append(("YOLOv11 Detector", test_yolo11_detector()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Demo Modules", test_demo_modules()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:30s} {status}")
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! AlphAction is ready to use.")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

