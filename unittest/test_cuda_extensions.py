#!/usr/bin/env python3
"""
Test script to verify CUDA extensions work correctly with PyTorch 2.x
"""

import torch
import alphaction._custom_cuda_ext as ext
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_roi_align_3d():
    """Test ROI Align 3D forward and backward"""
    logging.info("Testing ROI Align 3D...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping test")
        return
    
    # Create test input
    batch_size, channels, length, height, width = 2, 64, 4, 112, 112
    input_tensor = torch.randn(batch_size, channels, length, height, width).cuda()
    
    # Create ROIs: [batch_idx, x1, y1, x2, y2]
    num_rois = 4
    rois = torch.tensor([
        [0, 10, 10, 50, 50],
        [0, 20, 20, 60, 60],
        [1, 15, 15, 55, 55],
        [1, 25, 25, 65, 65],
    ], dtype=torch.float32).cuda()
    
    # Test forward
    try:
        output = ext.roi_align_3d_forward(
            input_tensor, rois,
            spatial_scale=1.0,
            pooled_height=7,
            pooled_width=7,
            sampling_ratio=2
        )
        logging.info(f"✅ ROI Align 3D Forward: Output shape = {output.shape}")
        assert output.shape == (num_rois, channels, length, 7, 7)
    except Exception as e:
        logging.error(f"❌ ROI Align 3D Forward failed: {e}")
        return False
    
    # Test backward
    try:
        grad_output = torch.randn_like(output)
        grad_input = ext.roi_align_3d_backward(
            grad_output, rois,
            spatial_scale=1.0,
            pooled_height=7,
            pooled_width=7,
            batch_size=batch_size,
            channels=channels,
            length=length,
            height=height,
            width=width,
            sampling_ratio=2
        )
        logging.info(f"✅ ROI Align 3D Backward: Grad shape = {grad_input.shape}")
        assert grad_input.shape == input_tensor.shape
    except Exception as e:
        logging.error(f"❌ ROI Align 3D Backward failed: {e}")
        return False
    
    return True


def test_roi_pool_3d():
    """Test ROI Pool 3D forward and backward"""
    logging.info("Testing ROI Pool 3D...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping test")
        return
    
    # Create test input
    batch_size, channels, length, height, width = 2, 64, 4, 112, 112
    input_tensor = torch.randn(batch_size, channels, length, height, width).cuda()
    
    # Create ROIs
    num_rois = 4
    rois = torch.tensor([
        [0, 10, 10, 50, 50],
        [0, 20, 20, 60, 60],
        [1, 15, 15, 55, 55],
        [1, 25, 25, 65, 65],
    ], dtype=torch.float32).cuda()
    
    # Test forward
    try:
        output, argmax = ext.roi_pool_3d_forward(
            input_tensor, rois,
            spatial_scale=1.0,
            pooled_height=7,
            pooled_width=7
        )
        logging.info(f"✅ ROI Pool 3D Forward: Output shape = {output.shape}")
        assert output.shape == (num_rois, channels, length, 7, 7)
    except Exception as e:
        logging.error(f"❌ ROI Pool 3D Forward failed: {e}")
        return False
    
    # Test backward
    try:
        grad_output = torch.randn_like(output)
        grad_input = ext.roi_pool_3d_backward(
            grad_output, input_tensor, rois, argmax,
            spatial_scale=1.0,
            pooled_height=7,
            pooled_width=7,
            batch_size=batch_size,
            channels=channels,
            length=length,
            height=height,
            width=width
        )
        logging.info(f"✅ ROI Pool 3D Backward: Grad shape = {grad_input.shape}")
        assert grad_input.shape == input_tensor.shape
    except Exception as e:
        logging.error(f"❌ ROI Pool 3D Backward failed: {e}")
        return False
    
    return True


def test_sigmoid_focal_loss():
    """Test Sigmoid Focal Loss forward and backward"""
    logging.info("Testing Sigmoid Focal Loss...")
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping test")
        return
    
    # Create test input
    batch_size, num_classes = 8, 80
    logits = torch.randn(batch_size, num_classes).cuda()
    targets = torch.randn(batch_size, num_classes).cuda()
    
    # Test forward
    try:
        losses = ext.sigmoid_focalloss_forward(
            logits, targets,
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.25
        )
        logging.info(f"✅ Sigmoid Focal Loss Forward: Loss shape = {losses.shape}")
        assert losses.shape == (batch_size * num_classes,)
    except Exception as e:
        logging.error(f"❌ Sigmoid Focal Loss Forward failed: {e}")
        return False
    
    # Test backward
    try:
        d_losses = torch.ones_like(losses)
        d_logits = ext.sigmoid_focalloss_backward(
            logits, targets, d_losses,
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.25
        )
        logging.info(f"✅ Sigmoid Focal Loss Backward: Grad shape = {d_logits.shape}")
        assert d_logits.shape == logits.shape
    except Exception as e:
        logging.error(f"❌ Sigmoid Focal Loss Backward failed: {e}")
        return False
    
    return True


def test_softmax_focal_loss():
    """Test Softmax Focal Loss forward and backward"""
    logging.info("Testing Softmax Focal Loss...")
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping test")
        return
    
    # Create test input
    batch_size, num_classes = 8, 80
    logits = torch.randn(batch_size, num_classes).cuda()
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.int32).cuda()
    
    # Test forward
    try:
        losses, P = ext.softmax_focalloss_forward(
            logits, targets,
            gamma=2.0,
            alpha=0.25
        )
        logging.info(f"✅ Softmax Focal Loss Forward: Loss shape = {losses.shape}, P shape = {P.shape}")
        assert losses.shape == (batch_size,)
        assert P.shape == (batch_size, num_classes)
    except Exception as e:
        logging.error(f"❌ Softmax Focal Loss Forward failed: {e}")
        return False
    
    # Test backward
    try:
        d_losses = torch.ones_like(losses)
        d_logits = ext.softmax_focalloss_backward(
            logits, targets, P, d_losses,
            gamma=2.0,
            alpha=0.25
        )
        logging.info(f"✅ Softmax Focal Loss Backward: Grad shape = {d_logits.shape}")
        assert d_logits.shape == logits.shape
    except Exception as e:
        logging.error(f"❌ Softmax Focal Loss Backward failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    logging.info("="*60)
    logging.info("CUDA Extensions Test Suite for PyTorch 2.x")
    logging.info("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"PyTorch Version: {torch.__version__}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logging.error("❌ CUDA not available!")
        return
    
    logging.info("="*60)
    
    # Run tests
    tests = [
        ("ROI Align 3D", test_roi_align_3d),
        ("ROI Pool 3D", test_roi_pool_3d),
        ("Sigmoid Focal Loss", test_sigmoid_focal_loss),
        ("Softmax Focal Loss", test_softmax_focal_loss),
    ]
    
    results = []
    for test_name, test_func in tests:
        logging.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logging.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logging.info("")
    logging.info("="*60)
    logging.info("Test Summary")
    logging.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logging.info(f"{status}: {test_name}")
    
    logging.info("="*60)
    logging.info(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        logging.info("🎉 All tests passed! CUDA extensions are working correctly with PyTorch 2.x")
    else:
        logging.error("Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()

