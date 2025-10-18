# -----------------------------------------------------
# YOLO11 Configuration with BoT-SORT Tracker
# -----------------------------------------------------

"""Configuration for YOLO11 detector"""


class ConfigYOLO11:
    """Configuration class for YOLO11 detector with BoT-SORT"""
    
    def __init__(self):
        # Model weights path (will auto-download if not present)
        self.WEIGHTS = 'yolo11x.pt'
        
        # Detection confidence threshold
        self.CONFIDENCE = 0.1
        
        # NMS IoU threshold
        self.NMS_THRES = 0.4
        
        # Tracker configuration (None uses default BoT-SORT config)
        self.TRACKER_CONFIG = None
        
        # Model input size (YOLO11 handles this automatically)
        self.IMG_SIZE = 640
        
        # Number of classes (1 for person only)
        self.NUM_CLASSES = 1
    
    def get(self, key, default=None):
        """Get configuration value"""
        return getattr(self, key, default)


# Global configuration instance
cfg = ConfigYOLO11()


