import cv2
import numpy as np
import sys
import os

# Add classifier to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'classifier'))

from ModelTraining.YOLOIDDetector import YOLOIDDetector
from config import YOLO_MODEL_PATHS

class ContentDetector:
    def __init__(self):
        self.yolo_detector = None
        self._initialize_yolo()
    
    def _initialize_yolo(self):
        """Initialize YOLO detector with trained model"""
        try:
            # Try to find existing trained model
            for model_path in YOLO_MODEL_PATHS:
                if os.path.exists(model_path):
                    self.yolo_detector = YOLOIDDetector(model_path)
                    print(f"✅ Loaded YOLO model: {model_path}")
                    return
            
            # Fallback to pre-trained
            self.yolo_detector = YOLOIDDetector()
            print("⚠️ Using pre-trained YOLO (no trained model found)")
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLO: {e}")
            self.yolo_detector = None
    
    def detect_content_type(self, image):
        """
        Detect content type in image
        Returns: 'id_card', 'picture', or 'text'
        """
        # Convert for YOLO if grayscale
        if len(image.shape) == 2:
            image_for_yolo = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_for_yolo = image
        
        # Try YOLO detection first
        if self.yolo_detector:
            try:
                detections = self.yolo_detector.detect_id_cards(image_for_yolo, confidence=0.4)
                
                if detections:
                    # Check what was detected
                    for detection in detections:
                        if detection['class'] == 'ID':
                            return 'id_card'
                        elif detection['class'] == 'Picture':
                            return 'picture'
            except Exception as e:
                print(f"⚠️ YOLO detection error: {e}")
        
        # Default to text if no YOLO detections
        return 'text'