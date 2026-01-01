"""
Person Detector Module
Uses YOLOv8-nano for fast and accurate person detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

# Try to import ultralytics YOLO, with fallback to OpenCV DNN
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Using OpenCV HOG detector as fallback.")


class PersonDetector:
    """Detects persons in images/frames using YOLO or HOG fallback"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize person detector
        
        Args:
            model_path: Path to YOLO model (auto-downloads if not present)
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.hog_detector = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(model_path)
                print(f"Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self._init_hog_fallback()
        else:
            self._init_hog_fallback()
    
    def _init_hog_fallback(self):
        """Initialize OpenCV HOG detector as fallback"""
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("Using OpenCV HOG detector (fallback)")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in a frame
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for each detected person
        """
        if self.yolo_model is not None:
            return self._detect_yolo(frame)
        elif self.hog_detector is not None:
            return self._detect_hog(frame)
        else:
            return []
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect using YOLO"""
        results = self.yolo_model(frame, verbose=False, classes=[0])  # class 0 = person
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    detections.append((int(x1), int(y1), int(x2), int(y2), confidence))
        
        return detections
    
    def _detect_hog(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect using HOG (fallback)"""
        # Resize for faster processing
        scale = 1.0
        if frame.shape[1] > 640:
            scale = 640 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        boxes, weights = self.hog_detector.detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            if weight >= self.confidence_threshold:
                # Scale back to original size
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w) / scale)
                y2 = int((y + h) / scale)
                detections.append((x1, y1, x2, y2, float(weight)))
        
        return detections
    
    def crop_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                    padding: float = 0.1) -> np.ndarray:
        """
        Crop person from frame with optional padding
        
        Args:
            frame: Source frame
            bbox: (x1, y1, x2, y2) bounding box
            padding: Padding ratio to add around detection
            
        Returns:
            Cropped person image
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        return frame[y1:y2, x1:x2].copy()


# Global instance
person_detector = PersonDetector()
