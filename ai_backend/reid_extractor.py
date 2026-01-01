"""
Re-ID Feature Extractor Module
Extracts 512-dimensional feature vectors from person crops for re-identification

Supports:
- OSNet (if torchreid available)
- Lightweight CNN feature extraction (fallback)
- Histogram-based features (ultra-lightweight fallback)
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import os

# Try to import deep learning libraries
TORCH_AVAILABLE = False
TORCHREID_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    
    try:
        import torchreid
        TORCHREID_AVAILABLE = True
        print("torchreid available - using OSNet for Re-ID")
    except ImportError:
        print("torchreid not installed - using lightweight Re-ID")
except ImportError:
    print("PyTorch not available - using histogram-based features")


class ReIDExtractor:
    """
    Extracts 512-dimensional feature vectors from person images
    
    These vectors capture:
    - Body shape and proportions
    - Clothing texture patterns
    - Overall appearance features
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize Re-ID feature extractor
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.feature_dim = 512
        self.device = 'cpu'
        self.model = None
        self.transform = None
        
        if TORCH_AVAILABLE:
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
            
            if TORCHREID_AVAILABLE:
                self._init_osnet()
            else:
                self._init_lightweight_cnn()
        else:
            self._init_histogram_features()
    
    def _init_osnet(self):
        """Initialize OSNet-AIN model from torchreid"""
        try:
            # Use OSNet-AIN which handles cross-domain color variations
            self.model = torchreid.models.build_model(
                name='osnet_ain_x1_0',
                num_classes=1,  # We only need feature extraction
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),  # Standard Re-ID size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("OSNet-AIN initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize OSNet: {e}")
            self._init_lightweight_cnn()
    
    def _init_lightweight_cnn(self):
        """Initialize a lightweight CNN for feature extraction"""
        try:
            import torchvision.models as models
            
            # Use MobileNetV2 as lightweight backbone
            self.model = models.mobilenet_v2(pretrained=True)
            
            # Replace classifier with feature extractor
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.last_channel, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("Lightweight CNN (MobileNetV2) initialized")
            
        except Exception as e:
            print(f"Failed to initialize lightweight CNN: {e}")
            self._init_histogram_features()
    
    def _init_histogram_features(self):
        """
        Initialize histogram-based feature extraction (no deep learning)
        
        This creates 512-d vectors from:
        - Color histograms (HSV)
        - Edge histograms (HOG-like)
        - Spatial color distribution
        """
        self.use_histogram_features = True
        print("Using histogram-based features (no deep learning)")
    
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-dimensional feature vector from person crop
        
        Args:
            person_crop: BGR image of cropped person
            
        Returns:
            512-dimensional numpy array (L2 normalized)
        """
        if person_crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        if self.model is not None:
            return self._extract_deep_features(person_crop)
        else:
            return self._extract_histogram_features(person_crop)
    
    def _extract_deep_features(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract features using deep learning model"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_tensor = self.transform(rgb_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_batch)
            
            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]
            
            features = features.cpu().numpy().flatten()
        
        # Ensure 512-d output
        if len(features) != self.feature_dim:
            # Resize/pad if necessary
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def _extract_histogram_features(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-d features using histograms (no deep learning)
        
        Feature breakdown:
        - 90 bins: H histogram (x 3 regions = 270)
        - 64 bins: S histogram (x 2 regions = 128)
        - 64 bins: V histogram (x 1 = 64)
        - 50 bins: Edge orientation histogram
        Total: 512 dimensions
        """
        features = []
        
        # Resize to standard size
        resized = cv2.resize(person_crop, (128, 256))
        
        # Convert to HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Split into 3 horizontal regions (upper, middle, lower body)
        h, w = resized.shape[:2]
        regions = [
            hsv[0:h//3, :],        # Upper
            hsv[h//3:2*h//3, :],   # Middle
            hsv[2*h//3:h, :]       # Lower
        ]
        
        # Hue histograms for each region (90 bins x 3 = 270)
        for region in regions:
            h_hist = cv2.calcHist([region], [0], None, [90], [0, 180])
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
            features.extend(h_hist)
        
        # Saturation histograms for upper and lower (64 bins x 2 = 128)
        for region in [regions[0], regions[2]]:
            s_hist = cv2.calcHist([region], [1], None, [64], [0, 256])
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
            features.extend(s_hist)
        
        # Value histogram (64 bins)
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
        features.extend(v_hist)
        
        # Edge orientation histogram (50 bins)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi + 180
        
        # Weighted histogram by magnitude
        edge_hist, _ = np.histogram(
            orientation.flatten(), 
            bins=50, 
            range=(0, 360),
            weights=magnitude.flatten()
        )
        edge_hist = edge_hist / (edge_hist.sum() + 1e-6)
        features.extend(edge_hist)
        
        features = np.array(features, dtype=np.float32)
        
        # Ensure exactly 512 dimensions
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            features1: First 512-d feature vector
            features2: Second 512-d feature vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Features are already L2 normalized, so dot product = cosine similarity
        similarity = np.dot(features1, features2)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, float(similarity)))


# Global instance
reid_extractor = ReIDExtractor()
