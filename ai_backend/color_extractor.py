"""
Color Extractor Module
Extracts clothing colors from person crops with color constancy handling
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter


class ColorExtractor:
    """Extracts dominant colors from clothing regions with illumination invariance"""
    
    # Color name mappings for common colors (in HSV ranges)
    COLOR_RANGES = {
        'red': [(0, 100, 100), (10, 255, 255)],      # Red wraps around 0
        'red2': [(160, 100, 100), (180, 255, 255)],  # Red continuation
        'orange': [(10, 100, 100), (25, 255, 255)],
        'yellow': [(25, 100, 100), (35, 255, 255)],
        'green': [(35, 100, 100), (85, 255, 255)],
        'cyan': [(85, 100, 100), (95, 255, 255)],
        'blue': [(95, 100, 100), (130, 255, 255)],
        'purple': [(130, 100, 100), (160, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'black': [(0, 0, 0), (180, 255, 50)],
        'gray': [(0, 0, 50), (180, 30, 200)],
        'brown': [(10, 100, 50), (25, 255, 150)],
        'pink': [(140, 50, 150), (170, 255, 255)],
    }
    
    def __init__(self, n_colors: int = 3):
        """
        Initialize color extractor
        
        Args:
            n_colors: Number of dominant colors to extract
        """
        self.n_colors = n_colors
    
    def white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gray World white balance correction
        
        This corrects for different camera color temperatures,
        making a red shirt look red regardless of warm/cool lighting.
        """
        img = image.astype(np.float32)
        
        # Calculate average of each channel
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])
        
        # Calculate overall average
        avg_all = (avg_b + avg_g + avg_r) / 3
        
        # Avoid division by zero
        if avg_b < 1: avg_b = 1
        if avg_g < 1: avg_g = 1
        if avg_r < 1: avg_r = 1
        
        # Scale each channel
        img[:, :, 0] = np.clip(img[:, :, 0] * (avg_all / avg_b), 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * (avg_all / avg_g), 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * (avg_all / avg_r), 0, 255)
        
        return img.astype(np.uint8)
    
    def extract_clothing_regions(self, person_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split person crop into upper and lower body regions
        
        Args:
            person_crop: Cropped person image (full body)
            
        Returns:
            Dict with 'upper' and 'lower' region images
        """
        h, w = person_crop.shape[:2]
        
        # Upper body: top 45% of crop (excludes head ~top 15%, focuses on torso)
        upper_start = int(h * 0.15)  # Skip head
        upper_end = int(h * 0.50)    # Torso region
        
        # Lower body: bottom 50% of crop
        lower_start = int(h * 0.50)
        lower_end = h
        
        return {
            'upper': person_crop[upper_start:upper_end, :].copy(),
            'lower': person_crop[lower_start:lower_end, :].copy()
        }
    
    def get_dominant_colors(self, image: np.ndarray, n_colors: int = None) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using K-means clustering in LAB space
        
        LAB color space is more perceptually uniform and handles
        lighting variations better than RGB.
        """
        if n_colors is None:
            n_colors = self.n_colors
        
        if image.size == 0:
            return [(0, 0, 0)]
        
        # Apply white balance correction
        image = self.white_balance_correction(image)
        
        # Convert to LAB color space for illumination invariance
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Reshape to list of pixels
        pixels = lab_image.reshape(-1, 3)
        
        # Remove pixels with very low saturation (background/shadows)
        # L channel > 20 and < 235 (not too dark or too bright)
        mask = (pixels[:, 0] > 20) & (pixels[:, 0] < 235)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < n_colors * 10:
            filtered_pixels = pixels
        
        if len(filtered_pixels) < n_colors:
            return [(0, 0, 0)]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers and convert back to BGR
        centers_lab = kmeans.cluster_centers_.astype(np.uint8)
        centers_bgr = []
        
        for center in centers_lab:
            lab_pixel = np.array([[center]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            centers_bgr.append(tuple(bgr_pixel[0, 0].tolist()))
        
        # Sort by frequency (most common first)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        sorted_centers = [centers_bgr[i] for i, _ in label_counts.most_common()]
        
        return sorted_centers
    
    def color_to_name(self, bgr_color: Tuple[int, int, int]) -> str:
        """Convert BGR color to human-readable name using HSV ranges"""
        # Convert to HSV
        pixel = np.array([[bgr_color]], dtype=np.uint8)
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0, 0]
        
        # Check each color range
        for color_name, (lower, upper) in self.COLOR_RANGES.items():
            if color_name == 'red2':
                continue  # Handle red separately
            
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                if color_name == 'red' or (h >= 160):  # Combine red ranges
                    return 'red'
                return color_name
        
        # Check red2 range
        lower, upper = self.COLOR_RANGES['red2']
        if (lower[0] <= h <= upper[0] and 
            lower[1] <= s <= upper[1] and 
            lower[2] <= v <= upper[2]):
            return 'red'
        
        return 'unknown'
    
    def extract_color_features(self, person_crop: np.ndarray) -> Dict[str, any]:
        """
        Extract comprehensive color features from a person crop
        
        Returns:
            Dict containing:
            - upper_colors: List of dominant BGR colors for upper body
            - lower_colors: List of dominant BGR colors for lower body
            - upper_color_names: Human-readable color names
            - lower_color_names: Human-readable color names
            - upper_histogram: Color histogram for upper body (for matching)
            - lower_histogram: Color histogram for lower body
        """
        regions = self.extract_clothing_regions(person_crop)
        
        upper_colors = self.get_dominant_colors(regions['upper'])
        lower_colors = self.get_dominant_colors(regions['lower'])
        
        # Get color names
        upper_names = [self.color_to_name(c) for c in upper_colors]
        lower_names = [self.color_to_name(c) for c in lower_colors]
        
        # Get Hue histograms for detailed matching
        upper_hist = self._get_hue_histogram(regions['upper'])
        lower_hist = self._get_hue_histogram(regions['lower'])
        
        return {
            'upper_colors': upper_colors,
            'lower_colors': lower_colors,
            'upper_color_names': upper_names,
            'lower_color_names': lower_names,
            'upper_histogram': upper_hist,
            'lower_histogram': lower_hist,
            'primary_upper': upper_names[0] if upper_names else 'unknown',
            'primary_lower': lower_names[0] if lower_names else 'unknown'
        }
    
    def _get_hue_histogram(self, image: np.ndarray) -> np.ndarray:
        """Get normalized Hue histogram (illumination invariant)"""
        if image.size == 0:
            return np.zeros(180)
        
        # Apply white balance
        image = self.white_balance_correction(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for significant saturation (ignore gray/white/black)
        mask = hsv[:, :, 1] > 50
        
        # Calculate Hue histogram
        hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8) * 255, [180], [0, 180])
        
        # Normalize
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        return hist.flatten()
    
    def compare_colors(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compare two color histograms using correlation
        
        Returns:
            Similarity score between 0 and 1
        """
        if hist1.sum() == 0 or hist2.sum() == 0:
            return 0.0
        
        # Use correlation comparison
        correlation = cv2.compareHist(
            hist1.astype(np.float32).reshape(-1, 1),
            hist2.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        
        # Normalize to 0-1 range
        return max(0, (correlation + 1) / 2)
    
    def match_color_description(self, features: Dict, description: str) -> float:
        """
        Match extracted colors against text description
        
        Args:
            features: Color features from extract_color_features()
            description: Text like "red shirt" or "blue jeans"
            
        Returns:
            Match score between 0 and 1
        """
        description_lower = description.lower()
        
        # Check for color keywords
        color_keywords = list(self.COLOR_RANGES.keys())
        color_keywords.remove('red2')  # Internal use only
        
        matches = []
        for color in color_keywords:
            if color in description_lower:
                # Check if color matches upper or lower clothing
                if color in features.get('upper_color_names', []):
                    matches.append(1.0)
                elif color in features.get('lower_color_names', []):
                    matches.append(1.0)
                else:
                    matches.append(0.0)
        
        if not matches:
            return 0.5  # No color specified, neutral score
        
        return sum(matches) / len(matches)


# Global instance
color_extractor = ColorExtractor()
