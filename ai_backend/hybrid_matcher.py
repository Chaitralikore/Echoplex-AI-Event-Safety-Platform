"""
Hybrid Matcher Module
Combines Re-ID features, clothing colors, and metadata for robust person matching

Matching Score Breakdown:
- Re-ID Feature Similarity: 50% (body shape, texture, appearance)
- Upper Clothing Color: 20% (top color match)
- Lower Clothing Color: 15% (bottom color match)
- Height/Proportions: 15% (body aspect ratio)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2

from person_detector import person_detector
from color_extractor import color_extractor
from reid_extractor import reid_extractor


class HybridMatcher:
    """
    Hybrid person matching using multiple attributes
    
    Combines:
    - Deep Re-ID features (512-d vectors)
    - Clothing color analysis
    - Body proportions
    """
    
    # Weights for different matching components
    WEIGHTS = {
        'reid_features': 0.50,    # Body shape, texture, overall appearance
        'upper_color': 0.20,      # Top clothing color
        'lower_color': 0.15,      # Bottom clothing color
        'body_ratio': 0.15        # Height/width proportions
    }
    
    def __init__(self):
        """Initialize hybrid matcher with all sub-modules"""
        self.detector = person_detector
        self.color_extractor = color_extractor
        self.reid_extractor = reid_extractor
    
    def extract_person_features(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from a person crop
        
        Args:
            person_crop: BGR image of cropped person
            
        Returns:
            Dict containing:
            - reid_vector: 512-d feature vector
            - color_features: Clothing color information
            - body_ratio: Height/width aspect ratio
        """
        if person_crop.size == 0:
            return None
        
        h, w = person_crop.shape[:2]
        
        # Extract Re-ID features (512-d vector)
        reid_vector = self.reid_extractor.extract_features(person_crop)
        
        # Extract clothing colors
        color_features = self.color_extractor.extract_color_features(person_crop)
        
        # Calculate body aspect ratio
        body_ratio = h / w if w > 0 else 2.0  # Typical person ratio ~2.5-3.5
        
        return {
            'reid_vector': reid_vector,
            'color_features': color_features,
            'body_ratio': body_ratio,
            'primary_upper': color_features.get('primary_upper', 'unknown'),
            'primary_lower': color_features.get('primary_lower', 'unknown')
        }
    
    def compute_match_score(self, 
                           query_features: Dict[str, Any],
                           candidate_features: Dict[str, Any],
                           text_description: Optional[str] = None) -> Dict[str, float]:
        """
        Compute hybrid match score between query and candidate
        
        Args:
            query_features: Features of the person being searched for
            candidate_features: Features of a detected person
            text_description: Optional text description (e.g., "red shirt, blue jeans")
            
        Returns:
            Dict with:
            - total_score: Combined weighted score (0-1)
            - reid_score: Re-ID feature similarity
            - upper_color_score: Upper clothing match
            - lower_color_score: Lower clothing match
            - body_ratio_score: Body proportions match
        """
        scores = {}
        
        # 1. Re-ID Feature Similarity (cosine similarity)
        scores['reid_score'] = self.reid_extractor.compute_similarity(
            query_features['reid_vector'],
            candidate_features['reid_vector']
        )
        
        # 2. Upper Clothing Color Match
        scores['upper_color_score'] = self.color_extractor.compare_colors(
            query_features['color_features']['upper_histogram'],
            candidate_features['color_features']['upper_histogram']
        )
        
        # 3. Lower Clothing Color Match
        scores['lower_color_score'] = self.color_extractor.compare_colors(
            query_features['color_features']['lower_histogram'],
            candidate_features['color_features']['lower_histogram']
        )
        
        # 4. Body Ratio Similarity
        ratio_diff = abs(query_features['body_ratio'] - candidate_features['body_ratio'])
        scores['body_ratio_score'] = max(0, 1 - ratio_diff / 2)  # Allow up to 2.0 difference
        
        # 5. Text Description Match (bonus if provided)
        if text_description:
            text_match = self.color_extractor.match_color_description(
                candidate_features['color_features'],
                text_description
            )
            # Boost score if text description matches
            scores['text_match_score'] = text_match
        else:
            scores['text_match_score'] = 0.5  # Neutral
        
        # Calculate weighted total score
        total_score = (
            scores['reid_score'] * self.WEIGHTS['reid_features'] +
            scores['upper_color_score'] * self.WEIGHTS['upper_color'] +
            scores['lower_color_score'] * self.WEIGHTS['lower_color'] +
            scores['body_ratio_score'] * self.WEIGHTS['body_ratio']
        )
        
        # Apply text description bonus (up to 10% boost)
        if text_description and scores['text_match_score'] > 0.5:
            text_bonus = (scores['text_match_score'] - 0.5) * 0.2
            total_score = min(1.0, total_score + text_bonus)
        
        scores['total_score'] = total_score
        
        return scores
    
    def find_matches_in_frame(self,
                              frame: np.ndarray,
                              query_features: Dict[str, Any],
                              text_description: Optional[str] = None,
                              min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find all matching persons in a video frame
        
        Args:
            frame: BGR video frame
            query_features: Features of person being searched for
            text_description: Optional clothing description
            min_score: Minimum match score threshold
            
        Returns:
            List of matches with bounding boxes and scores
        """
        matches = []
        
        # Detect all persons in frame
        detections = self.detector.detect_persons(frame)
        
        for x1, y1, x2, y2, detection_conf in detections:
            # Crop person
            person_crop = self.detector.crop_person(frame, (x1, y1, x2, y2))
            
            if person_crop.size == 0:
                continue
            
            # Extract features
            candidate_features = self.extract_person_features(person_crop)
            
            if candidate_features is None:
                continue
            
            # Compute match score
            scores = self.compute_match_score(
                query_features,
                candidate_features,
                text_description
            )
            
            if scores['total_score'] >= min_score:
                matches.append({
                    'bbox': (x1, y1, x2, y2),
                    'detection_confidence': detection_conf,
                    'match_scores': scores,
                    'total_score': scores['total_score'],
                    'candidate_features': candidate_features
                })
        
        # Sort by total score (best matches first)
        matches.sort(key=lambda x: x['total_score'], reverse=True)
        
        return matches
    
    def get_match_explanation(self, scores: Dict[str, float]) -> str:
        """
        Generate human-readable explanation of match scores
        
        Args:
            scores: Score dictionary from compute_match_score()
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Overall confidence
        confidence = int(scores['total_score'] * 100)
        explanations.append(f"Overall Match: {confidence}%")
        
        # Component breakdown
        if scores['reid_score'] > 0.7:
            explanations.append(f"✓ Strong body/appearance match ({int(scores['reid_score']*100)}%)")
        elif scores['reid_score'] > 0.5:
            explanations.append(f"○ Moderate appearance match ({int(scores['reid_score']*100)}%)")
        
        if scores['upper_color_score'] > 0.7:
            explanations.append(f"✓ Top clothing color matches ({int(scores['upper_color_score']*100)}%)")
        
        if scores['lower_color_score'] > 0.7:
            explanations.append(f"✓ Bottom clothing matches ({int(scores['lower_color_score']*100)}%)")
        
        if scores.get('text_match_score', 0) > 0.7:
            explanations.append("✓ Description matches detected clothing")
        
        return " | ".join(explanations)


# Global instance
hybrid_matcher = HybridMatcher()
