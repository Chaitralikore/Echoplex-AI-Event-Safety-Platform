"""
Precision Guardrails Module
Prevents false positives through:
- Dynamic thresholds based on video quality
- Negative vector blacklist (from False Match feedback)
- Attribute filtering (color mismatch rejection)
- Multi-frame consistency requirements
- Temporal smoothing

PHILOSOPHY: Default answer is "NO MATCH" - only flag when undeniable
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import time
import json
import os


class PrecisionGuardrails:
    """
    Precision-first matching with strict false positive prevention
    """
    
    # Minimum thresholds based on video quality
    THRESHOLD_HIGH_QUALITY = 0.85    # Clear, good lighting
    THRESHOLD_MEDIUM_QUALITY = 0.90  # Some blur
    THRESHOLD_LOW_QUALITY = 0.95     # Blurry, poor lighting
    THRESHOLD_VERY_LOW = 0.97        # Very poor quality
    
    # Consistency requirements (reduced for faster analysis)
    MIN_CONSISTENT_FRAMES = 5         # ~0.3s at 15fps sample rate
    MIN_TRACK_DURATION_SECONDS = 1.0  # 1 second of stable detection
    MAX_VECTOR_VARIANCE = 0.20        # Maximum variance in consecutive detections
    
    # Color match requirements
    COLOR_MATCH_REQUIRED = True      # Reject if described color doesn't match
    COLOR_MISMATCH_PENALTY = 0.30    # Reduce score by 30% if color doesn't match
    
    def __init__(self, blacklist_path: str = "./vector_blacklist.json"):
        """
        Initialize precision guardrails
        
        Args:
            blacklist_path: Path to persist blacklisted vectors
        """
        self.blacklist_path = blacklist_path
        
        # Blacklist: case_id -> list of blacklisted vectors
        self.blacklisted_vectors: Dict[str, List[np.ndarray]] = {}
        self.blacklist_metadata: Dict[str, List[Dict]] = {}
        
        # Track consistency across frames
        self.frame_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Load existing blacklist
        self._load_blacklist()
        
        print("Precision Guardrails initialized")
        print(f"  - Min frames for match: {self.MIN_CONSISTENT_FRAMES}")
        print(f"  - Min duration: {self.MIN_TRACK_DURATION_SECONDS}s")
        print(f"  - Base threshold: {self.THRESHOLD_HIGH_QUALITY}")
    
    def _load_blacklist(self):
        """Load blacklisted vectors from disk"""
        if os.path.exists(self.blacklist_path):
            try:
                with open(self.blacklist_path, 'r') as f:
                    data = json.load(f)
                
                for case_id, vectors in data.get('vectors', {}).items():
                    self.blacklisted_vectors[case_id] = [
                        np.array(v, dtype=np.float32) for v in vectors
                    ]
                
                self.blacklist_metadata = data.get('metadata', {})
                print(f"Loaded {sum(len(v) for v in self.blacklisted_vectors.values())} blacklisted vectors")
            except Exception as e:
                print(f"Could not load blacklist: {e}")
    
    def _save_blacklist(self):
        """Save blacklisted vectors to disk"""
        try:
            data = {
                'vectors': {
                    case_id: [v.tolist() for v in vectors]
                    for case_id, vectors in self.blacklisted_vectors.items()
                },
                'metadata': self.blacklist_metadata
            }
            with open(self.blacklist_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Could not save blacklist: {e}")
    
    def estimate_video_quality(self, frame: np.ndarray = None, 
                                frame_width: int = 0, frame_height: int = 0,
                                blur_score: float = None) -> str:
        """
        Estimate video quality to determine threshold
        
        Args:
            frame: Video frame (optional)
            frame_width: Frame width
            frame_height: Frame height
            blur_score: Pre-computed blur score (0-1, lower = blurrier)
            
        Returns:
            Quality level: 'high', 'medium', 'low', 'very_low'
        """
        quality_score = 1.0
        
        # Resolution factor
        if frame_width > 0 and frame_height > 0:
            pixels = frame_width * frame_height
            if pixels < 320 * 240:
                quality_score *= 0.5
            elif pixels < 640 * 480:
                quality_score *= 0.7
            elif pixels < 1280 * 720:
                quality_score *= 0.85
        
        # Blur factor (if provided)
        if blur_score is not None:
            quality_score *= blur_score
        
        # Compute blur from frame if available
        if frame is not None:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Laplacian variance: <100 is blurry, >500 is sharp
            if laplacian_var < 50:
                quality_score *= 0.5
            elif laplacian_var < 100:
                quality_score *= 0.7
            elif laplacian_var < 200:
                quality_score *= 0.85
        
        # Determine quality level
        if quality_score >= 0.8:
            return 'high'
        elif quality_score >= 0.6:
            return 'medium'
        elif quality_score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def get_dynamic_threshold(self, quality: str = 'medium') -> float:
        """Get match threshold based on video quality"""
        thresholds = {
            'high': self.THRESHOLD_HIGH_QUALITY,
            'medium': self.THRESHOLD_MEDIUM_QUALITY,
            'low': self.THRESHOLD_LOW_QUALITY,
            'very_low': self.THRESHOLD_VERY_LOW
        }
        return thresholds.get(quality, self.THRESHOLD_MEDIUM_QUALITY)
    
    def add_to_blacklist(self, case_id: str, vector: np.ndarray, 
                         reason: str = "false_match") -> bool:
        """
        Add a vector to the blacklist for a specific case
        
        Called when user clicks "False Match"
        
        Args:
            case_id: The case this vector was incorrectly matched to
            vector: The 512-d vector to blacklist
            reason: Reason for blacklisting
        """
        if case_id not in self.blacklisted_vectors:
            self.blacklisted_vectors[case_id] = []
            self.blacklist_metadata[case_id] = []
        
        self.blacklisted_vectors[case_id].append(vector.astype(np.float32))
        self.blacklist_metadata[case_id].append({
            'reason': reason,
            'timestamp': time.time()
        })
        
        self._save_blacklist()
        print(f"[BLACKLIST] Added vector to blacklist for case {case_id} (reason: {reason})")
        return True
    
    def is_blacklisted(self, case_id: str, vector: np.ndarray, 
                       similarity_threshold: float = 0.90) -> bool:
        """
        Check if a vector is blacklisted (too similar to a known false positive)
        
        Args:
            case_id: Case to check blacklist for
            vector: Vector to check
            similarity_threshold: How similar to blacklisted vector to reject
            
        Returns:
            True if vector should be rejected
        """
        if case_id not in self.blacklisted_vectors:
            return False
        
        vector = vector.astype(np.float32)
        
        for blacklisted in self.blacklisted_vectors[case_id]:
            similarity = float(np.dot(vector, blacklisted))
            if similarity >= similarity_threshold:
                print(f"[BLACKLIST] Rejected vector (similarity {similarity:.2f} to blacklisted)")
                return True
        
        return False
    
    def check_attribute_match(self, detected_attrs: Dict[str, str],
                               reported_attrs: Dict[str, str]) -> Tuple[bool, float]:
        """
        Check if detected attributes match reported description
        
        Args:
            detected_attrs: Colors/attributes detected in video
            reported_attrs: Colors/attributes from missing person report
            
        Returns:
            (passes_filter, penalty) - True if passes, penalty to apply
        """
        penalty = 0.0
        critical_mismatch = False
        
        # Check upper clothing color
        reported_upper = (reported_attrs.get('upper_clothing') or 
                         reported_attrs.get('upperClothingColor') or '').lower().strip()
        detected_upper = (detected_attrs.get('primary_upper') or 
                         detected_attrs.get('upper_color') or '').lower().strip()
        
        if reported_upper and reported_upper != 'unknown':
            if detected_upper and detected_upper != 'unknown':
                if reported_upper not in detected_upper and detected_upper not in reported_upper:
                    penalty += 0.20
                    print(f"[ATTR] Upper color mismatch: reported={reported_upper}, detected={detected_upper}")
                    
                    # Critical colors that should never mismatch
                    critical_colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'pink']
                    if any(c in reported_upper for c in critical_colors):
                        if not any(c in detected_upper for c in critical_colors if c in reported_upper):
                            critical_mismatch = True
        
        # Check lower clothing color
        reported_lower = (reported_attrs.get('lower_clothing') or 
                         reported_attrs.get('lowerClothingColor') or '').lower().strip()
        detected_lower = (detected_attrs.get('primary_lower') or 
                         detected_attrs.get('lower_color') or '').lower().strip()
        
        if reported_lower and reported_lower != 'unknown':
            if detected_lower and detected_lower != 'unknown':
                if reported_lower not in detected_lower and detected_lower not in reported_lower:
                    penalty += 0.15
                    print(f"[ATTR] Lower color mismatch: reported={reported_lower}, detected={detected_lower}")
        
        # Check gender if specified
        reported_gender = (reported_attrs.get('gender') or '').lower().strip()
        detected_gender = (detected_attrs.get('gender') or '').lower().strip()
        
        if reported_gender and reported_gender not in ['', 'unknown', 'other']:
            if detected_gender and detected_gender not in ['', 'unknown', 'other']:
                if reported_gender != detected_gender:
                    penalty += 0.25
                    print(f"[ATTR] Gender mismatch: reported={reported_gender}, detected={detected_gender}")
        
        # If COLOR_MATCH_REQUIRED and we have a critical mismatch, reject
        if self.COLOR_MATCH_REQUIRED and critical_mismatch:
            return (False, 1.0)  # Reject completely
        
        return (True, penalty)
    
    def check_temporal_consistency(self, case_id: str, 
                                    current_vector: np.ndarray,
                                    current_score: float,
                                    timestamp: float,
                                    frame_number: int) -> Dict[str, Any]:
        """
        Check if detection is consistent across multiple frames
        
        Args:
            case_id: Case being matched
            current_vector: Current detection's 512-d vector
            current_score: Current match score
            timestamp: Current video timestamp
            frame_number: Current frame number
            
        Returns:
            Dict with consistency info and whether to trigger match
        """
        track_key = f"{case_id}"
        
        # Add current detection to history
        self.frame_history[track_key].append({
            'vector': current_vector,
            'score': current_score,
            'timestamp': timestamp,
            'frame': frame_number
        })
        
        # Keep only recent history (last 5 seconds worth)
        history = self.frame_history[track_key]
        if len(history) > 1:
            cutoff_time = timestamp - 5.0
            history = [h for h in history if h['timestamp'] >= cutoff_time]
            self.frame_history[track_key] = history
        
        # Check consistency
        result = {
            'consistent_frames': len(history),
            'required_frames': self.MIN_CONSISTENT_FRAMES,
            'duration': 0.0,
            'required_duration': self.MIN_TRACK_DURATION_SECONDS,
            'score_variance': 0.0,
            'should_trigger': False,
            'reason': ''
        }
        
        if len(history) < 2:
            result['reason'] = 'Insufficient frames'
            return result
        
        # Calculate duration
        first_timestamp = min(h['timestamp'] for h in history)
        last_timestamp = max(h['timestamp'] for h in history)
        result['duration'] = last_timestamp - first_timestamp
        
        # Check frame count requirement
        if len(history) < self.MIN_CONSISTENT_FRAMES:
            result['reason'] = f'Only {len(history)}/{self.MIN_CONSISTENT_FRAMES} frames'
            return result
        
        # Check duration requirement
        if result['duration'] < self.MIN_TRACK_DURATION_SECONDS:
            result['reason'] = f'Only {result["duration"]:.1f}s / {self.MIN_TRACK_DURATION_SECONDS}s required'
            return result
        
        # Check score consistency (variance should be low)
        scores = [h['score'] for h in history[-self.MIN_CONSISTENT_FRAMES:]]
        result['score_variance'] = np.std(scores)
        
        if result['score_variance'] > 0.15:
            result['reason'] = f'Score too variable ({result["score_variance"]:.2f})'
            return result
        
        # Check vector consistency
        if len(history) >= 5:
            recent_vectors = [h['vector'] for h in history[-10:]]
            stacked = np.stack(recent_vectors)
            vector_variance = np.var(stacked, axis=0).mean()
            
            if vector_variance > self.MAX_VECTOR_VARIANCE:
                result['reason'] = f'Vector too variable ({vector_variance:.3f})'
                return result
        
        # All checks passed!
        result['should_trigger'] = True
        result['reason'] = 'Consistent detection confirmed'
        result['average_score'] = np.mean(scores)
        
        return result
    
    def validate_match(self, case_id: str, case_data: Dict,
                       detection_vector: np.ndarray,
                       detection_attrs: Dict[str, str],
                       raw_score: float,
                       video_quality: str = 'medium',
                       timestamp: float = 0,
                       frame_number: int = 0) -> Dict[str, Any]:
        """
        Full validation pipeline for a potential match
        
        Returns whether match should be reported and adjusted score
        """
        result = {
            'is_valid': False,
            'original_score': raw_score,
            'adjusted_score': raw_score,
            'threshold': 0.0,
            'rejections': [],
            'warnings': [],
            'temporal_status': None
        }
        
        # 1. Get dynamic threshold based on video quality
        threshold = self.get_dynamic_threshold(video_quality)
        result['threshold'] = threshold
        
        # 2. Check blacklist
        if self.is_blacklisted(case_id, detection_vector):
            result['rejections'].append('Vector blacklisted (previous false match)')
            result['adjusted_score'] = 0.0
            return result
        
        # 3. Check attribute match
        reported_attrs = case_data.get('metadata', {})
        passes_attr, attr_penalty = self.check_attribute_match(detection_attrs, reported_attrs)
        
        if not passes_attr:
            result['rejections'].append(f'Critical attribute mismatch')
            result['adjusted_score'] = 0.0
            return result
        
        if attr_penalty > 0:
            result['adjusted_score'] = raw_score - attr_penalty
            result['warnings'].append(f'Attribute mismatch penalty: -{attr_penalty:.0%}')
        
        # 4. Check temporal consistency
        temporal = self.check_temporal_consistency(
            case_id, detection_vector, raw_score, timestamp, frame_number
        )
        result['temporal_status'] = temporal
        
        if not temporal['should_trigger']:
            result['rejections'].append(f'Temporal: {temporal["reason"]}')
            # Don't return yet - we still want to track this detection
            # Just don't flag it as a match
        
        # 5. Check against threshold
        if result['adjusted_score'] < threshold:
            result['rejections'].append(f'Score {result["adjusted_score"]:.1%} < threshold {threshold:.1%}')
            return result
        
        # 6. Final validation - must pass temporal check
        if temporal['should_trigger']:
            result['is_valid'] = True
            # Use average score from consistent detections
            if 'average_score' in temporal:
                result['adjusted_score'] = temporal['average_score'] - attr_penalty
        
        return result
    
    def reset_frame_history(self, case_id: str = None):
        """Reset frame history for fresh analysis"""
        if case_id:
            self.frame_history[case_id] = []
        else:
            self.frame_history.clear()


# Global instance
precision_guardrails = PrecisionGuardrails()
