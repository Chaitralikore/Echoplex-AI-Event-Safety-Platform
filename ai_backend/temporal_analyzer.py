"""
Temporal Analyzer Module
Implements Temporal Feature Aggregation (Super-Vectors) and Spatio-Temporal Filtering

Features:
- Super-Vector creation from consecutive detections
- Spatio-temporal camera prioritization
- Track ID management for persistent person tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime, timedelta
import time


class TemporalTrack:
    """
    Represents a tracked person across multiple frames
    
    Accumulates features from consecutive detections to create
    high-confidence "Super-Vectors"
    """
    
    def __init__(self, track_id: str, initial_features: Dict[str, Any], 
                 frame_number: int, timestamp: float):
        self.track_id = track_id
        self.feature_history: List[np.ndarray] = [initial_features['reid_vector']]
        self.color_history: List[Dict] = [initial_features['color_features']]
        self.frame_numbers: List[int] = [frame_number]
        self.timestamps: List[float] = [timestamp]
        self.last_bbox: Tuple[int, int, int, int] = None
        self.last_update_frame = frame_number
        self.confidence_score = 0.0
        
        # Metadata from initial detection
        self.primary_upper_color = initial_features.get('primary_upper', 'unknown')
        self.primary_lower_color = initial_features.get('primary_lower', 'unknown')
    
    def update(self, features: Dict[str, Any], frame_number: int, 
               timestamp: float, bbox: Tuple[int, int, int, int]):
        """Add new detection to track"""
        self.feature_history.append(features['reid_vector'])
        self.color_history.append(features['color_features'])
        self.frame_numbers.append(frame_number)
        self.timestamps.append(timestamp)
        self.last_bbox = bbox
        self.last_update_frame = frame_number
        
        # Update dominant colors if we have enough samples
        if len(self.color_history) >= 3:
            self._update_dominant_colors()
    
    def _update_dominant_colors(self):
        """Update color based on most frequent detection"""
        upper_colors = [c.get('primary_upper', 'unknown') for c in self.color_history]
        lower_colors = [c.get('primary_lower', 'unknown') for c in self.color_history]
        
        # Get most common
        from collections import Counter
        upper_counter = Counter(upper_colors)
        lower_counter = Counter(lower_colors)
        
        self.primary_upper_color = upper_counter.most_common(1)[0][0]
        self.primary_lower_color = lower_counter.most_common(1)[0][0]
    
    def get_super_vector(self, min_detections: int = 5) -> Optional[np.ndarray]:
        """
        Generate Super-Vector from accumulated features
        
        Super-Vector is a weighted average of all feature vectors,
        with more recent detections weighted higher.
        
        Args:
            min_detections: Minimum detections required for Super-Vector
            
        Returns:
            512-d Super-Vector or None if insufficient detections
        """
        if len(self.feature_history) < min_detections:
            return None
        
        # Create weights (exponential decay, recent = higher weight)
        n = len(self.feature_history)
        weights = np.exp(np.linspace(-1, 0, n))  # Exponential weights
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of all feature vectors
        stacked = np.stack(self.feature_history)
        super_vector = np.average(stacked, axis=0, weights=weights)
        
        # L2 normalize
        norm = np.linalg.norm(super_vector)
        if norm > 0:
            super_vector = super_vector / norm
        
        return super_vector.astype(np.float32)
    
    def get_confidence(self) -> float:
        """
        Calculate confidence score based on track quality
        
        Factors:
        - Number of consecutive detections
        - Temporal consistency
        - Feature stability (low variance = high confidence)
        """
        n = len(self.feature_history)
        
        if n < 3:
            return 0.3  # Low confidence for short tracks
        
        # Base confidence from detection count (saturates at 20)
        count_confidence = min(n / 20, 1.0)
        
        # Feature stability (lower variance = higher confidence)
        if n >= 5:
            stacked = np.stack(self.feature_history[-10:])  # Last 10 features
            variance = np.var(stacked, axis=0).mean()
            stability_confidence = max(0, 1 - variance * 10)
        else:
            stability_confidence = 0.5
        
        # Temporal continuity (consistent frame intervals)
        if len(self.frame_numbers) >= 3:
            frame_diffs = np.diff(self.frame_numbers)
            regularity = 1 / (1 + np.std(frame_diffs) / np.mean(frame_diffs))
        else:
            regularity = 0.5
        
        # Combined confidence
        confidence = (
            count_confidence * 0.4 +
            stability_confidence * 0.4 +
            regularity * 0.2
        )
        
        self.confidence_score = confidence
        return confidence
    
    @property
    def duration(self) -> float:
        """Track duration in seconds"""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def detection_count(self) -> int:
        return len(self.feature_history)


class SpatioTemporalFilter:
    """
    Spatio-Temporal Filtering for camera prioritization
    
    When a match is found at a location, prioritize searching
    nearby cameras for a time window (movement prediction).
    """
    
    # Camera topology (define which cameras are near each other)
    # Format: camera_id -> list of (nearby_camera_id, travel_time_seconds)
    CAMERA_TOPOLOGY = {
        'main_entrance': [
            ('lobby_cam_1', 30),
            ('lobby_cam_2', 30),
            ('exit_gate_a', 60),
            ('parking_entrance', 90),
        ],
        'lobby_cam_1': [
            ('main_entrance', 30),
            ('lobby_cam_2', 15),
            ('food_court_1', 45),
            ('vip_section', 60),
        ],
        'lobby_cam_2': [
            ('main_entrance', 30),
            ('lobby_cam_1', 15),
            ('food_court_2', 45),
            ('restrooms', 30),
        ],
        'food_court': [
            ('lobby_cam_1', 45),
            ('lobby_cam_2', 45),
            ('stage_area', 90),
        ],
        'vip_section': [
            ('lobby_cam_1', 60),
            ('stage_area', 30),
            ('exit_gate_b', 45),
        ],
        'exit_gate_a': [
            ('main_entrance', 60),
            ('parking_lot', 30),
        ],
        'exit_gate_b': [
            ('vip_section', 45),
            ('parking_lot', 45),
        ],
        'stage_area': [
            ('food_court', 90),
            ('vip_section', 30),
            ('general_seating', 60),
        ],
    }
    
    def __init__(self, search_window_seconds: int = 300):
        """
        Initialize spatio-temporal filter
        
        Args:
            search_window_seconds: Time window to search after detection (default 5 min)
        """
        self.search_window = search_window_seconds
        
        # Last known sightings: person_id -> (camera_id, timestamp, confidence)
        self.sightings: Dict[str, Tuple[str, float, float]] = {}
    
    def register_sighting(self, person_id: str, camera_id: str, 
                         timestamp: float, confidence: float):
        """Register a confirmed sighting at a camera location"""
        self.sightings[person_id] = (camera_id, timestamp, confidence)
    
    def get_priority_cameras(self, person_id: str, 
                             current_time: float = None) -> List[Tuple[str, float]]:
        """
        Get prioritized list of cameras to search for a person
        
        Based on last sighting location and travel time predictions.
        
        Args:
            person_id: ID of person being searched
            current_time: Current timestamp (default: now)
            
        Returns:
            List of (camera_id, priority_score) sorted by priority
        """
        if current_time is None:
            current_time = time.time()
        
        if person_id not in self.sightings:
            # No prior sighting - return all cameras with equal priority
            return [(cam, 1.0) for cam in self.CAMERA_TOPOLOGY.keys()]
        
        last_camera, last_time, last_confidence = self.sightings[person_id]
        elapsed = current_time - last_time
        
        # If too much time has passed, return all cameras
        if elapsed > self.search_window:
            return [(cam, 1.0) for cam in self.CAMERA_TOPOLOGY.keys()]
        
        priority_cameras = []
        
        # Add the last camera with high priority (person might still be there)
        if elapsed < 60:  # Less than 1 minute
            priority_cameras.append((last_camera, 1.0))
        else:
            priority_cameras.append((last_camera, 0.6))
        
        # Add nearby cameras based on travel time
        if last_camera in self.CAMERA_TOPOLOGY:
            for nearby_cam, travel_time in self.CAMERA_TOPOLOGY[last_camera]:
                # Check if person could have reached this camera
                if elapsed >= travel_time * 0.5:  # Could have arrived
                    # Priority decreases as window closes
                    time_remaining = self.search_window - elapsed
                    priority = min(1.0, time_remaining / self.search_window)
                    
                    # Boost priority if travel time matches elapsed time
                    time_match = 1 - abs(elapsed - travel_time) / travel_time
                    priority *= (0.5 + 0.5 * max(0, time_match))
                    
                    priority_cameras.append((nearby_cam, priority))
        
        # Sort by priority (highest first)
        priority_cameras.sort(key=lambda x: x[1], reverse=True)
        
        return priority_cameras
    
    def get_search_region(self, person_id: str, 
                          frame_width: int, frame_height: int,
                          current_time: float = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Get region of interest in frame based on predicted movement
        
        Returns:
            (x1, y1, x2, y2) region or None for full frame search
        """
        # For now, return None (search full frame)
        # In production, this could predict where person might appear
        # based on direction of travel from last detection
        return None


class TemporalAnalyzer:
    """
    Main temporal analysis component
    
    Manages person tracks across frames and generates Super-Vectors
    """
    
    def __init__(self, 
                 min_track_length: int = 10,
                 max_gap_frames: int = 15,
                 iou_threshold: float = 0.3):
        """
        Initialize temporal analyzer
        
        Args:
            min_track_length: Minimum frames for Super-Vector
            max_gap_frames: Maximum frame gap before track is ended
            iou_threshold: IoU threshold for track association
        """
        self.min_track_length = min_track_length
        self.max_gap_frames = max_gap_frames
        self.iou_threshold = iou_threshold
        
        self.active_tracks: Dict[str, TemporalTrack] = {}
        self.completed_tracks: List[TemporalTrack] = []
        self.next_track_id = 0
        
        self.spatio_temporal_filter = SpatioTemporalFilter()
        
        # Re-ID extractor for feature comparison
        try:
            from reid_extractor import reid_extractor
            self.reid = reid_extractor
        except ImportError:
            self.reid = None
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _find_matching_track(self, features: Dict, bbox: Tuple, 
                             current_frame: int) -> Optional[str]:
        """Find existing track that matches this detection"""
        best_match_id = None
        best_score = 0.0
        
        for track_id, track in self.active_tracks.items():
            # Skip stale tracks
            frame_gap = current_frame - track.last_update_frame
            if frame_gap > self.max_gap_frames:
                continue
            
            # Compute IoU if we have previous bbox
            if track.last_bbox:
                iou = self._compute_iou(bbox, track.last_bbox)
            else:
                iou = 0.0
            
            # Compute feature similarity
            if self.reid and len(track.feature_history) > 0:
                similarity = self.reid.compute_similarity(
                    features['reid_vector'],
                    track.feature_history[-1]  # Compare with last feature
                )
            else:
                similarity = 0.5
            
            # Combined score
            score = iou * 0.4 + similarity * 0.6
            
            if score > best_score and score > 0.4:
                best_score = score
                best_match_id = track_id
        
        return best_match_id
    
    def process_detections(self, detections: List[Dict], 
                          frame_number: int, 
                          timestamp: float) -> List[Dict]:
        """
        Process all detections in a frame
        
        Associates detections with existing tracks or creates new ones.
        
        Args:
            detections: List of detection dicts with 'features' and 'bbox'
            frame_number: Current frame number
            timestamp: Current timestamp in video
            
        Returns:
            List of detection dicts with added 'track_id' and 'super_vector'
        """
        processed = []
        matched_tracks = set()
        
        for det in detections:
            features = det.get('features')
            bbox = det.get('bbox')
            
            if features is None or bbox is None:
                continue
            
            # Try to match with existing track
            track_id = self._find_matching_track(features, bbox, frame_number)
            
            if track_id and track_id not in matched_tracks:
                # Update existing track
                self.active_tracks[track_id].update(features, frame_number, timestamp, bbox)
                matched_tracks.add(track_id)
            else:
                # Create new track
                track_id = f"track_{self.next_track_id}"
                self.next_track_id += 1
                
                self.active_tracks[track_id] = TemporalTrack(
                    track_id, features, frame_number, timestamp
                )
                self.active_tracks[track_id].last_bbox = bbox
            
            # Get Super-Vector if available
            track = self.active_tracks[track_id]
            super_vector = track.get_super_vector(self.min_track_length)
            
            processed.append({
                **det,
                'track_id': track_id,
                'super_vector': super_vector,
                'track_confidence': track.get_confidence(),
                'track_length': track.detection_count,
                'track_duration': track.duration,
                'primary_upper_color': track.primary_upper_color,
                'primary_lower_color': track.primary_lower_color
            })
        
        # Clean up stale tracks
        self._cleanup_stale_tracks(frame_number)
        
        return processed
    
    def _cleanup_stale_tracks(self, current_frame: int):
        """Move stale tracks to completed list"""
        stale_ids = []
        
        for track_id, track in self.active_tracks.items():
            if current_frame - track.last_update_frame > self.max_gap_frames:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            track = self.active_tracks.pop(track_id)
            if track.detection_count >= self.min_track_length:
                self.completed_tracks.append(track)
    
    def get_high_confidence_tracks(self, min_confidence: float = 0.7) -> List[TemporalTrack]:
        """Get all tracks with high confidence Super-Vectors"""
        tracks = []
        
        for track in list(self.active_tracks.values()) + self.completed_tracks:
            if track.get_confidence() >= min_confidence:
                if track.get_super_vector() is not None:
                    tracks.append(track)
        
        return tracks
    
    def get_best_match_track(self, query_features: Dict, 
                              min_similarity: float = 0.6) -> Optional[Tuple[TemporalTrack, float]]:
        """
        Find the best matching track for a query
        
        Uses Super-Vectors for more accurate matching.
        
        Returns:
            (track, similarity) or None
        """
        best_track = None
        best_similarity = 0.0
        
        query_vector = query_features['reid_vector']
        
        for track in self.get_high_confidence_tracks():
            super_vector = track.get_super_vector()
            
            if super_vector is not None and self.reid:
                similarity = self.reid.compute_similarity(query_vector, super_vector)
                
                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_track = track
        
        if best_track:
            return (best_track, best_similarity)
        return None
    
    def reset(self):
        """Reset all tracks"""
        self.active_tracks.clear()
        self.completed_tracks.clear()
        self.next_track_id = 0


# Global instance
temporal_analyzer = TemporalAnalyzer()
