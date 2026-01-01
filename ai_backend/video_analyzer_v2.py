"""
Enhanced Video Analyzer Module (v2.0)
Complete Re-ID Pipeline with Temporal Analysis

Features:
- Person detection (YOLO)
- Re-ID feature extraction (512-d vectors)
- Clothing color analysis with white balance
- Temporal feature aggregation (Super-Vectors)
- Spatio-temporal filtering
- Hybrid matching (features + colors + body)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
import time

# Try to import new Re-ID modules
REID_AVAILABLE = False
GUARDRAILS_AVAILABLE = False

try:
    from person_detector import person_detector
    from color_extractor import color_extractor
    from reid_extractor import reid_extractor
    from temporal_analyzer import temporal_analyzer, TemporalTrack
    from hybrid_matcher import hybrid_matcher
    REID_AVAILABLE = True
    print("Re-ID pipeline loaded successfully")
except ImportError as e:
    print(f"Re-ID modules not fully available: {e}")
    print("Falling back to face_recognition")

# Import precision guardrails
try:
    from precision_guardrails import precision_guardrails
    GUARDRAILS_AVAILABLE = True
    print("Precision guardrails loaded - false positive prevention active")
except ImportError as e:
    print(f"Precision guardrails not available: {e}")

# Fallback to face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class EnhancedVideoAnalyzer:
    """
    Enhanced video analyzer with full Re-ID pipeline
    
    Uses:
    - YOLO for person detection
    - OSNet/MobileNet for 512-d feature extraction
    - Clothing color analysis
    - Temporal Super-Vectors for reduced false positives
    """
    
    def __init__(self, frame_sample_rate: int = 5):
        """
        Initialize enhanced analyzer
        
        Args:
            frame_sample_rate: Analyze every Nth frame (~6/sec at 30fps)
        """
        self.frame_sample_rate = frame_sample_rate
        self.registered_cases: Dict[str, Dict[str, Any]] = {}
        
        # Precision guardrails for false positive prevention
        self.guardrails = precision_guardrails if GUARDRAILS_AVAILABLE else None
        
        # Track video quality for dynamic thresholds
        self.current_video_quality = 'medium'
        
        # Use Re-ID modules if available
        if REID_AVAILABLE:
            self.detector = person_detector
            self.color_extractor = color_extractor
            self.reid_extractor = reid_extractor
            self.temporal = temporal_analyzer
            self.matcher = hybrid_matcher
        else:
            self.detector = None
            self.color_extractor = None
            self.reid_extractor = None
            self.temporal = None
            self.matcher = None
    
    def register_case(self, case_id: str, name: str, photo_path: str, 
                      metadata: Dict[str, Any] = None) -> bool:
        """Register missing person with reference photo"""
        try:
            image = cv2.imread(photo_path)
            if image is None:
                print(f"Could not load image: {photo_path}")
                return False
            
            return self._register_case_from_image(case_id, name, image, metadata)
        except Exception as e:
            print(f"Error registering case: {e}")
            return False
    
    def register_case_from_base64(self, case_id: str, name: str, 
                                   base64_photo: str, metadata: Dict[str, Any] = None) -> bool:
        """Register missing person from base64 encoded photo"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Remove data URL prefix if present
            if ',' in base64_photo:
                base64_photo = base64_photo.split(',')[1]
            
            # Decode to image
            image_data = base64.b64decode(base64_photo)
            pil_image = Image.open(io.BytesIO(image_data))
            image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            return self._register_case_from_image(case_id, name, image, metadata)
        except Exception as e:
            print(f"Error registering case from base64: {e}")
            return False
    
    def _register_case_from_image(self, case_id: str, name: str, 
                                   image: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Internal: Register case from CV2 image"""
        if REID_AVAILABLE:
            return self._register_with_reid(case_id, name, image, metadata)
        elif FACE_RECOGNITION_AVAILABLE:
            return self._register_with_face(case_id, name, image, metadata)
        else:
            print("No recognition backend available")
            return False
    
    def _register_with_reid(self, case_id: str, name: str, 
                            image: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Register using Re-ID pipeline"""
        # Detect person in reference photo
        detections = self.detector.detect_persons(image)
        
        if not detections:
            # Use full image as person crop
            person_crop = image
        else:
            # Use largest detection
            largest = max(detections, key=lambda d: (d[2]-d[0]) * (d[3]-d[1]))
            x1, y1, x2, y2, _ = largest
            person_crop = image[y1:y2, x1:x2]
        
        # Extract features
        features = self.matcher.extract_person_features(person_crop)
        
        if features is None:
            print(f"Could not extract features for {name}")
            return False
        
        # Store registration
        self.registered_cases[case_id] = {
            'name': name,
            'reid_vector': features['reid_vector'],
            'color_features': features['color_features'],
            'features': features,
            'metadata': metadata or {},
            'description': metadata.get('description', '') if metadata else ''
        }
        
        print(f"Registered {name} with Re-ID features (upper: {features['primary_upper']}, lower: {features['primary_lower']})")
        return True
    
    def _register_with_face(self, case_id: str, name: str, 
                            image: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Fallback: Register using face_recognition"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        
        if not encodings:
            print(f"No face detected for {name}")
            return False
        
        self.registered_cases[case_id] = {
            'name': name,
            'encoding': encodings[0],
            'metadata': metadata or {}
        }
        
        print(f"Registered {name} with face encoding (fallback)")
        return True
    
    def analyze_video_bytes(self, video_bytes: bytes, tolerance: float = 0.7) -> Dict[str, Any]:
        """Analyze video from bytes"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            return self.analyze_video(tmp_path, tolerance)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def analyze_video(self, video_path: str, tolerance: float = 0.7) -> Dict[str, Any]:
        """
        Analyze video with full Re-ID pipeline
        
        Args:
            video_path: Path to video file
            tolerance: Match threshold (0-1, higher = stricter)
            
        Returns:
            Analysis results with matches, tracks, and timing
        """
        results = {
            'frames_analyzed': 0,
            'persons_detected': 0,
            'faces_detected': 0,
            'tracks_created': 0,
            'super_vectors_generated': 0,
            'matches': [],
            'analysis_complete': True,
            'error': None,
            'method': 'reid' if REID_AVAILABLE else 'face_recognition'
        }
        
        if not self.registered_cases:
            results['error'] = 'No cases registered'
            return results
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results['error'] = 'Could not open video'
            results['analysis_complete'] = False
            return results
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[ANALYZE] Video: {total_frames} frames at {fps:.1f} FPS")
        
        # Reset temporal analyzer
        if self.temporal:
            self.temporal.reset()
        
        frame_count = 0
        all_matches: Dict[str, List[Dict]] = {case_id: [] for case_id in self.registered_cases}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames
            if frame_count % self.frame_sample_rate != 0:
                continue
            
            results['frames_analyzed'] += 1
            timestamp = frame_count / fps
            
            if REID_AVAILABLE:
                self._analyze_frame_reid(frame, frame_count, timestamp, results, all_matches, tolerance)
            elif FACE_RECOGNITION_AVAILABLE:
                self._analyze_frame_face(frame, frame_count, timestamp, results, all_matches, tolerance)
        
        cap.release()
        
        # Compile final matches from all detections
        for case_id, case_data in self.registered_cases.items():
            case_matches = all_matches[case_id]
            
            if case_matches:
                # Get best match using Super-Vector if available
                if self.temporal and REID_AVAILABLE:
                    best_match = self._get_best_temporal_match(case_id, case_matches)
                else:
                    best_match = max(case_matches, key=lambda m: m['score'])
                
                results['matches'].append({
                    'case_id': case_id,
                    'name': case_data['name'],
                    'confidence': round(best_match['score'] * 100, 1),
                    'detection_count': len(case_matches),
                    'first_seen': round(min(m['timestamp'] for m in case_matches), 1),
                    'last_seen': round(max(m['timestamp'] for m in case_matches), 1),
                    'metadata': case_data.get('metadata', {}),
                    'match_details': best_match.get('details', {}),
                    'super_vector_used': best_match.get('super_vector_used', False)
                })
        
        # Track statistics
        if self.temporal:
            results['tracks_created'] = self.temporal.next_track_id
            results['super_vectors_generated'] = len(self.temporal.get_high_confidence_tracks())
        
        return results
    
    def _analyze_frame_reid(self, frame: np.ndarray, frame_num: int, 
                            timestamp: float, results: Dict, 
                            all_matches: Dict, tolerance: float):
        """Analyze frame using Re-ID pipeline"""
        # Detect persons
        detections = self.detector.detect_persons(frame)
        results['persons_detected'] += len(detections)
        
        frame_detections = []
        
        for x1, y1, x2, y2, det_conf in detections:
            person_crop = self.detector.crop_person(frame, (x1, y1, x2, y2))
            
            if person_crop.size == 0:
                continue
            
            # Extract features
            features = self.matcher.extract_person_features(person_crop)
            
            if features is None:
                continue
            
            frame_detections.append({
                'bbox': (x1, y1, x2, y2),
                'features': features,
                'detection_confidence': det_conf
            })
        
        # Process through temporal analyzer
        if self.temporal and frame_detections:
            processed = self.temporal.process_detections(frame_detections, frame_num, timestamp)
        else:
            processed = frame_detections
        
        # Match against registered cases
        for det in processed:
            features = det['features']
            super_vector = det.get('super_vector')
            track_confidence = det.get('track_confidence', 0.5)
            
            for case_id, case_data in self.registered_cases.items():
                # Use Super-Vector if available and high confidence
                if super_vector is not None and track_confidence > 0.6:
                    similarity = self.reid_extractor.compute_similarity(
                        case_data['reid_vector'], super_vector
                    )
                    super_vector_used = True
                else:
                    similarity = self.reid_extractor.compute_similarity(
                        case_data['reid_vector'], features['reid_vector']
                    )
                    super_vector_used = False
                
                # Also score clothing colors
                color_score = self.color_extractor.compare_colors(
                    case_data['color_features']['upper_histogram'],
                    features['color_features']['upper_histogram']
                ) * 0.5 + self.color_extractor.compare_colors(
                    case_data['color_features']['lower_histogram'],
                    features['color_features']['lower_histogram']
                ) * 0.5
                
                # Combined score (weighted)
                combined_score = similarity * 0.6 + color_score * 0.4
                
                # Boost score if track has high confidence
                if track_confidence > 0.7:
                    combined_score = min(1.0, combined_score * 1.1)
                
                # ====== PRECISION GUARDRAILS ======
                if self.guardrails:
                    # Build detected attributes
                    detected_attrs = {
                        'primary_upper': features.get('primary_upper', 'unknown'),
                        'primary_lower': features.get('primary_lower', 'unknown'),
                        'upper_color': features.get('primary_upper', 'unknown'),
                        'lower_color': features.get('primary_lower', 'unknown')
                    }
                    
                    # Validate through guardrails
                    validation = self.guardrails.validate_match(
                        case_id=case_id,
                        case_data=case_data,
                        detection_vector=features['reid_vector'],
                        detection_attrs=detected_attrs,
                        raw_score=combined_score,
                        video_quality=self.current_video_quality,
                        timestamp=timestamp,
                        frame_number=frame_num
                    )
                    
                    # Log rejections for debugging
                    if validation['rejections']:
                        for rejection in validation['rejections']:
                            print(f"[GUARDRAIL] Rejected {case_data['name']}: {rejection}")
                    
                    # Only accept if guardrails validate
                    if not validation['is_valid']:
                        # Still track this detection for temporal consistency
                        # but don't flag as a match yet
                        continue
                    
                    # Use adjusted score from guardrails
                    combined_score = validation['adjusted_score']
                else:
                    # Without guardrails, use tolerance as threshold  
                    if combined_score < tolerance:
                        continue
                
                # ====== VALID MATCH ======
                all_matches[case_id].append({
                    'score': combined_score,
                    'timestamp': timestamp,
                    'frame': frame_num,
                    'bbox': det['bbox'],
                    'track_id': det.get('track_id'),
                    'track_confidence': track_confidence,
                    'super_vector_used': super_vector_used,
                    'details': {
                        'reid_score': float(similarity),
                        'color_score': float(color_score),
                        'upper_color': det.get('primary_upper_color', features['primary_upper']),
                        'lower_color': det.get('primary_lower_color', features['primary_lower'])
                    }
                })
    
    def _analyze_frame_face(self, frame: np.ndarray, frame_num: int,
                            timestamp: float, results: Dict,
                            all_matches: Dict, tolerance: float):
        """Fallback: Analyze frame using face_recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        results['faces_detected'] += len(face_locations)
        results['persons_detected'] += len(face_locations)
        
        if not face_locations:
            return
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for encoding in face_encodings:
            for case_id, case_data in self.registered_cases.items():
                if 'encoding' not in case_data:
                    continue
                
                distance = face_recognition.face_distance([case_data['encoding']], encoding)[0]
                similarity = 1 - distance
                
                if similarity >= tolerance:
                    all_matches[case_id].append({
                        'score': similarity,
                        'timestamp': timestamp,
                        'frame': frame_num,
                        'bbox': None,
                        'super_vector_used': False,
                        'details': {'face_distance': float(distance)}
                    })
    
    def _get_best_temporal_match(self, case_id: str, matches: List[Dict]) -> Dict:
        """Get best match considering temporal information"""
        if not matches:
            return {}
        
        # Group by track
        track_matches: Dict[str, List[Dict]] = {}
        for m in matches:
            track_id = m.get('track_id', 'unknown')
            if track_id not in track_matches:
                track_matches[track_id] = []
            track_matches[track_id].append(m)
        
        # Find best track
        best_match = None
        best_score = 0
        
        for track_id, track_ms in track_matches.items():
            # Score based on consistency and Super-Vector usage
            track_score = max(m['score'] for m in track_ms)
            track_length = len(track_ms)
            
            # Boost for longer tracks (more consistent detection)
            if track_length >= 10:
                track_score *= 1.15
            elif track_length >= 5:
                track_score *= 1.05
            
            # Boost for Super-Vector usage
            if any(m.get('super_vector_used') for m in track_ms):
                track_score *= 1.1
            
            track_score = min(1.0, track_score)
            
            if track_score > best_score:
                best_score = track_score
                best_match = max(track_ms, key=lambda m: m['score'])
                best_match['score'] = track_score
        
        return best_match or max(matches, key=lambda m: m['score'])
    
    def get_registered_cases(self) -> List[Dict[str, Any]]:
        """Get list of registered cases"""
        return [
            {
                'case_id': case_id,
                'name': data['name'],
                'has_reid': 'reid_vector' in data,
                'metadata': data.get('metadata', {})
            }
            for case_id, data in self.registered_cases.items()
        ]
    
    def remove_case(self, case_id: str) -> bool:
        """Remove a registered case"""
        if case_id in self.registered_cases:
            del self.registered_cases[case_id]
            return True
        return False


# Global instance (replaces old video_analyzer)
# Use sample_rate=15 for faster analysis (~2 fps)
enhanced_video_analyzer = EnhancedVideoAnalyzer(frame_sample_rate=15)

# Alias for backward compatibility
video_analyzer = enhanced_video_analyzer
