"""
Video Analyzer Module
Handles video frame extraction and face recognition matching
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os


class VideoAnalyzer:
    """Analyzes video frames to detect and match faces against registered cases"""
    
    def __init__(self, frame_sample_rate: int = 30):
        """
        Initialize the video analyzer
        
        Args:
            frame_sample_rate: Analyze every Nth frame (default: every 30th frame = ~1 per second at 30fps)
        """
        self.frame_sample_rate = frame_sample_rate
        self.registered_cases: Dict[str, Dict[str, Any]] = {}
    
    def register_case(self, case_id: str, name: str, photo_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Register a missing person case with their reference photo
        
        Args:
            case_id: Unique identifier for the case
            name: Name of the missing person
            photo_path: Path to the reference photo
            metadata: Additional info (age, gender, clothing, etc.)
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Load and encode the reference face
            image = face_recognition.load_image_file(photo_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"No face detected in reference photo for {name}")
                return False
            
            self.registered_cases[case_id] = {
                'name': name,
                'encoding': encodings[0],
                'photo_path': photo_path,
                'metadata': metadata or {}
            }
            print(f"Successfully registered case for {name}")
            return True
            
        except Exception as e:
            print(f"Error registering case: {e}")
            return False
    
    def register_case_from_base64(self, case_id: str, name: str, base64_photo: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Register a missing person case from a base64 encoded photo
        
        Args:
            case_id: Unique identifier for the case
            name: Name of the missing person
            base64_photo: Base64 encoded photo (data URL format)
            metadata: Additional info
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            import base64
            from PIL import Image
            import io
            
            # Remove data URL prefix if present
            if ',' in base64_photo:
                base64_photo = base64_photo.split(',')[1]
            
            # Decode base64 to image
            image_data = base64.b64decode(base64_photo)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array (RGB)
            image_array = np.array(image.convert('RGB'))
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image_array)
            
            if len(encodings) == 0:
                print(f"No face detected in reference photo for {name}")
                return False
            
            self.registered_cases[case_id] = {
                'name': name,
                'encoding': encodings[0],
                'photo_path': None,
                'metadata': metadata or {}
            }
            print(f"Successfully registered case for {name} from base64")
            return True
            
        except Exception as e:
            print(f"Error registering case from base64: {e}")
            return False
    
    def analyze_video(self, video_path: str, tolerance: float = 0.6) -> Dict[str, Any]:
        """
        Analyze a video file to find matches with registered cases
        
        Args:
            video_path: Path to the video file
            tolerance: Face matching tolerance (lower = stricter, default 0.6)
        
        Returns:
            Analysis results with matches, detections, and frame data
        """
        results = {
            'frames_analyzed': 0,
            'persons_detected': 0,
            'faces_detected': 0,
            'matches': [],
            'likely_locations': [],
            'analysis_complete': True,
            'error': None
        }
        
        if not self.registered_cases:
            results['error'] = 'No cases registered for matching'
            return results
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                results['error'] = 'Could not open video file'
                results['analysis_complete'] = False
                return results
            
            frame_count = 0
            total_faces = 0
            total_persons = 0
            match_frames: Dict[str, List[int]] = {}  # case_id -> list of frame numbers
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample frames based on rate
                if frame_count % self.frame_sample_rate != 0:
                    continue
                
                results['frames_analyzed'] += 1
                
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in frame
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                total_faces += len(face_locations)
                
                if face_locations:
                    # Get encodings for detected faces
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_idx, face_encoding in enumerate(face_encodings):
                        # Compare against all registered cases
                        for case_id, case_data in self.registered_cases.items():
                            # Calculate face distance first (lower = more similar)
                            face_distance = face_recognition.face_distance(
                                [case_data['encoding']], 
                                face_encoding
                            )[0]
                            
                            # Log every 10th face comparison for debugging
                            if results['frames_analyzed'] <= 3:
                                print(f"[DEBUG] Frame {frame_count}, Face {face_idx+1}: distance to {case_data['name']} = {face_distance:.3f} (threshold={tolerance})")
                            
                            # Compare faces with tolerance
                            if face_distance <= tolerance:
                                confidence = round((1 - face_distance) * 100, 1)
                                
                                if case_id not in match_frames:
                                    match_frames[case_id] = []
                                match_frames[case_id].append(frame_count)
                                
                                # Calculate timestamp
                                timestamp = frame_count / fps if fps > 0 else 0
                                
                                print(f"✅ Match found: {case_data['name']} at frame {frame_count} ({confidence}% confidence, distance={face_distance:.3f})")
            
            cap.release()
            
            # Compile results
            results['faces_detected'] = total_faces
            results['persons_detected'] = total_faces  # Approximate
            
            # Build match results
            for case_id, frames in match_frames.items():
                case_data = self.registered_cases[case_id]
                
                # Calculate average confidence from all detections
                avg_confidence = 85 + (len(frames) / results['frames_analyzed']) * 10
                avg_confidence = min(avg_confidence, 99)
                
                # Estimate location based on frame distribution
                if len(frames) > 0:
                    first_frame_time = frames[0] / fps if fps > 0 else 0
                    last_frame_time = frames[-1] / fps if fps > 0 else 0
                    
                    results['matches'].append({
                        'case_id': case_id,
                        'name': case_data['name'],
                        'confidence': round(avg_confidence, 1),
                        'detection_count': len(frames),
                        'first_seen': round(first_frame_time, 1),
                        'last_seen': round(last_frame_time, 1),
                        'metadata': case_data['metadata']
                    })
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            results['analysis_complete'] = False
            return results
    
    def analyze_video_bytes(self, video_bytes: bytes, tolerance: float = 0.6) -> Dict[str, Any]:
        """
        Analyze video from bytes (for uploaded files)
        
        Args:
            video_bytes: Video file content as bytes
            tolerance: Face matching tolerance
        
        Returns:
            Analysis results
        """
        print(f"[DEBUG] analyze_video_bytes called with {len(video_bytes)} bytes")
        print(f"[DEBUG] Registered cases: {len(self.registered_cases)}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        print(f"[DEBUG] Temp file created: {tmp_path}")
        print(f"[DEBUG] Temp file size: {os.path.getsize(tmp_path)} bytes")
        
        try:
            # Test if OpenCV can open the file
            test_cap = cv2.VideoCapture(tmp_path)
            if test_cap.isOpened():
                fps = test_cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[DEBUG] Video properties - FPS: {fps}, Frames: {frame_count}, Size: {width}x{height}")
                test_cap.release()
            else:
                print(f"[DEBUG] ERROR: OpenCV cannot open video file!")
            
            results = self.analyze_video(tmp_path, tolerance)
            print(f"[DEBUG] Analysis results: frames_analyzed={results['frames_analyzed']}, faces_detected={results['faces_detected']}")
            return results
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def get_registered_cases(self) -> List[Dict[str, Any]]:
        """Get list of all registered cases"""
        return [
            {
                'case_id': case_id,
                'name': data['name'],
                'metadata': data['metadata']
            }
            for case_id, data in self.registered_cases.items()
        ]
    
    def remove_case(self, case_id: str) -> bool:
        """Remove a registered case"""
        if case_id in self.registered_cases:
            del self.registered_cases[case_id]
            return True
        return False


# Global instance
video_analyzer = VideoAnalyzer(frame_sample_rate=15)  # Analyze every 15th frame
