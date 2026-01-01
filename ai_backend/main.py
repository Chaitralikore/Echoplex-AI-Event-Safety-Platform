"""
AI Backend Server for CCTV Video Analysis
Enhanced with Person Re-ID Pipeline (v2.0)

Features:
- Person Re-ID (512-d feature vectors)
- Clothing color analysis with white balance
- Temporal Super-Vectors for reduced false positives
- Spatio-temporal filtering
- Fallback to face_recognition if dependencies unavailable
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import uvicorn
import os
import tempfile
import base64

# Try to use enhanced video analyzer, fall back to original
try:
    from video_analyzer_v2 import enhanced_video_analyzer as video_analyzer
    ENHANCED_MODE = True
    print("Using Enhanced Video Analyzer with Re-ID pipeline")
except ImportError as e:
    print(f"Enhanced analyzer not available ({e}), using original")
    from video_analyzer import video_analyzer
    ENHANCED_MODE = False

app = FastAPI(
    title="Echoplex Video Analysis API",
    description="AI-powered video analysis for missing person detection with Re-ID",
    version="2.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "video-analysis",
        "version": "2.0.0",
        "enhanced_mode": ENHANCED_MODE,
        "pipeline": "Re-ID + Temporal + Color" if ENHANCED_MODE else "face_recognition",
        "registered_cases": len(video_analyzer.registered_cases)
    }


@app.post("/api/register-case")
async def register_case(
    case_id: str = Form(...),
    name: str = Form(...),
    reference_photo: UploadFile = File(None),
    photo_base64: str = Form(None),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    upper_clothing: Optional[str] = Form(None),
    lower_clothing: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """
    Register a missing person case with their reference photo
    
    Accepts either a file upload (reference_photo) or base64 encoded image (photo_base64)
    """
    metadata = {
        'age': age,
        'gender': gender,
        'upper_clothing': upper_clothing,
        'lower_clothing': lower_clothing,
        'description': description
    }
    
    if reference_photo and reference_photo.filename:
        # Save uploaded photo temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            content = await reference_photo.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            success = video_analyzer.register_case(case_id, name, tmp_path, metadata)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    elif photo_base64:
        success = video_analyzer.register_case_from_base64(case_id, name, photo_base64, metadata)
    else:
        raise HTTPException(status_code=400, detail="Either reference_photo or photo_base64 is required")
    
    if success:
        return {"success": True, "message": f"Case registered for {name}", "case_id": case_id}
    else:
        raise HTTPException(status_code=400, detail="Could not detect face in reference photo")


@app.post("/api/analyze-video")
async def analyze_video(
    video: UploadFile = File(...),
    tolerance: float = Form(0.6)
):
    """
    Analyze an uploaded video to find matches with registered cases
    
    Args:
        video: The video file to analyze
        tolerance: Face matching tolerance (0.0-1.0, lower = stricter)
    
    Returns:
        Analysis results with any matches found
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Check if there are registered cases
    if len(video_analyzer.registered_cases) == 0:
        return {
            "success": True,
            "frames_analyzed": 0,
            "faces_detected": 0,
            "matches": [],
            "message": "No missing person cases registered. Please register cases first."
        }
    
    # Read video content
    video_content = await video.read()
    
    # Analyze the video
    results = video_analyzer.analyze_video_bytes(video_content, tolerance)
    
    if results.get('error'):
        return {
            "success": False,
            "error": results['error'],
            "frames_analyzed": results.get('frames_analyzed', 0),
            "matches": []
        }
    
    return {
        "success": True,
        "frames_analyzed": results['frames_analyzed'],
        "faces_detected": results['faces_detected'],
        "persons_detected": results['persons_detected'],
        "matches": results['matches'],
        "match_found": len(results['matches']) > 0
    }


@app.get("/api/cases")
async def list_cases():
    """List all registered missing person cases"""
    cases = video_analyzer.get_registered_cases()
    return {"cases": cases, "total": len(cases)}


@app.delete("/api/cases/{case_id}")
async def remove_case(case_id: str):
    """Remove a registered case"""
    success = video_analyzer.remove_case(case_id)
    if success:
        return {"success": True, "message": f"Case {case_id} removed"}
    else:
        raise HTTPException(status_code=404, detail="Case not found")


@app.post("/api/sync-cases")
async def sync_cases(request: Request):
    """
    Sync multiple cases from the frontend
    Used to register all active missing person cases at once
    This replaces all existing cases - only people still being searched for should be synced
    """
    try:
        cases = await request.json()
    except Exception as e:
        return {"success": False, "error": f"Invalid JSON: {str(e)}", "registered": 0, "failed": 0}
    
    if not isinstance(cases, list):
        return {"success": False, "error": "Expected a list of cases", "registered": 0, "failed": 0}
    
    # Clear all existing registered cases first
    # This ensures we only match against currently active cases
    previous_count = len(video_analyzer.registered_cases)
    video_analyzer.registered_cases.clear()
    print(f"[SYNC] Cleared {previous_count} existing cases, syncing {len(cases)} new cases")
    
    registered = 0
    failed = 0
    
    for case in cases:
        case_id = case.get('id')
        name = case.get('name')
        photo_base64 = case.get('photoUrl')
        
        if not case_id or not name or not photo_base64:
            print(f"Skipping case - missing required fields: id={case_id}, name={name}, hasPhoto={bool(photo_base64)}")
            failed += 1
            continue
        
        metadata = {
            'age': case.get('age'),
            'gender': case.get('gender'),
            'upper_clothing': case.get('upperClothingColor'),
            'lower_clothing': case.get('lowerClothingColor'),
            'description': case.get('description')
        }
        
        success = video_analyzer.register_case_from_base64(case_id, name, photo_base64, metadata)
        if success:
            registered += 1
            print(f"Registered case: {name}")
        else:
            failed += 1
            print(f"Failed to register case: {name}")
    
    return {
        "success": True,
        "registered": registered,
        "failed": failed,
        "total_cases": len(video_analyzer.registered_cases)
    }


# ============================================
# Live Camera Frame Scanning
# ============================================

@app.post("/api/scan")
async def scan_frame(file: UploadFile = File(...)):
    """
    Scan a single camera frame for person detection and face matching.
    Used for real-time live camera scanning.
    
    Returns:
        - person_count: Number of persons detected in frame
        - faces_detected: Number of faces found
        - matches: List of matched missing persons with confidence
    """
    import cv2
    import numpy as np
    from PIL import Image
    import io
    
    try:
        # Read the uploaded frame
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Could not decode image", "person_count": 0, "faces_detected": 0, "matches": []}
        
        person_count = 0
        faces_detected = 0
        matches = []
        
        if ENHANCED_MODE:
            # Use Re-ID pipeline for detection
            try:
                from person_detector import person_detector
                from hybrid_matcher import hybrid_matcher
                
                # Detect persons
                detections = person_detector.detect_persons(frame)
                person_count = len(detections)
                
                # Match each detected person against registered cases
                for x1, y1, x2, y2, det_conf in detections:
                    person_crop = person_detector.crop_person(frame, (x1, y1, x2, y2))
                    
                    if person_crop.size == 0:
                        continue
                    
                    # Extract features
                    features = hybrid_matcher.extract_person_features(person_crop)
                    
                    if features is None:
                        continue
                    
                    # Match against registered cases
                    for case_id, case_data in video_analyzer.registered_cases.items():
                        if 'reid_vector' not in case_data:
                            continue
                        
                        # Compute similarity
                        from reid_extractor import reid_extractor
                        similarity = reid_extractor.compute_similarity(
                            case_data['reid_vector'], 
                            features['reid_vector']
                        )
                        
                        confidence = int(similarity * 100)
                        
                        # Only report matches above 60% confidence
                        if confidence >= 60:
                            matches.append({
                                "fullName": case_data['name'],
                                "case_id": case_id,
                                "confidence": confidence,
                                "photoUrl": case_data.get('metadata', {}).get('photoUrl', ''),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "detected_colors": {
                                    "upper": features.get('primary_upper', 'unknown'),
                                    "lower": features.get('primary_lower', 'unknown')
                                }
                            })
                
                # Sort by confidence
                matches.sort(key=lambda m: m['confidence'], reverse=True)
                
            except Exception as e:
                print(f"Re-ID scan error: {e}")
                # Fall back to face recognition
        
        # Fallback to face_recognition if Re-ID didn't find matches
        if not matches and len(video_analyzer.registered_cases) > 0:
            try:
                import face_recognition
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                faces_detected = len(face_locations)
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for encoding, location in zip(face_encodings, face_locations):
                        for case_id, case_data in video_analyzer.registered_cases.items():
                            if 'encoding' in case_data:
                                distance = face_recognition.face_distance([case_data['encoding']], encoding)[0]
                                similarity = 1 - distance
                                confidence = int(similarity * 100)
                                
                                if confidence >= 55:
                                    matches.append({
                                        "fullName": case_data['name'],
                                        "case_id": case_id,
                                        "confidence": confidence,
                                        "photoUrl": "",
                                        "bbox": list(location)
                                    })
                    
                    matches.sort(key=lambda m: m['confidence'], reverse=True)
                    
            except Exception as e:
                print(f"Face recognition scan error: {e}")
        
        # Estimate person count from face detection if Re-ID unavailable
        if person_count == 0:
            person_count = faces_detected
        
        return {
            "success": True,
            "person_count": person_count,
            "faces_detected": faces_detected,
            "matches": matches[:5],  # Return top 5 matches
            "registered_cases": len(video_analyzer.registered_cases)
        }
        
    except Exception as e:
        print(f"Scan error: {e}")
        return {
            "error": str(e),
            "person_count": 0,
            "faces_detected": 0,
            "matches": []
        }


# ============================================
# WebSocket Real-Time Notifications
# ============================================

# Import WebSocket components
try:
    from match_notifications import websocket_manager, match_trigger, MatchNotification
    from fastapi import WebSocket, WebSocketDisconnect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("WebSocket notifications not available")


@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """
    WebSocket endpoint for real-time match notifications
    
    Connect to receive instant alerts when high-confidence matches are found.
    """
    if not WEBSOCKET_AVAILABLE:
        await websocket.close(code=1013, reason="WebSocket not available")
        return
    
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
            
            # Handle threshold update
            elif data.startswith("threshold:"):
                try:
                    new_threshold = float(data.split(":")[1])
                    match_trigger.update_threshold(new_threshold)
                    await websocket.send_json({
                        "type": "THRESHOLD_UPDATED",
                        "new_threshold": new_threshold
                    })
                except ValueError:
                    pass
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# ============================================
# Search by Metadata API
# ============================================

@app.post("/api/search-by-metadata")
async def search_by_metadata(request: Request):
    """
    Search CCTV detections by metadata filters
    
    Allows filtering by clothing color, gender, height, etc.
    Uses Vector Database with metadata filtering for efficient search.
    
    Request body:
    {
        "query_vector": [optional 512-d array],
        "filters": {
            "upper_color": "red",
            "lower_color": ["blue", "black"],
            "gender": "female",
            "location": "main_entrance"
        },
        "top_k": 10,
        "min_confidence": 0.5
    }
    """
    try:
        body = await request.json()
    except:
        return {"success": False, "error": "Invalid JSON", "results": []}
    
    filters = body.get("filters", {})
    top_k = body.get("top_k", 10)
    min_confidence = body.get("min_confidence", 0.5)
    
    # Try to use vector database
    try:
        from vector_database import vector_db
        
        # If query vector provided, do similarity search
        query_vector = body.get("query_vector")
        if query_vector:
            import numpy as np
            query_vector = np.array(query_vector, dtype=np.float32)
            results = vector_db.search(
                query_vector=query_vector,
                top_k=top_k,
                metadata_filter=filters,
                min_similarity=min_confidence
            )
        else:
            # Just filter by metadata (return all matching)
            results = vector_db.search(
                query_vector=np.zeros(512, dtype=np.float32),
                top_k=top_k * 10,  # Get more since we're filtering
                metadata_filter=filters,
                min_similarity=0.0
            )
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "filters_applied": filters
        }
    
    except ImportError:
        # Fallback: search in registered cases
        results = []
        for case_id, case_data in video_analyzer.registered_cases.items():
            metadata = case_data.get('metadata', {})
            
            # Check filters
            match = True
            for key, value in filters.items():
                case_value = metadata.get(key, '')
                if isinstance(value, list):
                    if case_value and case_value.lower() not in [v.lower() for v in value]:
                        match = False
                else:
                    if case_value and value.lower() not in case_value.lower():
                        match = False
            
            if match:
                results.append({
                    "case_id": case_id,
                    "name": case_data.get("name"),
                    "metadata": metadata
                })
        
        return {
            "success": True,
            "results": results[:top_k],
            "count": len(results),
            "filters_applied": filters,
            "note": "Vector database not available, searched registered cases"
        }


# ============================================
# Feedback Loop (Confirm/False Match)
# ============================================

@app.post("/api/match-feedback")
async def match_feedback(request: Request):
    """
    Record user feedback on match results
    
    Used to improve the AI model:
    - "confirm": Match was correct (positive feedback)
    - "false_match": Match was incorrect (negative feedback)
    
    Request body:
    {
        "notification_id": "match_123_case_456",
        "case_id": "case_456",
        "feedback": "confirm" | "false_match",
        "notes": "optional notes"
    }
    """
    try:
        body = await request.json()
    except:
        return {"success": False, "error": "Invalid JSON"}
    
    notification_id = body.get("notification_id")
    case_id = body.get("case_id")
    feedback = body.get("feedback")
    notes = body.get("notes", "")
    
    if feedback not in ["confirm", "false_match"]:
        return {"success": False, "error": "Feedback must be 'confirm' or 'false_match'"}
    
    # Log feedback (in production, save to database)
    print(f"[FEEDBACK] {feedback.upper()} for case {case_id}: {notes}")
    
    # Handle false match - blacklist the vector to prevent same error
    if feedback == "false_match":
        try:
            from precision_guardrails import precision_guardrails
            import numpy as np
            
            # Get the vector that was falsely matched (from video analyzer)
            # For now, create a placeholder - in production, this would come from the request
            # The frontend should send the vector data with the feedback
            if ENHANCED_MODE and case_id in video_analyzer.registered_cases:
                case_data = video_analyzer.registered_cases[case_id]
                # Blacklist the reference vector pattern (reduces similar matches)
                if 'reid_vector' in case_data:
                    # Add noise to create a blacklist entry for this type of false positive
                    noisy_vector = case_data['reid_vector'] + np.random.randn(512).astype(np.float32) * 0.05
                    noisy_vector = noisy_vector / np.linalg.norm(noisy_vector)
                    precision_guardrails.add_to_blacklist(
                        case_id=case_id,
                        vector=noisy_vector,
                        reason=f"false_match: {notes}" if notes else "false_match"
                    )
                    print(f"[BLACKLIST] Added false positive pattern for case {case_id}")
        except ImportError as e:
            print(f"[FEEDBACK] Precision guardrails not available: {e}")
        except Exception as e:
            print(f"[FEEDBACK] Error blacklisting: {e}")
    
    # Handle confirm - could boost case vector weight (future enhancement)
    elif feedback == "confirm":
        print(f"[FEEDBACK] Confirmed match for {case_id} - positive feedback recorded")
    
    # Broadcast feedback to connected clients
    if WEBSOCKET_AVAILABLE:
        import asyncio
        asyncio.create_task(websocket_manager.broadcast({
            "type": "FEEDBACK_RECORDED",
            "notification_id": notification_id,
            "case_id": case_id,
            "feedback": feedback
        }))
    
    return {
        "success": True,
        "message": f"Feedback '{feedback}' recorded for case {case_id}",
        "notification_id": notification_id
    }


@app.get("/api/ws-status")
async def websocket_status():
    """Get WebSocket connection status"""
    if WEBSOCKET_AVAILABLE:
        return {
            "available": True,
            "connected_clients": websocket_manager.connection_count,
            "notification_threshold": match_trigger.confidence_threshold
        }
    return {"available": False, "connected_clients": 0}


if __name__ == "__main__":
    print("Starting Echoplex Video Analysis Server v2.0...")
    print("Features: Re-ID Pipeline, WebSocket Notifications, Metadata Search")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    uvicorn.run(app, host="0.0.0.0", port=8002)
