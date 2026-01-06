from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from typing import List, Optional
import numpy as np

from ultralytics import YOLO
from PIL import Image

import os
import uuid
import cv2
import json

# Try to import DeepFace, but make it optional
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace loaded successfully - face recognition enabled")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print(f"DeepFace not available - face recognition disabled: {e}")

import firebase_admin
from firebase_admin import credentials, db as realtime_db


# ---------------------------
# Firebase Admin init (using Realtime Database)
# ---------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://echoplex-final-default-rtdb.asia-southeast1.firebasedatabase.app'
})


# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ---------------------------
# Load YOLO model
# ---------------------------
yolo_model = YOLO("yolov8n.pt")


# ---------------------------
# Existing YOLO detection endpoint
# ---------------------------
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    results = yolo_model(image)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": results.names[cls_id],
            }
        )

    return {"detections": detections}


# ---------------------------
# Missing person case creation (using Realtime Database)
# ---------------------------
@app.post("/cases")
async def create_case(
    fullName: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    topColor: str = Form(...),
    bottomColor: str = Form(...),
    description: str = Form(""),
    lastSeenLocation: str = Form(""),
    reportedBy: str = Form(""),
    referencePhoto: UploadFile = File(...)
):
    # 1) Save image to uploads/
    file_ext = referencePhoto.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join("uploads", filename)
    with open(file_path, "wb") as f:
        f.write(await referencePhoto.read())

    # 2) Compute face embedding (if DeepFace is available)
    embedding = None
    if DEEPFACE_AVAILABLE:
        try:
            embedding_obj = DeepFace.represent(img_path=file_path, model_name="ArcFace")[0]
            embedding = embedding_obj["embedding"]
            print(f"Face embedding extracted successfully for {fullName}")
        except Exception as e:
            print(f"Face embedding failed: {e}")

    # 3) Save case to Realtime Database
    case_id = str(uuid.uuid4())
    photo_url = f"http://127.0.0.1:8001/uploads/{filename}"
    case_data = {
        "fullName": fullName,
        "age": age,
        "gender": gender,
        "topColor": topColor,
        "bottomColor": bottomColor,
        "description": description,
        "lastSeenLocation": lastSeenLocation,
        "reportedBy": reportedBy,
        "photoUrl": photo_url,
        "active": True,
    }
    if embedding:
        case_data["embedding"] = embedding
    
    # Save to Realtime Database under 'faceRecognitionCases'
    ref = realtime_db.reference(f'faceRecognitionCases/{case_id}')
    ref.set(case_data)
    print(f"Case saved to Realtime Database: {case_id}")

    return {"caseId": case_id, "status": "ok", "photoUrl": photo_url}


# ---------------------------
# Face matching endpoint - compare camera frame against stored cases
# ---------------------------
@app.post("/api/match-face")
async def match_face(file: UploadFile = File(...)):
    """
    Takes a camera frame, extracts faces, and compares against all stored case embeddings.
    Returns list of potential matches with confidence scores.
    """
    if not DEEPFACE_AVAILABLE:
        return {"matches": [], "faces_detected": 0, "message": "Face recognition not available"}
    
    try:
        # 1) Save temp image
        contents = await file.read()
        temp_path = f"uploads/temp_{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        matches = []
        frame_embeddings = []
        
        try:
            # 2) Extract face embeddings from camera frame
            frame_embeddings = DeepFace.represent(
                img_path=temp_path, 
                model_name="ArcFace",
                enforce_detection=False
            )
            
            if not frame_embeddings:
                os.remove(temp_path)
                return {"matches": [], "faces_detected": 0, "message": "No faces detected in frame"}

            # 3) Get all active cases from Realtime Database
            ref = realtime_db.reference('faceRecognitionCases')
            cases_data = ref.get() or {}
            
            # 4) Compare each detected face against stored embeddings
            for frame_emb in frame_embeddings:
                frame_vector = np.array(frame_emb["embedding"])
                
                for case_id, case_data in cases_data.items():
                    if not case_data.get("active", False):
                        continue
                    if "embedding" not in case_data:
                        continue
                    
                    case_vector = np.array(case_data["embedding"])
                    
                    # Compute cosine similarity
                    similarity = np.dot(frame_vector, case_vector) / (
                        np.linalg.norm(frame_vector) * np.linalg.norm(case_vector)
                    )
                    
                    # Convert to percentage
                    confidence = float(similarity) * 100
                    
                    if confidence > 40:  # Threshold for potential match
                        matches.append({
                            "caseId": case_id,
                            "fullName": case_data.get("fullName", "Unknown"),
                            "confidence": round(confidence, 2),
                            "photoUrl": case_data.get("photoUrl", ""),
                            "age": case_data.get("age", ""),
                            "description": case_data.get("description", "")
                        })
            
            # Sort by confidence
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            print(f"Face detection error: {e}")
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "matches": matches[:5],
            "faces_detected": len(frame_embeddings),
            "message": "Scan complete"
        }
        
    except Exception as e:
        return {"matches": [], "faces_detected": 0, "error": str(e)}


# ---------------------------
# Combined detection + face matching (for efficiency)
# ---------------------------
def _scan_image_bytes(contents: bytes):
    """
    Internal helper that runs YOLO + DeepFace matching on a single image.
    Returns a dict with detections, person_count, faces_detected and matches.
    """
    # 1) YOLO detection
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = yolo_model(image)[0]
    
    detections = []
    person_count = 0
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        
        if class_name == "person":
            person_count += 1
            
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf,
            "class_id": cls_id,
            "class_name": class_name,
        })
    
    # 2) Face matching (only if DeepFace is available)
    matches = []
    faces_detected = 0
    
    if person_count > 0 and DEEPFACE_AVAILABLE:
        try:
            # Save temp image for DeepFace
            temp_path = f"uploads/temp_{uuid.uuid4()}.jpg"
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            try:
                frame_embeddings = DeepFace.represent(
                    img_path=temp_path,
                    model_name="ArcFace", 
                    enforce_detection=False
                )
                faces_detected = len(frame_embeddings)
                
                if frame_embeddings:
                    # Get active cases from Realtime Database
                    ref = realtime_db.reference('faceRecognitionCases')
                    cases_data = ref.get() or {}
                    
                    for frame_emb in frame_embeddings:
                        frame_vector = np.array(frame_emb["embedding"])
                        
                        for case_id, case_data in cases_data.items():
                            if not isinstance(case_data, dict):
                                continue
                            if not case_data.get("active", False):
                                continue
                            if "embedding" not in case_data:
                                continue
                            
                            case_vector = np.array(case_data["embedding"])
                            similarity = np.dot(frame_vector, case_vector) / (
                                np.linalg.norm(frame_vector) * np.linalg.norm(case_vector)
                            )
                            confidence = float(similarity) * 100
                            
                            if confidence > 40:
                                matches.append({
                                    "caseId": case_id,
                                    "fullName": case_data.get("fullName", "Unknown"),
                                    "confidence": round(confidence, 2),
                                    "photoUrl": case_data.get("photoUrl", ""),
                                })
                
            except Exception as e:
                print(f"Face matching error: {e}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Scan error: {e}")
    
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "detections": detections,
        "person_count": person_count,
        "faces_detected": faces_detected,
        "matches": matches[:5],
    }


@app.post("/api/scan")
async def scan_frame(file: UploadFile = File(...)):
    """
    Combined endpoint: YOLO person detection + face matching for a single image.
    """
    contents = await file.read()
    return _scan_image_bytes(contents)


# ---------------------------
# Video scanning endpoint
# ---------------------------
@app.post("/api/scan-video")
async def scan_video(file: UploadFile = File(...)):
    """
    Analyze an uploaded video, sampling frames and performing the same
    person + face matching used for single-frame scans.
    Returns aggregated matches across the whole video.
    """
    if not DEEPFACE_AVAILABLE:
        return {"matches": [], "frames_analyzed": 0, "message": "Face recognition not available"}
    
    # Save video to temporary file
    video_ext = file.filename.split(".")[-1] if file.filename else "mp4"
    temp_video_path = os.path.join("uploads", f"temp_video_{uuid.uuid4()}.{video_ext}")
    with open(temp_video_path, "wb") as f:
        f.write(await file.read())
    
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"matches": [], "frames_analyzed": 0, "message": "Could not open video file"}
    
    frame_index = 0
    frames_analyzed = 0
    aggregated_matches: Dict[str, Dict[str, Any]] = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            
            # Sample every 15th frame to keep processing time reasonable on long videos
            if frame_index % 15 != 0:
                continue
            
            frames_analyzed += 1
            
            # Convert BGR (OpenCV) to RGB and then to bytes for reuse of _scan_image_bytes
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = rgb_frame.shape[:2] if rgb_frame is not None else (0, 1)
            pil_image = Image.fromarray(rgb_frame)
            buf = BytesIO()
            pil_image.save(buf, format="JPEG")
            frame_bytes = buf.getvalue()
            
            result = _scan_image_bytes(frame_bytes)

            # Derive a coarse \"side of frame\" hint (left/center/right) from detected persons
            frame_position = None
            detections = result.get("detections", [])
            person_centers = []
            for det in detections:
                if det.get("class_name") == "person":
                    cx = 0.5 * (det.get("x1", 0.0) + det.get("x2", 0.0))
                    person_centers.append(cx)
            if person_centers and frame_width:
                avg_cx = sum(person_centers) / len(person_centers)
                rel = avg_cx / float(frame_width)
                if rel < 0.33:
                    frame_position = "left"
                elif rel < 0.66:
                    frame_position = "center"
                else:
                    frame_position = "right"
            
            for m in result.get("matches", []):
                case_id = m["caseId"]
                existing = aggregated_matches.get(case_id, {
                    "caseId": case_id,
                    "fullName": m.get("fullName", "Unknown"),
                    "photoUrl": m.get("photoUrl", ""),
                    "bestConfidence": 0.0,
                    "hits": 0,
                    "positionCounts": {"left": 0, "center": 0, "right": 0},
                    "position": None,
                })
                existing["hits"] += 1
                if m["confidence"] > existing["bestConfidence"]:
                    existing["bestConfidence"] = m["confidence"]
                # Track which side of the frame this match most often appears on
                if frame_position:
                    counts = existing.get("positionCounts", {"left": 0, "center": 0, "right": 0})
                    counts[frame_position] = counts.get(frame_position, 0) + 1
                    existing["positionCounts"] = counts
                    # Update dominant position
                    existing["position"] = max(counts, key=counts.get)
                aggregated_matches[case_id] = existing
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    # Convert aggregated dict to sorted list
    matches_list = list(aggregated_matches.values())
    matches_list.sort(key=lambda x: x["bestConfidence"], reverse=True)
    
    return {
        "frames_analyzed": frames_analyzed,
        "matches": matches_list,
    }
