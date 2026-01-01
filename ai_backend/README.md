# Echoplex AI Video Analysis Backend

This Python backend provides real face recognition and video analysis for the Lost & Found feature.

## Prerequisites

1. **Python 3.8+** installed
2. **CMake** and **C++ compiler** (required for dlib/face_recognition)
   - Windows: Install Visual Studio Build Tools with C++ workload
   - Mac: `xcode-select --install`
   - Linux: `sudo apt-get install build-essential cmake`

## Quick Setup

### Windows (PowerShell)

```powershell
cd ai_backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

### Mac/Linux

```bash
cd ai_backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/register-case` | POST | Register a missing person with photo |
| `/api/analyze-video` | POST | Analyze video for face matches |
| `/api/sync-cases` | POST | Sync multiple cases from frontend |
| `/api/cases` | GET | List all registered cases |
| `/api/cases/{id}` | DELETE | Remove a case |

## How It Works

1. **Case Registration**: When a missing person is reported with a reference photo, the photo is sent to this backend which extracts and encodes the face.

2. **Video Analysis**: When a CCTV video is uploaded:
   - Extracts frames at regular intervals
   - Detects faces in each frame using HOG detector
   - Compares detected faces against registered faces using deep learning embeddings
   - Returns matches with confidence scores

3. **Face Matching**: Uses the `face_recognition` library which is built on dlib's state-of-the-art face recognition model (99.38% accuracy on LFW dataset).

## Troubleshooting

### "face_recognition" installation fails
This usually means dlib can't compile. Ensure you have:
- CMake installed: `pip install cmake`
- C++ compiler available

### Server won't start
Make sure port 8002 is not in use:
```bash
netstat -ano | findstr :8002
```

### No faces detected
- Ensure the video has clear, front-facing shots of faces
- The reference photo should have a clear, well-lit face
- Try adjusting the `tolerance` parameter (lower = stricter matching)

## Running with Frontend

1. Start this backend: `python main.py` (runs on port 8002)
2. Start the main app: `npm run dev:full` (frontend + express server)
3. Go to Lost & Found, add a missing person with photo
4. Upload a video to analyze
