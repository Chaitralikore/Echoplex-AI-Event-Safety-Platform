import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import {
  Search,
  Camera,
  MapPin,
  Clock,
  User,
  AlertCircle,
  CheckCircle,
  Eye,
  Zap,
  Trash2,
  StopCircle,
  Video,
  X,
  Upload,
  Play
} from 'lucide-react';
import { db } from '../firebase';
import { ref, onValue, push, set, update, runTransaction, remove } from 'firebase/database';

interface CameraMatch {
  cameraId: string;
  location: string;
  confidence: number;
  timestamp: number;
  imageUrl?: string;
}

interface MissingPerson {
  id: string;
  name: string;
  age: number;
  description: string;
  lastSeen: string;
  reportedTime: number;
  status: 'searching' | 'found' | 'potential-match';
  reportedBy: string;
  photoUrl?: string;
  gender?: 'male' | 'female' | 'other';
  heightRange?: 'short' | 'medium' | 'tall';
  upperClothingColor?: string;
  lowerClothingColor?: string;
  aiMatchConfidence?: number;
  currentLocation?: string;
  cameraMatches?: CameraMatch[];
}

const videoConstraints = {
  facingMode: 'user'
};

const LostAndFound: React.FC = () => {
  const [missingPersons, setMissingPersons] = useState<MissingPerson[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState<string>('');
  const [detectedPersons, setDetectedPersons] = useState<number>(0);
  const [matchResult, setMatchResult] = useState<{
    name: string;
    confidence: number;
    photoUrl?: string;
  } | null>(null);
  const webcamRef = useRef<Webcam>(null);
  const scanIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [newReport, setNewReport] = useState({
    name: '',
    age: '',
    description: '',
    lastSeen: '',
    reportedBy: '',
    gender: '',
    heightRange: '',
    upperClothingColor: '',
    lowerClothingColor: '',
    photoFile: null as File | null,
    photoUrl: ''
  });
  const [aiScanResults, setAiScanResults] = useState({
    totalScans: 0,
    facesDetected: 0,
    matchAttempts: 0,
    successRate: 0
  });

  // Video upload state
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string>('');
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const [videoAnalysisResult, setVideoAnalysisResult] = useState<{
    likelyLocation: string;
    matchedPerson: string | null;
    matchConfidence: number;
    detectedDetails: {
      gender: string;
      estimatedAge: string;
      upperClothing: string;
      lowerClothing: string;
    };
    relevantInfo: string[];
    framesAnalyzed: number;
    personsDetected: number;
    // NEW: Confidence breakdown from hybrid matcher
    confidenceBreakdown?: {
      overall_match: number;
      reid_features: number;
      upper_clothing: number;
      lower_clothing: number;
      body_shape: number;
      super_vector_boost: string;
    };
    superVectorUsed?: boolean;
    trackInfo?: {
      trackId: string;
      trackLength: number;
    };
  } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // WebSocket State for Real-Time Notifications
  const [wsConnected, setWsConnected] = useState(false);
  const [realtimeNotifications, setRealtimeNotifications] = useState<{
    type: string;
    notification_id: string;
    case_id: string;
    case_name: string;
    confidence: number;
    confidence_breakdown: Record<string, number | string>;
    location: string;
    created_at: string;
  }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Connect to WebSocket for real-time notifications
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://127.0.0.1:8002/ws/notifications');

        ws.onopen = () => {
          console.log('[WS] Connected to notification server');
          setWsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('[WS] Received:', data);

            if (data.type === 'POTENTIAL_MATCH') {
              // Show real-time match notification
              setRealtimeNotifications(prev => [data, ...prev].slice(0, 10));

              // Also update the video analysis result if we're analyzing
              if (isAnalyzingVideo || videoAnalysisResult) {
                setVideoAnalysisResult(prev => prev ? {
                  ...prev,
                  matchedPerson: data.case_name,
                  matchConfidence: data.confidence,
                  confidenceBreakdown: data.confidence_breakdown,
                  superVectorUsed: data.super_vector_used
                } : null);
              }
            }
          } catch (e) {
            console.error('[WS] Parse error:', e);
          }
        };

        ws.onclose = () => {
          console.log('[WS] Disconnected');
          setWsConnected(false);
          // Attempt to reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };

        ws.onerror = (error) => {
          console.error('[WS] Error:', error);
        };

        wsRef.current = ws;
      } catch (e) {
        console.log('[WS] Connection failed, will retry');
        setTimeout(connectWebSocket, 5000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Handle video file upload
  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedVideo(file);
      setVideoPreviewUrl(URL.createObjectURL(file));
      setVideoAnalysisResult(null);
    }
  };

  // Clear uploaded video
  const clearUploadedVideo = () => {
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setUploadedVideo(null);
    setVideoPreviewUrl('');
    setVideoAnalysisResult(null);
  };

  // Helper function to format video timestamp (seconds to MM:SS)
  const formatVideoTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Analyze uploaded video using real AI backend
  const analyzeVideo = async () => {
    if (!uploadedVideo) return;

    setIsAnalyzingVideo(true);

    const AI_BACKEND_URL = 'http://127.0.0.1:8002';
    const locations = ['Main Entrance', 'Food Court', 'VIP Section', 'General Area', 'Exit Gate B'];

    try {
      // First, sync all active cases with photos to the AI backend
      const activeCases = missingPersons.filter(p => p.status === 'searching' && p.photoUrl);

      if (activeCases.length > 0) {
        // Sync cases with the AI backend
        const syncData = activeCases.map(p => ({
          id: p.id,
          name: p.name,
          photoUrl: p.photoUrl,
          age: p.age,
          gender: p.gender,
          upperClothingColor: p.upperClothingColor,
          lowerClothingColor: p.lowerClothingColor,
          description: p.description
        }));

        try {
          await fetch(`${AI_BACKEND_URL}/api/sync-cases`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(syncData)
          });
        } catch (err) {
          console.warn('Could not sync cases with AI backend:', err);
        }
      }

      // Send video to AI backend for analysis
      const formData = new FormData();
      formData.append('video', uploadedVideo);
      formData.append('tolerance', '0.8');

      const response = await fetch(`${AI_BACKEND_URL}/api/analyze-video`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Video analysis failed');
      }

      const data = await response.json();

      let matchedPerson = null;
      let matchedCase: MissingPerson | null = null;
      let matchConfidence = 0;

      // Check if we found any matches
      if (data.matches && data.matches.length > 0) {
        const topMatch = data.matches[0];
        matchedPerson = topMatch.name;
        matchConfidence = topMatch.confidence;

        // Find the matching case in our local state
        matchedCase = missingPersons.find(p => p.id === topMatch.case_id) ||
          missingPersons.find(p => p.name.toLowerCase() === topMatch.name.toLowerCase()) || null;
      }

      const result = {
        likelyLocation: matchedCase?.lastSeen || locations[Math.floor(Math.random() * locations.length)],
        matchedPerson,
        matchConfidence,
        detectedDetails: {
          gender: matchedCase?.gender
            ? matchedCase.gender.charAt(0).toUpperCase() + matchedCase.gender.slice(1)
            : (data.faces_detected > 0 ? 'Detected' : 'Unknown'),
          estimatedAge: matchedCase
            ? `${matchedCase.age} years`
            : 'Unknown',
          upperClothing: matchedCase?.upperClothingColor
            ? matchedCase.upperClothingColor + (matchedCase.upperClothingColor.includes('top') ? '' : ' top')
            : 'Unknown',
          lowerClothing: matchedCase?.lowerClothingColor
            ? matchedCase.lowerClothingColor + (matchedCase.lowerClothingColor.includes('bottom') || matchedCase.lowerClothingColor.includes('pants') ? '' : ' bottom')
            : 'Unknown',
        },
        relevantInfo: [
          `Frames analyzed: ${data.frames_analyzed}`,
          `Persons detected: ${data.persons_detected || data.faces_detected || 0}`,
          data.matches && data.matches.length > 0
            ? `Seen in video: ${formatVideoTime(data.matches[0].first_seen)} - ${formatVideoTime(data.matches[0].last_seen)}`
            : `Analyzed at: ${new Date().toLocaleTimeString()}`,
          matchedPerson
            ? `✅ Face matched with ${matchedPerson}`
            : ((data.persons_detected || data.faces_detected || 0) > 0 ? '❌ No match with registered cases' : '⚠️ No persons detected in video'),
        ],
        framesAnalyzed: data.frames_analyzed || 0,
        personsDetected: data.persons_detected || data.faces_detected || 0,
      };

      setVideoAnalysisResult(result);

      // Auto-update status to potential-match if a match was found
      if (matchedCase && matchedPerson) {
        handleStatusUpdate(matchedCase.id, 'potential-match');
      }

    } catch (err) {
      console.error('AI backend error:', err);

      // Show error result
      setVideoAnalysisResult({
        likelyLocation: 'Unknown',
        matchedPerson: null,
        matchConfidence: 0,
        detectedDetails: {
          gender: 'Unknown',
          estimatedAge: 'Unknown',
          upperClothing: 'Unknown',
          lowerClothing: 'Unknown',
        },
        relevantInfo: [
          '⚠️ AI Backend not available',
          'Make sure the Python AI server is running:',
          'cd ai_backend && pip install -r requirements.txt && python main.py',
          'Server should be running on http://127.0.0.1:8002'
        ],
        framesAnalyzed: 0,
        personsDetected: 0,
      });
    }

    setIsAnalyzingVideo(false);
  };

  // Convert uploaded photo to base64 data URL for storage
  const uploadPhoto = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result as string);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // call YOLO backend with uploaded photo (optional - gracefully fails if server unavailable)
  const runYoloOnPhoto = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://127.0.0.1:8001/api/detect', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        console.error('YOLO API error', await res.text());
        return [];
      }

      const data = await res.json();
      console.log('YOLO raw response:', data);
      return data.detections as Array<{
        x1: number;
        y1: number;
        x2: number;
        y2: number;
        confidence: number;
        class_id: number;
        class_name: string;
      }>;
    } catch (err) {
      console.warn('YOLO server not available, skipping AI detection:', err);
      return [];
    }
  };

  // Convert base64 to blob for sending to API
  const base64ToBlob = (base64: string): Blob => {
    const byteString = atob(base64.split(',')[1]);
    const mimeString = base64.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
  };

  // Scan frame from webcam - detect persons AND match faces
  const scanFrame = useCallback(async () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const blob = base64ToBlob(imageSrc);
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      // Use the combined scan endpoint for person detection + face matching
      // AI backend runs on port 8002
      const res = await fetch('http://127.0.0.1:8002/api/scan', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        setScanStatus('Detection server error');
        return;
      }

      const data = await res.json();
      const personCount = data.person_count || 0;
      const facesDetected = data.faces_detected || 0;
      const matches = data.matches || [];

      setDetectedPersons(personCount);

      // Update stats in Firebase
      const scansRef = ref(db, 'stats/aiFaceScans');
      await runTransaction(scansRef, (current) => (current || 0) + 1);

      if (facesDetected > 0) {
        const facesRef = ref(db, 'stats/facesDetected');
        await runTransaction(facesRef, (current) => (current || 0) + facesDetected);
      }

      // Check for face matches
      if (matches.length > 0) {
        const topMatch = matches[0];
        setScanStatus(`🎯 MATCH FOUND: ${topMatch.fullName} (${topMatch.confidence}% confidence)`);

        // Store match result for display
        setMatchResult({
          name: topMatch.fullName,
          confidence: topMatch.confidence,
          photoUrl: topMatch.photoUrl
        });

        // Update the matched person's status in Firebase
        const matchesRef = ref(db, 'stats/matchesConfirmed');
        await runTransaction(matchesRef, (current) => (current || 0) + 1);

        // Find and update the matching person in missingPersons
        const matchingPerson = missingPersons.find(p =>
          p.name.toLowerCase() === topMatch.fullName.toLowerCase()
        );
        if (matchingPerson && matchingPerson.status === 'searching') {
          const personRef = ref(db, `missingPersons/${matchingPerson.id}`);
          await update(personRef, {
            status: 'potential-match',
            aiMatchConfidence: topMatch.confidence / 100,
            currentLocation: 'Camera Feed'
          });
        }
      } else if (personCount > 0) {
        setScanStatus(`Scanning... ${personCount} person(s), ${facesDetected} face(s) detected`);
      } else {
        setScanStatus('Scanning... No persons in frame');
      }
    } catch (err) {
      console.warn('Scan error:', err);
      setScanStatus('Detection server not available. Start the backend with: python main.py');
    }
  }, [missingPersons]);

  // Start continuous scanning
  const startScanning = useCallback(async () => {
    if (isScanning) return;

    setIsScanning(true);
    setScanStatus('Syncing cases with AI backend...');

    // AUTO-SYNC: Register all searching cases with AI backend before scanning
    try {
      const searchingCases = missingPersons.filter(p => p.status === 'searching');
      if (searchingCases.length > 0) {
        const syncData = searchingCases.map(p => ({
          id: p.id,
          name: p.name,
          photoUrl: p.photoUrl,
          age: p.age,
          gender: p.gender,
          upperClothingColor: p.upperClothingColor,
          lowerClothingColor: p.lowerClothingColor,
          description: p.description
        }));

        const syncRes = await fetch('http://127.0.0.1:8002/api/sync-cases', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(syncData)
        });

        if (syncRes.ok) {
          const syncResult = await syncRes.json();
          console.log(`Synced ${syncResult.registered} cases for live scanning`);
          setScanStatus(`Ready! ${syncResult.registered} case(s) registered. Starting scan...`);
        }
      } else {
        setScanStatus('No active cases to scan for. Add a missing person first.');
        setIsScanning(false);
        return;
      }
    } catch (err) {
      console.warn('Could not sync cases with AI backend:', err);
      setScanStatus('AI backend not available. Start with: python main.py');
      setIsScanning(false);
      return;
    }

    // Give backend 500ms to process registrations
    await new Promise(resolve => setTimeout(resolve, 500));

    setScanStatus('Scanning...');

    // Scan every 2 seconds
    scanIntervalRef.current = setInterval(() => {
      scanFrame();
    }, 2000);

    // Run first scan immediately
    scanFrame();
  }, [isScanning, scanFrame, missingPersons]);

  // Stop scanning
  const stopScanning = useCallback(() => {
    setIsScanning(false);
    setScanStatus('');
    setDetectedPersons(0);
    setMatchResult(null);

    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const missingRef = ref(db, 'missingPersons');
    const unsubscribe = onValue(missingRef, (snapshot) => {
      if (snapshot.exists()) {
        const data = snapshot.val();
        const list: MissingPerson[] = Object.entries(data).map(
          ([id, value]: [string, any]) => ({
            id,
            ...(value as any)
          })
        );
        setMissingPersons(list);
      } else {
        setMissingPersons([]);
      }
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    const statsRef = ref(db, 'stats');
    const unsubscribe = onValue(statsRef, (snapshot) => {
      if (snapshot.exists()) {
        const s = snapshot.val();
        const totalScans = s.aiFaceScans || 0;
        const facesDetected = s.facesDetected || 0;
        const matchesConfirmed = s.matchesConfirmed || 0;
        // Success rate: matches / total scans (works with Re-ID body detection)
        const successRate =
          totalScans === 0 ? 0 : matchesConfirmed / totalScans;

        setAiScanResults({
          totalScans,
          facesDetected,
          matchAttempts: matchesConfirmed,
          successRate
        });
      } else {
        setAiScanResults({
          totalScans: 0,
          facesDetected: 0,
          matchAttempts: 0,
          successRate: 0
        });
      }
    });
    return () => unsubscribe();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'found':
        return 'bg-green-900/20 text-green-400';
      case 'potential-match':
        return 'bg-yellow-900/20 text-yellow-400';
      default:
        return 'bg-red-900/20 text-red-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'found':
        return CheckCircle;
      case 'potential-match':
        return AlertCircle;
      default:
        return Search;
    }
  };

  const handleSubmitReport = async () => {
    if (!newReport.name || !newReport.age || !newReport.description) return;

    try {
      let photoUrl: string | undefined;

      // If photo is provided, register with backend for face recognition
      if (newReport.photoFile) {
        const formData = new FormData();
        formData.append('fullName', newReport.name);
        formData.append('age', newReport.age);
        formData.append('gender', newReport.gender || 'unknown');
        formData.append('topColor', newReport.upperClothingColor || 'unknown');
        formData.append('bottomColor', newReport.lowerClothingColor || 'unknown');
        formData.append('description', newReport.description);
        formData.append('lastSeenLocation', newReport.lastSeen || 'Unknown');
        formData.append('reportedBy', newReport.reportedBy || 'Anonymous');
        formData.append('referencePhoto', newReport.photoFile);

        try {
          const res = await fetch('http://127.0.0.1:8001/cases', {
            method: 'POST',
            body: formData
          });

          if (res.ok) {
            const data = await res.json();
            console.log('Case registered with backend for face matching:', data);
            photoUrl = `http://127.0.0.1:8001/uploads/${data.caseId}`;
          } else {
            console.warn('Backend registration failed, using base64 fallback');
            // Fallback: convert to base64 for local storage
            photoUrl = await uploadPhoto(newReport.photoFile);
          }
        } catch (err) {
          console.warn('Backend not available, using base64 fallback:', err);
          // Fallback: convert to base64 for local storage
          photoUrl = await uploadPhoto(newReport.photoFile);
        }
      }

      // Build report object, only including fields that have values
      // Firebase doesn't accept undefined values
      const report: Record<string, any> = {
        name: newReport.name,
        age: parseInt(newReport.age),
        description: newReport.description,
        lastSeen: newReport.lastSeen || 'Unknown',
        reportedTime: Date.now(),
        status: 'searching',
        reportedBy: newReport.reportedBy || 'Anonymous'
      };

      // Only add optional fields if they have values
      if (newReport.gender) report.gender = newReport.gender;
      if (newReport.heightRange) report.heightRange = newReport.heightRange;
      if (newReport.upperClothingColor) report.upperClothingColor = newReport.upperClothingColor;
      if (newReport.lowerClothingColor) report.lowerClothingColor = newReport.lowerClothingColor;
      if (photoUrl) report.photoUrl = photoUrl;

      const missingRef = ref(db, 'missingPersons');
      const newRef = push(missingRef);
      await set(newRef, report);

      const activeCasesRef = ref(db, 'stats/activeCases');
      await runTransaction(activeCasesRef, (current) => (current || 0) + 1);

      setNewReport({
        name: '',
        age: '',
        description: '',
        lastSeen: '',
        reportedBy: '',
        gender: '',
        heightRange: '',
        upperClothingColor: '',
        lowerClothingColor: '',
        photoFile: null,
        photoUrl: ''
      });
      alert('Report submitted successfully!');
    } catch (err: any) {
      console.error('Error submitting report:', err);
      const errorMessage = err?.message || err?.code || 'Unknown error occurred';
      alert(`Could not submit report: ${errorMessage}`);
    }
  };

  const handleStatusUpdate = async (
    id: string,
    newStatus: MissingPerson['status']
  ) => {
    const personRef = ref(db, `missingPersons/${id}`);
    await update(personRef, { status: newStatus });

    if (newStatus === 'found') {
      const matchesRef = ref(db, 'stats/matchesConfirmed');
      await runTransaction(matchesRef, (current) => (current || 0) + 1);

      const activeCasesRef = ref(db, 'stats/activeCases');
      await runTransaction(activeCasesRef, (current) =>
        Math.max((current || 0) - 1, 0)
      );
    }
  };

  const handleDelete = async (id: string, status: string) => {
    if (!confirm('Are you sure you want to delete this entry?')) return;

    try {
      const personRef = ref(db, `missingPersons/${id}`);
      await remove(personRef);

      // Update active cases count if the person wasn't already found
      if (status !== 'found') {
        const activeCasesRef = ref(db, 'stats/activeCases');
        await runTransaction(activeCasesRef, (current) =>
          Math.max((current || 0) - 1, 0)
        );
      }
    } catch (err) {
      console.error('Error deleting entry:', err);
      alert('Could not delete entry');
    }
  };

  // Reset all AI stats
  const handleResetStats = async () => {
    if (!confirm('Are you sure you want to reset all AI stats to 0?')) return;

    try {
      const statsRef = ref(db, 'stats');
      await set(statsRef, {
        aiFaceScans: 0,
        facesDetected: 0,
        matchesConfirmed: 0,
        activeCases: missingPersons.filter(p => p.status !== 'found').length
      });
      alert('Stats reset successfully!');
    } catch (err) {
      console.error('Error resetting stats:', err);
      alert('Could not reset stats');
    }
  };

  const filteredPersons = missingPersons.filter((person) =>
    person.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    person.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (person.upperClothingColor &&
      person.upperClothingColor.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (person.lowerClothingColor &&
      person.lowerClothingColor.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="space-y-6">
      {/* Video Upload (Left) + Live Camera with Analytics (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">

        {/* LEFT COLUMN: Video Upload + Analysis */}
        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm text-gray-300 flex items-center gap-2">
                <Video className="h-4 w-4 text-purple-400" />
                CCTV Video Analysis
              </h3>
              {uploadedVideo && (
                <button
                  onClick={clearUploadedVideo}
                  className="text-gray-400 hover:text-red-400 transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            {videoPreviewUrl ? (
              <div className="space-y-3">
                <video
                  ref={videoRef}
                  src={videoPreviewUrl}
                  controls
                  className="w-full rounded-lg bg-black"
                  style={{ maxHeight: '250px' }}
                />
                <div className="flex gap-2">
                  <button
                    onClick={analyzeVideo}
                    disabled={isAnalyzingVideo}
                    className="flex-1 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 disabled:opacity-50 text-white py-2 px-4 rounded-lg transition-all flex items-center justify-center"
                  >
                    {isAnalyzingVideo ? (
                      <>
                        <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Analyze Video
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              <label className="flex flex-col items-center justify-center bg-gray-900/50 rounded-lg py-12 cursor-pointer hover:bg-gray-900/70 transition-colors border-2 border-dashed border-gray-600 hover:border-purple-500">
                <Upload className="h-10 w-10 text-gray-500 mb-3" />
                <p className="text-gray-400 mb-1">Upload CCTV Footage</p>
                <p className="text-gray-500 text-xs">MP4, WebM, AVI supported</p>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* Video Analysis Results */}
          {videoAnalysisResult && (
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 space-y-4">
              <h4 className="text-sm font-medium text-white flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-400" />
                Analysis Results
              </h4>

              {/* Match Result */}
              {videoAnalysisResult.matchedPerson && (
                <div className="bg-gradient-to-r from-green-900/60 to-green-800/40 rounded-lg p-3 border border-green-500/50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-green-400" />
                      <span className="text-green-300 font-medium">Potential Match Found</span>
                      {videoAnalysisResult.superVectorUsed && (
                        <span className="text-xs bg-purple-600/50 px-2 py-0.5 rounded text-purple-200">Super-Vector</span>
                      )}
                    </div>
                    <span className="text-2xl font-bold text-white">{videoAnalysisResult.matchConfidence}%</span>
                  </div>
                  <p className="text-white mt-1 font-semibold">{videoAnalysisResult.matchedPerson}</p>

                  {/* Confidence Breakdown */}
                  {videoAnalysisResult.confidenceBreakdown && (
                    <div className="mt-3 pt-3 border-t border-green-700/50">
                      <p className="text-green-300 text-xs mb-2 font-medium">Confidence Breakdown:</p>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-gray-400 text-xs w-24">Re-ID Features</span>
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full transition-all"
                              style={{ width: `${videoAnalysisResult.confidenceBreakdown.reid_features}%` }}
                            />
                          </div>
                          <span className="text-white text-xs w-10">{videoAnalysisResult.confidenceBreakdown.reid_features}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-400 text-xs w-24">Upper Clothing</span>
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-orange-500 h-2 rounded-full transition-all"
                              style={{ width: `${videoAnalysisResult.confidenceBreakdown.upper_clothing}%` }}
                            />
                          </div>
                          <span className="text-white text-xs w-10">{videoAnalysisResult.confidenceBreakdown.upper_clothing}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-400 text-xs w-24">Lower Clothing</span>
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-cyan-500 h-2 rounded-full transition-all"
                              style={{ width: `${videoAnalysisResult.confidenceBreakdown.lower_clothing}%` }}
                            />
                          </div>
                          <span className="text-white text-xs w-10">{videoAnalysisResult.confidenceBreakdown.lower_clothing}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-gray-400 text-xs w-24">Body Shape</span>
                          <div className="flex-1 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-purple-500 h-2 rounded-full transition-all"
                              style={{ width: `${videoAnalysisResult.confidenceBreakdown.body_shape}%` }}
                            />
                          </div>
                          <span className="text-white text-xs w-10">{videoAnalysisResult.confidenceBreakdown.body_shape}%</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Feedback Buttons */}
                  <div className="mt-3 flex gap-2">
                    <button
                      onClick={async () => {
                        const matchedCase = missingPersons.find(p => p.name.toLowerCase() === videoAnalysisResult.matchedPerson?.toLowerCase());
                        if (matchedCase) {
                          await fetch('http://127.0.0.1:8002/api/match-feedback', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              case_id: matchedCase.id,
                              feedback: 'confirm'
                            })
                          });
                          handleStatusUpdate(matchedCase.id, 'found');
                        }
                      }}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm py-1.5 px-3 rounded-lg flex items-center justify-center gap-1"
                    >
                      <CheckCircle className="h-4 w-4" /> Confirm Found
                    </button>
                    <button
                      onClick={async () => {
                        const matchedCase = missingPersons.find(p => p.name.toLowerCase() === videoAnalysisResult.matchedPerson?.toLowerCase());
                        if (matchedCase) {
                          await fetch('http://127.0.0.1:8002/api/match-feedback', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              case_id: matchedCase.id,
                              feedback: 'false_match'
                            })
                          });
                        }
                      }}
                      className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm py-1.5 px-3 rounded-lg flex items-center justify-center gap-1"
                    >
                      <X className="h-4 w-4" /> False Match
                    </button>
                  </div>
                </div>
              )}

              {/* Likely Location */}
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <MapPin className="h-4 w-4 text-amber-400" />
                  <span className="text-gray-300 text-sm">Likely Location</span>
                </div>
                <p className="text-white font-medium">{videoAnalysisResult.likelyLocation}</p>
              </div>

              {/* Detected Person Details */}
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <User className="h-4 w-4 text-blue-400" />
                  <span className="text-gray-300 text-sm">Detected Person Details</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div><span className="text-gray-400">Gender:</span> <span className="text-white">{videoAnalysisResult.detectedDetails.gender}</span></div>
                  <div><span className="text-gray-400">Age:</span> <span className="text-white">{videoAnalysisResult.detectedDetails.estimatedAge}</span></div>
                  <div><span className="text-gray-400">Top:</span> <span className="text-white">{videoAnalysisResult.detectedDetails.upperClothing}</span></div>
                  <div><span className="text-gray-400">Bottom:</span> <span className="text-white">{videoAnalysisResult.detectedDetails.lowerClothing}</span></div>
                </div>
              </div>

              {/* Relevant Info */}
              <div className="bg-gray-700/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="h-4 w-4 text-cyan-400" />
                  <span className="text-gray-300 text-sm">Relevant Information</span>
                </div>
                <ul className="text-sm space-y-1">
                  {videoAnalysisResult.relevantInfo.map((info, idx) => (
                    <li key={idx} className="text-gray-300 flex items-start gap-2">
                      <span className="text-cyan-400">•</span> {info}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-blue-900/30 rounded-lg p-2 text-center">
                  <p className="text-xl font-bold text-white">{videoAnalysisResult.framesAnalyzed}</p>
                  <p className="text-xs text-blue-300">Frames Analyzed</p>
                </div>
                <div className="bg-green-900/30 rounded-lg p-2 text-center">
                  <p className="text-xl font-bold text-white">{videoAnalysisResult.personsDetected}</p>
                  <p className="text-xs text-green-300">Persons Detected</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Live Camera + Analytics */}
        <div className="space-y-4">
          {/* Live Camera Preview */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
            <h3 className="text-sm text-gray-300 mb-2 flex items-center gap-2">
              <Camera className="h-4 w-4 text-cyan-400" />
              Live Camera Preview
            </h3>
            {cameraEnabled ? (
              <div className="space-y-3">
                <div className="relative">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="rounded-lg w-full"
                  />
                  {isScanning && (
                    <div className="absolute top-2 left-2 bg-red-600 text-white px-2 py-1 rounded-lg text-xs flex items-center animate-pulse">
                      <div className="w-2 h-2 bg-white rounded-full mr-2 animate-ping" />
                      SCANNING
                    </div>
                  )}
                  {detectedPersons > 0 && (
                    <div className="absolute top-2 right-2 bg-green-600 text-white px-2 py-1 rounded-lg text-xs">
                      {detectedPersons} person(s) detected
                    </div>
                  )}
                </div>

                {/* Match Result Display */}
                {matchResult && (
                  <div className="bg-gradient-to-r from-green-900/80 to-green-800/80 border-2 border-green-500 rounded-lg p-4 animate-pulse">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <CheckCircle className="h-8 w-8 text-green-400" />
                        <div>
                          <p className="text-green-300 text-sm font-medium">MATCH FOUND!</p>
                          <p className="text-white text-lg font-bold">{matchResult.name}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-green-300 text-sm">Confidence</p>
                        <p className="text-3xl font-bold text-white">{matchResult.confidence}%</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setMatchResult(null)}
                      className="mt-3 w-full bg-green-600 hover:bg-green-700 text-white py-1 px-3 rounded text-sm"
                    >
                      Dismiss
                    </button>
                  </div>
                )}

                {/* Scan Status */}
                {scanStatus && !matchResult && (
                  <div className={`text-sm px-3 py-2 rounded-lg ${scanStatus.includes('MATCH FOUND')
                    ? 'bg-green-900/50 text-green-300 font-bold'
                    : scanStatus.includes('error') || scanStatus.includes('not available')
                      ? 'bg-red-900/50 text-red-300'
                      : 'bg-blue-900/50 text-blue-300'
                    }`}>
                    {scanStatus}
                  </div>
                )}

                <div className="flex gap-2">
                  {!isScanning ? (
                    <button
                      onClick={startScanning}
                      className="flex-1 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white py-2 px-4 rounded-lg transition-all flex items-center justify-center"
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Start Scan
                    </button>
                  ) : (
                    <button
                      onClick={stopScanning}
                      className="flex-1 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white py-2 px-4 rounded-lg transition-all flex items-center justify-center"
                    >
                      <StopCircle className="h-4 w-4 mr-2" />
                      Stop Scan
                    </button>
                  )}
                  <button
                    onClick={() => {
                      stopScanning();
                      setCameraEnabled(false);
                    }}
                    className="bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded-lg transition-colors"
                  >
                    Close
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center bg-gray-900/50 rounded-lg py-12">
                <Camera className="h-12 w-12 text-gray-500 mb-4" />
                <p className="text-gray-400 mb-4">Camera is off</p>
                <button
                  onClick={() => setCameraEnabled(true)}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center"
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Start Camera
                </button>
              </div>
            )}
          </div>

          {/* AI Analytics Dashboard - Below Camera */}
          <div className="grid grid-cols-3 gap-3 relative">
            <button
              onClick={handleResetStats}
              className="absolute -top-2 -right-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs px-2 py-1 rounded z-10"
              title="Reset Stats"
            >
              Reset
            </button>
            <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-4 border border-blue-700/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-blue-300">Face Scans</p>
                  <p className="text-xl font-bold text-white">
                    {aiScanResults.totalScans.toLocaleString()}
                  </p>
                </div>
                <Eye className="h-6 w-6 text-blue-400" />
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-4 border border-green-700/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-green-300">Success Rate</p>
                  <p className="text-xl font-bold text-white">
                    {Math.round(aiScanResults.successRate * 100)}%
                  </p>
                </div>
                <Zap className="h-6 w-6 text-green-400" />
              </div>
            </div>

            <div className="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 rounded-xl p-4 border border-yellow-700/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-yellow-300">Active Cases</p>
                  <p className="text-xl font-bold text-white">
                    {missingPersons.filter((p) => p.status !== 'found').length}
                  </p>
                </div>
                <Search className="h-6 w-6 text-yellow-400" />
              </div>
            </div>
          </div>

          {/* Report Missing Person - moved here to fill space */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50">
            <h3 className="text-sm font-semibold mb-3 flex items-center">
              <User className="h-4 w-4 mr-2 text-blue-400" />
              Report Missing Person
            </h3>
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  placeholder="Full Name"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.name}
                  onChange={(e) =>
                    setNewReport((prev) => ({ ...prev, name: e.target.value }))
                  }
                />
                <input
                  type="number"
                  placeholder="Age"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.age}
                  onChange={(e) =>
                    setNewReport((prev) => ({ ...prev, age: e.target.value }))
                  }
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <select
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.gender}
                  onChange={(e) =>
                    setNewReport((prev) => ({
                      ...prev,
                      gender: e.target.value as 'male' | 'female' | 'other'
                    }))
                  }
                >
                  <option value="">Gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
                <select
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.heightRange}
                  onChange={(e) =>
                    setNewReport((prev) => ({ ...prev, heightRange: e.target.value }))
                  }
                >
                  <option value="">Height</option>
                  <option value="short">Short</option>
                  <option value="medium">Medium</option>
                  <option value="tall">Tall</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  placeholder="Top color"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.upperClothingColor}
                  onChange={(e) =>
                    setNewReport((prev) => ({
                      ...prev,
                      upperClothingColor: e.target.value
                    }))
                  }
                />
                <input
                  type="text"
                  placeholder="Bottom color"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.lowerClothingColor}
                  onChange={(e) =>
                    setNewReport((prev) => ({
                      ...prev,
                      lowerClothingColor: e.target.value
                    }))
                  }
                />
              </div>
              <input
                type="text"
                placeholder="Description (features, last seen location)"
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.description}
                onChange={(e) =>
                  setNewReport((prev) => ({
                    ...prev,
                    description: e.target.value
                  }))
                }
              />
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  placeholder="Last Seen Location"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.lastSeen}
                  onChange={(e) =>
                    setNewReport((prev) => ({
                      ...prev,
                      lastSeen: e.target.value
                    }))
                  }
                />
                <input
                  type="text"
                  placeholder="Reported By"
                  className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  value={newReport.reportedBy}
                  onChange={(e) =>
                    setNewReport((prev) => ({
                      ...prev,
                      reportedBy: e.target.value
                    }))
                  }
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 mb-1 block">
                  Reference Photo
                </label>
                <input
                  type="file"
                  accept="image/*"
                  className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-2 py-1 text-sm text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  onChange={(e) => {
                    const file = e.target.files?.[0] || null;
                    setNewReport((prev) => ({ ...prev, photoFile: file }));
                  }}
                />
              </div>
              <button
                onClick={handleSubmitReport}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-2 px-4 rounded-lg text-sm transition-all duration-200 transform hover:scale-[1.02]"
              >
                Submit Report & Start AI Search
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Search Section - Now standalone */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Search Interface */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Search className="h-5 w-5 mr-2 text-blue-400" />
            Search Missing Persons
          </h3>
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by name, clothing color, or description..."
              className="w-full pl-10 pr-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="text-sm text-gray-400">
            {filteredPersons.length} of {missingPersons.length} cases shown
          </div>
        </div>
      </div>

      {/* Missing Persons List */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold flex items-center">
            <AlertCircle className="h-5 w-5 mr-2 text-blue-400" />
            Active Missing Persons Cases
          </h3>
          <div className="text-sm text-gray-400">
            AI facial recognition active on{' '}
            {missingPersons.filter((p) => p.status !== 'found').length} cases
          </div>
        </div>

        <div className="space-y-4">
          {filteredPersons.map((person) => {
            const StatusIcon = getStatusIcon(person.status);
            return (
              <div
                key={person.id}
                className="bg-gray-700/30 rounded-xl p-6 border border-gray-600/30 hover:border-gray-500/50 transition-all"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    {/* Reference Photo */}
                    {person.photoUrl && (
                      <div className="flex-shrink-0">
                        <img
                          src={person.photoUrl}
                          alt={`${person.name}'s photo`}
                          className="w-16 h-16 rounded-lg object-cover border-2 border-gray-600"
                        />
                      </div>
                    )}
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="text-lg font-semibold text-white">
                          {person.name}
                        </h4>
                        <span className="text-sm text-gray-400">
                          Age {person.age}
                        </span>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(
                            person.status
                          )}`}
                        >
                          {person.status.replace('-', ' ').toUpperCase()}
                        </span>
                      </div>

                      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400 mb-1">
                        {person.gender && <span>Gender: {person.gender}</span>}
                        {person.heightRange && (
                          <span>Height: {person.heightRange}</span>
                        )}
                        {(person.upperClothingColor ||
                          person.lowerClothingColor) && (
                            <span>
                              Clothes: {person.upperClothingColor || '?'} top,{' '}
                              {person.lowerClothingColor || '?'} bottom
                            </span>
                          )}
                        {typeof person.aiMatchConfidence === 'number' && (
                          <span className="text-green-400">
                            AI confidence:{' '}
                            {Math.round(person.aiMatchConfidence * 100)}%
                          </span>
                        )}
                      </div>

                      <p className="text-gray-300 mb-2">
                        {person.description}
                      </p>
                      <div className="flex items-center space-x-4 text-sm text-gray-400">
                        <div className="flex items-center">
                          <MapPin className="h-4 w-4 mr-1" />
                          Last seen: {person.lastSeen}
                        </div>
                        <div className="flex items-center">
                          <Clock className="h-4 w-4 mr-1" />
                          {new Date(person.reportedTime).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  </div>
                  <StatusIcon className="h-6 w-6 text-blue-400" />
                </div>

                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">
                    Reported by: {person.reportedBy}
                  </div>
                  <div className="flex space-x-2">
                    {person.status === 'potential-match' && (
                      <>
                        <button
                          onClick={() =>
                            handleStatusUpdate(person.id, 'found')
                          }
                          className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                        >
                          Confirm Found
                        </button>
                        <button
                          onClick={() =>
                            handleStatusUpdate(person.id, 'searching')
                          }
                          className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                        >
                          False Match
                        </button>
                      </>
                    )}
                    {person.status === 'searching' && (
                      <>
                        <button
                          onClick={() =>
                            handleStatusUpdate(person.id, 'potential-match')
                          }
                          className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                        >
                          Mark Potential Match
                        </button>
                        <button
                          onClick={() =>
                            handleStatusUpdate(person.id, 'found')
                          }
                          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                        >
                          Mark as Found
                        </button>
                      </>
                    )}
                    <button
                      onClick={() => handleDelete(person.id, person.status)}
                      className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm transition-colors flex items-center"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default LostAndFound;
