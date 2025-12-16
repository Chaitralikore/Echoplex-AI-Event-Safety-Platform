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
  StopCircle
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

  // stub – no real upload yet
  const uploadPhoto = async (file: File): Promise<string> => {
    return '';
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
      const res = await fetch('http://127.0.0.1:8001/api/scan', {
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
  const startScanning = useCallback(() => {
    if (isScanning) return;
    
    setIsScanning(true);
    setScanStatus('Starting scan...');
    
    // Scan every 2 seconds
    scanIntervalRef.current = setInterval(() => {
      scanFrame();
    }, 2000);
    
    // Run first scan immediately
    scanFrame();
  }, [isScanning, scanFrame]);

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
        const successRate =
          facesDetected === 0 ? 0 : matchesConfirmed / facesDetected;

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
            console.warn('Backend registration failed, continuing with Firebase only');
          }
        } catch (err) {
          console.warn('Backend not available for face registration:', err);
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
    } catch (err) {
      console.error('Error submitting report:', err);
      alert('Could not submit report');
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
      {/* Live Camera + AI Analytics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
          <h3 className="text-sm text-gray-300 mb-2">Live Camera Preview</h3>
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
                <div className={`text-sm px-3 py-2 rounded-lg ${
                  scanStatus.includes('MATCH FOUND') 
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
            <div className="flex flex-col items-center justify-center bg-gray-900/50 rounded-lg py-16">
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

        {/* AI Analytics Dashboard */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 relative">
          <button
            onClick={handleResetStats}
            className="absolute -top-2 -right-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs px-2 py-1 rounded z-10"
            title="Reset Stats"
          >
            Reset Stats
          </button>
          <div className="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-6 border border-blue-700/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-300">Face Scans</p>
                <p className="text-2xl font-bold text-white">
                  {aiScanResults.totalScans.toLocaleString()}
                </p>
              </div>
              <Eye className="h-8 w-8 text-blue-400" />
            </div>
            <div className="text-sm text-blue-300">Frames analyzed</div>
          </div>

          <div className="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-700/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-300">Success Rate</p>
                <p className="text-2xl font-bold text-white">
                  {Math.round(aiScanResults.successRate * 100)}%
                </p>
              </div>
              <Zap className="h-8 w-8 text-green-400" />
            </div>
            <div className="text-sm text-green-300">Accuracy</div>
          </div>

          <div className="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 rounded-xl p-6 border border-yellow-700/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-yellow-300">Active Cases</p>
                <p className="text-2xl font-bold text-white">
                  {missingPersons.filter((p) => p.status !== 'found').length}
                </p>
              </div>
              <Search className="h-8 w-8 text-yellow-400" />
            </div>
            <div className="text-sm text-yellow-300">Currently searching</div>
          </div>
        </div>
      </div>

      {/* Search and Report */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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

        {/* New Report Form */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <User className="h-5 w-5 mr-2 text-blue-400" />
            Report Missing Person
          </h3>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <input
                type="text"
                placeholder="Full Name"
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.name}
                onChange={(e) =>
                  setNewReport((prev) => ({ ...prev, name: e.target.value }))
                }
              />
              <input
                type="number"
                placeholder="Age"
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.age}
                onChange={(e) =>
                  setNewReport((prev) => ({ ...prev, age: e.target.value }))
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <select
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.gender}
                onChange={(e) =>
                  setNewReport((prev) => ({ ...prev, gender: e.target.value }))
                }
              >
                <option value="">Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>

              <select
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.heightRange}
                onChange={(e) =>
                  setNewReport((prev) => ({
                    ...prev,
                    heightRange: e.target.value
                  }))
                }
              >
                <option value="">Height</option>
                <option value="short">Short</option>
                <option value="medium">Medium</option>
                <option value="tall">Tall</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <input
                type="text"
                placeholder="Top color (e.g. black)"
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
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
                placeholder="Bottom color (e.g. blue)"
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
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
              placeholder="Description (clothing, distinguishing features)"
              className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              value={newReport.description}
              onChange={(e) =>
                setNewReport((prev) => ({
                  ...prev,
                  description: e.target.value
                }))
              }
            />

            <div className="grid grid-cols-2 gap-3">
              <input
                type="text"
                placeholder="Last Seen Location"
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
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
                className="bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                value={newReport.reportedBy}
                onChange={(e) =>
                  setNewReport((prev) => ({
                    ...prev,
                    reportedBy: e.target.value
                  }))
                }
              />
            </div>

            {/* Photo Upload */}
            <div>
              <label className="text-sm text-gray-300 mb-2 block">
                Reference Photo (optional)
              </label>
              <input
                type="file"
                accept="image/*"
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                onChange={(e) => {
                  const file = e.target.files?.[0] || null;
                  setNewReport((prev) => ({ ...prev, photoFile: file }));
                }}
              />
            </div>

            <button
              onClick={handleSubmitReport}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-2 px-4 rounded-lg transition-all duration-200 transform hover:scale-[1.02]"
            >
              Submit Report & Start AI Search
            </button>
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
