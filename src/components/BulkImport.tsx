import React, { useState } from 'react';
<<<<<<< HEAD
import { Upload, CheckCircle, XCircle, FileText, Trash2, LogIn, LogOut, X } from 'lucide-react';
=======
import { Upload, Users, CheckCircle, XCircle, FileText, Trash2 } from 'lucide-react';
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb

interface ImportResult {
  success: boolean;
  message: string;
  data?: {
    imported: number;
    failed: number;
    failedRecords: any[];
  };
}

const BulkImport: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [bulkCheckingIn, setBulkCheckingIn] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [eventId] = useState('EVT-2024-001');

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

  // -------- Helpers --------
  const handleClearAttendees = async () => {
    if (!confirm('Are you sure you want to clear ALL attendees? This action cannot be undone.')) {
      return;
    }

    setClearing(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/attendees/clear/${eventId}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (data.success) {
        setResult({
          success: true,
          message: `Cleared ${data.data.deletedCount} attendees. You can now re-import.`,
        });
      } else {
        setResult({
          success: false,
          message: data.message || 'Failed to clear attendees',
        });
      }
    } catch (error: any) {
      setResult({
        success: false,
        message: 'Failed to connect to server',
      });
    } finally {
      setClearing(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  // Normalize header text so we can accept variants like
  // "S. No.", "S No", "email_ID", "Ticket_Id", etc.
  const normalizeHeader = (raw: string) => {
    const cleaned = raw.toLowerCase().replace(/\./g, '').replace(/[_\s]/g, '');
    // Treat "email_id" and similar as "email"
    if (cleaned === 'emailid') return 'email';
    return cleaned;
  };

  const parseCSV = (text: string) => {
<<<<<<< HEAD
    const lines = text.split('\n').filter(line => line.trim());
=======
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb

    if (lines.length < 2) {
      throw new Error('CSV file is empty or invalid');
    }

<<<<<<< HEAD
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/[\s_]/g, ''));

    // Map various column name formats to standard names
    const columnMap: Record<string, string> = {
      'name': 'name',
      'email': 'email',
      'emailid': 'email',
      'email_id': 'email',
      'ticketid': 'ticketId',
      'ticket_id': 'ticketId',
      'phone': 'phone',
      'phoneno': 'phone',
      'phone_no': 'phone',
      'location': 'location',
      'zone': 'location',
      's.no': 'sno',
      'sno': 'sno',
      'serialno': 'sno'
    };

    // Normalize headers
    const normalizedHeaders = headers.map(h => columnMap[h] || h);

    // Validate required headers (name, email, ticketId)
    const hasName = normalizedHeaders.includes('name');
    const hasEmail = normalizedHeaders.includes('email');
    const hasTicketId = normalizedHeaders.includes('ticketId');

    if (!hasName || !hasEmail || !hasTicketId) {
      throw new Error('CSV must have columns: Name, email_ID, Ticket_Id');
=======
    const rawHeaders = lines[0].split(',').map((h) => h.trim());
    const normalizedHeaders = rawHeaders.map(normalizeHeader);

    // Required logical columns (after normalization)
    const requiredKeys = ['sno', 'name', 'email', 'phoneno', 'ticketid', 'location'];
    const missing: string[] = [];

    for (const key of requiredKeys) {
      if (!normalizedHeaders.includes(key)) {
        missing.push(key);
      }
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
    }

    if (missing.length > 0) {
      throw new Error('CSV must have columns: s.no, name, email, phone no, ticketId, location');
    }

    const attendees: any[] = [];

    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;

<<<<<<< HEAD
      const values = lines[i].split(',').map(v => v.trim());
      const attendee: any = {};

      normalizedHeaders.forEach((header, index) => {
        if (header !== 'sno') { // Skip serial number column
          attendee[header] = values[index] || '';
=======
      const values = lines[i].split(',').map((v) => v.trim());
      const attendee: any = {};

      rawHeaders.forEach((header, index) => {
        const norm = normalizeHeader(header);
        const value = values[index];

        if (norm === 'ticketid') {
          attendee.ticketId = value;
        } else if (norm === 'phoneno') {
          attendee.phone = value;
        } else if (norm === 'sno') {
          attendee.serialNo = value;
        } else if (norm === 'email') {
          attendee.email = value;
        } else if (norm === 'name') {
          attendee.name = value;
        } else if (norm === 'location') {
          attendee.location = value;
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
        }
      });

      attendees.push(attendee);
    }

    return attendees;
  };

  const handleClearFile = () => {
    setFile(null);
    setResult(null);
    const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };

  const handleBulkAction = async (action: 'import' | 'check-in' | 'check-out') => {
    if (!file) return;

    setLoading(true);
    setResult(null);

    try {
      const text = await file.text();
      const rows = parseCSV(text); // validation happens here
      let endpoint = '';
      let body: any = { eventId };

      if (action === 'import') {
        endpoint = '/attendees/bulk-import';
        body.attendeeList = rows;
      } else if (action === 'check-in') {
        endpoint = '/attendees/bulk-check-in';
        body.ticketIds = rows.map(r => r.ticketId);
      } else if (action === 'check-out') {
        endpoint = '/attendees/bulk-check-out';
        body.ticketIds = rows.map(r => r.ticketId);
      }

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
<<<<<<< HEAD
        body: JSON.stringify(body),
=======
        body: JSON.stringify({
          attendeeList,
          eventId,
        }),
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
      });

      const data = await response.json();
      setResult(data);

<<<<<<< HEAD
      if (data.success && action === 'import') {
        // Only clear file on successful import if desired, but user might want to check-in immediately?
        // Let's keep file selected for flexibility, unless it's import.
        // Actually prompt says "Clear file button alongside" implying manual clear.
        // Existing behavior clears file on success. I'll modify to ONLY clear on explicit action or maybe just import.
        // Let's keep existing behavior for import, but for check-in/out maybe keep it?
        // User request "clear file button", so I'll trust that for clearing.
        // But for import, let's stick to existing behavior of clearing on success, OR remove it since we have a clear button now.
        // I will REMOVE the auto-clear on success so user can do Import -> CheckIn sequence if they want.
        // Wait, if they import, they aren't checked in? Usually "Import" just registers.
        // So Import -> Check In flow makes sense.
=======
      if (data.success) {
        setFile(null);
        const fileInput = document.getElementById('csv-upload') as HTMLInputElement | null;
        if (fileInput) fileInput.value = '';
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
      }
    } catch (error: any) {
      setResult({
        success: false,
<<<<<<< HEAD
        message: error.message || `Failed to ${action} attendees`
=======
        message: error.message || 'Failed to process CSV file',
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
      });
    } finally {
      setLoading(false);
    }
  };

<<<<<<< HEAD
  const handleUpload = () => handleBulkAction('import');
  const handleBulkCheckIn = () => handleBulkAction('check-in');
  const handleBulkCheckOut = () => handleBulkAction('check-out');

  /*const downloadSampleCSV = () => {
     const csv = `name,email,phone,ticketId
 Anbreen Shabir,anbreen@example.com,+919234563265,TKT-001
 Ankita Sawai,ankita@example.com,+919390631629,TKT-002
 Chaitrali Kore,chaitrali@example.com,+917869872312,TKT-003
 Anjali Sharma,anjali@example.com,+918345678934,TKT-004
 Sneha Chavan,sneha@example.com,+918361678954,TKT-005`;
 
     const blob = new Blob([csv], { type: 'text/csv' });
     const url = window.URL.createObjectURL(blob);
     const a = document.createElement('a');
     a.href = url;
     a.download = 'sample-attendees.csv';
     a.click();
     window.URL.revokeObjectURL(url);
   }; */
=======
  const handleBulkCheckIn = async () => {
    if (!confirm('Check in ALL registered attendees for this event now?')) {
      return;
    }

    setBulkCheckingIn(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/attendees/bulk-check-in/${eventId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // No body needed; backend uses existing per-attendee location from CSV
        body: JSON.stringify({}),
      });

      const data = await response.json();
      setResult(data);
    } catch (error: any) {
      setResult({
        success: false,
        message: error.message || 'Failed to perform bulk check-in',
      });
    } finally {
      setBulkCheckingIn(false);
    }
  };
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
          <Upload className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Bulk Import Attendees</h2>
          <p className="text-slate-400 text-sm">Upload CSV file to register multiple attendees at once</p>
        </div>
      </div>

<<<<<<< HEAD


      {/* File Upload */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Upload Attendee List (CSV File)
        </label>
        <div className="flex flex-wrap gap-3 w-full">
=======
      {/* File Upload */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-slate-300 mb-2">Upload Attendee List (CSV File)</label>
        <div className="flex flex-col md:flex-row gap-3">
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
          <input
            id="csv-upload"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="flex-1 min-w-[200px] px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-purple-500 file:text-white hover:file:bg-purple-600 file:cursor-pointer"
          />
          {file && (
            <button
              onClick={handleClearFile}
              className="px-4 py-3 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-colors flex items-center justify-center"
              title="Clear file"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-3 mt-3">
          <button
            onClick={handleUpload}
<<<<<<< HEAD
            disabled={!file || loading || clearing}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
          >
            {loading ? (
              <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
=======
            disabled={!file || loading || clearing || bulkCheckingIn}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
          >
            {loading ? (
              <>
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                Importing...
              </>
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
            ) : (
              <Upload className="w-5 h-5" />
            )}
            Import
          </button>

          <button
            onClick={handleBulkCheckIn}
            disabled={!file || loading || clearing}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
          >
            {loading ? (
              <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
            ) : (
              <LogIn className="w-5 h-5" />
            )}
            Check In
          </button>

          <button
            onClick={handleBulkCheckOut}
            disabled={!file || loading || clearing}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-amber-500 hover:bg-amber-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
          >
            {loading ? (
              <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
            ) : (
              <LogOut className="w-5 h-5" />
            )}
            Check Out
          </button>

          <button
            onClick={handleBulkCheckIn}
            disabled={loading || clearing || bulkCheckingIn}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
            title="Mark all registered attendees as checked in"
          >
            {bulkCheckingIn ? (
              <>
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                Bulk Check-In...
              </>
            ) : (
              <>
                <Users className="w-5 h-5" />
                Bulk Check-In
              </>
            )}
          </button>
          <button
            onClick={handleClearAttendees}
<<<<<<< HEAD
            disabled={loading || clearing}
            className="px-6 py-3 bg-red-500 hover:bg-red-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
            title="Clear all attendees from database"
=======
            disabled={loading || clearing || bulkCheckingIn}
            className="flex items-center justify-center gap-2 px-6 py-3 bg-red-500 hover:bg-red-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors whitespace-nowrap"
            title="Clear all attendees before re-importing"
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
          >
            {clearing ? (
              <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
            ) : (
              <Trash2 className="w-5 h-5" />
            )}
          </button>
        </div>
        {file && !loading && (
          <p className="text-slate-400 text-sm mt-2">
            Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
          </p>
        )}
      </div>

      {/* Result Display */}
      {result && (
        <div
          className={`p-4 rounded-lg border ${result.success
            ? 'bg-emerald-500/10 border-emerald-500/20'
            : 'bg-red-500/10 border-red-500/20'
            }`}
        >
          <div className="flex items-center gap-3 mb-3">
            {result.success ? (
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            ) : (
              <XCircle className="w-6 h-6 text-red-400" />
            )}
            <p className={`font-medium ${result.success ? 'text-emerald-400' : 'text-red-400'}`}>
              {result.message}
            </p>
          </div>

          {result.data && (
            <>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <span className="text-slate-400 text-sm">Success</span>
                  </div>
                  <p className="text-3xl font-bold text-emerald-400">{result.data.imported || result.data.successfulCount}</p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <XCircle className="w-5 h-5 text-red-400" />
                    <span className="text-slate-400 text-sm">Failed</span>
                  </div>
                  <p className="text-3xl font-bold text-red-400">{result.data.failed || result.data.failedCount}</p>
                </div>
              </div>

<<<<<<< HEAD
              {/* Show failed records if any */}
              {(result.data.failedRecords || result.data.failedDetails) && (result.data.failedRecords?.length > 0 || result.data.failedDetails?.length > 0) && (
=======
              {result.data.failedRecords && result.data.failedRecords.length > 0 && (
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
                <div className="mt-4 bg-slate-700/50 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-3">Failed Records:</h4>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {(result.data.failedRecords || result.data.failedDetails).map((record: any, index: number) => (
                      <div key={index} className="text-sm bg-slate-800 rounded p-2">
                        <p className="text-red-400">{record.reason}</p>
<<<<<<< HEAD
                        <p className="text-slate-400 text-xs mt-1">
                          {record.ticketId ? `Ticket ID: ${record.ticketId}` : JSON.stringify(record.attendee)}
                        </p>
=======
                        <p className="text-slate-400 text-xs mt-1">{JSON.stringify(record.attendee)}</p>
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="mt-6 bg-slate-700/50 rounded-lg p-4 border border-slate-600">
        <h4 className="text-white font-medium mb-3 flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          CSV Format Requirements
        </h4>
        <div className="space-y-2 text-sm text-slate-300">
          <div className="flex items-start gap-2">
            <span className="text-emerald-400 font-bold">✓</span>
<<<<<<< HEAD
            <span><strong>Required columns:</strong> Name, email_ID, Ticket_Id</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span><strong>Optional columns:</strong> Phone no, Location</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span><strong>Location</strong> updates Zone Occupancy on check-in</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span>Supports <strong>10,000+</strong> attendees per file</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span>First row must contain column headers</span>
=======
            <span>
              <strong>Required columns:</strong> s.no, name, email, phone no, ticketId, location
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span>
              First row must contain these column headers (e.g. S. No., Name, email_ID, Phone no, Ticket_Id,
              Location)
            </span>
>>>>>>> 7fb4a4900d0f088d04c029527320a5d892089ebb
          </div>
          <div className="flex items-start gap-2">
            <span className="text-cyan-400 font-bold">•</span>
            <span>Each Ticket_Id must be unique</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BulkImport;
