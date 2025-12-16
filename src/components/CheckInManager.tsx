// src/components/CheckInManager.tsx
import React, { useState, useEffect } from 'react';
import { UserCheck, UserX, Search, Clock, MapPin, Users, RefreshCw } from 'lucide-react';

interface Attendee {
  id: string;
  name: string;
  email: string;
  phone: string;
  ticketId: string;
  checkInTime: string | null;
  checkOutTime: string | null;
  status: 'not_checked_in' | 'checked_in' | 'checked_out';
  location?: string;
  eventId: string;
}

const CheckInManager: React.FC = () => {
  const [ticketId, setTicketId] = useState('');
  const [eventId, setEventId] = useState('EVT-2024-001');
  const [location, setLocation] = useState('Main Entrance');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [attendeeData, setAttendeeData] = useState<Attendee | null>(null);
  const [checkedInList, setCheckedInList] = useState<Attendee[]>([]);
  const [loadingList, setLoadingList] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';

  // Fetch all checked-in attendees
  const fetchCheckedInAttendees = async () => {
    setLoadingList(true);
    try {
      const response = await fetch(`${API_BASE_URL}/attendees/checked-in/${eventId}`);
      const data = await response.json();
      if (data.success) {
        setCheckedInList(data.data || []);
      }
    } catch (error) {
      console.error('Failed to fetch checked-in list:', error);
    } finally {
      setLoadingList(false);
    }
  };

  // Fetch list on mount and after each check-in/check-out
  useEffect(() => {
    fetchCheckedInAttendees();
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchCheckedInAttendees, 10000);
    return () => clearInterval(interval);
  }, [eventId]);

  const handleCheckIn = async () => {
    if (!ticketId.trim()) {
      setMessage({ type: 'error', text: 'Please enter a ticket ID' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/attendees/check-in`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticketId: ticketId.trim(),
          eventId,
          location
        }),
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ type: 'success', text: data.message });
        setAttendeeData(data.data.attendee);
        setTicketId('');
        // Refresh the checked-in list
        fetchCheckedInAttendees();
      } else {
        setMessage({ type: 'error', text: data.message });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to server. Please try again.' });
      console.error('Check-in error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCheckOut = async () => {
    if (!ticketId.trim()) {
      setMessage({ type: 'error', text: 'Please enter a ticket ID' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/attendees/check-out`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticketId: ticketId.trim(),
          eventId
        }),
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ 
          type: 'success', 
          text: `${data.message} - Duration: ${data.data.durationMinutes} minutes` 
        });
        setAttendeeData(data.data.attendee);
        setTicketId('');
        // Refresh the checked-in list
        fetchCheckedInAttendees();
      } else {
        setMessage({ type: 'error', text: data.message });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to server. Please try again.' });
      console.error('Check-out error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Quick check-out directly from the checked-in list
  const handleRowCheckOut = async (ticketIdFromRow: string) => {
    // Reuse existing logic but with the row's ticket ID
    setTicketId(ticketIdFromRow);
    await handleCheckOut();
  };

  const handleCheckStatus = async () => {
    if (!ticketId.trim()) {
      setMessage({ type: 'error', text: 'Please enter a ticket ID' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/attendees/status/${ticketId.trim()}/${eventId}`
      );

      const data = await response.json();

      if (data.success) {
        setAttendeeData(data.data);
        setMessage({ type: 'success', text: 'Attendee status retrieved' });
      } else {
        setMessage({ type: 'error', text: data.message });
        setAttendeeData(null);
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to server. Please try again.' });
      console.error('Status check error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const badges = {
      not_checked_in: 'bg-slate-600 text-slate-300',
      checked_in: 'bg-emerald-500 text-white',
      checked_out: 'bg-cyan-500 text-white'
    };
    return badges[status as keyof typeof badges] || badges.not_checked_in;
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-teal-500 rounded-lg flex items-center justify-center">
          <UserCheck className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Check-In Manager</h2>
          <p className="text-slate-400 text-sm">Manage attendee entry and exit</p>
        </div>
      </div>

      {/* Input Section */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Ticket ID
          </label>
          <input
            type="text"
            value={ticketId}
            onChange={(e) => setTicketId(e.target.value)}
            placeholder="Enter ticket ID (e.g., TKT-12345)"
            className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Event ID
            </label>
            <input
              type="text"
              value={eventId}
              onChange={(e) => setEventId(e.target.value)}
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Location
            </label>
            <select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
              disabled={loading}
            >
              <option value="Main Entrance">Main Entrance</option>
              <option value="VIP Section">VIP Section</option>
              <option value="General Area">General Area</option>
              <option value="Food Court">Food Court</option>
            </select>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <button
          onClick={handleCheckIn}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          <UserCheck className="w-5 h-5" />
          Check In
        </button>

        <button
          onClick={handleCheckOut}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          <UserX className="w-5 h-5" />
          Check Out
        </button>

        <button
          onClick={handleCheckStatus}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-teal-500 hover:bg-teal-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          <Search className="w-5 h-5" />
          Status
        </button>
      </div>

      {/* Message Display */}
      {message && (
        <div
          className={`p-4 rounded-lg mb-6 ${
            message.type === 'success'
              ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
              : 'bg-red-500/10 border border-red-500/20 text-red-400'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Attendee Info Display */}
      {attendeeData && (
        <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">{attendeeData.name}</h3>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBadge(attendeeData.status)}`}>
              {attendeeData.status.replace('_', ' ').toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-slate-400">Email</p>
              <p className="text-white">{attendeeData.email}</p>
            </div>
            <div>
              <p className="text-slate-400">Phone</p>
              <p className="text-white">{attendeeData.phone || 'N/A'}</p>
            </div>
            <div>
              <p className="text-slate-400">Ticket ID</p>
              <p className="text-white">{attendeeData.ticketId}</p>
            </div>
            <div>
              <p className="text-slate-400">Attendee ID</p>
              <p className="text-white">{attendeeData.id}</p>
            </div>
          </div>

          {attendeeData.checkInTime && (
            <div className="mt-4 pt-4 border-t border-slate-600">
              <div className="flex items-center gap-2 text-emerald-400 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Check-in Details</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-slate-400">Time</p>
                  <p className="text-white">
                    {new Date(attendeeData.checkInTime).toLocaleString()}
                  </p>
                </div>
                {attendeeData.location && (
                  <div>
                    <p className="text-slate-400">Location</p>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-cyan-400" />
                      <p className="text-white">{attendeeData.location}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {attendeeData.checkOutTime && (
            <div className="mt-4 pt-4 border-t border-slate-600">
              <div className="flex items-center gap-2 text-cyan-400 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Check-out Details</span>
              </div>
              <p className="text-slate-400 text-sm">Time</p>
              <p className="text-white text-sm">
                {new Date(attendeeData.checkOutTime).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}

      {/* All Checked-In Attendees List */}
      <div className="mt-6 bg-slate-700 rounded-lg border border-slate-600">
        <div className="flex items-center justify-between p-4 border-b border-slate-600">
          <div className="flex items-center gap-3">
            <Users className="w-5 h-5 text-emerald-400" />
            <h3 className="text-lg font-semibold text-white">
              All Checked-In Attendees ({checkedInList.length})
            </h3>
          </div>
          <button
            onClick={fetchCheckedInAttendees}
            disabled={loadingList}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-sm transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loadingList ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
        
        {checkedInList.length === 0 ? (
          <div className="p-8 text-center text-slate-400">
            <Users className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No attendees checked in yet</p>
          </div>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full">
              <thead className="bg-slate-800 sticky top-0">
                <tr>
                  <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">Name</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">Ticket ID</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">Location</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">Check-In Time</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-600">
                {checkedInList.map((attendee) => (
                  <tr key={attendee.id} className="hover:bg-slate-600/50 transition-colors">
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-white font-medium">{attendee.name}</p>
                        <p className="text-slate-400 text-xs">{attendee.email}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-cyan-400 font-mono text-sm">{attendee.ticketId}</span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1 text-slate-300 text-sm">
                        <MapPin className="w-3 h-3 text-emerald-400" />
                        {attendee.location || 'Unknown'}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-slate-300 text-sm">
                      {attendee.checkInTime 
                        ? new Date(attendee.checkInTime).toLocaleString()
                        : 'N/A'
                      }
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleRowCheckOut(attendee.ticketId)}
                        disabled={loading || loadingList}
                        className="px-3 py-1.5 text-xs font-medium rounded-lg bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white transition-colors"
                      >
                        Check Out
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default CheckInManager;