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

  useEffect(() => {
    fetchCheckedInAttendees();
    const interval = setInterval(fetchCheckedInAttendees, 60000);
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticketId: ticketId.trim(), eventId, location }),
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ type: 'success', text: data.message });
        setAttendeeData(data.data.attendee);
        setTicketId('');
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticketId: ticketId.trim(), eventId }),
      });

      const data = await response.json();

      if (data.success) {
        setMessage({
          type: 'success',
          text: `${data.message} - Duration: ${data.data.durationMinutes} minutes`
        });
        setAttendeeData(data.data.attendee);
        setTicketId('');
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

  const handleRowCheckOut = async (ticketIdFromRow: string) => {
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
      not_checked_in: 'bg-surfaceHover text-textSecondary',
      checked_in: 'bg-green-600 text-white',
      checked_out: 'bg-accent text-black font-semibold'
    };
    return badges[status as keyof typeof badges] || badges.not_checked_in;
  };

  return (
    <div className="bg-surface rounded-xl p-6 border border-surfaceHover">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
          <UserCheck className="w-6 h-6 text-black" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-textPrimary">Check-In Manager</h2>
          <p className="text-textSecondary text-sm">Manage attendee entry and exit</p>
        </div>
      </div>

      {/* Input Section */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-textSecondary mb-2">
            Ticket ID
          </label>
          <input
            type="text"
            value={ticketId}
            onChange={(e) => setTicketId(e.target.value)}
            placeholder="Enter ticket ID (e.g., TKT-12345)"
            className="w-full px-4 py-2 bg-base border border-surfaceHover rounded-lg text-textPrimary placeholder-textSecondary focus:outline-none focus:ring-2 focus:ring-accent"
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-textSecondary mb-2">
              Event ID
            </label>
            <input
              type="text"
              value={eventId}
              onChange={(e) => setEventId(e.target.value)}
              className="w-full px-4 py-2 bg-base border border-surfaceHover rounded-lg text-textPrimary focus:outline-none focus:ring-2 focus:ring-accent"
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-textSecondary mb-2">
              Location
            </label>
            <select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="w-full px-4 py-2 bg-base border border-surfaceHover rounded-lg text-textPrimary focus:outline-none focus:ring-2 focus:ring-accent"
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

      {/* Action Buttons — green=check in (entry), surfaceHover=check out (neutral exit), accent=status (primary query) */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        <button
          onClick={handleCheckIn}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-surfaceHover disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          <UserCheck className="w-5 h-5" />
          Check In
        </button>

        <button
          onClick={handleCheckOut}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-surfaceHover hover:bg-surfaceHover/70 disabled:opacity-50 disabled:cursor-not-allowed text-textPrimary rounded-lg font-medium transition-colors border border-surfaceHover"
        >
          <UserX className="w-5 h-5" />
          Check Out
        </button>

        <button
          onClick={handleCheckStatus}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-3 bg-accent hover:bg-accent/80 disabled:opacity-50 disabled:cursor-not-allowed text-black font-semibold rounded-lg transition-colors"
        >
          <Search className="w-5 h-5" />
          Status
        </button>
      </div>

      {/* Message Display — green/red kept: functional success/error */}
      {message && (
        <div
          className={`p-4 rounded-lg mb-6 ${message.type === 'success'
            ? 'bg-green-500/10 border border-green-500/20 text-green-400'
            : 'bg-red-500/10 border border-red-500/20 text-red-400'
            }`}
        >
          {message.text}
        </div>
      )}

      {/* Attendee Info Display */}
      {attendeeData && (
        <div className="bg-base rounded-lg p-4 border border-surfaceHover">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-textPrimary">{attendeeData.name}</h3>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBadge(attendeeData.status)}`}>
              {attendeeData.status.replace('_', ' ').toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-textSecondary">Email</p>
              <p className="text-textPrimary">{attendeeData.email}</p>
            </div>
            <div>
              <p className="text-textSecondary">Phone</p>
              <p className="text-textPrimary">{attendeeData.phone || 'N/A'}</p>
            </div>
            <div>
              <p className="text-textSecondary">Ticket ID</p>
              <p className="text-textPrimary">{attendeeData.ticketId}</p>
            </div>
            <div>
              <p className="text-textSecondary">Attendee ID</p>
              <p className="text-textPrimary">{attendeeData.id}</p>
            </div>
          </div>

          {attendeeData.checkInTime && (
            <div className="mt-4 pt-4 border-t border-surfaceHover">
              <div className="flex items-center gap-2 text-accent mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Check-in Details</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-textSecondary">Time</p>
                  <p className="text-textPrimary">
                    {new Date(attendeeData.checkInTime).toLocaleString()}
                  </p>
                </div>
                {attendeeData.location && (
                  <div>
                    <p className="text-textSecondary">Location</p>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-accent" />
                      <p className="text-textPrimary">{attendeeData.location}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {attendeeData.checkOutTime && (
            <div className="mt-4 pt-4 border-t border-surfaceHover">
              <div className="flex items-center gap-2 text-accent mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Check-out Details</span>
              </div>
              <p className="text-textSecondary text-sm">Time</p>
              <p className="text-textPrimary text-sm">
                {new Date(attendeeData.checkOutTime).toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}

      {/* All Checked-In Attendees List */}
      <div className="mt-6 bg-base rounded-lg border border-surfaceHover">
        <div className="flex items-center justify-between p-4 border-b border-surfaceHover">
          <div className="flex items-center gap-3">
            <Users className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-semibold text-textPrimary">
              All Checked-In Attendees ({checkedInList.length})
            </h3>
          </div>
          <button
            onClick={fetchCheckedInAttendees}
            disabled={loadingList}
            className="flex items-center gap-2 px-3 py-1.5 bg-surfaceHover hover:bg-surfaceHover/70 text-textPrimary rounded-lg text-sm transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loadingList ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {checkedInList.length === 0 ? (
          <div className="p-8 text-center text-textSecondary">
            <Users className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No attendees checked in yet</p>
          </div>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full">
              <thead className="bg-surface sticky top-0">
                <tr>
                  <th className="text-left px-4 py-3 text-sm font-medium text-textSecondary">Name</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-textSecondary">Ticket ID</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-textSecondary">Location</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-textSecondary">Check-In Time</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-textSecondary">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surfaceHover">
                {checkedInList.map((attendee) => (
                  <tr key={attendee.id} className="hover:bg-surfaceHover/30 transition-colors">
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-textPrimary font-medium">{attendee.name}</p>
                        <p className="text-textSecondary text-xs">{attendee.email}</p>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="text-accent font-mono text-sm">{attendee.ticketId}</span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1 text-textSecondary text-sm">
                        <MapPin className="w-3 h-3 text-accent" />
                        {attendee.location || 'Unknown'}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-textSecondary text-sm">
                      {attendee.checkInTime
                        ? new Date(attendee.checkInTime).toLocaleString()
                        : 'N/A'
                      }
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleRowCheckOut(attendee.ticketId)}
                        disabled={loading || loadingList}
                        className="px-3 py-1.5 text-xs font-medium rounded-lg bg-surfaceHover hover:bg-surfaceHover/70 disabled:opacity-50 disabled:cursor-not-allowed text-textPrimary transition-colors"
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
