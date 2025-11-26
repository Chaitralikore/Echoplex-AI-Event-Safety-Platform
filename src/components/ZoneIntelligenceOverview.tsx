import React, { useState, useEffect } from 'react';
import { MapPin, TrendingUp } from 'lucide-react';

interface ZoneStats {
  totalCheckedIn: number;
  zones: Record<string, number>;
}

const ZoneIntelligenceOverview: React.FC = () => {
  const [zoneStats, setZoneStats] = useState<ZoneStats | null>(null);
  const [loading, setLoading] = useState(true);
  
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';
  const eventId = 'EVT-2024-001';

  // Zone definitions with capacities
  const zoneDefinitions = [
    { id: 'ZONE-A', name: 'Main Entrance', capacity: 5000 },
    { id: 'ZONE-B', name: 'VIP Section', capacity: 500 },
    { id: 'ZONE-C', name: 'General Area', capacity: 10000 },
    { id: 'ZONE-D', name: 'Food Court', capacity: 2000 },
  ];

  const fetchZoneData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/attendees/zones/${eventId}`);
      const data = await response.json();

      if (data.success) {
        setZoneStats(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch zone data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchZoneData();
    const interval = setInterval(fetchZoneData, 5000);
    return () => clearInterval(interval);
  }, []);

  const getRiskLevel = (current: number, capacity: number) => {
    const percentage = (current / capacity) * 100;
    if (percentage >= 80) return 'HIGH RISK';
    if (percentage >= 50) return 'MEDIUM RISK';
    return 'LOW RISK';
  };

  const getRiskColor = (current: number, capacity: number) => {
    const percentage = (current / capacity) * 100;
    if (percentage >= 80) return 'bg-red-500';
    if (percentage >= 50) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  const getPercentageFull = (current: number, capacity: number) => {
    return Math.min(Math.round((current / capacity) * 100), 100);
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-teal-500 rounded-lg flex items-center justify-center">
          <MapPin className="w-6 h-6 text-white" />
        </div>
        <h2 className="text-xl font-semibold text-white">Zone Intelligence Overview</h2>
      </div>

      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full mx-auto"></div>
          <p className="text-slate-400 mt-4">Loading zone data...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {zoneDefinitions.map((zone) => {
            const currentAttendees = zoneStats?.zones[zone.name] || 0;
            const percentageFull = getPercentageFull(currentAttendees, zone.capacity);
            const riskLevel = getRiskLevel(currentAttendees, zone.capacity);
            const riskColor = getRiskColor(currentAttendees, zone.capacity);

            return (
              <div key={zone.id} className="bg-slate-700 rounded-xl p-5 border border-slate-600 hover:border-cyan-500 transition-colors">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">{zone.name}</h3>
                  <div className="flex items-center gap-2">
                    {percentageFull >= 80 && (
                      <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></span>
                    )}
                    {percentageFull >= 50 && percentageFull < 80 && (
                      <span className="w-3 h-3 bg-amber-500 rounded-full"></span>
                    )}
                    {percentageFull < 50 && (
                      <span className="w-3 h-3 bg-emerald-500 rounded-full"></span>
                    )}
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-white">
                      {currentAttendees.toLocaleString()}
                    </span>
                    <span className="text-slate-400 text-sm">
                      / {zone.capacity.toLocaleString()}
                    </span>
                  </div>
                  <p className="text-slate-500 text-xs mt-1">capacity</p>
                </div>

                <div className="mb-4">
                  <div className="w-full bg-slate-600 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-full ${riskColor} transition-all duration-500`}
                      style={{ width: `${percentageFull}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      riskLevel === 'HIGH RISK'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                        : riskLevel === 'MEDIUM RISK'
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    }`}
                  >
                    {riskLevel}
                  </span>
                  <span className="text-slate-400 text-sm font-medium">
                    {percentageFull}% full
                  </span>
                </div>

                {percentageFull >= 80 && (
                  <div className="mt-3 pt-3 border-t border-slate-600">
                    <div className="flex items-center gap-2 text-red-400 text-xs">
                      <TrendingUp className="w-4 h-4" />
                      <span className="font-medium">Near Capacity Alert</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6 flex items-center justify-center gap-2 text-slate-500 text-sm">
        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
        <span>Real-time updates every 5 seconds</span>
      </div>
    </div>
  );
};

export default ZoneIntelligenceOverview;