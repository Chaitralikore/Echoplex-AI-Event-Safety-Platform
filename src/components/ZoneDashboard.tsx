import React, { useState, useEffect } from 'react';
import { MapPin, Users, TrendingUp } from 'lucide-react';

interface ZoneStats {
  totalCheckedIn: number;
  zones: Record<string, number>;
}

const ZoneDashboard: React.FC = () => {
  const [zoneStats, setZoneStats] = useState<ZoneStats | null>(null);
  const [loading, setLoading] = useState(true);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';
  const eventId = 'EVT-2024-001';

  const zones = [
    { id: 'ZONE-A', name: 'Main Entrance', capacity: 5000 },
    { id: 'ZONE-B', name: 'VIP Section', capacity: 500 },
    { id: 'ZONE-C', name: 'General Area', capacity: 10000 },
    { id: 'ZONE-D', name: 'Food Court', capacity: 2000 },
  ];

  const fetchZoneStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/attendees/zones/${eventId}`);
      const data = await response.json();

      if (data.success) {
        setZoneStats(data.data);
      }
    } catch (error) {
      console.error('Failed to fetch zone stats:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchZoneStats();
    const interval = setInterval(fetchZoneStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const getOccupancyPercentage = (zoneName: string, capacity: number) => {
    const count = zoneStats?.zones[zoneName] || 0;
    return Math.min((count / capacity) * 100, 100);
  };

  // Kept semantic: red=near capacity, amber=warning, green=normal
  const getOccupancyColor = (percentage: number) => {
    if (percentage >= 90) return 'bg-red-500';
    if (percentage >= 70) return 'bg-amber-500';
    return 'bg-accent';
  };

  return (
    <div className="bg-surface rounded-xl p-6 border border-surfaceHover">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
            <MapPin className="w-6 h-6 text-black" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-textPrimary">Zone Occupancy</h2>
            <p className="text-textSecondary text-sm">Real-time attendance tracking</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-base px-4 py-2 rounded-lg border border-surfaceHover">
          <Users className="w-5 h-5 text-accent" />
          <span className="text-textPrimary font-semibold">{zoneStats?.totalCheckedIn || 0}</span>
          <span className="text-textSecondary text-sm">Total</span>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-8">
          <div className="animate-spin w-8 h-8 border-4 border-accent border-t-transparent rounded-full mx-auto"></div>
          <p className="text-textSecondary mt-4">Loading zone data...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {zones.map((zone) => {
            const count = zoneStats?.zones[zone.name] || 0;
            const percentage = getOccupancyPercentage(zone.name, zone.capacity);

            return (
              <div key={zone.id} className="bg-base rounded-lg p-4 border border-surfaceHover">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-textPrimary font-medium">{zone.name}</h3>
                    <p className="text-textSecondary text-sm">{zone.id}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-textPrimary">{count}</p>
                    <p className="text-textSecondary text-xs">of {zone.capacity.toLocaleString()}</p>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-surfaceHover rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full ${getOccupancyColor(percentage)} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between mt-2">
                  <span className="text-textSecondary text-sm">{percentage.toFixed(1)}% occupied</span>
                  {percentage >= 90 && (
                    <span className="text-red-400 text-sm font-medium flex items-center gap-1">
                      <TrendingUp className="w-4 h-4" />
                      Near Capacity
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ZoneDashboard;
