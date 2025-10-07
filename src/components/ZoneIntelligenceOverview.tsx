import React, { useState, useEffect } from 'react';
import { MapPin, Users, TrendingUp, AlertTriangle, QrCode, Smartphone, Activity, Zap } from 'lucide-react';
import { zoneService, Zone, ZoneEvent } from '../services/zoneService';
import QRScanner from './QRScanner';

const ZoneIntelligenceOverview: React.FC = () => {
  const [zones, setZones] = useState<Zone[]>([]);
  const [recentEvents, setRecentEvents] = useState<ZoneEvent[]>([]);
  const [showScanner, setShowScanner] = useState(false);
  const [scanResult, setScanResult] = useState<string>('');
  const [userId] = useState(`user_${Math.random().toString(36).substr(2, 9)}`);
  const [userCurrentZone, setUserCurrentZone] = useState<Zone | undefined>();

  useEffect(() => {
    // Initial load
    setZones(zoneService.getZones());
    setRecentEvents(zoneService.getZoneEvents(undefined, 10));
    setUserCurrentZone(zoneService.getUserCurrentZone(userId));

    // Subscribe to real-time updates
    const unsubscribe = zoneService.subscribe((updatedZones) => {
      setZones(updatedZones);
      setRecentEvents(zoneService.getZoneEvents(undefined, 10));
      setUserCurrentZone(zoneService.getUserCurrentZone(userId));
    });

    return unsubscribe;
  }, [userId]);

  const handleQRScan = (qrData: string) => {
    const result = zoneService.processQRScan(qrData, userId);
    setScanResult(result.message);
    
    if (result.success) {
      // Update UI immediately
      setZones(zoneService.getZones());
      setUserCurrentZone(zoneService.getUserCurrentZone(userId));
    }
  };

  const getZoneColor = (risk: string) => {
    switch (risk) {
      case 'HIGH': return 'from-red-500 to-red-600 border-red-500/30';
      case 'MEDIUM': return 'from-amber-500 to-amber-600 border-amber-500/30';
      default: return 'from-emerald-500 to-emerald-600 border-emerald-500/30';
    }
  };

  const getTrendIcon = (zone: Zone) => {
    // Simulate trend based on current occupancy
    if (zone.percentFull > 85) return { icon: '↗️', trend: 'increasing', color: 'text-red-400' };
    if (zone.percentFull < 50) return { icon: '↘️', trend: 'decreasing', color: 'text-green-400' };
    return { icon: '➡️', trend: 'stable', color: 'text-yellow-400' };
  };

  const totalCapacity = zones.reduce((sum, zone) => sum + zone.capacity, 0);
  const totalOccupancy = zones.reduce((sum, zone) => sum + zone.currentAttendees, 0);
  const averageOccupancy = zones.length > 0 ? Math.round((totalOccupancy / totalCapacity) * 100) : 0;
  const highRiskZones = zones.filter(zone => zone.riskLevel === 'HIGH').length;

  return (
    <div className="space-y-6">
      {/* Real-time Metrics Header */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-cyan-900/60 to-cyan-800/40 rounded-xl p-6 border border-cyan-700/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-cyan-300">Total Occupancy</p>
              <p className="text-3xl font-bold text-white">{totalOccupancy.toLocaleString()}</p>
            </div>
            <Users className="h-10 w-10 text-cyan-400" />
          </div>
          <div className="text-sm text-cyan-300 mt-2">
            {totalCapacity.toLocaleString()} capacity • {averageOccupancy}% full
          </div>
        </div>

        <div className="bg-gradient-to-br from-emerald-900/60 to-emerald-800/40 rounded-xl p-6 border border-emerald-700/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-emerald-300">Active Zones</p>
              <p className="text-3xl font-bold text-white">{zones.length}</p>
            </div>
            <MapPin className="h-10 w-10 text-emerald-400" />
          </div>
          <div className="text-sm text-emerald-300 mt-2">Real-time monitoring</div>
        </div>

        <div className="bg-gradient-to-br from-amber-900/60 to-amber-800/40 rounded-xl p-6 border border-amber-700/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-amber-300">High Risk Zones</p>
              <p className="text-3xl font-bold text-white">{highRiskZones}</p>
            </div>
            <AlertTriangle className="h-10 w-10 text-amber-400" />
          </div>
          <div className="text-sm text-amber-300 mt-2">Requiring attention</div>
        </div>

        <div className="bg-gradient-to-br from-indigo-900/60 to-indigo-800/40 rounded-xl p-6 border border-indigo-700/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-indigo-300">QR Scans Today</p>
              <p className="text-3xl font-bold text-white">{recentEvents.length * 12}</p>
            </div>
            <QrCode className="h-10 w-10 text-indigo-400" />
          </div>
          <div className="text-sm text-indigo-300 mt-2">Check-ins/check-outs</div>
        </div>
      </div>

      {/* User Status & QR Scanner */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-teal-600 rounded-xl flex items-center justify-center">
              <Smartphone className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Zone Check-In System</h3>
              <p className="text-slate-400">
                {userCurrentZone 
                  ? `Currently in: ${userCurrentZone.name}` 
                  : 'Not checked into any zone'
                }
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowScanner(true)}
            className="bg-gradient-to-r from-cyan-600 to-teal-700 hover:from-cyan-700 hover:to-teal-800 text-white px-6 py-3 rounded-lg transition-all duration-200 transform hover:scale-[1.02] flex items-center space-x-2"
          >
            <QrCode className="h-5 w-5" />
            <span>Scan QR Code</span>
          </button>
        </div>
        
        {scanResult && (
          <div className="mt-4 p-3 bg-emerald-900/20 border border-emerald-700/30 rounded-lg">
            <p className="text-emerald-400 text-sm">{scanResult}</p>
          </div>
        )}
      </div>

      {/* Enhanced Zone Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {zones.map((zone) => {
          const trend = getTrendIcon(zone);
          return (
            <div 
              key={zone.id} 
              className={`bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border-2 transition-all hover:scale-[1.02] ${getZoneColor(zone.riskLevel)}`}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="font-semibold text-white text-lg">{zone.name}</h4>
                  <div className="flex items-center text-sm text-slate-400 mt-1">
                    <MapPin className="h-4 w-4 mr-1" />
                    {zone.location}
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{trend.icon}</span>
                  <div className={`h-3 w-3 rounded-full bg-gradient-to-r ${getZoneColor(zone.riskLevel).split(' ')[0]} ${getZoneColor(zone.riskLevel).split(' ')[1]}`}></div>
                </div>
              </div>

              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Occupancy</span>
                  <span className="text-white font-medium">
                    {zone.currentAttendees.toLocaleString()} / {zone.capacity.toLocaleString()}
                  </span>
                </div>
                
                <div className="w-full bg-slate-700/50 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full bg-gradient-to-r ${getZoneColor(zone.riskLevel).split(' ')[0]} ${getZoneColor(zone.riskLevel).split(' ')[1]} transition-all duration-500`}
                    style={{ width: `${zone.percentFull}%` }}
                  ></div>
                </div>

                <div className="flex items-center justify-between">
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    zone.riskLevel === 'HIGH' ? 'bg-red-900/40 text-red-400' :
                    zone.riskLevel === 'MEDIUM' ? 'bg-amber-900/40 text-amber-400' :
                    'bg-emerald-900/40 text-emerald-400'
                  }`}>
                    {zone.riskLevel} RISK
                  </span>
                  <span className="text-white font-bold text-lg">{zone.percentFull}%</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-700/30 rounded-lg p-2">
                  <div className="text-slate-400">Trend</div>
                  <div className={`font-medium ${trend.color}`}>{trend.trend}</div>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-2">
                  <div className="text-slate-400">Entrances</div>
                  <div className="text-white font-medium">{zone.entrances.length}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Real-time Activity Feed */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-semibold mb-4 flex items-center">
          <Activity className="h-6 w-6 mr-3 text-cyan-400" />
          Real-Time Zone Activity
        </h3>
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {recentEvents.map((event, index) => {
            const zone = zones.find(z => z.id === event.zoneId);
            return (
              <div key={event.id} className="flex items-center justify-between bg-slate-700/30 rounded-lg p-3">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    event.action === 'check-in' ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}></div>
                  <div>
                    <div className="text-white text-sm">
                      User {event.action === 'check-in' ? 'entered' : 'exited'} {zone?.name}
                    </div>
                    <div className="text-slate-400 text-xs">
                      {event.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  event.action === 'check-in' ? 'bg-emerald-900/40 text-emerald-400' : 'bg-amber-900/40 text-amber-400'
                }`}>
                  {event.action.toUpperCase()}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* QR Scanner Modal */}
      <QRScanner
        isOpen={showScanner}
        onScan={handleQRScan}
        onClose={() => setShowScanner(false)}
      />
    </div>
  );
};

export default ZoneIntelligenceOverview;