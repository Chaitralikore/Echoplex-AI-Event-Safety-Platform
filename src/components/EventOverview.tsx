import React, { useState, useEffect } from 'react';
import { TrendingUp, MapPin, Users, Clock, AlertTriangle, Camera, Bone as Drone, Activity, Zap, Shield, QrCode } from 'lucide-react';
import ZoneIntelligenceOverview from './ZoneIntelligenceOverview';
import ZoneQRManager from './ZoneQRManager';

const EventOverview: React.FC = () => {
  const [activeView, setActiveView] = useState<'overview' | 'zones' | 'qr-manager'>('overview');
  const [crowdDensity, setCrowdDensity] = useState(0.72);
  const [weatherRisk, setWeatherRisk] = useState(0.15);
  const [securityScore, setSecurityScore] = useState(0.91);
  const [emergencyReadiness, setEmergencyReadiness] = useState(0.94);

  useEffect(() => {
    const interval = setInterval(() => {
      setCrowdDensity(prev => Math.max(0.1, Math.min(1, prev + (Math.random() - 0.5) * 0.05)));
      setWeatherRisk(prev => Math.max(0, Math.min(1, prev + (Math.random() - 0.5) * 0.02)));
      setSecurityScore(prev => Math.max(0.5, Math.min(1, prev + (Math.random() - 0.5) * 0.02)));
      setEmergencyReadiness(prev => Math.max(0.8, Math.min(1, prev + (Math.random() - 0.5) * 0.01)));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  if (activeView === 'zones') {
    return (
      <div className="space-y-6">
        {/* Navigation */}
        <div className="flex space-x-4">
          <button
            onClick={() => setActiveView('overview')}
            className="px-4 py-2 bg-slate-700/50 text-slate-300 hover:text-white rounded-lg transition-colors"
          >
            ← Back to Overview
          </button>
          <button
            onClick={() => setActiveView('qr-manager')}
            className="px-4 py-2 bg-cyan-600/20 text-cyan-400 hover:bg-cyan-600/30 rounded-lg transition-colors flex items-center"
          >
            <QrCode className="h-4 w-4 mr-2" />
            QR Code Manager
          </button>
        </div>
        <ZoneIntelligenceOverview />
      </div>
    );
  }

  if (activeView === 'qr-manager') {
    return (
      <div className="space-y-6">
        {/* Navigation */}
        <div className="flex space-x-4">
          <button
            onClick={() => setActiveView('overview')}
            className="px-4 py-2 bg-slate-700/50 text-slate-300 hover:text-white rounded-lg transition-colors"
          >
            ← Back to Overview
          </button>
          <button
            onClick={() => setActiveView('zones')}
            className="px-4 py-2 bg-cyan-600/20 text-cyan-400 hover:bg-cyan-600/30 rounded-lg transition-colors flex items-center"
          >
            <MapPin className="h-4 w-4 mr-2" />
            Live Zone Intelligence
          </button>
        </div>
        <ZoneQRManager />
      </div>
    );
  }

  const zones = [
    { name: 'Main Stage', capacity: 25000, current: 23400, risk: 'high', trend: 'increasing' },
    { name: 'Food Court', capacity: 8000, current: 6200, risk: 'medium', trend: 'stable' },
    { name: 'West Gate', capacity: 5000, current: 2800, risk: 'low', trend: 'decreasing' },
    { name: 'VIP Area', capacity: 1000, current: 890, risk: 'low', trend: 'stable' },
    { name: 'Parking Lot A', capacity: 3000, current: 2900, risk: 'medium', trend: 'increasing' },
    { name: 'Emergency Exit 1', capacity: 2000, current: 150, risk: 'low', trend: 'stable' },
  ];

  const getZoneColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'from-red-500 to-red-600';
      case 'medium': return 'from-amber-500 to-amber-600';
      default: return 'from-emerald-500 to-emerald-600';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return '↗️';
      case 'decreasing': return '↘️';
      default: return '➡️';
    }
  };

  const aiInsights = [
    {
      priority: 'High',
      type: 'Crowd Surge Prediction',
      message: 'Main Stage expected to reach critical density in 12 minutes',
      confidence: 89,
      action: 'Deploy overflow management'
    },
    {
      priority: 'Medium',
      type: 'Weather Alert',
      message: 'Light precipitation possible in 45 minutes',
      confidence: 67,
      action: 'Prepare covered areas'
    },
    {
      priority: 'Low',
      type: 'Resource Optimization',
      message: 'Medical team reallocation recommended for Food Court',
      confidence: 78,
      action: 'Reassign MED-03'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Enhanced Real-time Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-cyan-900/60 to-cyan-800/40 rounded-xl p-6 border border-cyan-700/30 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-cyan-300">Crowd Density</p>
              <p className="text-3xl font-bold text-white">{Math.round(crowdDensity * 100)}%</p>
            </div>
            <Users className="h-10 w-10 text-cyan-400" />
          </div>
          <div className="w-full bg-cyan-900/30 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${
                crowdDensity > 0.8 ? 'bg-gradient-to-r from-red-500 to-red-600' : 
                crowdDensity > 0.6 ? 'bg-gradient-to-r from-amber-500 to-amber-600' : 
                'bg-gradient-to-r from-emerald-500 to-emerald-600'
              }`}
              style={{ width: `${crowdDensity * 100}%` }}
            ></div>
          </div>
          <div className="text-sm text-cyan-300 mt-2">
            {crowdDensity > 0.8 ? 'Critical Level' : crowdDensity > 0.6 ? 'High Density' : 'Normal Level'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-amber-900/60 to-amber-800/40 rounded-xl p-6 border border-amber-700/30 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-amber-300">Weather Risk</p>
              <p className="text-3xl font-bold text-white">{Math.round(weatherRisk * 100)}%</p>
            </div>
            <TrendingUp className="h-10 w-10 text-amber-400" />
          </div>
          <div className="w-full bg-amber-900/30 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all duration-500 ${
                weatherRisk > 0.5 ? 'bg-gradient-to-r from-red-500 to-red-600' : 
                weatherRisk > 0.3 ? 'bg-gradient-to-r from-amber-500 to-amber-600' : 
                'bg-gradient-to-r from-emerald-500 to-emerald-600'
              }`}
              style={{ width: `${weatherRisk * 100}%` }}
            ></div>
          </div>
          <div className="text-sm text-amber-300 mt-2">
            {weatherRisk > 0.5 ? 'High Risk' : weatherRisk > 0.3 ? 'Moderate Risk' : 'Low Risk'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-emerald-900/60 to-emerald-800/40 rounded-xl p-6 border border-emerald-700/30 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-emerald-300">Security Score</p>
              <p className="text-3xl font-bold text-white">{Math.round(securityScore * 100)}%</p>
            </div>
            <Shield className="h-10 w-10 text-emerald-400" />
          </div>
          <div className="w-full bg-emerald-900/30 rounded-full h-3">
            <div 
              className="h-3 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-600 transition-all duration-500"
              style={{ width: `${securityScore * 100}%` }}
            ></div>
          </div>
          <div className="text-sm text-emerald-300 mt-2">Excellent Status</div>
        </div>

        <div className="bg-gradient-to-br from-teal-900/60 to-teal-800/40 rounded-xl p-6 border border-teal-700/30 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-teal-300">Emergency Readiness</p>
              <p className="text-3xl font-bold text-white">{Math.round(emergencyReadiness * 100)}%</p>
            </div>
            <AlertTriangle className="h-10 w-10 text-teal-400" />
          </div>
          <div className="w-full bg-teal-900/30 rounded-full h-3">
            <div 
              className="h-3 rounded-full bg-gradient-to-r from-teal-500 to-teal-600 transition-all duration-500"
              style={{ width: `${emergencyReadiness * 100}%` }}
            ></div>
          </div>
          <div className="text-sm text-teal-300 mt-2">All Systems Ready</div>
        </div>
      </div>

      {/* AI-Powered Insights */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-semibold mb-6 flex items-center">
          <Zap className="h-6 w-6 mr-3 text-cyan-400" />
          AI-Powered Predictive Insights
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {aiInsights.map((insight, index) => (
            <div key={index} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30">
              <div className="flex items-center justify-between mb-3">
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                  insight.priority === 'High' ? 'bg-red-900/40 text-red-400' :
                  insight.priority === 'Medium' ? 'bg-amber-900/40 text-amber-400' :
                  'bg-cyan-900/40 text-cyan-400'
                }`}>
                  {insight.priority}
                </span>
                <span className="text-xs text-slate-400">{insight.confidence}% confidence</span>
              </div>
              <div className="text-sm font-medium text-cyan-400 mb-2">{insight.type}</div>
              <div className="text-sm text-slate-300 mb-3">{insight.message}</div>
              <button className="text-xs bg-cyan-600/20 text-cyan-400 hover:bg-cyan-600/30 px-3 py-1 rounded-full transition-colors">
                {insight.action}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Zone Status */}
      <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold flex items-center">
            <MapPin className="h-6 w-6 mr-3 text-cyan-400" />
            Zone Intelligence Overview
          </h3>
          <div className="flex space-x-3">
            <button
              onClick={() => setActiveView('zones')}
              className="bg-gradient-to-r from-cyan-600 to-teal-700 hover:from-cyan-700 hover:to-teal-800 text-white px-4 py-2 rounded-lg transition-all duration-200 transform hover:scale-[1.02] flex items-center space-x-2"
            >
              <QrCode className="h-4 w-4" />
              <span>Live Zone Intelligence</span>
            </button>
            <button
              onClick={() => setActiveView('qr-manager')}
              className="bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
            >
              <QrCode className="h-4 w-4" />
              <span>QR Manager</span>
            </button>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {zones.map((zone) => (
            <div key={zone.name} className="bg-slate-700/30 rounded-xl p-5 border border-slate-600/30 hover:border-slate-500/50 transition-all">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-white text-lg">{zone.name}</h4>
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getTrendIcon(zone.trend)}</span>
                  <div className={`h-3 w-3 rounded-full bg-gradient-to-r ${getZoneColor(zone.risk)}`}></div>
                </div>
              </div>
              <div className="text-sm text-slate-400 mb-3">
                <span className="text-white font-medium">{zone.current.toLocaleString()}</span> / {zone.capacity.toLocaleString()} capacity
              </div>
              <div className="w-full bg-slate-600/50 rounded-full h-3 mb-3">
                <div 
                  className={`h-3 rounded-full bg-gradient-to-r ${getZoneColor(zone.risk)} transition-all duration-500`}
                  style={{ width: `${(zone.current / zone.capacity) * 100}%` }}
                ></div>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  zone.risk === 'high' ? 'bg-red-900/40 text-red-400' :
                  zone.risk === 'medium' ? 'bg-amber-900/40 text-amber-400' :
                  'bg-emerald-900/40 text-emerald-400'
                }`}>
                  {zone.risk.toUpperCase()} RISK
                </span>
                <span className="text-slate-400">{Math.round((zone.current / zone.capacity) * 100)}% full</span>
              </div>
            </div>
          ))}
        </div>
        
        {/* QR Code Integration Notice */}
        <div className="mt-6 bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <QrCode className="h-6 w-6 text-cyan-400" />
            <div>
              <div className="text-cyan-400 font-medium">QR Code Zone Tracking Available</div>
              <div className="text-slate-300 text-sm">Real-time occupancy tracking via QR code check-ins at zone entrances. Click "Live Zone Intelligence" for detailed monitoring.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Live Feeds and Drone Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-semibold mb-6 flex items-center">
            <Camera className="h-6 w-6 mr-3 text-cyan-400" />
            Live AI Camera Network
          </h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30">
                <div className="bg-slate-600/50 h-24 rounded-lg mb-3 flex items-center justify-center relative overflow-hidden">
                  <Camera className="h-8 w-8 text-slate-400" />
                  <div className="absolute top-2 right-2 w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                </div>
                <div className="text-sm text-slate-300 font-medium">Camera {i}</div>
                <div className="text-xs text-emerald-400">AI Processing Active</div>
              </div>
            ))}
          </div>
          <div className="bg-cyan-900/20 rounded-lg p-3 border border-cyan-700/30">
            <div className="text-sm text-cyan-400 font-medium">Network Status</div>
            <div className="text-xs text-slate-300">47/50 cameras online • 94% uptime • AI analysis: Active</div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-semibold mb-6 flex items-center">
            <Drone className="h-6 w-6 mr-3 text-cyan-400" />
            Autonomous Drone Fleet
          </h3>
          <div className="space-y-4">
            {[
              { id: 'D-001', location: 'Main Stage Perimeter', battery: 85, status: 'Patrolling', mission: 'Crowd monitoring' },
              { id: 'D-002', location: 'Food Court Area', battery: 62, status: 'Investigating', mission: 'Anomaly response' },
              { id: 'D-003', location: 'West Gate', battery: 91, status: 'Standby', mission: 'Ready for dispatch' },
              { id: 'D-004', location: 'Base Station', battery: 23, status: 'Charging', mission: 'Battery replacement' },
            ].map((drone) => (
              <div key={drone.id} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-teal-600 rounded-lg flex items-center justify-center">
                      <Drone className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <span className="font-semibold text-white">{drone.id}</span>
                      <div className="text-xs text-slate-400">{drone.mission}</div>
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    drone.status === 'Patrolling' ? 'bg-emerald-900/40 text-emerald-400' :
                    drone.status === 'Investigating' ? 'bg-amber-900/40 text-amber-400' :
                    drone.status === 'Standby' ? 'bg-cyan-900/40 text-cyan-400' :
                    'bg-slate-900/40 text-slate-400'
                  }`}>
                    {drone.status}
                  </span>
                </div>
                <div className="text-sm text-slate-300 mb-2">{drone.location}</div>
                <div className="flex items-center space-x-3">
                  <div className="flex-1">
                    <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                      <span>Battery</span>
                      <span>{drone.battery}%</span>
                    </div>
                    <div className="w-full bg-slate-600/50 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${
                          drone.battery > 50 ? 'bg-gradient-to-r from-emerald-500 to-emerald-600' : 
                          drone.battery > 20 ? 'bg-gradient-to-r from-amber-500 to-amber-600' : 
                          'bg-gradient-to-r from-red-500 to-red-600'
                        }`}
                        style={{ width: `${drone.battery}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EventOverview;