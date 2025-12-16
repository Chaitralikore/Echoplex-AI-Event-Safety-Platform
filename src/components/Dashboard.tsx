import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, Users, MapPin, Camera, Radio, Search, Bone as Drone, Shield, Zap, Brain, Eye, MonitorCheck} from 'lucide-react';
//import CrowdMonitor from './CrowdMonitor';
//import RiskAssessment from './RiskAssessment';
//import IncidentManagement from './IncidentManagement';
//import ResourceDispatch from './ResourceDispatch';
//import AnomalyDetection from './AnomalyDetection';
import LostAndFound from './LostAndFound';
//import CommandInterface from './CommandInterface';
//import EventOverview from './EventOverview';
import CheckInSection from './CheckInSection';
import AICrowdPredictor from './AICrowdPredictor';

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('CheckIn');
  const [totalAttendees, setTotalAttendees] = useState(0);
  const [activeIncidents, setActiveIncidents] = useState(0);
  const [currentTime, setCurrentTime] = useState(new Date());

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';
  const eventId = 'EVT-2024-001';

  const fetchTotalAttendees = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/attendees/zones/${eventId}`);
      const data = await response.json();

      if (data.success) {
        setTotalAttendees(data.data.totalCheckedIn);
      }
    } catch (error) {
      console.error('Failed to fetch total attendees:', error);
    }
  };

 
  useEffect(() => {
    // Update time every second
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    // Fetch total attendees immediately
  fetchTotalAttendees();
  
  // Update total attendees every 5 seconds
  const attendeeInterval = setInterval(() => {
    fetchTotalAttendees();
  }, 5000);
    

    // Simulate real-time updates
    const dataInterval = setInterval(() => {
      //Simulate incident count changes
      if (Math.random() < 0.15) {
        setActiveIncidents(prev => Math.max(0, Math.min(10, prev + Math.floor(Math.random() * 3 - 1))));
      }
    }, 3000); 

    return () => {
      
      clearInterval(timeInterval);
      clearInterval(attendeeInterval);
      clearInterval(dataInterval);
    };
  }, []); 

  const tabs = [
    { id: 'CheckIn', label: 'CheckIn Section', icon: MonitorCheck, description: 'Real-time attendance count'},
    //{ id: 'overview', label: 'Event Overview', icon: Activity, description: 'Real-time event status' },
    //{ id: 'crowd', label: 'Crowd Intelligence', icon: Users, description: 'AI crowd analysis' },
    //{ id: 'risk', label: 'Risk Assessment', icon: Shield, description: 'Predictive safety analytics' },
    //{ id: 'incidents', label: 'Incident Command', icon: AlertTriangle, description: 'Active incident management' },
    //{ id: 'resources', label: 'Resource Dispatch', icon: Radio, description: 'Dynamic resource allocation' },
    //{ id: 'anomalies', label: 'AI Detection', icon: Zap, description: 'Multimodal anomaly detection' },
    { id: 'lost-found', label: 'Lost & Found', icon: Search, description: 'Facial recognition search' },
    { id: 'ai-predictor', label: 'Crowd Predictor', icon: Brain, description: 'ML-powered surge prediction' },
  ];

  const renderActiveComponent = () => {
    switch (activeTab) {
     
      case 'CheckIn' : return <CheckInSection />;
      //case 'overview': return <EventOverview />;
      //case 'crowd': return <CrowdMonitor />;
      //case 'risk': return <RiskAssessment />;
      //case 'incidents': return <IncidentManagement />;
      //case 'resources': return <ResourceDispatch />;
      //case 'anomalies': return <AnomalyDetection />;
      case 'lost-found': return <LostAndFound />;
      case 'ai-predictor': return <AICrowdPredictor />;
      default: return <CheckInSection />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900 text-white">
      {/* Header */}
      <header className="bg-slate-800/80 backdrop-blur-md border-b border-slate-700/50 p-4 sticky top-0 z-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 via-teal-500 to-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
                <Eye className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 bg-clip-text text-transparent">
                  Echoplex
                </h1>
                <div className="text-sm text-slate-400">AI-Powered Event Safety Intelligence</div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-8">
            <div className="text-right">
              <div className="text-sm text-slate-400">Current Time</div>
              <div className="text-lg font-mono font-bold text-cyan-400">
                {currentTime.toLocaleTimeString()}
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-slate-400">Total Attendees</div>
              <div className="text-xl font-bold text-white">{totalAttendees.toLocaleString()}</div>
            
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700/30 px-4 py-3 overflow-x-auto">
        <div className="flex space-x-1 min-w-max">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center space-x-3 px-6 py-3 rounded-xl transition-all duration-200 whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600 text-white shadow-lg transform scale-[1.02]'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Icon className="h-5 w-5" />
                <div className="text-left">
                  <div className="text-sm font-semibold">{tab.label}</div>
                  <div className="text-xs opacity-80">{tab.description}</div>
                </div>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6 pb-20">
        <div className="max-w-7xl mx-auto">
          {renderActiveComponent()}
        </div>
      </main>
    </div>
  );
};


export default Dashboard;