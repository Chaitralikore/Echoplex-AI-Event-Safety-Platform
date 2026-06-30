import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, Users, MapPin, Camera, Radio, Search, Bone as Drone, Shield, Zap, Brain, Eye, MonitorCheck} from 'lucide-react';
import LostAndFound from './LostAndFound';
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
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    fetchTotalAttendees();

    const attendeeInterval = setInterval(() => {
      fetchTotalAttendees();
    }, 5000);

    const dataInterval = setInterval(() => {
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
    { id: 'lost-found', label: 'Lost & Found', icon: Search, description: 'Facial recognition search' },
    { id: 'ai-predictor', label: 'Crowd Predictor', icon: Brain, description: 'ML-powered surge prediction' },
  ];

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'CheckIn' : return <CheckInSection />;
      case 'lost-found': return <LostAndFound />;
      case 'ai-predictor': return <AICrowdPredictor />;
      default: return <CheckInSection />;
    }
  };

  return (
    <div className="min-h-screen bg-base text-textPrimary">
      {/* Header */}
      <header className="bg-surface border-b border-surfaceHover p-4 sticky top-0 z-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-accent rounded-xl flex items-center justify-center">
                <Eye className="h-6 w-6 text-black" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-textPrimary">
                  Echoplex
                </h1>
                <div className="text-sm text-textSecondary">AI-Powered Event Safety Intelligence</div>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-8">
            <div className="text-right">
              <div className="text-sm text-textSecondary">Current Time</div>
              <div className="text-lg font-mono font-bold text-accent">
                {currentTime.toLocaleTimeString()}
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-textSecondary">Total Attendees</div>
              <div className="text-xl font-bold text-textPrimary">{totalAttendees.toLocaleString()}</div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-surface border-b border-surfaceHover px-4 py-3 overflow-x-auto">
        <div className="flex space-x-1 min-w-max">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center space-x-3 px-6 py-3 rounded-xl transition-all duration-200 whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-accent text-black shadow-lg'
                    : 'text-textSecondary hover:text-textPrimary hover:bg-surfaceHover'
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
