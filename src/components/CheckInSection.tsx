import React from 'react';
import { UserCheck, QrCode, Upload, BarChart3 } from 'lucide-react';
import CheckInManager from './CheckInManager';
import QRCodeGenerator from './QRCodeGenerator';
import BulkImport from './BulkImport';
import ZoneDashboard from './ZoneDashboard';

const CheckInSection: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="bg-surface rounded-xl p-6 border border-surfaceHover">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 bg-accent rounded-xl flex items-center justify-center">
              <UserCheck className="w-8 h-8 text-black" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-textPrimary mb-1">
                Check-In Manager
              </h1>
              <p className="text-textSecondary text-sm">
                Comprehensive attendee management and QR code-based check-in system
              </p>
            </div>
          </div>

          {/* Feature Icons */}
          <div className="hidden lg:flex items-center gap-3">
            <div className="flex items-center gap-2 bg-surfaceHover px-4 py-2 rounded-lg border border-surfaceHover">
              <QrCode className="w-5 h-5 text-accent" />
              <span className="text-textSecondary text-sm font-medium">QR Codes</span>
            </div>
            <div className="flex items-center gap-2 bg-surfaceHover px-4 py-2 rounded-lg border border-surfaceHover">
              <Upload className="w-5 h-5 text-accent" />
              <span className="text-textSecondary text-sm font-medium">Bulk Import</span>
            </div>
            <div className="flex items-center gap-2 bg-surfaceHover px-4 py-2 rounded-lg border border-surfaceHover">
              <BarChart3 className="w-5 h-5 text-accent" />
              <span className="text-textSecondary text-sm font-medium">Live Analytics</span>
            </div>
          </div>
        </div>

        {/* Quick Stats Bar */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-base rounded-lg p-4 border border-surfaceHover">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-surfaceHover rounded-lg flex items-center justify-center">
                <UserCheck className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="text-textSecondary text-xs">Manual Check-In</p>
                <p className="text-textPrimary font-semibold">Available</p>
              </div>
            </div>
          </div>

          <div className="bg-base rounded-lg p-4 border border-surfaceHover">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-surfaceHover rounded-lg flex items-center justify-center">
                <QrCode className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="text-textSecondary text-xs">QR Code System</p>
                <p className="text-textPrimary font-semibold">4 Zones Active</p>
              </div>
            </div>
          </div>

          <div className="bg-base rounded-lg p-4 border border-surfaceHover">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-surfaceHover rounded-lg flex items-center justify-center">
                <Upload className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="text-textSecondary text-xs">Bulk Import</p>
                <p className="text-textPrimary font-semibold">CSV Ready</p>
              </div>
            </div>
          </div>

          <div className="bg-base rounded-lg p-4 border border-surfaceHover">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-surfaceHover rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="text-textSecondary text-xs">Live Updates</p>
                <p className="text-textPrimary font-semibold">Real-Time</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sub-section Headers and Components */}

      {/* 1. Manual Check-In */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-1 h-8 bg-accent rounded-full"></div>
          <h2 className="text-2xl font-bold text-textPrimary">Manual Check-In / Check-Out</h2>
        </div>
        <CheckInManager />
      </div>

      {/* 2. Zone Occupancy Dashboard */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-1 h-8 bg-accent rounded-full"></div>
          <h2 className="text-2xl font-bold text-textPrimary">Zone Occupancy Dashboard</h2>
        </div>
        <ZoneDashboard />
      </div>

      {/* 3. QR Code Management */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-1 h-8 bg-accent rounded-full"></div>
          <h2 className="text-2xl font-bold text-textPrimary">QR Code Generation</h2>
        </div>
        <QRCodeGenerator />
      </div>

      {/* 4. Bulk Import */}
      <div>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-1 h-8 bg-accent rounded-full"></div>
          <h2 className="text-2xl font-bold text-textPrimary">Bulk Attendee Import</h2>
        </div>
        <BulkImport />
      </div>
    </div>
  );
};

export default CheckInSection;
