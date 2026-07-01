import React, { useState } from 'react';
import { QrCode, Download, Copy } from 'lucide-react';

interface Zone {
  id: string;
  name: string;
  capacity: number;
}

interface QRCodeGeneratorProps {
  value?: string;
  size?: number;
}

const zones: Zone[] = [
  { id: 'ZONE-A', name: 'Main Entrance', capacity: 5000 },
  { id: 'ZONE-B', name: 'VIP Section', capacity: 500 },
  { id: 'ZONE-C', name: 'General Area', capacity: 10000 },
  { id: 'ZONE-D', name: 'Food Court', capacity: 2000 },
];

const QRCodeGenerator: React.FC<QRCodeGeneratorProps> = ({ value, size = 200 }) => {
  const [eventId] = useState('EVT-2024-001');
  const [copiedZone, setCopiedZone] = useState<string | null>(null);

  const generateQRUrl = (zoneId: string, zoneName: string) => {
    const baseUrl = window.location.origin;
    return `${baseUrl}/check-in?eventId=${eventId}&zoneId=${zoneId}&zoneName=${encodeURIComponent(zoneName)}`;
  };

  const getQRCodeImageUrl = (zoneId: string, zoneName: string) => {
    const url = generateQRUrl(zoneId, zoneName);
    return `https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=${encodeURIComponent(url)}`;
  };

  const copyToClipboard = (zoneId: string, zoneName: string) => {
    const url = generateQRUrl(zoneId, zoneName);
    navigator.clipboard.writeText(url);
    setCopiedZone(zoneId);
    setTimeout(() => setCopiedZone(null), 2000);
  };

  const downloadQRCode = (zoneId: string, zoneName: string) => {
    const qrUrl = getQRCodeImageUrl(zoneId, zoneName);
    const link = document.createElement('a');
    link.href = qrUrl;
    link.download = `${zoneId}-QRCode.png`;
    link.click();
  };

  // If value prop is provided, render a simple QR code image
  if (value) {
    const qrImageUrl = `https://api.qrserver.com/v1/create-qr-code/?size=${size}x${size}&data=${encodeURIComponent(value)}`;
    return (
      <img
        src={qrImageUrl}
        alt="QR Code"
        width={size}
        height={size}
      />
    );
  }

  return (
    <div className="bg-surface rounded-xl p-6 border border-surfaceHover">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
          <QrCode className="w-6 h-6 text-black" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-textPrimary">Zone QR Codes</h2>
          <p className="text-textSecondary text-sm">Generate and manage entry QR codes</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {zones.map((zone) => (
          <div key={zone.id} className="bg-base rounded-lg p-4 border border-surfaceHover">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-textPrimary">{zone.name}</h3>
                <p className="text-sm text-textSecondary">
                  {zone.id} • Capacity: {zone.capacity.toLocaleString()}
                </p>
              </div>
            </div>

            {/* QR Code Display — white background kept intentionally: QR codes require high contrast */}
            <div className="bg-white p-4 rounded-lg mb-4 flex items-center justify-center">
              <img
                src={getQRCodeImageUrl(zone.id, zone.name)}
                alt={`${zone.name} QR Code`}
                className="w-48 h-48"
              />
            </div>

            {/* URL Display */}
            <div className="bg-surfaceHover rounded p-3 mb-3">
              <p className="text-xs text-textSecondary break-all">
                {generateQRUrl(zone.id, zone.name)}
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => copyToClipboard(zone.id, zone.name)}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-accent hover:bg-accent/80 text-black font-semibold rounded-lg transition-colors"
              >
                <Copy className="w-4 h-4" />
                {copiedZone === zone.id ? 'Copied!' : 'Copy URL'}
              </button>
              <button
                onClick={() => downloadQRCode(zone.id, zone.name)}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-surfaceHover hover:bg-surfaceHover/70 text-textPrimary rounded-lg font-medium transition-colors border border-surfaceHover"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 bg-base rounded-lg p-4 border border-surfaceHover">
        <h4 className="text-textPrimary font-medium mb-2">Instructions:</h4>
        <ol className="text-sm text-textSecondary space-y-1 list-decimal list-inside">
          <li>Download or print QR codes for each zone</li>
          <li>Display QR codes at zone entrances</li>
          <li>Attendees scan QR code with their phone</li>
          <li>They enter their ticket ID to check in</li>
          <li>System tracks zone-wise attendance automatically</li>
        </ol>
      </div>
    </div>
  );
};

export default QRCodeGenerator;
