import { v4 as uuidv4 } from 'uuid';

export interface Zone {
  id: string;
  name: string;
  capacity: number;
  currentAttendees: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  percentFull: number;
  qrCode: string;
  location: string;
  entrances: string[];
}

export interface ZoneEvent {
  id: string;
  userId: string;
  zoneId: string;
  action: 'check-in' | 'check-out';
  timestamp: Date;
  location?: string;
}

export interface User {
  id: string;
  currentZone?: string;
  checkInHistory: ZoneEvent[];
}

class ZoneService {
  private zones: Map<string, Zone> = new Map();
  private users: Map<string, User> = new Map();
  private eventHistory: ZoneEvent[] = [];
  private listeners: ((zones: Zone[]) => void)[] = [];

  constructor() {
    this.initializeZones();
    this.simulateRealTimeActivity();
  }

  private initializeZones() {
    const initialZones: Omit<Zone, 'currentAttendees' | 'riskLevel' | 'percentFull' | 'qrCode'>[] = [
      {
        id: 'main_stage',
        name: 'Main Stage',
        capacity: 25000,
        location: 'Central Arena',
        entrances: ['North Gate', 'South Gate', 'VIP Entrance']
      },
      {
        id: 'food_court',
        name: 'Food Court',
        capacity: 8000,
        location: 'East Wing',
        entrances: ['East Gate', 'Food Court Bridge']
      },
      {
        id: 'west_gate',
        name: 'West Gate',
        capacity: 5000,
        location: 'West Entrance',
        entrances: ['West Gate Main', 'West Gate Secondary']
      },
      {
        id: 'vip_area',
        name: 'VIP Area',
        capacity: 1000,
        location: 'Premium Section',
        entrances: ['VIP Entrance', 'VIP Lounge']
      },
      {
        id: 'parking_lot_a',
        name: 'Parking Lot A',
        capacity: 3000,
        location: 'North Parking',
        entrances: ['Parking Entry A1', 'Parking Entry A2']
      },
      {
        id: 'emergency_exit_1',
        name: 'Emergency Exit 1',
        capacity: 2000,
        location: 'South Emergency',
        entrances: ['Emergency Exit 1']
      }
    ];

    initialZones.forEach(zoneData => {
      const zone: Zone = {
        ...zoneData,
        currentAttendees: Math.floor(Math.random() * zoneData.capacity * 0.8),
        riskLevel: 'LOW',
        percentFull: 0,
        qrCode: this.generateQRCodeData(zoneData.id, zoneData.name)
      };
      
      this.updateZoneMetrics(zone);
      this.zones.set(zone.id, zone);
    });
  }

  private generateQRCodeData(zoneId: string, zoneName: string): string {
    return JSON.stringify({
      type: 'echoplex_zone',
      zoneId,
      zoneName,
      timestamp: Date.now(),
      version: '1.0'
    });
  }

  private updateZoneMetrics(zone: Zone) {
    zone.percentFull = Math.round((zone.currentAttendees / zone.capacity) * 100);
    
    if (zone.percentFull >= 95) {
      zone.riskLevel = 'HIGH';
    } else if (zone.percentFull >= 80) {
      zone.riskLevel = 'MEDIUM';
    } else {
      zone.riskLevel = 'LOW';
    }
  }

  private simulateRealTimeActivity() {
    setInterval(() => {
      // Simulate random check-ins and check-outs
      if (Math.random() < 0.3) {
        const zones = Array.from(this.zones.values());
        const randomZone = zones[Math.floor(Math.random() * zones.length)];
        const action = Math.random() > 0.6 ? 'check-in' : 'check-out';
        const userId = `sim_user_${Math.floor(Math.random() * 1000)}`;
        
        this.processZoneEvent(userId, randomZone.id, action);
      }
    }, 2000);
  }

  public processZoneEvent(userId: string, zoneId: string, action: 'check-in' | 'check-out'): boolean {
    const zone = this.zones.get(zoneId);
    if (!zone) return false;

    let user = this.users.get(userId);
    if (!user) {
      user = {
        id: userId,
        checkInHistory: []
      };
      this.users.set(userId, user);
    }

    // Prevent duplicate check-ins
    if (action === 'check-in' && user.currentZone === zoneId) {
      return false;
    }

    // Prevent check-out if not in zone
    if (action === 'check-out' && user.currentZone !== zoneId) {
      return false;
    }

    const event: ZoneEvent = {
      id: uuidv4(),
      userId,
      zoneId,
      action,
      timestamp: new Date()
    };

    // Update zone occupancy
    if (action === 'check-in') {
      zone.currentAttendees = Math.min(zone.capacity, zone.currentAttendees + 1);
      user.currentZone = zoneId;
    } else {
      zone.currentAttendees = Math.max(0, zone.currentAttendees - 1);
      user.currentZone = undefined;
    }

    // Update metrics
    this.updateZoneMetrics(zone);

    // Store event
    this.eventHistory.push(event);
    user.checkInHistory.push(event);

    // Notify listeners
    this.notifyListeners();

    return true;
  }

  public processQRScan(qrData: string, userId: string): { success: boolean; message: string; zone?: Zone } {
    try {
      const data = JSON.parse(qrData);
      
      if (data.type !== 'echoplex_zone') {
        return { success: false, message: 'Invalid QR code format' };
      }

      const zone = this.zones.get(data.zoneId);
      if (!zone) {
        return { success: false, message: 'Zone not found' };
      }

      const user = this.users.get(userId);
      const isCurrentlyInZone = user?.currentZone === data.zoneId;
      const action = isCurrentlyInZone ? 'check-out' : 'check-in';

      const success = this.processZoneEvent(userId, data.zoneId, action);
      
      if (success) {
        return {
          success: true,
          message: `Successfully ${action === 'check-in' ? 'checked into' : 'checked out of'} ${zone.name}`,
          zone
        };
      } else {
        return { success: false, message: 'Failed to process zone event' };
      }
    } catch (error) {
      return { success: false, message: 'Invalid QR code data' };
    }
  }

  public getZones(): Zone[] {
    return Array.from(this.zones.values());
  }

  public getZone(zoneId: string): Zone | undefined {
    return this.zones.get(zoneId);
  }

  public getUserCurrentZone(userId: string): Zone | undefined {
    const user = this.users.get(userId);
    if (!user?.currentZone) return undefined;
    return this.zones.get(user.currentZone);
  }

  public getZoneEvents(zoneId?: string, limit: number = 50): ZoneEvent[] {
    let events = this.eventHistory;
    
    if (zoneId) {
      events = events.filter(event => event.zoneId === zoneId);
    }
    
    return events
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  public subscribe(callback: (zones: Zone[]) => void) {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback);
    };
  }

  private notifyListeners() {
    const zones = this.getZones();
    this.listeners.forEach(callback => callback(zones));
  }

  // Analytics methods
  public getZoneAnalytics(zoneId: string, hours: number = 24) {
    const zone = this.zones.get(zoneId);
    if (!zone) return null;

    const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);
    const recentEvents = this.eventHistory.filter(
      event => event.zoneId === zoneId && event.timestamp >= cutoffTime
    );

    const checkIns = recentEvents.filter(e => e.action === 'check-in').length;
    const checkOuts = recentEvents.filter(e => e.action === 'check-out').length;

    return {
      zone,
      checkIns,
      checkOuts,
      netFlow: checkIns - checkOuts,
      peakOccupancy: Math.max(zone.currentAttendees, ...recentEvents.map(() => zone.currentAttendees)),
      averageOccupancy: zone.currentAttendees * 0.8, // Simplified calculation
      events: recentEvents
    };
  }
}

export const zoneService = new ZoneService();