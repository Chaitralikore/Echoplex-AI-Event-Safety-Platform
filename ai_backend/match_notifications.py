"""
Match Notification System
Real-time WebSocket notifications for high-confidence matches

Features:
- WebSocket manager for connected clients
- Match notification trigger with confidence threshold
- Cosine similarity comparison for Super-Vectors
- Event broadcasting to all connected frontends
"""

import asyncio
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import numpy as np
from datetime import datetime
import time


class MatchNotification:
    """Represents a match notification to be sent to frontend"""
    
    def __init__(self, 
                 case_id: str,
                 case_name: str,
                 confidence: float,
                 confidence_breakdown: Dict[str, float],
                 location: str,
                 timestamp: float,
                 frame_number: int,
                 super_vector_used: bool = False,
                 metadata: Dict[str, Any] = None):
        self.case_id = case_id
        self.case_name = case_name
        self.confidence = confidence
        self.confidence_breakdown = confidence_breakdown
        self.location = location
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.super_vector_used = super_vector_used
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.notification_id = f"match_{int(time.time() * 1000)}_{case_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "POTENTIAL_MATCH",
            "notification_id": self.notification_id,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "location": self.location,
            "video_timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "super_vector_used": self.super_vector_used,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


class WebSocketManager:
    """Manages WebSocket connections for real-time notifications"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat()
        }
        print(f"[WS] Client connected: {self.connection_metadata[websocket]['client_id']}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "client_id": self.connection_metadata[websocket]['client_id'],
            "message": "Connected to Echoplex Real-time Match Notifications"
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket"""
        if websocket in self.active_connections:
            client_id = self.connection_metadata.get(websocket, {}).get('client_id', 'unknown')
            self.active_connections.discard(websocket)
            self.connection_metadata.pop(websocket, None)
            print(f"[WS] Client disconnected: {client_id}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"[WS] Error sending to client: {e}")
                disconnected.add(connection)
        
        # Clean up failed connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_match_notification(self, notification: MatchNotification):
        """Send a match notification to all clients"""
        await self.broadcast(notification.to_dict())
        print(f"[WS] Broadcasted match notification: {notification.case_name} ({notification.confidence}%)")
    
    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


class MatchNotificationTrigger:
    """
    Triggers match notifications when Super-Vector matches exceed threshold
    
    Compares Active Report vectors against CCTV Super-Vectors using
    Cosine Similarity and triggers notifications for high-confidence matches.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.85,
                 websocket_manager: WebSocketManager = None):
        """
        Initialize match notification trigger
        
        Args:
            confidence_threshold: Minimum confidence to trigger notification (0-1)
            websocket_manager: WebSocket manager for sending notifications
        """
        self.confidence_threshold = confidence_threshold
        self.ws_manager = websocket_manager or WebSocketManager()
        
        # Track recent notifications to avoid duplicates
        self.recent_notifications: Dict[str, float] = {}  # case_id -> last_notification_time
        self.notification_cooldown = 30.0  # seconds between notifications for same case
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1: First 512-d feature vector (reference photo)
            vec2: Second 512-d feature vector (CCTV Super-Vector)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize vectors (they should already be normalized)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Cosine similarity = dot product of normalized vectors
        similarity = float(np.dot(vec1_normalized, vec2_normalized))
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))
    
    async def check_and_notify(self,
                               case_id: str,
                               case_name: str,
                               reference_vector: np.ndarray,
                               super_vector: np.ndarray,
                               match_scores: Dict[str, float],
                               location: str,
                               video_timestamp: float,
                               frame_number: int,
                               metadata: Dict[str, Any] = None) -> Optional[MatchNotification]:
        """
        Check if match exceeds threshold and trigger notification
        
        Args:
            case_id: Unique ID of the missing person case
            case_name: Name of the missing person
            reference_vector: 512-d vector from reference photo
            super_vector: 512-d Super-Vector from CCTV (aggregated from multiple frames)
            match_scores: Breakdown of matching scores (reid, color, etc.)
            location: Camera/location ID
            video_timestamp: Time in video where match occurred
            frame_number: Frame number of detection
            metadata: Additional case metadata
            
        Returns:
            MatchNotification if triggered, None otherwise
        """
        # Compute final cosine similarity with Super-Vector
        similarity = self.cosine_similarity(reference_vector, super_vector)
        
        # Build confidence breakdown
        confidence_breakdown = {
            "overall_match": round(similarity * 100, 1),
            "reid_features": round(match_scores.get('reid_score', similarity) * 100, 1),
            "upper_clothing": round(match_scores.get('upper_color_score', 0.5) * 100, 1),
            "lower_clothing": round(match_scores.get('lower_color_score', 0.5) * 100, 1),
            "body_shape": round(match_scores.get('body_ratio_score', 0.5) * 100, 1),
            "super_vector_boost": "Active" if similarity > match_scores.get('reid_score', 0) else "N/A"
        }
        
        # Combined confidence (weighted)
        combined_confidence = (
            match_scores.get('reid_score', similarity) * 0.5 +
            match_scores.get('upper_color_score', 0.5) * 0.2 +
            match_scores.get('lower_color_score', 0.5) * 0.15 +
            match_scores.get('body_ratio_score', 0.5) * 0.15
        )
        
        # Boost for Super-Vector (aggregated from multiple frames = higher confidence)
        combined_confidence = min(1.0, combined_confidence * 1.1)
        
        # Check threshold
        if combined_confidence < self.confidence_threshold:
            return None
        
        # Check cooldown (avoid spam)
        current_time = time.time()
        last_notification = self.recent_notifications.get(case_id, 0)
        if current_time - last_notification < self.notification_cooldown:
            return None
        
        # Create notification
        notification = MatchNotification(
            case_id=case_id,
            case_name=case_name,
            confidence=round(combined_confidence * 100, 1),
            confidence_breakdown=confidence_breakdown,
            location=location,
            timestamp=video_timestamp,
            frame_number=frame_number,
            super_vector_used=True,
            metadata=metadata
        )
        
        # Update cooldown tracker
        self.recent_notifications[case_id] = current_time
        
        # Send via WebSocket
        if self.ws_manager:
            await self.ws_manager.send_match_notification(notification)
        
        return notification
    
    def update_threshold(self, new_threshold: float):
        """Update the confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        print(f"[TRIGGER] Updated confidence threshold to {self.confidence_threshold}")


# Global instances
websocket_manager = WebSocketManager()
match_trigger = MatchNotificationTrigger(
    confidence_threshold=0.85,
    websocket_manager=websocket_manager
)
