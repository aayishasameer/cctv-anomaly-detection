#!/usr/bin/env python3
"""
Complete Dual Window CCTV System
Real-time video + Detailed anomaly information display
Integrates: Stealing Detection, ReID, Adaptive Zones, Anomaly Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from improved_reid_system import ImprovedReIDTracker
from vae_anomaly_detector import AnomalyDetector, FeatureExtractor
from typing import Dict, List, Tuple, Optional
import time
import json
import os
from collections import defaultdict, deque
from datetime import datetime

class DualWindowCCTVSystem:
    """Complete CCTV system with dual window display"""
    
    def __init__(self, camera_id: str = "camera_1"):
        self.camera_id = camera_id
        
        print(f"🚀 Initializing Complete Dual Window CCTV System")
        print("=" * 70)
        
        # Initialize all components
        print("📹 Loading YOLO person detection...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        print("🔍 Loading improved ReID tracker...")
        self.reid_tracker = ImprovedReIDTracker()
        
        print("🧠 Loading VAE anomaly detector...")
        try:
            self.anomaly_detector = AnomalyDetector()
            self.anomaly_detector.load_model()  # Fixed: no argument needed
            self.feature_extractor = FeatureExtractor()
            self.anomaly_enabled = True
            print("✅ VAE anomaly detector loaded")
        except Exception as e:
            print(f"⚠️  VAE not available: {e}")
            self.anomaly_detector = None
            self.feature_extractor = None
            self.anomaly_enabled = False
        
        print("🎯 Loading adaptive interaction zones...")
        self.interaction_zones = self._load_interaction_zones()
        
        # Behavior tracking
        self.person_behaviors = defaultdict(lambda: {
            'positions': deque(maxlen=90),
            'timestamps': deque(maxlen=90),
            'speeds': deque(maxlen=90),
            'zone_time': 0.0,
            'first_seen': None,
            'anomaly_scores': deque(maxlen=30),
            'risk_history': deque(maxlen=30),
            'alerts': []
        })
        
        # System statistics
        self.stats = {
            'frames_processed': 0,
            'total_persons': 0,
            'active_persons': 0,
            'stealing_alerts': 0,
            'high_risk_alerts': 0,
            'suspicious_alerts': 0,
            'reid_matches': 0,
            'start_time': time.time()
        }
        
        # Alert log
        self.alert_log = deque(maxlen=50)
        
        # Display settings
        self.info_panel_width = 600
        self.info_panel_height = 720
        
        print("✅ Complete system initialized!")
        print(f"   🎥 Camera ID: {camera_id}")
        print(f"   🔍 ReID: Improved consistency tracking")
        print(f"   🧠 Anomaly Detection: {'Enabled' if self.anomaly_enabled else 'Disabled'}")
        print(f"   🎯 Adaptive Zones: {len(self.interaction_zones)} zones loaded")
        print(f"   🖥️  Dual Window: Real-time + Info Panel")
    
    def _load_interaction_zones(self) -> List[Dict]:
        """Load learned interaction zones"""
        zones_path = "models/learned_interaction_zones.json"
        
        if os.path.exists(zones_path):
            try:
                with open(zones_path, 'r') as f:
                    data = json.load(f)
                zones = data.get('zones', [])
                print(f"✅ Loaded {len(zones)} learned interaction zones")
                return zones
            except Exception as e:
                print(f"⚠️  Could not load zones: {e}")
        
        # Default zone
        return [{
            'id': 'default_zone',
            'bbox': [100, 100, 540, 380],
            'center': [320, 240],
            'density': 0.5
        }]
    
    def check_zone_interaction(self, bbox: List[float]) -> Tuple[bool, Optional[str]]:
        """Check if person is in interaction zone"""
        person_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        for zone in self.interaction_zones:
            zone_bbox = zone['bbox']
            if (zone_bbox[0] <= person_center[0] <= zone_bbox[2] and
                zone_bbox[1] <= person_center[1] <= zone_bbox[3]):
                return True, zone['id']
        
        return False, None
    
    def analyze_behavior(self, global_id: int, timestamp: float) -> Dict:
        """Comprehensive behavior analysis"""
        behavior = self.person_behaviors[global_id]
        
        if behavior['first_seen'] is None:
            return {'risk_level': 'unknown', 'score': 0.0, 'reasons': []}
        
        risk_score = 0.0
        reasons = []
        
        # Duration analysis
        duration = timestamp - behavior['first_seen']
        
        # Loitering detection
        if duration > 5.0 and len(behavior['positions']) > 30:
            positions = np.array(list(behavior['positions']))
            movement_range = np.ptp(positions, axis=0).sum()
            
            if movement_range < 100:
                risk_score += 0.3
                reasons.append(f"Loitering ({duration:.1f}s)")
        
        # Speed analysis
        if len(behavior['speeds']) > 5:
            speeds = list(behavior['speeds'])
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            if max_speed > 100:
                risk_score += 0.2
                reasons.append(f"Rapid movement")
            
            if len(speeds) > 10:
                speed_var = np.var(speeds)
                if speed_var > 500:
                    risk_score += 0.15
                    reasons.append("Erratic movement")
        
        # Zone interaction
        if behavior['zone_time'] > 3.0:
            risk_score += 0.25
            reasons.append(f"Extended zone presence ({behavior['zone_time']:.1f}s)")
        
        # Anomaly score integration
        if len(behavior['anomaly_scores']) > 0:
            avg_anomaly = np.mean(list(behavior['anomaly_scores']))
            if avg_anomaly > 0.6:
                risk_score += 0.3
                reasons.append(f"High anomaly score ({avg_anomaly:.2f})")
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = 'STEALING'
        elif risk_score >= 0.6:
            risk_level = 'HIGH_RISK'
        elif risk_score >= 0.4:
            risk_level = 'SUSPICIOUS'
        else:
            risk_level = 'NORMAL'
        
        return {
            'risk_level': risk_level,
            'score': min(risk_score, 1.0),
            'reasons': reasons,
            'duration': duration
        }
    
    def create_info_panel(self, frame_idx: int, fps: int, active_persons: Dict) -> np.ndarray:
        """Create detailed information panel"""
        
        panel = np.zeros((self.info_panel_height, self.info_panel_width, 3), dtype=np.uint8)
        panel.fill(30)  # Dark background
        
        y_offset = 20
        line_height = 25
        
        # Title
        cv2.putText(panel, "CCTV ANOMALY DETECTION SYSTEM", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
        
        # Draw separator
        cv2.line(panel, (10, y_offset), (590, y_offset), (100, 100, 100), 2)
        y_offset += 20
        
        # System Status
        cv2.putText(panel, "SYSTEM STATUS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        runtime = time.time() - self.stats['start_time']
        status_info = [
            f"Frame: {frame_idx}",
            f"Runtime: {runtime:.1f}s",
            f"FPS: {fps}",
            f"Active Persons: {len(active_persons)}",
            f"Total Detected: {self.stats['total_persons']}",
            f"ReID Matches: {self.reid_tracker.get_statistics()['match_rate']:.1%}"
        ]
        
        for info in status_info:
            cv2.putText(panel, info, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height - 5
        
        y_offset += 10
        cv2.line(panel, (10, y_offset), (590, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        # Alert Summary
        cv2.putText(panel, "ALERT SUMMARY", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        alert_summary = [
            (f"Stealing: {self.stats['stealing_alerts']}", (0, 0, 255)),
            (f"High Risk: {self.stats['high_risk_alerts']}", (0, 165, 255)),
            (f"Suspicious: {self.stats['suspicious_alerts']}", (0, 255, 255))
        ]
        
        for text, color in alert_summary:
            cv2.putText(panel, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height - 5
        
        y_offset += 10
        cv2.line(panel, (10, y_offset), (590, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        # Active Persons Details
        cv2.putText(panel, "ACTIVE PERSONS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        for global_id, person_data in list(active_persons.items())[:8]:  # Show max 8
            risk_level = person_data['risk_level']
            risk_score = person_data['risk_score']
            
            # Color based on risk
            if risk_level == 'STEALING':
                color = (0, 0, 255)
                emoji = "🔴"
            elif risk_level == 'HIGH_RISK':
                color = (0, 165, 255)
                emoji = "🟠"
            elif risk_level == 'SUSPICIOUS':
                color = (0, 255, 255)
                emoji = "🟡"
            else:
                color = (0, 255, 0)
                emoji = "🟢"
            
            # Person header
            cv2.putText(panel, f"ID {global_id}: {risk_level}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height - 5
            
            # Risk score bar
            bar_x = 30
            bar_y = y_offset - 10
            bar_width = 200
            bar_height = 15
            
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            fill_width = int(bar_width * risk_score)
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
            cv2.putText(panel, f"{risk_score:.2f}", (bar_x + bar_width + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y_offset += line_height
            
            # Reasons
            reasons = person_data.get('reasons', [])
            if reasons:
                reason_text = ", ".join(reasons[:2])
                cv2.putText(panel, f"  {reason_text}", (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                y_offset += line_height - 10
            
            y_offset += 5
        
        # Recent Alerts Log
        if y_offset < self.info_panel_height - 200:
            y_offset += 10
            cv2.line(panel, (10, y_offset), (590, y_offset), (100, 100, 100), 1)
            y_offset += 20
            
            cv2.putText(panel, "RECENT ALERTS", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += line_height
            
            for alert in list(self.alert_log)[-5:]:
                alert_text = f"{alert['time']} - ID{alert['id']}: {alert['type']}"
                color = (0, 0, 255) if alert['type'] == 'STEALING' else (0, 165, 255)
                cv2.putText(panel, alert_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += line_height - 10
        
        # Footer
        footer_y = self.info_panel_height - 30
        cv2.line(panel, (10, footer_y - 10), (590, footer_y - 10), (100, 100, 100), 1)
        cv2.putText(panel, f"Camera: {self.camera_id} | {datetime.now().strftime('%H:%M:%S')}",
                   (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return panel
    
    def process_video_dual_window(self, video_path: str, output_path: str = None):
        """Process video with dual window display"""
        
        print(f"\n🎬 Processing: {os.path.basename(video_path)}")
        print("=" * 70)
        print("🖥️  Dual Window Mode:")
        print("   Left: Real-time video with detections")
        print("   Right: Detailed anomaly information")
        print("\nPress 'q' to quit, 'SPACE' to pause")
        print("=" * 70)
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output if needed
        if output_path:
            combined_width = width + self.info_panel_width
            combined_height = max(height, self.info_panel_height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
        
        frame_idx = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_idx / fps
                    
                    # Detect and track persons
                    results = self.yolo_model.track(
                        source=frame,
                        tracker="botsort.yaml",
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        verbose=False
                    )
                    
                    # Prepare detections
                    detections = []
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        confidences = results[0].boxes.conf.cpu().numpy()
                        
                        for box, track_id, conf in zip(boxes, track_ids, confidences):
                            detections.append({
                                'track_id': track_id,
                                'bbox': box.tolist(),
                                'confidence': float(conf)
                            })
                    
                    # Update ReID tracker
                    track_mapping = self.reid_tracker.update(frame, detections, timestamp)
                    
                    # Process each person
                    active_persons = {}
                    
                    for detection in detections:
                        track_id = detection['track_id']
                        bbox = detection['bbox']
                        
                        # Get consistent global ID
                        global_id = track_mapping.get(track_id, track_id)
                        
                        # Update statistics
                        if global_id not in self.person_behaviors or \
                           self.person_behaviors[global_id]['first_seen'] is None:
                            self.stats['total_persons'] += 1
                        
                        # Calculate center
                        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        
                        # Check zone interaction
                        in_zone, zone_id = self.check_zone_interaction(bbox)
                        
                        # Update behavior tracking
                        behavior = self.person_behaviors[global_id]
                        
                        if behavior['first_seen'] is None:
                            behavior['first_seen'] = timestamp
                        
                        behavior['positions'].append(center)
                        behavior['timestamps'].append(timestamp)
                        
                        # Calculate speed
                        if len(behavior['positions']) > 1:
                            prev_pos = behavior['positions'][-2]
                            speed = np.linalg.norm(np.array(center) - np.array(prev_pos))
                            behavior['speeds'].append(speed)
                        
                        # Update zone time
                        if in_zone:
                            behavior['zone_time'] += 1.0 / fps
                        
                        # Get anomaly score
                        anomaly_score = 0.0
                        if self.anomaly_enabled and self.feature_extractor:
                            try:
                                features = self.feature_extractor.extract_features(
                                    global_id, bbox, frame_idx
                                )
                                if features is not None:
                                    anomaly_score = self.anomaly_detector.detect_anomaly(features)
                                    behavior['anomaly_scores'].append(anomaly_score)
                            except:
                                pass
                        
                        # Analyze behavior
                        analysis = self.analyze_behavior(global_id, timestamp)
                        
                        # Combined risk
                        combined_risk = (0.6 * analysis['score'] + 0.4 * anomaly_score)
                        risk_level = analysis['risk_level']
                        
                        # Update statistics
                        if risk_level == 'STEALING':
                            self.stats['stealing_alerts'] += 1
                            self.alert_log.append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'id': global_id,
                                'type': 'STEALING'
                            })
                        elif risk_level == 'HIGH_RISK':
                            self.stats['high_risk_alerts'] += 1
                        elif risk_level == 'SUSPICIOUS':
                            self.stats['suspicious_alerts'] += 1
                        
                        # Store for display
                        active_persons[global_id] = {
                            'bbox': bbox,
                            'risk_level': risk_level,
                            'risk_score': combined_risk,
                            'reasons': analysis['reasons'],
                            'in_zone': in_zone
                        }
                        
                        # Draw on frame
                        x1, y1, x2, y2 = [int(c) for c in bbox]
                        
                        # Color based on risk
                        if risk_level == 'STEALING':
                            color = (0, 0, 255)
                        elif risk_level == 'HIGH_RISK':
                            color = (0, 165, 255)
                        elif risk_level == 'SUSPICIOUS':
                            color = (0, 255, 255)
                        else:
                            color = (0, 255, 0)
                        
                        thickness = 3 if risk_level in ['STEALING', 'HIGH_RISK'] else 2
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label
                        label = f"ID:{global_id} | {risk_level}"
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw risk bar
                        bar_width = x2 - x1
                        bar_height = 8
                        bar_fill = int(bar_width * combined_risk)
                        cv2.rectangle(frame, (x1, y2+5), (x1+bar_width, y2+5+bar_height),
                                    (100, 100, 100), -1)
                        cv2.rectangle(frame, (x1, y2+5), (x1+bar_fill, y2+5+bar_height),
                                    color, -1)
                        
                        # Draw zone indicator
                        if in_zone:
                            cv2.circle(frame, (x2-10, y1+10), 5, (255, 0, 255), -1)
                    
                    # Draw zones
                    for zone in self.interaction_zones:
                        zx1, zy1, zx2, zy2 = [int(c) for c in zone['bbox']]
                        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 2)
                        cv2.putText(frame, zone['id'], (zx1, zy1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    # Update stats
                    self.stats['frames_processed'] = frame_idx
                    self.stats['active_persons'] = len(active_persons)
                    
                    frame_idx += 1
                
                # Create info panel
                info_panel = self.create_info_panel(frame_idx, fps, active_persons)
                
                # Resize frame to match info panel height if needed
                if frame.shape[0] != self.info_panel_height:
                    scale = self.info_panel_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    frame_resized = cv2.resize(frame, (new_width, self.info_panel_height))
                else:
                    frame_resized = frame
                
                # Combine frames
                combined = np.hstack([frame_resized, info_panel])
                
                # Display
                cv2.imshow('Complete CCTV System - Dual Window', combined)
                
                # Save if output specified
                if output_path and not paused:
                    out.write(combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final statistics"""
        print(f"\n🎉 PROCESSING COMPLETED!")
        print("=" * 70)
        print(f"📊 FINAL STATISTICS:")
        print(f"   Frames processed: {self.stats['frames_processed']}")
        print(f"   Total persons detected: {self.stats['total_persons']}")
        print(f"   Stealing alerts: {self.stats['stealing_alerts']}")
        print(f"   High risk alerts: {self.stats['high_risk_alerts']}")
        print(f"   Suspicious alerts: {self.stats['suspicious_alerts']}")
        
        reid_stats = self.reid_tracker.get_statistics()
        print(f"\n🔍 REID STATISTICS:")
        print(f"   Total detections: {reid_stats['total_detections']}")
        print(f"   ReID matches: {reid_stats['reid_matches']}")
        print(f"   Match rate: {reid_stats['match_rate']:.2%}")
        print(f"   New IDs created: {reid_stats['new_ids']}")
        print(f"   Active tracks: {reid_stats['active_tracks']}")

def main():
    """Run complete dual window system"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Dual Window CCTV System')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--camera-id', '-c', default='camera_1', help='Camera ID')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Video not found: {args.input}")
        return
    
    # Initialize system
    system = DualWindowCCTVSystem(camera_id=args.camera_id)
    
    # Process video
    system.process_video_dual_window(
        video_path=args.input,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
