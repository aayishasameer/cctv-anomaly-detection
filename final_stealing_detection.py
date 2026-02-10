#!/usr/bin/env python3
"""
Final Stealing Detection System with Consistent ReID
Combines improved person tracking with stealing behavior detection
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

class StealingBehaviorAnalyzer:
    """Analyze stealing behavior patterns"""
    
    def __init__(self):
        # Behavior thresholds
        self.loitering_threshold = 5.0  # seconds
        self.rapid_movement_threshold = 100  # pixels per frame
        self.zone_interaction_threshold = 3.0  # seconds
        
        # Track behavior history
        self.person_behaviors = defaultdict(lambda: {
            'positions': deque(maxlen=90),  # 3 seconds at 30fps
            'timestamps': deque(maxlen=90),
            'speeds': deque(maxlen=90),
            'zone_interactions': 0,
            'suspicious_actions': 0,
            'first_seen': None,
            'last_zone_time': None
        })
    
    def update_behavior(self, global_id: int, position: Tuple[float, float], 
                       timestamp: float, in_zone: bool = False):
        """Update behavior tracking for a person"""
        
        behavior = self.person_behaviors[global_id]
        
        if behavior['first_seen'] is None:
            behavior['first_seen'] = timestamp
        
        # Add current data
        behavior['positions'].append(position)
        behavior['timestamps'].append(timestamp)
        
        # Calculate speed if we have previous position
        if len(behavior['positions']) > 1:
            prev_pos = behavior['positions'][-2]
            curr_pos = behavior['positions'][-1]
            speed = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            behavior['speeds'].append(speed)
        
        # Track zone interactions
        if in_zone:
            if behavior['last_zone_time'] is None:
                behavior['last_zone_time'] = timestamp
            behavior['zone_interactions'] += 1
    
    def analyze_person(self, global_id: int, timestamp: float) -> Dict:
        """Analyze person's behavior and return risk assessment"""
        
        behavior = self.person_behaviors[global_id]
        
        if behavior['first_seen'] is None:
            return {'risk_level': 'unknown', 'score': 0.0, 'reasons': []}
        
        risk_score = 0.0
        reasons = []
        
        # Calculate duration
        duration = timestamp - behavior['first_seen']
        
        # Check loitering
        if duration > self.loitering_threshold:
            if len(behavior['positions']) > 0:
                # Check if person is relatively stationary
                positions = np.array(list(behavior['positions']))
                movement_range = np.ptp(positions, axis=0).sum()
                
                if movement_range < 100:  # Less than 100 pixels total movement
                    risk_score += 0.3
                    reasons.append(f"Loitering ({duration:.1f}s)")
        
        # Check rapid movements
        if len(behavior['speeds']) > 0:
            avg_speed = np.mean(list(behavior['speeds']))
            max_speed = np.max(list(behavior['speeds']))
            
            if max_speed > self.rapid_movement_threshold:
                risk_score += 0.2
                reasons.append(f"Rapid movement (speed: {max_speed:.1f})")
            
            # Check for erratic movement (high speed variance)
            if len(behavior['speeds']) > 10:
                speed_variance = np.var(list(behavior['speeds']))
                if speed_variance > 500:
                    risk_score += 0.15
                    reasons.append("Erratic movement pattern")
        
        # Check zone interactions
        if behavior['zone_interactions'] > 10:
            risk_score += 0.25
            reasons.append(f"Multiple zone interactions ({behavior['zone_interactions']})")
        
        # Check time in zone
        if behavior['last_zone_time'] is not None:
            zone_duration = timestamp - behavior['last_zone_time']
            if zone_duration > self.zone_interaction_threshold:
                risk_score += 0.2
                reasons.append(f"Extended zone presence ({zone_duration:.1f}s)")
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.5:
            risk_level = 'medium'
        elif risk_score >= 0.3:
            risk_level = 'low'
        else:
            risk_level = 'normal'
        
        return {
            'risk_level': risk_level,
            'score': min(risk_score, 1.0),
            'reasons': reasons,
            'duration': duration,
            'zone_interactions': behavior['zone_interactions']
        }

class FinalStealingDetectionSystem:
    """Complete stealing detection system with consistent ReID"""
    
    def __init__(self, camera_id: str = "camera_1"):
        self.camera_id = camera_id
        
        print(f"🚀 Initializing Final Stealing Detection System")
        print("=" * 60)
        
        # Initialize YOLO
        print("📹 Loading YOLO person detection...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Initialize improved ReID tracker
        print("🔍 Loading improved ReID tracker...")
        self.reid_tracker = ImprovedReIDTracker()
        
        # Initialize VAE anomaly detector
        print("🧠 Loading VAE anomaly detector...")
        try:
            self.anomaly_detector = AnomalyDetector()
            self.anomaly_detector.load_model("models/vae_anomaly_detector.pth")
            self.feature_extractor = FeatureExtractor()
            print("✅ VAE anomaly detector loaded")
        except Exception as e:
            print(f"⚠️  VAE not available: {e}")
            self.anomaly_detector = None
            self.feature_extractor = None
        
        # Initialize behavior analyzer
        print("🎯 Initializing behavior analyzer...")
        self.behavior_analyzer = StealingBehaviorAnalyzer()
        
        # Detection zones (simplified - can be loaded from learned zones)
        self.interaction_zones = self._load_interaction_zones()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'persons_detected': 0,
            'stealing_alerts': 0,
            'reid_consistency': []
        }
        
        print("✅ System initialized successfully!")
        print(f"   Camera ID: {camera_id}")
        print(f"   ReID: Improved tracker with consistency")
        print(f"   Anomaly Detection: {'Enabled' if self.anomaly_detector else 'Disabled'}")
        print(f"   Behavior Analysis: Enabled")
    
    def _load_interaction_zones(self) -> List[Dict]:
        """Load interaction zones"""
        zones_path = "models/learned_interaction_zones.json"
        
        if os.path.exists(zones_path):
            try:
                with open(zones_path, 'r') as f:
                    data = json.load(f)
                zones = data.get('zones', [])
                print(f"✅ Loaded {len(zones)} interaction zones")
                return zones
            except:
                pass
        
        # Default zone (center of frame)
        return [{
            'id': 'default_zone',
            'bbox': [100, 100, 540, 380],
            'center': [320, 240]
        }]
    
    def check_zone_interaction(self, bbox: List[float]) -> bool:
        """Check if person is in interaction zone"""
        
        person_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        for zone in self.interaction_zones:
            zone_bbox = zone['bbox']
            if (zone_bbox[0] <= person_center[0] <= zone_bbox[2] and
                zone_bbox[1] <= person_center[1] <= zone_bbox[3]):
                return True
        
        return False
    
    def process_video(self, video_path: str, output_path: str = None, 
                     display: bool = False) -> Dict:
        """Process video with stealing detection and consistent ReID"""
        
        print(f"\n🎬 Processing: {os.path.basename(video_path)}")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        # Tracking data
        stealing_alerts = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Detect and track persons
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.4,
                    verbose=False
                )
                
                # Prepare detections for ReID
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
                
                # Update ReID tracker - Get consistent global IDs
                track_mapping = self.reid_tracker.update(frame, detections, timestamp)
                
                # Process each person
                for detection in detections:
                    track_id = detection['track_id']
                    bbox = detection['bbox']
                    
                    # Get consistent global ID
                    global_id = track_mapping.get(track_id, track_id)
                    
                    # Calculate person center
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    
                    # Check zone interaction
                    in_zone = self.check_zone_interaction(bbox)
                    
                    # Update behavior
                    self.behavior_analyzer.update_behavior(
                        global_id, center, timestamp, in_zone
                    )
                    
                    # Analyze behavior
                    analysis = self.behavior_analyzer.analyze_person(global_id, timestamp)
                    
                    # Get anomaly score if available
                    anomaly_score = 0.0
                    if self.anomaly_detector and self.feature_extractor:
                        try:
                            features = self.feature_extractor.extract_features(
                                global_id, bbox, frame_idx
                            )
                            if features is not None:
                                anomaly_score = self.anomaly_detector.detect_anomaly(features)
                        except:
                            pass
                    
                    # Combined risk assessment
                    combined_risk = (0.6 * analysis['score'] + 0.4 * anomaly_score)
                    
                    # Determine final risk level
                    if combined_risk >= 0.7:
                        final_risk = 'STEALING'
                        color = (0, 0, 255)  # Red
                    elif combined_risk >= 0.5:
                        final_risk = 'HIGH RISK'
                        color = (0, 165, 255)  # Orange
                    elif combined_risk >= 0.3:
                        final_risk = 'SUSPICIOUS'
                        color = (0, 255, 255)  # Yellow
                    else:
                        final_risk = 'NORMAL'
                        color = (0, 255, 0)  # Green
                    
                    # Record stealing alerts
                    if final_risk in ['STEALING', 'HIGH RISK']:
                        stealing_alerts.append({
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'global_id': global_id,
                            'risk_level': final_risk,
                            'score': combined_risk,
                            'reasons': analysis['reasons']
                        })
                    
                    # Draw on frame
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    
                    # Draw bounding box
                    thickness = 3 if final_risk in ['STEALING', 'HIGH RISK'] else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw labels
                    label = f"ID:{global_id} | {final_risk}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw risk score bar
                    bar_width = x2 - x1
                    bar_height = 8
                    bar_fill = int(bar_width * combined_risk)
                    cv2.rectangle(frame, (x1, y2+5), (x1+bar_width, y2+5+bar_height), 
                                (100, 100, 100), -1)
                    cv2.rectangle(frame, (x1, y2+5), (x1+bar_fill, y2+5+bar_height), 
                                color, -1)
                    
                    # Draw reasons
                    if analysis['reasons']:
                        reason_text = ", ".join(analysis['reasons'][:2])
                        cv2.putText(frame, reason_text, (x1, y2+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw system info
                info_lines = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Active Persons: {len(detections)}",
                    f"Stealing Alerts: {len(stealing_alerts)}",
                    f"ReID Consistency: {self.reid_tracker.get_statistics()['match_rate']:.2%}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display or save
                if display:
                    cv2.imshow('Stealing Detection with Consistent ReID', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if output_path:
                    out.write(frame)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% | Alerts: {len(stealing_alerts)}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Calculate results
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        reid_stats = self.reid_tracker.get_statistics()
        
        results = {
            'frames_processed': frame_idx,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'stealing_alerts': len(stealing_alerts),
            'reid_statistics': reid_stats,
            'alerts_detail': stealing_alerts
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save detection results"""
        
        output_file = f"stealing_detection_results_{self.camera_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved: {output_file}")

def main():
    """Run final stealing detection demo"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Stealing Detection with Consistent ReID')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--display', '-d', action='store_true', help='Display video')
    parser.add_argument('--camera-id', '-c', default='camera_1', help='Camera ID')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FinalStealingDetectionSystem(camera_id=args.camera_id)
    
    # Set output path
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"final_stealing_detection_{base_name}.mp4"
    
    # Process video
    results = system.process_video(
        video_path=args.input,
        output_path=args.output,
        display=args.display
    )
    
    # Print results
    print(f"\n🎉 PROCESSING COMPLETED!")
    print(f"=" * 60)
    print(f"📊 RESULTS:")
    print(f"   Frames processed: {results['frames_processed']}")
    print(f"   Average FPS: {results['avg_fps']:.1f}")
    print(f"   Stealing alerts: {results['stealing_alerts']}")
    print(f"   ReID match rate: {results['reid_statistics']['match_rate']:.2%}")
    print(f"   Active tracks: {results['reid_statistics']['active_tracks']}")
    print(f"   New IDs created: {results['reid_statistics']['new_ids']}")
    
    print(f"\n💾 OUTPUTS:")
    print(f"   📹 Video: {args.output}")
    print(f"   📊 Results: stealing_detection_results_{args.camera_id}.json")

if __name__ == "__main__":
    main()
