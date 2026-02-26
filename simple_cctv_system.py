#!/usr/bin/env python3
"""
Simplified CCTV System - Single Camera with Local Tracker IDs Only
No Global ReID - Uses only YOLO+BotSORT tracker IDs for stable tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
# Removed: from person_reid_system import GlobalPersonTracker
from adaptive_zone_learning import ActivityZoneLearner
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import time
import json
import os

class SimpleCCTVSystem:
    """Simplified CCTV system using only local tracker IDs"""
    
    def __init__(self, camera_id: str = "cam1", model_path: str = "models/vae_anomaly_detector.pth"):
        print(f"🚀 Initializing Simplified CCTV System for {camera_id}")
        print("=" * 60)
        
        self.camera_id = camera_id
        
        # Initialize core components
        print("📹 Loading YOLO person detection...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        print("🧠 Loading VAE anomaly detector...")
        self.anomaly_detector = AnomalyDetector(model_path)
        try:
            self.anomaly_detector.load_model()
            print("✅ VAE anomaly detector loaded")
        except FileNotFoundError:
            print("❌ VAE model not found! Please train first.")
            raise
        
        # Removed: ReID tracker initialization
        # self.reid_tracker = GlobalPersonTracker()
        
        print("🤚 Initializing hand detection...")
        self.hand_detector = self._init_hand_detector()
        
        print("🎯 Loading adaptive interaction zones...")
        self.zone_detector = None  # Will be initialized with video dimensions
        
        # Tracking and anomaly data - keyed by track_id only
        self.person_data = {}  # track_id -> person data
        self.anomaly_histories = {}  # track_id -> anomaly history
        
        # Visualization parameters
        self.colors = {
            'normal': (0, 255, 0),      # Green - Normal behavior
            'suspicious': (0, 165, 255), # Orange - Suspicious behavior  
            'anomaly': (0, 0, 255)      # Red - Anomalous behavior
        }
        
        # Anomaly thresholds for 3-color system
        self.anomaly_thresholds = {
            'suspicious': 0.3,  # Above this = suspicious (orange)
            'anomaly': 0.7      # Above this = anomaly (red)
        }
        
        # Smoothing parameters
        self.anomaly_window_size = 15  # Frames for anomaly smoothing
        self.min_track_length = 10     # Minimum frames before showing anomaly
        
        # Statistics
        self.total_tracks_seen = 0
        self.active_tracks = set()
        
        print("✅ Simplified CCTV System initialized successfully!")
        print("📌 Using local tracker IDs only (no Global ReID)")
    
    def _init_hand_detector(self):
        """Initialize MediaPipe hand detection"""
        try:
            mp_hands = mp.solutions.hands
            return mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=6,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except AttributeError:
            print("⚠️  MediaPipe hand detection not available, using fallback")
            return None
    
    def _init_zone_detector(self, width: int, height: int):
        """Initialize adaptive zone detector with video dimensions"""
        try:
            from stealing_detection_system import AdaptiveZoneDetector
            self.zone_detector = AdaptiveZoneDetector(width, height)
            print(f"🎯 Loaded {len(self.zone_detector.interaction_zones)} learned interaction zones")
        except Exception as e:
            print(f"⚠️  Zone detector not available: {e}")
            print("🎯 Using fallback mode without interaction zones")
            self.zone_detector = None
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in frame"""
        if self.hand_detector is None:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(rgb_frame)
            
            hands_info = []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    
                    handedness = "Right"
                    if idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[idx].classification[0].label
                    
                    hands_info.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'center': [center_x, center_y],
                        'handedness': handedness,
                        'landmarks': hand_landmarks
                    })
            
            return hands_info
        except Exception as e:
            print(f"⚠️  Hand detection error: {e}")
            return []
    
    def get_person_hands(self, person_bbox: np.ndarray, all_hands: List[Dict]) -> List[Dict]:
        """Get hands belonging to a specific person"""
        person_hands = []
        px1, py1, px2, py2 = person_bbox
        
        for hand in all_hands:
            hx, hy = hand['center']
            
            # Check if hand is within person's bounding box (with tolerance)
            tolerance = 50
            if (px1 - tolerance <= hx <= px2 + tolerance and 
                py1 - tolerance <= hy <= py2 + tolerance):
                person_hands.append(hand)
        
        return person_hands
    
    def analyze_person_behavior(self, track_id: int, person_bbox: List[float], 
                               person_hands: List[Dict], frame_idx: int, fps: int) -> Dict:
        """Comprehensive person behavior analysis using track_id only"""
        
        timestamp = frame_idx / fps
        
        # Initialize person data if new track
        if track_id not in self.person_data:
            self.person_data[track_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'positions': [],
                'anomaly_scores': [],
                'behaviors': [],
                'interactions': [],
                'total_detections': 0
            }
            self.anomaly_histories[track_id] = []
            self.total_tracks_seen += 1
        
        person_info = self.person_data[track_id]
        
        # Update basic info
        person_info['last_seen'] = timestamp
        person_info['total_detections'] += 1
        self.active_tracks.add(track_id)
        
        # Update position history
        center_x = (person_bbox[0] + person_bbox[2]) / 2
        center_y = (person_bbox[1] + person_bbox[3]) / 2
        person_info['positions'].append([center_x, center_y, timestamp])
        
        # Keep only recent positions
        if len(person_info['positions']) > 100:
            person_info['positions'] = person_info['positions'][-100:]
        
        # 1. BEHAVIORAL ANOMALY DETECTION
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
            track_id, person_bbox, frame_idx
        )
        
        # Store anomaly score
        person_info['anomaly_scores'].append(anomaly_score)
        if len(person_info['anomaly_scores']) > 50:
            person_info['anomaly_scores'] = person_info['anomaly_scores'][-50:]
        
        # 2. INTERACTION ANALYSIS
        interaction_score = 0.0
        if self.zone_detector and person_hands:
            zone_interactions = self.zone_detector.detect_hand_interaction(
                person_hands, person_bbox
            )
            if zone_interactions['has_interaction']:
                interaction_score = zone_interactions['interaction_score']
                person_info['interactions'].append({
                    'timestamp': timestamp,
                    'score': interaction_score,
                    'zones': zone_interactions['interaction_zones']
                })
        
        # 3. MOTION ANALYSIS
        motion_score = 0.0
        if len(person_info['positions']) > 5:
            recent_positions = np.array([p[:2] for p in person_info['positions'][-5:]])
            
            # Calculate speed
            if len(recent_positions) > 1:
                distances = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
                avg_speed = np.mean(distances)
                
                # Normalize speed (higher speed = higher score)
                motion_score = min(avg_speed / 20.0, 1.0)
        
        # 4. COMBINED ANOMALY SCORE
        combined_score = (
            0.6 * anomaly_score +      # VAE anomaly (60%)
            0.3 * interaction_score +  # Zone interactions (30%)
            0.1 * motion_score         # Motion patterns (10%)
        )
        
        # 5. TEMPORAL SMOOTHING
        self.anomaly_histories[track_id].append(combined_score)
        if len(self.anomaly_histories[track_id]) > self.anomaly_window_size:
            self.anomaly_histories[track_id] = self.anomaly_histories[track_id][-self.anomaly_window_size:]
        
        # Calculate smoothed anomaly score
        if len(self.anomaly_histories[track_id]) >= self.min_track_length:
            smoothed_score = np.mean(self.anomaly_histories[track_id])
        else:
            smoothed_score = 0.0
        
        # 6. DETERMINE BEHAVIOR CATEGORY
        if smoothed_score >= self.anomaly_thresholds['anomaly']:
            behavior_category = 'anomaly'
            behavior_text = 'ANOMALY'
        elif smoothed_score >= self.anomaly_thresholds['suspicious']:
            behavior_category = 'suspicious'
            behavior_text = 'SUSPICIOUS'
        else:
            behavior_category = 'normal'
            behavior_text = 'NORMAL'
        
        # 7. ADDITIONAL BEHAVIOR DETAILS
        duration = timestamp - person_info['first_seen']
        is_loitering = duration > 10.0
        has_interactions = len(person_info['interactions']) > 0
        
        return {
            'track_id': track_id,  # Only track_id, no global_id
            'behavior_category': behavior_category,
            'behavior_text': behavior_text,
            'anomaly_score': smoothed_score,
            'raw_anomaly_score': combined_score,
            'duration': duration,
            'total_detections': person_info['total_detections'],
            'details': {
                'is_loitering': is_loitering,
                'has_interactions': has_interactions,
                'motion_score': motion_score,
                'interaction_score': interaction_score,
                'vae_score': anomaly_score
            }
        }
    
    def draw_person_visualization(self, frame: np.ndarray, person_bbox: np.ndarray, 
                                analysis: Dict, person_hands: List[Dict]) -> np.ndarray:
        """Draw comprehensive person visualization"""
        
        x1, y1, x2, y2 = person_bbox.astype(int)
        track_id = analysis['track_id']  # Only track_id
        
        # Choose color based on behavior category
        color = self.colors[analysis['behavior_category']]
        
        # Draw main bounding box with thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        
        # Create label - simplified to show only track_id
        behavior_text = analysis['behavior_text']
        anomaly_score = analysis['anomaly_score']
        
        # Main label with track ID only
        main_label = f"ID:{track_id} {behavior_text}"
        if anomaly_score > 0:
            main_label += f" ({anomaly_score:.2f})"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, main_label, (x1 + 5, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw additional info
        duration = analysis['duration']
        detections = analysis['total_detections']
        info_label = f"Time:{duration:.1f}s Det:{detections}"
        
        cv2.putText(frame, info_label, (x1 + 5, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw hands if detected
        for hand in person_hands:
            hx1, hy1, hx2, hy2 = hand['bbox']
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
            cv2.putText(frame, hand['handedness'], (hx1, hy1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw trajectory for anomalous persons
        if analysis['behavior_category'] != 'normal' and track_id in self.person_data:
            positions = self.person_data[track_id]['positions']
            if len(positions) > 2:
                # Draw trajectory line
                points = np.array([[int(p[0]), int(p[1])] for p in positions[-20:]])
                for i in range(len(points) - 1):
                    cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, 2)
        
        return frame
    
    def draw_interaction_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw learned interaction zones"""
        if self.zone_detector and hasattr(self.zone_detector, 'interaction_zones'):
            for zone in self.zone_detector.interaction_zones:
                x, y, w, h = zone['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.putText(frame, f"Zone {zone.get('id', '?')}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return frame
    
    def draw_statistics(self, frame: np.ndarray, frame_idx: int, total_frames: int,
                       active_persons: int, fps: float) -> np.ndarray:
        """Draw system statistics"""
        h, w = frame.shape[:2]
        
        # Statistics panel
        stats = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Active Persons: {active_persons}",
            f"Total Tracks: {self.total_tracks_seen}",
            f"FPS: {fps:.1f}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 30 + len(stats) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw statistics text
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = h - 120
        cv2.putText(frame, "Legend:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        legend_items = [
            ("Normal", self.colors['normal']),
            ("Suspicious", self.colors['suspicious']),
            ("Anomaly", self.colors['anomaly'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y = legend_y + 25 + i * 25
            cv2.rectangle(frame, (10, y-15), (30, y), color, -1)
            cv2.putText(frame, label, (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     display: bool = False) -> Dict:
        """Process video with simplified tracking"""
        
        print(f"\n🎬 Processing Video: {os.path.basename(video_path)}")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"📹 Camera ID: {self.camera_id}")
        print(f"🎯 Anomaly Visualization: 3-Color System")
        
        # Initialize zone detector
        self._init_zone_detector(width, height)
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                timestamp = frame_idx / fps
                
                # Person detection and tracking with improved stability
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort_stable.yaml",
                    persist=True,
                    imgsz=960,      # increased from 640 for better detection
                    conf=0.25,      # lower confidence threshold for better tracking
                    iou=0.5,        # IoU threshold for tracking
                    classes=[0],    # person only
                    verbose=False
                )
                
                # Hand detection
                hands = self.detect_hands(frame)
                
                # Process frame
                annotated_frame = frame.copy()
                
                # Draw interaction zones
                annotated_frame = self.draw_interaction_zones(annotated_frame)
                
                # Process each detected person
                active_persons = 0
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    active_persons = len(track_ids)
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        # Get hands for this person
                        person_hands = self.get_person_hands(box, hands)
                        
                        # Analyze person behavior using track_id only
                        analysis = self.analyze_person_behavior(
                            track_id, box.tolist(), person_hands, frame_idx, fps
                        )
                        
                        # Draw visualization
                        annotated_frame = self.draw_person_visualization(
                            annotated_frame, box, analysis, person_hands
                        )
                
                # Draw statistics
                current_fps = frame_idx / (time.time() - start_time)
                annotated_frame = self.draw_statistics(
                    annotated_frame, frame_idx, total_frames, active_persons, current_fps
                )
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('CCTV Anomaly Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% | FPS: {current_fps:.1f} | Persons: {active_persons} | Tracks: {self.total_tracks_seen}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        
        print(f"\n🎯 PROCESSING COMPLETE")
        print("=" * 50)
        print(f"📊 Frames processed: {frame_idx}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"👥 Total tracks seen: {self.total_tracks_seen}")
        
        if output_path:
            print(f"💾 Output saved to: {output_path}")
        
        return {
            'frames_processed': frame_idx,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'total_tracks': self.total_tracks_seen,
            'active_tracks': len(self.active_tracks)
        }

if __name__ == "__main__":
    # Example usage
    system = SimpleCCTVSystem(camera_id="demo")
    system.process_video(
        video_path="test_video.mp4",
        output_path="output_simple.mp4",
        display=False
    )
