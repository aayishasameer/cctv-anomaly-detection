#!/usr/bin/env python3
"""
Advanced Fix for Tracking Issues and False Anomalies
Addresses: ID switching, false positives, tracking consistency
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
import os
import argparse
from typing import Dict, Tuple, List
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

class AdvancedTrackingFixer:
    """Advanced system to fix tracking and anomaly detection issues"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth"):
        print("Initializing Advanced Tracking Fixer...")
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Load anomaly detector with modified threshold
        self.anomaly_detector = AnomalyDetector(model_path)
        try:
            self.anomaly_detector.load_model()
            # Adjust threshold to reduce false positives
            self.anomaly_detector.threshold *= 2.0  # Make it stricter
            print(f"✓ Anomaly model loaded, threshold adjusted to: {self.anomaly_detector.threshold:.4f}")
        except FileNotFoundError:
            print("❌ Anomaly detection model not found!")
            raise
        
        # Advanced tracking parameters
        self.global_tracks = {}                 # Global track management
        self.next_global_id = 1
        self.max_distance_threshold = 100       # Max distance for track association
        self.min_track_confidence = 0.7         # Minimum confidence for tracking
        self.track_memory_frames = 30           # Frames to remember lost tracks
        
        # Anomaly detection improvements
        self.anomaly_history_length = 50        # Longer history for stability
        self.anomaly_confirmation_threshold = 0.9  # 90% of frames must be anomalous
        self.min_anomaly_duration = 2.0        # Minimum 2 seconds of anomalous behavior
        self.anomaly_score_threshold = 1.0      # Minimum anomaly score
        
        # Track state management
        self.track_states = {}                  # Track state information
        self.lost_tracks = {}                   # Recently lost tracks for recovery
        
        # Colors
        self.colors = {
            'normal': (0, 255, 0),              # Green
            'warning': (0, 165, 255),           # Orange  
            'anomaly': (0, 0, 255),             # Red
            'tracking': (255, 255, 0),          # Cyan
            'lost': (128, 128, 128)             # Gray
        }
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_detections_to_tracks(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """Advanced detection-to-track association"""
        
        if not detections:
            return []
        
        # Get active tracks (seen recently)
        active_tracks = {}
        for global_id, track_info in self.global_tracks.items():
            if frame_idx - track_info['last_frame'] <= self.track_memory_frames:
                active_tracks[global_id] = track_info
        
        # Calculate cost matrix (distance + IoU)
        if not active_tracks:
            # No active tracks, create new ones
            associated_detections = []
            for det in detections:
                new_id = self.next_global_id
                self.next_global_id += 1
                
                det['global_id'] = new_id
                det['is_new'] = True
                
                # Initialize track
                self.global_tracks[new_id] = {
                    'positions': [det['center']],
                    'bboxes': [det['bbox']],
                    'confidences': [det['confidence']],
                    'last_frame': frame_idx,
                    'first_frame': frame_idx,
                    'anomaly_history': [],
                    'anomaly_scores': []
                }
                
                associated_detections.append(det)
            
            return associated_detections
        
        # Calculate association costs
        track_ids = list(active_tracks.keys())
        cost_matrix = np.full((len(detections), len(track_ids)), np.inf)
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track_info = active_tracks[track_id]
                
                # Calculate distance cost
                last_pos = track_info['positions'][-1]
                distance = np.linalg.norm(np.array(det['center']) - np.array(last_pos))
                
                # Calculate IoU cost
                last_bbox = track_info['bboxes'][-1]
                iou = self.calculate_iou(det['bbox'], last_bbox)
                
                # Combined cost (lower is better)
                if distance < self.max_distance_threshold:
                    cost_matrix[i, j] = distance * (1 - iou)
        
        # Hungarian algorithm (simplified greedy approach)
        associated_detections = []
        used_tracks = set()
        used_detections = set()
        
        # Greedy assignment
        while True:
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(len(detections)):
                if i in used_detections:
                    continue
                for j in range(len(track_ids)):
                    if j in used_tracks:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_cost == np.inf or min_cost > 200:  # No good matches
                break
            
            # Associate detection to track
            det = detections[min_i].copy()
            track_id = track_ids[min_j]
            
            det['global_id'] = track_id
            det['is_new'] = False
            
            # Update track
            track_info = self.global_tracks[track_id]
            track_info['positions'].append(det['center'])
            track_info['bboxes'].append(det['bbox'])
            track_info['confidences'].append(det['confidence'])
            track_info['last_frame'] = frame_idx
            
            # Keep limited history
            max_history = 100
            for key in ['positions', 'bboxes', 'confidences']:
                if len(track_info[key]) > max_history:
                    track_info[key] = track_info[key][-max_history:]
            
            associated_detections.append(det)
            used_tracks.add(min_j)
            used_detections.add(min_i)
        
        # Create new tracks for unassociated detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_id = self.next_global_id
                self.next_global_id += 1
                
                det['global_id'] = new_id
                det['is_new'] = True
                
                # Initialize new track
                self.global_tracks[new_id] = {
                    'positions': [det['center']],
                    'bboxes': [det['bbox']],
                    'confidences': [det['confidence']],
                    'last_frame': frame_idx,
                    'first_frame': frame_idx,
                    'anomaly_history': [],
                    'anomaly_scores': []
                }
                
                associated_detections.append(det)
        
        return associated_detections
    
    def advanced_anomaly_detection(self, global_id: int, bbox: List[float], frame_idx: int, fps: int) -> Tuple[bool, str, float]:
        """Advanced anomaly detection with temporal filtering"""
        
        # Get basic anomaly detection
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(global_id, bbox, frame_idx)
        
        # Get track info
        if global_id not in self.global_tracks:
            return False, "TRACKING", 0.0
        
        track_info = self.global_tracks[global_id]
        
        # Add to history
        track_info['anomaly_history'].append(is_anomaly)
        track_info['anomaly_scores'].append(anomaly_score)
        
        # Keep limited history
        if len(track_info['anomaly_history']) > self.anomaly_history_length:
            track_info['anomaly_history'] = track_info['anomaly_history'][-self.anomaly_history_length:]
            track_info['anomaly_scores'] = track_info['anomaly_scores'][-self.anomaly_history_length:]
        
        # Check track maturity
        track_duration = (frame_idx - track_info['first_frame']) / fps
        if track_duration < 1.0:  # Need at least 1 second of tracking
            return False, "TRACKING", anomaly_score
        
        # Analyze recent history
        recent_window = min(30, len(track_info['anomaly_history']))
        if recent_window < 15:  # Need sufficient history
            return False, "NORMAL", anomaly_score
        
        recent_anomalies = track_info['anomaly_history'][-recent_window:]
        recent_scores = track_info['anomaly_scores'][-recent_window:]
        
        # Calculate statistics
        anomaly_ratio = sum(recent_anomalies) / len(recent_anomalies)
        avg_score = np.mean(recent_scores)
        max_score = np.max(recent_scores)
        
        # Stricter anomaly confirmation
        if (anomaly_ratio >= self.anomaly_confirmation_threshold and 
            avg_score > self.anomaly_score_threshold and
            max_score > 2.0):
            
            # Check duration requirement
            anomaly_frames = sum(track_info['anomaly_history'])
            anomaly_duration = anomaly_frames / fps
            
            if anomaly_duration >= self.min_anomaly_duration:
                return True, "ANOMALY", avg_score
            else:
                return False, "WARNING", avg_score
        
        elif anomaly_ratio > 0.3 or avg_score > 0.5:
            return False, "WARNING", avg_score
        else:
            return False, "NORMAL", avg_score
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with advanced tracking and anomaly detection"""
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        anomaly_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection (no tracking, we'll do our own)
                results = self.yolo_model(
                    source=frame,
                    classes=[0],  # person only
                    conf=self.min_track_confidence,
                    verbose=False
                )
                
                # Extract detections
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        
                        detections.append({
                            'bbox': box.tolist(),
                            'center': [center_x, center_y],
                            'confidence': conf
                        })
                
                # Associate detections to tracks
                associated_detections = self.associate_detections_to_tracks(detections, frame_idx)
                
                # Process each track
                annotated_frame = frame.copy()
                
                for det in associated_detections:
                    global_id = det['global_id']
                    bbox = det['bbox']
                    confidence = det['confidence']
                    
                    # Advanced anomaly detection
                    is_anomaly, status, anomaly_score = self.advanced_anomaly_detection(
                        global_id, bbox, frame_idx, fps
                    )
                    
                    # Choose color
                    color = self.colors.get(status.lower(), self.colors['normal'])
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = [int(x) for x in bbox]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"ID:{global_id} {status}"
                    if anomaly_score > 0:
                        label += f" {anomaly_score:.2f}"
                    label += f" C:{confidence:.2f}"
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), 
                                color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Log confirmed anomalies
                    if is_anomaly:
                        anomaly_detections.append({
                            'frame': frame_idx,
                            'global_id': global_id,
                            'bbox': bbox,
                            'score': anomaly_score,
                            'confidence': confidence,
                            'timestamp': frame_idx / fps
                        })
                
                # Add info
                active_tracks = len([t for t in self.global_tracks.values() 
                                   if frame_idx - t['last_frame'] <= 5])
                
                info_lines = [
                    f"Frame: {frame_idx}/{total_frames}",
                    f"Active Tracks: {active_tracks}",
                    f"Total Tracks: {len(self.global_tracks)}",
                    f"Anomalies: {len(anomaly_detections)}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(annotated_frame, line, (10, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Advanced Tracking Fix', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 200 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Active: {active_tracks} | Anomalies: {len(anomaly_detections)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n=== Advanced Processing Complete ===")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total tracks created: {len(self.global_tracks)}")
        print(f"Total anomalies detected: {len(anomaly_detections)}")
        
        # Track statistics
        track_lengths = []
        for track_info in self.global_tracks.values():
            length = track_info['last_frame'] - track_info['first_frame'] + 1
            track_lengths.append(length)
        
        if track_lengths:
            print(f"Average track length: {np.mean(track_lengths):.1f} frames")
            print(f"Longest track: {np.max(track_lengths)} frames")
            print(f"Shortest track: {np.min(track_lengths)} frames")
        
        if anomaly_detections:
            print(f"\nConfirmed Anomalies:")
            for i, detection in enumerate(anomaly_detections[:5]):
                print(f"  {i+1}. ID {detection['global_id']} at {detection['timestamp']:.1f}s "
                      f"(Score: {detection['score']:.3f})")
        
        if output_path:
            print(f"Advanced output saved to: {output_path}")
        
        return {
            'anomaly_detections': anomaly_detections,
            'total_tracks': len(self.global_tracks),
            'track_lengths': track_lengths
        }

def main():
    parser = argparse.ArgumentParser(description='Advanced tracking and anomaly detection fix')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Initialize advanced tracker
    try:
        tracker = AdvancedTrackingFixer()
    except FileNotFoundError:
        return
    
    # Process video
    display = not args.no_display
    results = tracker.process_video(args.input, args.output, display)
    
    print(f"\nAdvanced processing completed!")

if __name__ == "__main__":
    main()