#!/usr/bin/env python3
"""
Dual Window CCTV System
Clean video output with separate control panel for system information
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
from improved_anomaly_detection import ImprovedAnomalyDetector, BalancedBehaviorAnalyzer
from person_reid_system import GlobalPersonTracker
import time
import json
import os
from typing import Dict, List, Tuple, Optional
import threading

class DualWindowCCTVSystem:
    """CCTV system with clean video output and separate control panel"""
    
    def __init__(self, camera_id: str = "cam1", model_path: str = "models/vae_anomaly_detector.pth"):
        print(f"🚀 Initializing Dual Window CCTV System for {camera_id}")
        print("=" * 60)
        
        self.camera_id = camera_id
        
        # Initialize core components
        print("📹 Loading YOLO person detection...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        print("🧠 Loading improved anomaly detector...")
        self.anomaly_detector = ImprovedAnomalyDetector(model_path)
        self.behavior_analyzer = BalancedBehaviorAnalyzer(self.anomaly_detector)
        print("✅ Improved anomaly detector loaded (reduces false positives)")
        
        print("🔍 Initializing Person ReID system...")
        self.reid_tracker = GlobalPersonTracker()
        
        # Initialize zone detector (fallback mode)
        self.zone_detector = None
        
        # Tracking and anomaly data
        self.person_data = {}
        self.anomaly_histories = {}
        
        # Visualization parameters
        self.colors = {
            'normal': (0, 255, 0),      # Green - Normal behavior
            'suspicious': (0, 165, 255), # Orange - Suspicious behavior  
            'anomaly': (0, 0, 255)      # Red - Anomalous behavior
        }
        
        # Anomaly thresholds for 3-color system
        self.anomaly_thresholds = {
            'suspicious': 0.3,
            'anomaly': 0.7
        }
        
        # Smoothing parameters
        self.anomaly_window_size = 15
        self.min_track_length = 10
        
        # Control panel data
        self.control_panel_data = {
            'frame_info': {'current': 0, 'total': 0, 'fps': 0},
            'person_counts': {'normal': 0, 'suspicious': 0, 'anomaly': 0, 'total': 0},
            'reid_stats': {'global_persons': 0, 'matches': 0, 'match_rate': 0.0},
            'system_status': {'camera_id': camera_id, 'reid_enabled': True, 'processing': False},
            'recent_alerts': [],
            'performance': {'avg_fps': 0, 'processing_time': 0}
        }
        
        print("✅ Dual Window CCTV System initialized successfully!")
    
    def _init_zone_detector(self, width: int, height: int):
        """Initialize zone detector with fallback"""
        try:
            # Create simple fallback zones
            self.zone_detector = {
                'zones': [
                    {'id': 'left_area', 'bbox': [0, int(height*0.2), int(width*0.4), int(height*0.8)]},
                    {'id': 'center_area', 'bbox': [int(width*0.3), int(height*0.3), int(width*0.7), int(height*0.7)]},
                    {'id': 'right_area', 'bbox': [int(width*0.6), int(height*0.2), width, int(height*0.8)]}
                ]
            }
            print(f"🎯 Using fallback interaction zones")
        except Exception as e:
            print(f"⚠️  Zone detector error: {e}")
            self.zone_detector = None
    
    def analyze_person_behavior(self, global_id: int, local_track_id: int, 
                               person_bbox: List[float], frame_idx: int, fps: int) -> Dict:
        """Analyze person behavior using improved anomaly detection"""
        
        timestamp = frame_idx / fps
        
        # Initialize person data if new
        if global_id not in self.person_data:
            self.person_data[global_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'cameras_seen': {self.camera_id},
                'local_tracks': {self.camera_id: local_track_id},
                'positions': [],
                'total_detections': 0
            }
        
        person_info = self.person_data[global_id]
        
        # Update basic info
        person_info['last_seen'] = timestamp
        person_info['cameras_seen'].add(self.camera_id)
        person_info['local_tracks'][self.camera_id] = local_track_id
        person_info['total_detections'] += 1
        
        # Update position history
        center_x = (person_bbox[0] + person_bbox[2]) / 2
        center_y = (person_bbox[1] + person_bbox[3]) / 2
        person_info['positions'].append([center_x, center_y, timestamp])
        
        # Keep only recent positions
        if len(person_info['positions']) > 100:
            person_info['positions'] = person_info['positions'][-100:]
        
        # Use improved behavior analysis
        behavior_analysis = self.behavior_analyzer.analyze_behavior(
            local_track_id, person_bbox, frame_idx
        )
        
        # Map to our color system
        category_mapping = {
            'normal': 'normal',
            'suspicious': 'suspicious', 
            'anomaly': 'anomaly'
        }
        
        behavior_category = category_mapping.get(behavior_analysis['category'], 'normal')
        anomaly_score = behavior_analysis['anomaly_score']
        
        # Additional details
        duration = timestamp - person_info['first_seen']
        is_loitering = duration > 10.0
        is_multi_camera = len(person_info['cameras_seen']) > 1
        
        return {
            'global_id': global_id,
            'local_track_id': local_track_id,
            'behavior_category': behavior_category,
            'behavior_text': behavior_analysis['category'].upper(),
            'anomaly_score': anomaly_score,
            'raw_anomaly_score': behavior_analysis['raw_score'],
            'duration': duration,
            'cameras_seen': len(person_info['cameras_seen']),
            'total_detections': person_info['total_detections'],
            'confidence': behavior_analysis['confidence'],
            'details': {
                'is_loitering': is_loitering,
                'is_multi_camera': is_multi_camera,
                'score_stability': behavior_analysis['score_stability'],
                'person_profile': behavior_analysis['person_profile']
            }
        }
    
    def draw_clean_person_visualization(self, frame: np.ndarray, person_bbox: np.ndarray, 
                                       analysis: Dict) -> np.ndarray:
        """Draw clean person visualization without system info"""
        
        x1, y1, x2, y2 = person_bbox.astype(int)
        global_id = analysis['global_id']
        local_id = analysis['local_track_id']
        
        # Choose color based on behavior category
        color = self.colors[analysis['behavior_category']]
        
        # Draw main bounding box with thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Simple, clean label
        behavior_text = analysis['behavior_text']
        anomaly_score = analysis['anomaly_score']
        
        # Main label with global ID only
        main_label = f"G:{global_id} {behavior_text}"
        if anomaly_score > 0.1:
            main_label += f" ({anomaly_score:.2f})"
        
        # Draw label with background
        label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), 
                     color, -1)
        cv2.putText(frame, main_label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw simple anomaly score bar (smaller)
        if anomaly_score > 0.1:
            bar_width = 60
            bar_height = 4
            bar_x = x1
            bar_y = y1 - label_size[1] - 20
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Score bar
            score_width = int(bar_width * min(anomaly_score, 1.0))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), color, -1)
        
        return frame
    
    def create_control_panel(self, width: int = 400, height: int = 600) -> np.ndarray:
        """Create control panel with system information"""
        
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(30)  # Dark background
        
        y_pos = 30
        line_height = 25
        
        # Title
        cv2.putText(panel, "CCTV CONTROL PANEL", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height * 2
        
        # Camera Info
        cv2.putText(panel, f"Camera: {self.control_panel_data['system_status']['camera_id']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        # Frame Info
        frame_info = self.control_panel_data['frame_info']
        cv2.putText(panel, f"Frame: {frame_info['current']}/{frame_info['total']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"FPS: {frame_info['fps']:.1f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height * 2
        
        # Person Counts
        cv2.putText(panel, "PERSON TRACKING", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_height
        
        counts = self.control_panel_data['person_counts']
        cv2.putText(panel, f"Total Persons: {counts['total']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        # Color-coded counts
        cv2.putText(panel, f"Normal: {counts['normal']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['normal'], 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Suspicious: {counts['suspicious']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['suspicious'], 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Anomalies: {counts['anomaly']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['anomaly'], 1)
        y_pos += line_height * 2
        
        # ReID Stats
        cv2.putText(panel, "RE-IDENTIFICATION", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += line_height
        
        reid_stats = self.control_panel_data['reid_stats']
        cv2.putText(panel, f"Global Persons: {reid_stats['global_persons']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"ReID Matches: {reid_stats['matches']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Match Rate: {reid_stats['match_rate']:.1%}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height * 2
        
        # Performance
        cv2.putText(panel, "PERFORMANCE", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_pos += line_height
        
        perf = self.control_panel_data['performance']
        cv2.putText(panel, f"Avg FPS: {perf['avg_fps']:.1f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Process Time: {perf['processing_time']:.1f}s", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        # Improved detection stats
        if hasattr(self, 'behavior_analyzer'):
            detection_stats = self.anomaly_detector.get_system_statistics()
            cv2.putText(panel, f"False Positive Reduction: {detection_stats['fp_reduction_rate']:.1%}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_pos += line_height
        
        y_pos += line_height
        
        # Recent Alerts
        cv2.putText(panel, "RECENT ALERTS", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += line_height
        
        alerts = self.control_panel_data['recent_alerts']
        if not alerts:
            cv2.putText(panel, "No recent alerts", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        else:
            for i, alert in enumerate(alerts[-5:]):  # Show last 5 alerts
                alert_text = f"G:{alert['global_id']} {alert['type']}"
                alert_color = self.colors.get(alert['category'], (255, 255, 255))
                cv2.putText(panel, alert_text, 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1)
                y_pos += 20
        
        # Legend at bottom
        legend_y = height - 120
        cv2.putText(panel, "COLOR LEGEND", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        legend_y += 20
        
        legend_items = [
            ("Green: Normal", self.colors['normal']),
            ("Orange: Suspicious", self.colors['suspicious']),
            ("Red: Anomaly", self.colors['anomaly'])
        ]
        
        for text, color in legend_items:
            cv2.putText(panel, text, (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            legend_y += 18
        
        # Controls
        controls_y = height - 40
        cv2.putText(panel, "Controls: Q=Quit, SPACE=Pause", (10, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def update_control_panel_data(self, frame_idx: int, total_frames: int, 
                                 current_fps: float, anomaly_counts: Dict, 
                                 reid_stats: Dict, processing_time: float):
        """Update control panel data"""
        
        self.control_panel_data['frame_info'] = {
            'current': frame_idx,
            'total': total_frames,
            'fps': current_fps
        }
        
        self.control_panel_data['person_counts'] = anomaly_counts
        self.control_panel_data['reid_stats'] = reid_stats
        self.control_panel_data['performance']['processing_time'] = processing_time
    
    def add_alert(self, global_id: int, alert_type: str, category: str):
        """Add alert to recent alerts"""
        
        alert = {
            'global_id': global_id,
            'type': alert_type,
            'category': category,
            'timestamp': time.time()
        }
        
        self.control_panel_data['recent_alerts'].append(alert)
        
        # Keep only recent alerts
        if len(self.control_panel_data['recent_alerts']) > 10:
            self.control_panel_data['recent_alerts'] = self.control_panel_data['recent_alerts'][-10:]
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with dual window display"""
        
        print(f"\n🎬 Processing Video: {os.path.basename(video_path)}")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"📹 Camera ID: {self.camera_id}")
        print(f"🎯 Dual Window Mode: Clean Video + Control Panel")
        
        # Initialize zone detector
        self._init_zone_detector(width, height)
        
        # Setup video writer for clean output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Clean output will be saved to: {output_path}")
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Person detection and tracking
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.4,
                    verbose=False
                )
                
                # Process frame - CLEAN VERSION
                clean_frame = frame.copy()
                
                # Process person detections
                anomaly_counts = {'normal': 0, 'suspicious': 0, 'anomaly': 0, 'total': 0}
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        if conf < 0.4:
                            continue
                        
                        anomaly_counts['total'] += 1
                        
                        # Global ReID processing
                        global_id = self.reid_tracker.update_global_tracking(
                            self.camera_id, track_id, frame, box.tolist(), conf, timestamp
                        )
                        
                        # Analyze person behavior
                        analysis = self.analyze_person_behavior(
                            global_id, track_id, box.tolist(), frame_idx, fps
                        )
                        
                        # Count anomaly categories
                        anomaly_counts[analysis['behavior_category']] += 1
                        
                        # Add alerts for anomalies
                        if analysis['behavior_category'] in ['suspicious', 'anomaly']:
                            self.add_alert(global_id, analysis['behavior_text'], analysis['behavior_category'])
                        
                        # Draw CLEAN person visualization (no system info)
                        clean_frame = self.draw_clean_person_visualization(
                            clean_frame, box, analysis
                        )
                
                # Calculate current FPS
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Get ReID statistics
                reid_stats = self.reid_tracker.get_tracking_statistics()
                reid_display_stats = {
                    'global_persons': reid_stats.get('total_global_persons', 0),
                    'matches': reid_stats.get('reid_matches', 0),
                    'match_rate': reid_stats.get('reid_match_rate', 0.0)
                }
                
                # Update control panel data
                elapsed_time = time.time() - start_time
                self.update_control_panel_data(
                    frame_idx, total_frames, current_fps, anomaly_counts, 
                    reid_display_stats, elapsed_time
                )
                
                # Create control panel
                control_panel = self.create_control_panel()
                
                # Write clean frame to output
                if writer:
                    writer.write(clean_frame)
                
                # Display both windows
                if display:
                    cv2.imshow('CCTV Video Feed - Clean Output', clean_frame)
                    cv2.imshow('CCTV Control Panel', control_panel)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️  Processing stopped by user")
                        break
                    elif key == ord(' '):
                        print("⏸️  Paused - Press any key to continue")
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Progress update (less frequent)
                if frame_idx % 200 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_idx / elapsed
                    progress = (frame_idx / total_frames) * 100
                    
                    print(f"📊 Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | "
                          f"Persons: {anomaly_counts['total']} | Anomalies: {anomaly_counts['anomaly']}")
                
                # Periodic cleanup
                if frame_idx % 1000 == 0:
                    self.reid_tracker.cleanup_old_tracks(timestamp)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        reid_stats = self.reid_tracker.get_tracking_statistics()
        
        print(f"\n🎯 PROCESSING COMPLETE")
        print("=" * 50)
        print(f"📊 Frames processed: {frame_idx}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🌍 Global persons tracked: {reid_stats['total_global_persons']}")
        print(f"🔄 ReID matches: {reid_stats['reid_matches']}")
        print(f"📈 ReID match rate: {reid_stats['reid_match_rate']:.2%}")
        
        # Save ReID data
        self.reid_tracker.save_reid_data(f"reid_data_{self.camera_id}.pkl")
        
        if output_path:
            print(f"💾 Clean output saved to: {output_path}")
        
        return {
            'frames_processed': frame_idx,
            'avg_fps': avg_fps,
            'reid_statistics': reid_stats,
            'total_persons': len(self.person_data)
        }

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Window CCTV System - Clean Video + Control Panel')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (clean, no overlays)')
    parser.add_argument('--camera-id', '-c', default='cam1', help='Camera ID for ReID')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input video not found: {args.input}")
        return
    
    # Set default output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"clean_output_{args.camera_id}_{base_name}.mp4"
    
    print("🚀 DUAL WINDOW CCTV SYSTEM")
    print("=" * 60)
    print("Features:")
    print("✅ Clean video output without system overlays")
    print("✅ Separate control panel window with all system info")
    print("✅ Global Person ReID across camera angles")
    print("✅ Real-time anomaly detection with VAE")
    print("✅ 3-Color behavior visualization (Green/Orange/Red)")
    print("✅ Real-time statistics and alerts")
    print("=" * 60)
    
    try:
        # Initialize dual window system
        system = DualWindowCCTVSystem(camera_id=args.camera_id)
        
        # Process video
        results = system.process_video(
            video_path=args.input,
            output_path=args.output,
            display=not args.no_display
        )
        
        print(f"\n🏆 DUAL WINDOW CCTV SYSTEM PROCESSING SUCCESSFUL!")
        print(f"📹 Clean video output: {args.output}")
        print(f"📊 System provided real-time monitoring in separate window")
        
    except FileNotFoundError as e:
        print(f"❌ Required model not found: {e}")
        print("Please ensure VAE model is trained:")
        print("python train_vae_model.py")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()