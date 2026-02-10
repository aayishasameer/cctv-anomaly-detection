#!/usr/bin/env python3
"""
Simplified Dual Window CCTV System - Trust BotSORT IDs
No ReID conflicts - just use BotSORT's stable tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
import time
import os
from collections import defaultdict

class SimpleDualWindowCCTV:
    """Simplified CCTV system using BotSORT IDs directly"""
    
    def __init__(self, camera_id: str = "cam1"):
        self.camera_id = camera_id
        
        # Load models
        print("🚀 Initializing CCTV System...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.anomaly_detector = AnomalyDetector("models/vae_anomaly_detector.pth")
        self.anomaly_detector.load_model()
        
        # Colors for threat levels
        self.colors = {
            'normal': (0, 255, 0),      # Green
            'suspicious': (0, 165, 255), # Orange
            'anomaly': (0, 0, 255)       # Red
        }
        
        # Control panel data
        self.control_panel_data = {
            'frame_info': {'current': 0, 'total': 0},
            'person_counts': {'normal': 0, 'suspicious': 0, 'anomaly': 0}
        }
    
    def create_control_panel(self, width: int = 400, height: int = 600) -> np.ndarray:
        """Create control panel with essential info"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_pos = 30
        line_height = 25
        
        # Title
        cv2.putText(panel, "CCTV CONTROL PANEL", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += line_height * 2
        
        # System Status
        cv2.putText(panel, "SYSTEM STATUS", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += line_height
        
        cv2.putText(panel, f"Camera: {self.camera_id}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        frame_info = self.control_panel_data['frame_info']
        cv2.putText(panel, f"Frame: {frame_info['current']}/{frame_info['total']}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height * 2
        
        # Person Counts
        cv2.putText(panel, "PERSON COUNTS", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += line_height
        
        counts = self.control_panel_data['person_counts']
        cv2.putText(panel, f"Normal: {counts['normal']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Suspicious: {counts['suspicious']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Anomaly: {counts['anomaly']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += line_height * 2
        
        # Color Legend
        cv2.putText(panel, "COLOR LEGEND", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height + 10
        
        # Draw color boxes with labels
        box_size = 20
        legend_items = [
            ("Normal", self.colors['normal']),
            ("Suspicious", self.colors['suspicious']),
            ("Anomaly", self.colors['anomaly'])
        ]
        
        for text, color in legend_items:
            cv2.rectangle(panel, (10, y_pos - 15), (10 + box_size, y_pos - 15 + box_size), color, -1)
            cv2.rectangle(panel, (10, y_pos - 15), (10 + box_size, y_pos - 15 + box_size), (255, 255, 255), 1)
            cv2.putText(panel, text, (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 30
        
        # Controls at bottom
        controls_y = height - 40
        cv2.putText(panel, "Controls: Q=Quit, SPACE=Pause", (10, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return panel
    
    def process_video(self, video_path: str, output_path: str = None, 
                     display: bool = True, fullscreen: bool = False, window_scale: float = 1.0):
        """Process video with simplified tracking"""
        
        print(f"\n🎬 Processing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup windows
        if display:
            cv2.namedWindow('CCTV Video Feed', cv2.WINDOW_NORMAL)
            cv2.namedWindow('CCTV Control Panel', cv2.WINDOW_NORMAL)
            
            if fullscreen:
                cv2.setWindowProperty('CCTV Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif window_scale != 1.0:
                scaled_w = int(width * window_scale)
                scaled_h = int(height * window_scale)
                cv2.resizeWindow('CCTV Video Feed', scaled_w, scaled_h)
                print(f"🖥️  Window: {scaled_w}x{scaled_h}")
        
        # Setup writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Track persons with BotSORT
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort_stable.yaml",
                    persist=True,
                    classes=[0],
                    conf=0.3,
                    verbose=False
                )
                
                # Create clean frame
                clean_frame = frame.copy()
                
                # Count persons by category
                counts = {'normal': 0, 'suspicious': 0, 'anomaly': 0}
                
                # Process detections
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        # Anomaly detection
                        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                            track_id, box.tolist(), frame_idx
                        )
                        
                        # Determine category
                        if anomaly_score > 1.0:
                            category = 'anomaly'
                        elif anomaly_score > 0.5:
                            category = 'suspicious'
                        else:
                            category = 'normal'
                        
                        counts[category] += 1
                        color = self.colors[category]
                        
                        # Draw on clean frame
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(clean_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Label
                        label = f"ID:{track_id} {category.upper()}"
                        if anomaly_score > 0:
                            label += f" ({anomaly_score:.2f})"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(clean_frame, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(clean_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update control panel data
                self.control_panel_data['frame_info'] = {'current': frame_idx, 'total': total_frames}
                self.control_panel_data['person_counts'] = counts
                
                # Create control panel
                control_panel = self.create_control_panel()
                
                # Write output
                if writer:
                    writer.write(clean_frame)
                
                # Display
                if display:
                    cv2.imshow('CCTV Video Feed', clean_frame)
                    cv2.imshow('CCTV Control Panel', control_panel)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                if frame_idx % 200 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_idx / elapsed
                    progress = (frame_idx / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% | FPS: {avg_fps:.1f} | Persons: {sum(counts.values())}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        
        print(f"\n🎯 COMPLETE")
        print(f"📊 Frames: {frame_idx}")
        print(f"⚡ FPS: {avg_fps:.1f}")
        print(f"⏱️  Time: {total_time:.1f}s")
        if output_path:
            print(f"💾 Output: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Dual Window CCTV - No ReID Conflicts')
    parser.add_argument('--input', '-i', required=True, help='Input video')
    parser.add_argument('--output', '-o', help='Output video')
    parser.add_argument('--camera-id', '-c', default='cam1', help='Camera ID')
    parser.add_argument('--no-display', action='store_true', help='No display')
    parser.add_argument('--fullscreen', '-f', action='store_true', help='Fullscreen')
    parser.add_argument('--window-scale', '-s', type=float, default=2.0, help='Window scale')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Video not found: {args.input}")
        return
    
    if not args.output:
        args.output = f"simple_output_{args.camera_id}.mp4"
    
    print("🚀 SIMPLIFIED DUAL WINDOW CCTV")
    print("=" * 60)
    print("✅ Uses BotSORT IDs directly (no ReID conflicts)")
    print("✅ Stable tracking through pose changes")
    print("✅ Clean dual window display")
    print("=" * 60)
    
    system = SimpleDualWindowCCTV(camera_id=args.camera_id)
    system.process_video(
        video_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        fullscreen=args.fullscreen,
        window_scale=args.window_scale
    )

if __name__ == "__main__":
    main()
