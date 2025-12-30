import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
import os
import argparse
from typing import Dict, Tuple

class AnomalyTracker:
    """Real-time anomaly detection with tracking visualization"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth"):
        print("Initializing Anomaly Tracker...")
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Load anomaly detector
        self.anomaly_detector = AnomalyDetector(model_path)
        try:
            self.anomaly_detector.load_model()
            print("✓ Anomaly detection model loaded successfully")
        except FileNotFoundError:
            print("❌ Anomaly detection model not found!")
            print("Please run 'python train_vae_model.py' first to train the model.")
            raise
        
        # Track anomaly states
        self.track_anomaly_scores = {}
        self.track_anomaly_history = {}
        self.anomaly_threshold_frames = 10  # Consecutive frames to confirm anomaly
        
        # Colors
        self.normal_color = (0, 255, 0)      # Green
        self.anomaly_color = (0, 0, 255)     # Red
        self.warning_color = (0, 165, 255)   # Orange
        
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """Process video with anomaly detection"""
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
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
                
                # Run tracking
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.3,
                    verbose=False
                )
                
                # Process detections
                annotated_frame = frame.copy()
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Detect anomaly
                        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                            track_id, box.tolist(), frame_idx
                        )
                        
                        # Update anomaly history
                        if track_id not in self.track_anomaly_history:
                            self.track_anomaly_history[track_id] = []
                        
                        self.track_anomaly_history[track_id].append(is_anomaly)
                        
                        # Keep only recent history
                        if len(self.track_anomaly_history[track_id]) > self.anomaly_threshold_frames:
                            self.track_anomaly_history[track_id] = \
                                self.track_anomaly_history[track_id][-self.anomaly_threshold_frames:]
                        
                        # Determine final anomaly status (require 70% of frames to be anomalous)
                        recent_anomalies = sum(self.track_anomaly_history[track_id])
                        is_confirmed_anomaly = recent_anomalies >= int(self.anomaly_threshold_frames * 0.7)
                        
                        # Choose color based on anomaly status
                        if is_confirmed_anomaly:
                            color = self.anomaly_color
                            status = "ANOMALY"
                        elif recent_anomalies > 0:
                            color = self.warning_color
                            status = "WARNING"
                        else:
                            color = self.normal_color
                            status = "NORMAL"
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw track ID and status
                        label = f"ID:{track_id} {status}"
                        if anomaly_score > 0:
                            label += f" ({anomaly_score:.3f})"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Log anomalies
                        if is_confirmed_anomaly:
                            anomaly_detections.append({
                                'frame': frame_idx,
                                'track_id': track_id,
                                'bbox': box.tolist(),
                                'score': anomaly_score,
                                'timestamp': frame_idx / fps
                            })
                
                # Add frame info
                info_text = f"Frame: {frame_idx}/{total_frames} | Anomalies: {len(anomaly_detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display legend
                legend_y = 60
                cv2.putText(annotated_frame, "Green: Normal", (10, legend_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.normal_color, 2)
                cv2.putText(annotated_frame, "Orange: Warning", (10, legend_y + 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.warning_color, 2)
                cv2.putText(annotated_frame, "Red: Anomaly", (10, legend_y + 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.anomaly_color, 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Anomaly Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Pause on spacebar
                        cv2.waitKey(0)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Anomalies detected: {len(anomaly_detections)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total anomalies detected: {len(anomaly_detections)}")
        
        if anomaly_detections:
            print(f"\nAnomaly Summary:")
            for i, detection in enumerate(anomaly_detections[:10]):  # Show first 10
                timestamp = detection['timestamp']
                print(f"  {i+1}. Track {detection['track_id']} at {timestamp:.1f}s "
                      f"(frame {detection['frame']}) - Score: {detection['score']:.3f}")
            
            if len(anomaly_detections) > 10:
                print(f"  ... and {len(anomaly_detections) - 10} more")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        
        return anomaly_detections

def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection on video')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--model', '-m', default='models/vae_anomaly_detector.pth', 
                       help='Path to trained VAE model')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Initialize tracker
    try:
        tracker = AnomalyTracker(args.model)
    except FileNotFoundError:
        return
    
    # Process video
    display = not args.no_display
    anomalies = tracker.process_video(args.input, args.output, display)
    
    print(f"\nProcessing completed successfully!")

if __name__ == "__main__":
    main()