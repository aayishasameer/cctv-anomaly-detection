#!/usr/bin/env python3
"""
Enhanced Anomaly Detection Tracker with Evaluation Metrics and ReID Integration
"""

import cv2
import numpy as np
from ultralytics import YOLO
from vae_anomaly_detector import AnomalyDetector
from evaluation_metrics import AnomalyEvaluator, MOTEvaluator, PerformanceProfiler
from multi_camera_reid import MultiCameraReID
import os
import argparse
import time
import json
from typing import Dict, Tuple, List

class EnhancedAnomalyTracker:
    """Enhanced tracker with evaluation and ReID capabilities"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth",
                 reid_model_path: str = "models/reid_model.pth",
                 enable_reid: bool = False):
        
        print("Initializing Enhanced Anomaly Tracker...")
        
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
        
        # Initialize ReID system if enabled
        self.enable_reid = enable_reid
        if enable_reid:
            self.reid_system = MultiCameraReID(reid_model_path)
            print("✓ ReID system initialized")
        else:
            self.reid_system = None
        
        # Initialize evaluators
        self.anomaly_evaluator = AnomalyEvaluator()
        self.mot_evaluator = MOTEvaluator()
        self.performance_profiler = PerformanceProfiler()
        
        # Track anomaly states
        self.track_anomaly_scores = {}
        self.track_anomaly_history = {}
        self.anomaly_threshold_frames = 10
        
        # Colors
        self.normal_color = (0, 255, 0)      # Green
        self.anomaly_color = (0, 0, 255)     # Red
        self.warning_color = (0, 165, 255)   # Orange
        
        # Performance tracking
        self.total_detections = 0
        self.total_anomalies = 0
        
    def extract_person_crop(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract person crop from frame for ReID"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add some padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        return person_crop
    
    def process_video_with_evaluation(self, video_path: str, output_path: str = None, 
                                    display: bool = True, ground_truth_file: str = None,
                                    camera_id: str = "cam1") -> Dict:
        """Process video with comprehensive evaluation"""
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Load ground truth if provided
        if ground_truth_file and os.path.exists(ground_truth_file):
            self.anomaly_evaluator.load_ground_truth(ground_truth_file)
            print(f"✓ Ground truth loaded from {ground_truth_file}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        anomaly_detections = []
        
        try:
            while True:
                self.performance_profiler.start_frame()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detection timing
                detection_start = time.time()
                
                # Run tracking
                results = self.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],  # person only
                    conf=0.3,
                    verbose=False
                )
                
                detection_time = time.time() - detection_start
                self.performance_profiler.add_detection_time(detection_time)
                
                # Process detections
                annotated_frame = frame.copy()
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    tracking_start = time.time()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Add to MOT evaluator
                        self.mot_evaluator.add_predicted_track(frame_idx, track_id, box.tolist(), conf)
                        
                        # Detect anomaly
                        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                            track_id, box.tolist(), frame_idx
                        )
                        
                        # ReID processing if enabled
                        global_track_id = track_id  # Default to local ID
                        if self.enable_reid and self.reid_system:
                            person_crop = self.extract_person_crop(frame, box)
                            global_track_id = self.reid_system.update_global_tracking(
                                camera_id, track_id, person_crop, box.tolist()
                            )
                        
                        # Update anomaly history
                        if track_id not in self.track_anomaly_history:
                            self.track_anomaly_history[track_id] = []
                        
                        self.track_anomaly_history[track_id].append(is_anomaly)
                        
                        # Keep only recent history
                        if len(self.track_anomaly_history[track_id]) > self.anomaly_threshold_frames:
                            self.track_anomaly_history[track_id] = \
                                self.track_anomaly_history[track_id][-self.anomaly_threshold_frames:]
                        
                        # Determine final anomaly status
                        recent_anomalies = sum(self.track_anomaly_history[track_id])
                        is_confirmed_anomaly = recent_anomalies >= int(self.anomaly_threshold_frames * 0.7)
                        
                        # Add to anomaly evaluator
                        timestamp = frame_idx / fps
                        self.anomaly_evaluator.add_prediction(
                            frame_idx, track_id, is_confirmed_anomaly, anomaly_score, timestamp
                        )
                        
                        # Choose color and status
                        if is_confirmed_anomaly:
                            color = self.anomaly_color
                            status = "ANOMALY"
                            self.total_anomalies += 1
                        elif recent_anomalies > 0:
                            color = self.warning_color
                            status = "WARNING"
                        else:
                            color = self.normal_color
                            status = "NORMAL"
                        
                        self.total_detections += 1
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with ReID info
                        if self.enable_reid and global_track_id != track_id:
                            label = f"L:{track_id} G:{global_track_id} {status}"
                        else:
                            label = f"ID:{track_id} {status}"
                        
                        if anomaly_score > 0:
                            label += f" ({anomaly_score:.3f})"
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Log confirmed anomalies
                        if is_confirmed_anomaly:
                            anomaly_detections.append({
                                'frame': frame_idx,
                                'track_id': track_id,
                                'global_track_id': global_track_id,
                                'bbox': box.tolist(),
                                'score': anomaly_score,
                                'timestamp': timestamp
                            })
                    
                    tracking_time = time.time() - tracking_start
                    self.performance_profiler.add_tracking_time(tracking_time)
                
                # Add performance info to frame
                perf_stats = self.performance_profiler.get_performance_stats()
                current_fps = perf_stats.get('average_fps', 0)
                
                info_text = f"Frame: {frame_idx}/{total_frames} | FPS: {current_fps:.1f} | Anomalies: {len(anomaly_detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add legend and ReID info
                legend_y = 60
                cv2.putText(annotated_frame, "Green: Normal", (10, legend_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.normal_color, 2)
                cv2.putText(annotated_frame, "Orange: Warning", (10, legend_y + 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.warning_color, 2)
                cv2.putText(annotated_frame, "Red: Anomaly", (10, legend_y + 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.anomaly_color, 2)
                
                if self.enable_reid:
                    cv2.putText(annotated_frame, "L:Local G:Global", (10, legend_y + 75), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Enhanced Anomaly Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Pause
                        cv2.waitKey(0)
                
                frame_idx += 1
                self.performance_profiler.end_frame()
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Current FPS: {current_fps:.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Generate comprehensive results
        results = self._generate_evaluation_results(
            video_path, anomaly_detections, ground_truth_file
        )
        
        return results
    
    def _generate_evaluation_results(self, video_path: str, anomaly_detections: List,
                                   ground_truth_file: str = None) -> Dict:
        """Generate comprehensive evaluation results"""
        
        video_name = os.path.basename(video_path)
        
        # Performance statistics
        perf_stats = self.performance_profiler.get_performance_stats()
        
        # Basic statistics
        basic_stats = {
            'total_frames_processed': perf_stats.get('total_frames', 0),
            'total_detections': self.total_detections,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / self.total_detections if self.total_detections > 0 else 0,
            'processing_fps': perf_stats.get('average_fps', 0),
            'average_detection_time': perf_stats.get('average_detection_time', 0),
            'average_tracking_time': perf_stats.get('average_tracking_time', 0)
        }
        
        results = {
            'video_name': video_name,
            'basic_statistics': basic_stats,
            'performance_metrics': perf_stats,
            'anomaly_detections': anomaly_detections
        }
        
        # Evaluation metrics if ground truth available
        if ground_truth_file and os.path.exists(ground_truth_file):
            try:
                # Anomaly detection metrics
                anomaly_metrics = self.anomaly_evaluator.evaluate_frame_level(video_name)
                temporal_metrics = self.anomaly_evaluator.evaluate_temporal_consistency(video_name)
                
                # MOT metrics (if ground truth tracking available)
                # mot_metrics = self.mot_evaluator.calculate_mota_motp()
                
                results.update({
                    'anomaly_detection_metrics': anomaly_metrics,
                    'temporal_consistency_metrics': temporal_metrics,
                    # 'tracking_metrics': mot_metrics
                })
                
                print(f"\n=== Evaluation Results ===")
                print(f"Accuracy: {anomaly_metrics['accuracy']:.3f}")
                print(f"Precision: {anomaly_metrics['precision']:.3f}")
                print(f"Recall: {anomaly_metrics['recall']:.3f}")
                print(f"F1-Score: {anomaly_metrics['f1_score']:.3f}")
                print(f"AUC-ROC: {anomaly_metrics['auc_roc']:.3f}")
                
            except Exception as e:
                print(f"Warning: Could not compute evaluation metrics: {e}")
        
        # ReID statistics if enabled
        if self.enable_reid and self.reid_system:
            reid_stats = self.reid_system.get_system_statistics()
            results['reid_statistics'] = reid_stats
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced anomaly detection with evaluation')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--ground-truth', '-gt', help='Ground truth JSON file')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--enable-reid', action='store_true', help='Enable ReID system')
    parser.add_argument('--camera-id', default='cam1', help='Camera ID for ReID')
    parser.add_argument('--results-file', '-r', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Check input
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return
    
    # Initialize enhanced tracker
    try:
        tracker = EnhancedAnomalyTracker(enable_reid=args.enable_reid)
    except FileNotFoundError:
        return
    
    # Process video
    display = not args.no_display
    results = tracker.process_video_with_evaluation(
        args.input, 
        args.output, 
        display, 
        args.ground_truth,
        args.camera_id
    )
    
    # Save results if requested
    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.results_file}")
    
    print(f"\nProcessing completed successfully!")

if __name__ == "__main__":
    main()