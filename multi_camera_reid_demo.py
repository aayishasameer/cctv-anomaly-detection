#!/usr/bin/env python3
"""
Multi-Camera ReID Demo
Demonstrates person re-identification across multiple camera angles
"""

import cv2
import numpy as np
from stealing_detection_system import StealingDetectionSystem
from person_reid_system import GlobalPersonTracker
import os
import time
import threading
from typing import Dict, List
import json

class MultiCameraReIDDemo:
    """Demo system for multi-camera person re-identification"""
    
    def __init__(self, camera_configs: Dict[str, Dict]):
        """
        Initialize multi-camera ReID demo
        
        Args:
            camera_configs: Dict with camera_id -> {'video_path': str, 'position': tuple}
        """
        self.camera_configs = camera_configs
        self.detectors = {}
        self.global_tracker = GlobalPersonTracker()
        
        # Shared data across cameras
        self.global_detections = {}
        self.frame_sync = {}
        self.running = True
        
        # Initialize detectors for each camera
        for camera_id, config in camera_configs.items():
            print(f"Initializing detector for {camera_id}...")
            try:
                detector = StealingDetectionSystem(
                    enable_reid=True,
                    camera_id=camera_id
                )
                # Share the same global tracker across all cameras
                detector.reid_tracker = self.global_tracker
                self.detectors[camera_id] = detector
                print(f"✅ {camera_id} detector ready")
            except Exception as e:
                print(f"❌ Failed to initialize {camera_id}: {e}")
        
        print(f"🎬 Multi-camera ReID demo initialized with {len(self.detectors)} cameras")
    
    def process_camera_stream(self, camera_id: str, video_path: str, 
                            output_path: str = None) -> Dict:
        """Process a single camera stream"""
        
        print(f"📹 Starting {camera_id} processing: {os.path.basename(video_path)}")
        
        detector = self.detectors[camera_id]
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        camera_detections = []
        
        try:
            while self.running and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Person detection and tracking
                results = detector.yolo_model.track(
                    source=frame,
                    tracker="botsort.yaml",
                    persist=True,
                    classes=[0],
                    conf=0.4,
                    verbose=False
                )
                
                # Hand detection
                hands = detector.hand_detector.detect_hands(frame)
                
                # Process frame
                annotated_frame = frame.copy()
                
                # Add camera ID overlay
                cv2.putText(annotated_frame, f"Camera: {camera_id}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                # Process detections with ReID
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        if conf < 0.4:
                            continue
                        
                        # Global ReID processing
                        global_id = self.global_tracker.update_global_tracking(
                            camera_id, track_id, frame, box.tolist(), conf, timestamp
                        )
                        
                        # Get person info
                        person_info = self.global_tracker.get_person_info(global_id)
                        cameras_seen = person_info.get('cameras', set()) if person_info else {camera_id}
                        
                        # Choose color based on global tracking status
                        if len(cameras_seen) > 1:
                            color = (0, 255, 0)  # Green for multi-camera tracked
                        elif global_id <= 5:  # First few persons get distinct colors
                            colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                            color = colors[global_id - 1]
                        else:
                            color = (128, 128, 128)  # Gray for others
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Create comprehensive label
                        label = f"L:{track_id} G:{global_id}"
                        if len(cameras_seen) > 1:
                            label += f" ({len(cameras_seen)} cams)"
                        
                        # Add quality info
                        if person_info:
                            avg_quality = np.mean(person_info.get('quality_scores', [0.5]))
                            label += f" Q:{avg_quality:.2f}"
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), 
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add camera list for multi-camera persons
                        if len(cameras_seen) > 1:
                            cam_list = ", ".join(sorted(cameras_seen))
                            cv2.putText(annotated_frame, f"Seen in: {cam_list}", 
                                      (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 255), 2)
                        
                        # Store detection data
                        detection_data = {
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'camera_id': camera_id,
                            'local_track_id': track_id,
                            'global_id': global_id,
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'cameras_seen': list(cameras_seen),
                            'multi_camera': len(cameras_seen) > 1
                        }
                        camera_detections.append(detection_data)
                
                # Add ReID statistics
                reid_stats = self.global_tracker.get_tracking_statistics()
                stats_text = [
                    f"Global Persons: {reid_stats['total_global_persons']}",
                    f"ReID Matches: {reid_stats['reid_matches']}",
                    f"Match Rate: {reid_stats['reid_match_rate']:.2%}"
                ]
                
                for i, text in enumerate(stats_text):
                    cv2.putText(annotated_frame, text, (10, height - 80 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Store frame for synchronized display
                self.frame_sync[camera_id] = annotated_frame.copy()
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 200 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"  {camera_id}: {progress:.1f}% | Global persons: {reid_stats['total_global_persons']}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        print(f"✅ {camera_id} processing completed: {len(camera_detections)} detections")
        return {
            'camera_id': camera_id,
            'detections': camera_detections,
            'total_frames': frame_idx
        }
    
    def run_synchronized_demo(self, display: bool = True, save_outputs: bool = True):
        """Run synchronized multi-camera demo"""
        
        print("🚀 Starting Multi-Camera ReID Demo")
        print("=" * 60)
        
        # Prepare output paths
        output_paths = {}
        if save_outputs:
            for camera_id in self.camera_configs.keys():
                output_paths[camera_id] = f"multicam_reid_{camera_id}_output.mp4"
        
        # Process each camera (in sequence for demo, could be parallel)
        all_results = {}
        
        for camera_id, config in self.camera_configs.items():
            video_path = config['video_path']
            output_path = output_paths.get(camera_id)
            
            if not os.path.exists(video_path):
                print(f"⚠️  Video not found for {camera_id}: {video_path}")
                continue
            
            # Process camera stream
            result = self.process_camera_stream(camera_id, video_path, output_path)
            all_results[camera_id] = result
        
        # Generate comprehensive report
        self.generate_reid_report(all_results)
        
        return all_results
    
    def generate_reid_report(self, results: Dict):
        """Generate comprehensive ReID analysis report"""
        
        print(f"\n📊 MULTI-CAMERA REID ANALYSIS REPORT")
        print("=" * 60)
        
        # Overall statistics
        total_detections = sum(len(r['detections']) for r in results.values())
        reid_stats = self.global_tracker.get_tracking_statistics()
        
        print(f"🎬 Cameras processed: {len(results)}")
        print(f"🎭 Total detections: {total_detections}")
        print(f"🌍 Global persons identified: {reid_stats['total_global_persons']}")
        print(f"🔄 ReID matches: {reid_stats['reid_matches']}")
        print(f"📈 Overall match rate: {reid_stats['reid_match_rate']:.2%}")
        
        # Per-camera breakdown
        print(f"\n📹 PER-CAMERA BREAKDOWN:")
        for camera_id, result in results.items():
            detections = result['detections']
            multi_cam_detections = [d for d in detections if d['multi_camera']]
            
            print(f"  {camera_id}:")
            print(f"    Detections: {len(detections)}")
            print(f"    Multi-camera persons: {len(set(d['global_id'] for d in multi_cam_detections))}")
            print(f"    Multi-camera rate: {len(multi_cam_detections)/max(1, len(detections)):.2%}")
        
        # Cross-camera analysis
        print(f"\n🔗 CROSS-CAMERA ANALYSIS:")
        
        # Find persons seen in multiple cameras
        global_person_cameras = {}
        for result in results.values():
            for detection in result['detections']:
                global_id = detection['global_id']
                if global_id not in global_person_cameras:
                    global_person_cameras[global_id] = set()
                global_person_cameras[global_id].add(detection['camera_id'])
        
        multi_camera_persons = {gid: cams for gid, cams in global_person_cameras.items() if len(cams) > 1}
        
        print(f"  Persons tracked across multiple cameras: {len(multi_camera_persons)}")
        
        if multi_camera_persons:
            print(f"  Multi-camera tracking details:")
            for global_id, cameras in sorted(multi_camera_persons.items()):
                person_info = self.global_tracker.get_person_info(global_id)
                duration = person_info.get('last_seen', 0) - person_info.get('first_seen', 0)
                print(f"    Global ID {global_id}: {sorted(cameras)} ({duration:.1f}s duration)")
        
        # Save detailed report
        report_data = {
            'summary': {
                'cameras_processed': len(results),
                'total_detections': total_detections,
                'global_persons': reid_stats['total_global_persons'],
                'reid_matches': reid_stats['reid_matches'],
                'match_rate': reid_stats['reid_match_rate']
            },
            'per_camera': {
                camera_id: {
                    'detections': len(result['detections']),
                    'frames_processed': result['total_frames']
                }
                for camera_id, result in results.items()
            },
            'multi_camera_persons': {
                str(gid): list(cams) for gid, cams in multi_camera_persons.items()
            },
            'reid_statistics': reid_stats
        }
        
        with open('multi_camera_reid_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: multi_camera_reid_report.json")
        print(f"🎬 Output videos saved with 'multicam_reid_' prefix")

def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-camera ReID demo')
    parser.add_argument('--videos', '-v', nargs='+', required=True,
                       help='Video files for different cameras')
    parser.add_argument('--camera-ids', '-c', nargs='+', 
                       help='Camera IDs (default: cam1, cam2, ...)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    
    args = parser.parse_args()
    
    # Validate inputs
    for video in args.videos:
        if not os.path.exists(video):
            print(f"❌ Video not found: {video}")
            return
    
    # Generate camera IDs if not provided
    if args.camera_ids:
        if len(args.camera_ids) != len(args.videos):
            print("❌ Number of camera IDs must match number of videos")
            return
        camera_ids = args.camera_ids
    else:
        camera_ids = [f"cam{i+1}" for i in range(len(args.videos))]
    
    # Create camera configurations
    camera_configs = {}
    for i, (camera_id, video_path) in enumerate(zip(camera_ids, args.videos)):
        camera_configs[camera_id] = {
            'video_path': video_path,
            'position': (i * 100, i * 100)  # Dummy positions
        }
    
    print(f"🎬 Multi-Camera ReID Demo Configuration:")
    for camera_id, config in camera_configs.items():
        print(f"  {camera_id}: {os.path.basename(config['video_path'])}")
    
    # Initialize and run demo
    try:
        demo = MultiCameraReIDDemo(camera_configs)
        results = demo.run_synchronized_demo(
            display=not args.no_display,
            save_outputs=True
        )
        
        print(f"\n🏆 Multi-camera ReID demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()