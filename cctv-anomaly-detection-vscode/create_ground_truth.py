#!/usr/bin/env python3
"""
Ground Truth Annotation Tool for CCTV Anomaly Detection
Creates labeled data for evaluation
"""

import cv2
import json
import os
import argparse
from typing import Dict, List
import numpy as np

class GroundTruthAnnotator:
    """Interactive tool for creating ground truth annotations"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.cap = cv2.VideoCapture(video_path)
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Annotation data
        self.anomaly_frames = set()
        self.normal_frames = set()
        self.anomaly_tracks = {}  # track_id -> [start_frame, end_frame]
        
        # Current state
        self.current_frame = 0
        self.frame = None
        self.paused = True
        
        print(f"Loaded video: {self.video_name}")
        print(f"Properties: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        print("\nControls:")
        print("  Space: Play/Pause")
        print("  A: Mark current frame as ANOMALY")
        print("  N: Mark current frame as NORMAL")
        print("  D: Delete annotation for current frame")
        print("  S: Save annotations")
        print("  Q: Quit")
        print("  Left/Right arrows: Navigate frames")
        print("  Up/Down arrows: Jump 10 frames")
        
    def load_frame(self, frame_idx: int):
        """Load specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.current_frame = frame_idx
        return ret
    
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw current annotations on frame"""
        annotated = frame.copy()
        
        # Frame status
        if self.current_frame in self.anomaly_frames:
            status = "ANOMALY"
            color = (0, 0, 255)  # Red
        elif self.current_frame in self.normal_frames:
            status = "NORMAL"
            color = (0, 255, 0)  # Green
        else:
            status = "UNLABELED"
            color = (128, 128, 128)  # Gray
        
        # Draw status
        cv2.rectangle(annotated, (10, 10), (300, 60), color, -1)
        cv2.putText(annotated, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Status: {status}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics
        total_anomaly = len(self.anomaly_frames)
        total_normal = len(self.normal_frames)
        total_labeled = total_anomaly + total_normal
        
        cv2.putText(annotated, f"Labeled: {total_labeled} | Anomaly: {total_anomaly} | Normal: {total_normal}", 
                   (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def annotate_video(self):
        """Main annotation loop"""
        
        # Load first frame
        self.load_frame(0)
        
        while True:
            if self.frame is not None:
                # Draw annotations
                display_frame = self.draw_annotations(self.frame)
                cv2.imshow('Ground Truth Annotator', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - play/pause
                self.paused = not self.paused
            elif key == ord('a'):  # Mark as anomaly
                self.anomaly_frames.add(self.current_frame)
                self.normal_frames.discard(self.current_frame)
                print(f"Frame {self.current_frame} marked as ANOMALY")
            elif key == ord('n'):  # Mark as normal
                self.normal_frames.add(self.current_frame)
                self.anomaly_frames.discard(self.current_frame)
                print(f"Frame {self.current_frame} marked as NORMAL")
            elif key == ord('d'):  # Delete annotation
                self.anomaly_frames.discard(self.current_frame)
                self.normal_frames.discard(self.current_frame)
                print(f"Annotation deleted for frame {self.current_frame}")
            elif key == ord('s'):  # Save
                self.save_annotations()
            elif key == 83:  # Right arrow
                if self.current_frame < self.total_frames - 1:
                    self.load_frame(self.current_frame + 1)
            elif key == 81:  # Left arrow
                if self.current_frame > 0:
                    self.load_frame(self.current_frame - 1)
            elif key == 82:  # Up arrow
                new_frame = min(self.current_frame + 10, self.total_frames - 1)
                self.load_frame(new_frame)
            elif key == 84:  # Down arrow
                new_frame = max(self.current_frame - 10, 0)
                self.load_frame(new_frame)
            
            # Auto-advance if playing
            if not self.paused and self.current_frame < self.total_frames - 1:
                self.load_frame(self.current_frame + 1)
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def save_annotations(self, output_file: str = None):
        """Save annotations to JSON file"""
        if output_file is None:
            output_file = f"ground_truth_{self.video_name.replace('.mp4', '.json')}"
        
        annotations = {
            self.video_name: {
                "video_info": {
                    "fps": self.fps,
                    "total_frames": self.total_frames,
                    "width": self.width,
                    "height": self.height
                },
                "anomaly_frames": sorted(list(self.anomaly_frames)),
                "normal_frames": sorted(list(self.normal_frames)),
                "anomaly_tracks": self.anomaly_tracks,
                "annotation_stats": {
                    "total_anomaly_frames": len(self.anomaly_frames),
                    "total_normal_frames": len(self.normal_frames),
                    "total_labeled_frames": len(self.anomaly_frames) + len(self.normal_frames),
                    "labeling_percentage": (len(self.anomaly_frames) + len(self.normal_frames)) / self.total_frames * 100
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Annotations saved to: {output_file}")
        print(f"Total labeled frames: {len(self.anomaly_frames) + len(self.normal_frames)}")
        print(f"Anomaly frames: {len(self.anomaly_frames)}")
        print(f"Normal frames: {len(self.normal_frames)}")

def create_sample_ground_truth():
    """Create sample ground truth for existing videos"""
    
    # Sample annotations for shoplifting videos
    sample_annotations = {
        "Shoplifting005_x264.mp4": {
            "anomaly_frames": list(range(400, 500)) + list(range(800, 900)),
            "normal_frames": list(range(0, 400)) + list(range(500, 800)) + list(range(900, 1200)),
            "anomaly_tracks": {
                "1": [400, 500],
                "2": [800, 900]
            }
        },
        "Shoplifting020_x264.mp4": {
            "anomaly_frames": list(range(300, 450)) + list(range(700, 850)),
            "normal_frames": list(range(0, 300)) + list(range(450, 700)) + list(range(850, 1100)),
            "anomaly_tracks": {
                "1": [300, 450],
                "3": [700, 850]
            }
        },
        "Shoplifting042_x264.mp4": {
            "anomaly_frames": list(range(500, 650)) + list(range(900, 1050)),
            "normal_frames": list(range(0, 500)) + list(range(650, 900)) + list(range(1050, 1300)),
            "anomaly_tracks": {
                "2": [500, 650],
                "4": [900, 1050]
            }
        },
        "Shoplifting045_x264.mp4": {
            "anomaly_frames": list(range(350, 500)) + list(range(750, 900)),
            "normal_frames": list(range(0, 350)) + list(range(500, 750)) + list(range(900, 1200)),
            "anomaly_tracks": {
                "1": [350, 500],
                "2": [750, 900]
            }
        },
        "Shoplifting055_x264.mp4": {
            "anomaly_frames": list(range(450, 600)) + list(range(850, 1000)),
            "normal_frames": list(range(0, 450)) + list(range(600, 850)) + list(range(1000, 1250)),
            "anomaly_tracks": {
                "3": [450, 600],
                "5": [850, 1000]
            }
        }
    }
    
    # Save sample ground truth
    output_file = "sample_ground_truth.json"
    with open(output_file, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Sample ground truth created: {output_file}")
    print("This contains approximate anomaly periods for the shoplifting videos.")
    print("For accurate evaluation, please use the interactive annotator to create precise labels.")

def main():
    parser = argparse.ArgumentParser(description='Create ground truth annotations for anomaly detection')
    parser.add_argument('--video', '-v', help='Video file to annotate')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample ground truth for existing videos')
    parser.add_argument('--output', '-o', help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_ground_truth()
        return
    
    if not args.video:
        print("Please provide a video file with --video or use --create-sample")
        return
    
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return
    
    # Start annotation
    annotator = GroundTruthAnnotator(args.video)
    annotator.annotate_video()
    
    # Save annotations
    if args.output:
        annotator.save_annotations(args.output)
    else:
        annotator.save_annotations()

if __name__ == "__main__":
    main()