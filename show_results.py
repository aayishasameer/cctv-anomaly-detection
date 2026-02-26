#!/usr/bin/env python3
"""
Extract and display sample frames from the processed video
"""

import cv2
import os
import sys

def extract_sample_frames(video_path, output_dir="result_frames", num_frames=10):
    """Extract sample frames from processed video"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Extract frames at regular intervals
    interval = total_frames // num_frames
    frame_count = 0
    saved_count = 0
    
    print(f"\n📸 Extracting {num_frames} sample frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at intervals
        if frame_count % interval == 0 and saved_count < num_frames:
            output_path = os.path.join(output_dir, f"frame_{saved_count:03d}_at_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            timestamp = frame_count / fps
            print(f"   ✅ Saved frame {saved_count+1}/{num_frames} at {timestamp:.1f}s -> {output_path}")
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n🎉 Extraction complete!")
    print(f"📁 Frames saved to: {output_dir}/")
    print(f"💡 You can view these frames to see the detection results")
    
    return output_dir

if __name__ == "__main__":
    video_path = "demo_output_1772090434.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    print("🎬 CCTV Detection Results Viewer")
    print("=" * 60)
    
    extract_sample_frames(video_path, num_frames=15)
