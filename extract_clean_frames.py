#!/usr/bin/env python3
"""Extract sample frames from clean display output"""
import cv2
import os

video_path = "clean_display_output.mp4"
output_dir = "clean_display_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract frames at different points
frame_positions = [0.2, 0.5, 0.8]

for i, pos in enumerate(frame_positions):
    frame_num = int(total_frames * pos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        output_path = f"{output_dir}/clean_frame_{i+1}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"✓ Saved: {output_path}")

cap.release()
print(f"\n✓ Extracted {len(frame_positions)} clean frames")
print(f"✓ Output video: {video_path}")
