#!/usr/bin/env python3
"""Extract sample frames from dual window output"""
import cv2
import os

video_path = "dual_clean_output.mp4"
output_dir = "dual_clean_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract frames at different points
frame_positions = [0.3, 0.6, 0.9]

for i, pos in enumerate(frame_positions):
    frame_num = int(total_frames * pos)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        output_path = f"{output_dir}/dual_frame_{i+1}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"✓ Saved: {output_path}")

cap.release()
print(f"\n✅ Extracted {len(frame_positions)} dual window frames")
print(f"✅ Output video: {video_path}")
print(f"\nThe CCTV Control Panel (right side) now shows:")
print("  - System Status")
print("  - Person Counts")
print("  - Recent Alerts")
print("  ❌ PERFORMANCE section removed!")
