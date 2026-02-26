#!/usr/bin/env python3
"""
Create a summary image showing detection results
"""

import cv2
import numpy as np
import os
import glob

def create_summary_grid(frames_dir="result_frames", output_path="detection_summary.jpg"):
    """Create a grid of sample frames"""
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not frame_files:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"📸 Found {len(frame_files)} frames")
    
    # Select 9 frames for 3x3 grid
    num_frames = min(9, len(frame_files))
    step = len(frame_files) // num_frames
    selected_frames = [frame_files[i * step] for i in range(num_frames)]
    
    # Load frames
    images = []
    for frame_path in selected_frames:
        img = cv2.imread(frame_path)
        if img is not None:
            # Add frame number text
            frame_num = os.path.basename(frame_path).split('_')[1]
            cv2.putText(img, f"Frame {frame_num}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            images.append(img)
    
    if not images:
        print("❌ Failed to load images")
        return
    
    # Create grid (3x3)
    rows = []
    for i in range(0, len(images), 3):
        row_images = images[i:i+3]
        # Pad if needed
        while len(row_images) < 3:
            row_images.append(np.zeros_like(images[0]))
        row = np.hstack(row_images)
        rows.append(row)
    
    # Stack rows
    grid = np.vstack(rows)
    
    # Add title
    title_height = 60
    title_img = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_img, "CCTV Anomaly Detection Results", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Combine
    final_img = np.vstack([title_img, grid])
    
    # Save
    cv2.imwrite(output_path, final_img)
    print(f"✅ Summary saved to: {output_path}")
    print(f"📊 Grid size: {final_img.shape[1]}x{final_img.shape[0]}")
    
    return output_path

if __name__ == "__main__":
    print("🎨 Creating Detection Summary")
    print("=" * 60)
    create_summary_grid()
