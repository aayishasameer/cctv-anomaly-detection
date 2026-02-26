#!/usr/bin/env python3
"""
Run Simplified CCTV System on Specific Video
Uses only local tracker IDs for stable tracking
"""

import os
import cv2
import time
from simple_cctv_system import SimpleCCTVSystem

def run_video_detection():
    """Run detection on specific video"""
    
    print("🚀 SIMPLIFIED CCTV ANOMALY DETECTION SYSTEM")
    print("=" * 70)
    print("📌 Using local tracker IDs only (no Global ReID)")
    print("")
    
    # Specific video path
    video_path = "cctv-anomaly-detection/cctv-anomaly-detection-1/working/test_anomaly/Shoplifting020_x264.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    cap.release()
    
    print(f"📹 INPUT VIDEO:")
    print(f"   File: {os.path.basename(video_path)}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {total_frames}")
    print(f"   Duration: {duration:.1f}s")
    
    # Initialize simplified system
    print(f"\n⚙️  INITIALIZING SIMPLIFIED SYSTEM...")
    try:
        system = SimpleCCTVSystem(camera_id="shoplifting_test")
        print("✅ System initialized with local tracking only")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Output path
    output_path = "simple_shoplifting_detection_output.mp4"
    
    print(f"\n🎬 PROCESSING VIDEO...")
    print(f"💾 Output: {output_path}")
    print(f"⏳ Estimated time: {duration/15:.0f}-{duration/20:.0f} minutes")
    print(f"\n{'='*70}")
    
    start_time = time.time()
    
    try:
        results = system.process_video(
            video_path=video_path,
            output_path=output_path,
            display=False  # Headless mode
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"🎉 PROCESSING COMPLETED!")
        print(f"{'='*70}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Processing time: {processing_time:.1f}s ({processing_time/60:.1f} min)")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Processing FPS: {results['avg_fps']:.1f}")
        print(f"   Speed ratio: {results['avg_fps']/fps:.2f}x realtime")
        
        print(f"\n👥 TRACKING STATISTICS:")
        print(f"   Total tracks: {results['total_tracks']}")
        print(f"   Active tracks: {results['active_tracks']}")
        print(f"   Tracking method: Local tracker IDs only")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   Video: {output_path}")
        
        # Check output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"   Size: {file_size:.1f} MB")
            print(f"   ✅ Video saved successfully!")
        
        print(f"\n{'='*70}")
        print(f"✅ DETECTION COMPLETE!")
        print(f"{'='*70}")
        
        print(f"\n📌 BENEFITS OF SIMPLIFIED SYSTEM:")
        print(f"   ✅ No Global ReID = More stable IDs")
        print(f"   ✅ Uses only tracker IDs = No ID reassignment")
        print(f"   ✅ Single camera optimized = Better performance")
        print(f"   ✅ Reduced complexity = More reliable")
        
        # Extract sample frames
        print(f"\n📸 Extracting sample frames for preview...")
        extract_frames(output_path)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

def extract_frames(video_path, num_frames=12):
    """Extract sample frames from output"""
    
    output_dir = "simple_detection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open output video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    
    saved = 0
    frame_idx = 0
    
    while saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            output_path = f"{output_dir}/result_frame_{saved:02d}.jpg"
            cv2.imwrite(output_path, frame)
            saved += 1
        
        frame_idx += 1
    
    cap.release()
    
    print(f"✅ Extracted {saved} frames to {output_dir}/")
    print(f"💡 View these frames to see stable tracking with local IDs")

if __name__ == "__main__":
    run_video_detection()
