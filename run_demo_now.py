#!/usr/bin/env python3
"""
Quick Demo Runner - Run Simplified CCTV System
Uses only local tracker IDs for stable single-camera tracking
"""

import os
import cv2
import time
from simple_cctv_system import SimpleCCTVSystem

def run_demo():
    """Run demo without GUI display"""
    
    print("🚀 SIMPLIFIED CCTV ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    print("📌 Using local tracker IDs only (no Global ReID)")
    print("")
    
    # Find test video
    test_video = "cctv-anomaly-detection/cctv-anomaly-detection-1/working/test_anomaly/Shoplifting020_x264.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Searching for videos...")
        import glob
        videos = glob.glob("**/Shoplifting*.mp4", recursive=True)
        if videos:
            test_video = videos[0]
            print(f"✅ Found: {test_video}")
        else:
            print("❌ No test videos found")
            return
    
    print(f"📹 Processing: {os.path.basename(test_video)}")
    
    # Initialize simplified system
    try:
        print("\n⚙️  Initializing simplified system...")
        system = SimpleCCTVSystem(camera_id="demo")
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process video without display
    output_path = f"simple_demo_output_{int(time.time())}.mp4"
    
    print(f"\n🎬 Processing video (headless mode)...")
    print(f"💾 Output will be saved to: {output_path}")
    print(f"⏳ This may take a few minutes...\n")
    
    try:
        results = system.process_video(
            video_path=test_video,
            output_path=output_path,
            display=False  # No GUI display
        )
        
        print(f"\n" + "=" * 60)
        print(f"🎉 PROCESSING COMPLETED!")
        print(f"=" * 60)
        print(f"\n📊 RESULTS:")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Processing FPS: {results['avg_fps']:.1f}")
        print(f"   Total runtime: {results.get('total_time', 0):.1f}s")
        print(f"   Total tracks: {results['total_tracks']}")
        print(f"   Active tracks: {results['active_tracks']}")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   Video: {output_path}")
        print(f"\n✅ Demo completed successfully!")
        print(f"\n📌 Note: This system uses only local tracker IDs")
        print(f"   No Global ReID = More stable tracking!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
