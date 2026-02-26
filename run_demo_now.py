#!/usr/bin/env python3
"""
Quick Demo Runner - Run CCTV System Now
"""

import os
import cv2
import time
from complete_cctv_system import CompleteCCTVSystem

def run_demo():
    """Run demo without GUI display"""
    
    print("🚀 CCTV ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    
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
    
    # Initialize system
    try:
        print("\n⚙️  Initializing system with trained models...")
        system = CompleteCCTVSystem(camera_id="demo")
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process video without display
    output_path = f"demo_output_{int(time.time())}.mp4"
    
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
        
        if 'reid_statistics' in results:
            print(f"\n👥 PERSON TRACKING:")
            print(f"   Total persons: {results['reid_statistics']['total_global_persons']}")
            print(f"   ReID matches: {results['reid_statistics']['reid_matches']}")
            print(f"   Match rate: {results['reid_statistics']['reid_match_rate']:.1%}")
        
        # Show anomaly statistics
        if 'anomaly_statistics' in results:
            stats = results['anomaly_statistics']
            print(f"\n🚨 ANOMALY DETECTION:")
            print(f"   Normal behavior: {stats.get('normal', 0)} detections")
            print(f"   Suspicious behavior: {stats.get('suspicious', 0)} detections") 
            print(f"   Anomalous behavior: {stats.get('anomaly', 0)} detections")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   Video: {output_path}")
        print(f"   ReID data: reid_data_demo.pkl")
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
