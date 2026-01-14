#!/usr/bin/env python3
"""
Simple CCTV Demo - No GUI
Process video and show results without display window
"""

import os
import cv2
import time
from complete_cctv_system import CompleteCCTVSystem

def run_simple_demo():
    """Run demo without GUI display"""
    
    print("🚀 CCTV ANOMALY DETECTION DEMO")
    print("=" * 50)
    
    # Use test video
    test_video = "working/test_anomaly/Shoplifting020_x264.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    print(f"📹 Processing: {os.path.basename(test_video)}")
    
    # Initialize system
    try:
        system = CompleteCCTVSystem(camera_id="demo")
        print("✅ System initialized with all trained models")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return
    
    # Process video without display
    output_path = f"demo_output_{int(time.time())}.mp4"
    
    print(f"\n🎬 Processing video (no GUI)...")
    print(f"💾 Output: {output_path}")
    
    try:
        results = system.process_video(
            video_path=test_video,
            output_path=output_path,
            display=False  # No GUI display
        )
        
        print(f"\n🎉 DEMO COMPLETED!")
        print(f"📊 RESULTS:")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Processing FPS: {results['avg_fps']:.1f}")
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
        
        print(f"\n💾 Output video: {output_path}")
        print(f"📁 ReID data: reid_data_demo.pkl")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_simple_demo()