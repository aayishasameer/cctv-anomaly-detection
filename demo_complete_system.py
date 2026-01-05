#!/usr/bin/env python3
"""
Demo: Complete CCTV System
Quick demonstration of the complete system with all features
"""

import os
import cv2
import numpy as np
from complete_cctv_system import CompleteCCTVSystem

def find_test_video():
    """Find a suitable test video"""
    
    search_paths = [
        "working/test_anomaly",
        "working", 
        "data",
        "."
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    return os.path.join(path, file)
    
    return None

def create_demo_video():
    """Create a simple demo video if no test video is found"""
    
    print("🎬 Creating demo video...")
    
    width, height = 640, 480
    fps = 30
    duration = 20  # seconds
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_video.mp4', fourcc, fps, (width, height))
    
    # Create demo with moving persons
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(30)  # Dark background
        
        t = frame_idx / fps
        
        # Person 1: Normal behavior (walking)
        if t < 15:
            x1 = int(50 + t * 30)
            y1 = int(200 + 10 * np.sin(t * 0.5))
            cv2.rectangle(frame, (x1, y1), (x1+40, y1+80), (0, 255, 0), -1)
            cv2.putText(frame, "Person 1", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person 2: Suspicious behavior (erratic movement)
        if t > 5 and t < 18:
            x2 = int(300 + 50 * np.sin(t * 2))
            y2 = int(150 + 30 * np.cos(t * 1.5))
            cv2.rectangle(frame, (x2, y2), (x2+40, y2+80), (0, 165, 255), -1)
            cv2.putText(frame, "Person 2", (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person 3: Anomalous behavior (rapid movement)
        if t > 10:
            x3 = int(500 + 100 * np.sin(t * 3))
            y3 = int(300 + 50 * np.cos(t * 4))
            cv2.rectangle(frame, (x3, y3), (x3+40, y3+80), (0, 0, 255), -1)
            cv2.putText(frame, "Person 3", (x3, y3-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame, f"Demo Frame {frame_idx} - Time: {t:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add behavior labels
        cv2.putText(frame, "Green: Normal | Orange: Suspicious | Red: Anomaly", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print("✅ Demo video created: demo_video.mp4")
    return 'demo_video.mp4'

def run_complete_demo():
    """Run the complete CCTV system demo"""
    
    print("🚀 COMPLETE CCTV SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases:")
    print("✅ Global Person ReID with consistent IDs")
    print("✅ Real-time anomaly detection")
    print("✅ 3-Color behavior visualization:")
    print("   🟢 Green: Normal behavior")
    print("   🟠 Orange: Suspicious behavior") 
    print("   🔴 Red: Anomalous behavior")
    print("✅ Anomaly scores with progress bars")
    print("✅ Multi-camera tracking capabilities")
    print("=" * 60)
    
    # Find or create test video
    test_video = find_test_video()
    
    if not test_video:
        print("⚠️  No test video found, creating demo video...")
        test_video = create_demo_video()
    else:
        print(f"📹 Using test video: {os.path.basename(test_video)}")
    
    # Check if VAE model exists
    vae_model_path = "models/vae_anomaly_detector.pth"
    if not os.path.exists(vae_model_path):
        print(f"❌ VAE model not found at {vae_model_path}")
        print("Please train the VAE model first:")
        print("python train_vae_model.py")
        return
    
    # Initialize complete system
    try:
        print("\n🔧 Initializing Complete CCTV System...")
        system = CompleteCCTVSystem(camera_id="demo_cam")
        print("✅ System initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Set output path
    output_path = f"complete_demo_output_{int(time.time())}.mp4"
    
    print(f"\n🎬 Starting video processing...")
    print(f"📹 Input: {test_video}")
    print(f"💾 Output: {output_path}")
    print(f"👀 Watch for:")
    print(f"   • Global IDs (G:X) that stay consistent")
    print(f"   • Local IDs (L:X) that may change")
    print(f"   • Color-coded behavior categories")
    print(f"   • Anomaly score bars above persons")
    print(f"   • ReID statistics in bottom-left")
    print(f"\nPress 'q' to quit, 'SPACE' to pause")
    print("=" * 60)
    
    try:
        # Process video with complete system
        results = system.process_video(
            video_path=test_video,
            output_path=output_path,
            display=True
        )
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"📊 Results Summary:")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Average FPS: {results['avg_fps']:.1f}")
        print(f"   Global persons: {results['reid_statistics']['total_global_persons']}")
        print(f"   ReID matches: {results['reid_statistics']['reid_matches']}")
        print(f"   Match rate: {results['reid_statistics']['reid_match_rate']:.2%}")
        
        print(f"\n💾 Output video saved: {output_path}")
        print(f"🔍 ReID data saved: reid_data_demo_cam.pkl")
        
        # Cleanup demo video if created
        if test_video == 'demo_video.mp4':
            os.remove('demo_video.mp4')
            print("🧹 Demo video cleaned up")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function"""
    
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Complete CCTV System Demo')
    parser.add_argument('--video', '-v', help='Specific video to use for demo')
    parser.add_argument('--camera-id', '-c', default='demo_cam', help='Camera ID')
    
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Video not found: {args.video}")
            return
        
        print(f"🎬 Running demo with: {args.video}")
        
        try:
            system = CompleteCCTVSystem(camera_id=args.camera_id)
            output_path = f"complete_demo_{args.camera_id}_{int(time.time())}.mp4"
            
            results = system.process_video(
                video_path=args.video,
                output_path=output_path,
                display=True
            )
            
            print(f"✅ Demo completed! Output: {output_path}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        run_complete_demo()

if __name__ == "__main__":
    main()