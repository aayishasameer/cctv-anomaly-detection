#!/usr/bin/env python3
"""
Demo: Dual Window CCTV System
Clean video output with separate control panel
"""

import os
from dual_window_cctv_system import DualWindowCCTVSystem

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

def run_dual_window_demo():
    """Run the dual window CCTV system demo"""
    
    print("🚀 DUAL WINDOW CCTV SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases:")
    print("✅ Clean video output without system overlays")
    print("✅ Separate control panel with:")
    print("   📊 Real-time statistics")
    print("   🎯 Person tracking counts")
    print("   🔍 ReID performance metrics")
    print("   🚨 Recent alerts")
    print("   📈 Performance monitoring")
    print("✅ 3-Color behavior visualization:")
    print("   🟢 Green: Normal behavior")
    print("   🟠 Orange: Suspicious behavior") 
    print("   🔴 Red: Anomalous behavior")
    print("=" * 60)
    
    # Find test video
    test_video = find_test_video()
    
    if not test_video:
        print("❌ No test video found!")
        print("Please place a video file in one of these directories:")
        print("  - working/test_anomaly/")
        print("  - working/")
        print("  - data/")
        return
    
    print(f"📹 Using test video: {os.path.basename(test_video)}")
    
    # Check if VAE model exists
    vae_model_path = "models/vae_anomaly_detector.pth"
    if not os.path.exists(vae_model_path):
        print(f"❌ VAE model not found at {vae_model_path}")
        print("Please train the VAE model first:")
        print("python train_vae_model.py")
        return
    
    # Initialize dual window system
    try:
        print("\n🔧 Initializing Dual Window CCTV System...")
        system = DualWindowCCTVSystem(camera_id="demo_cam")
        print("✅ System initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Set output path
    output_path = f"clean_demo_output.mp4"
    
    print(f"\n🎬 Starting dual window processing...")
    print(f"📹 Input: {test_video}")
    print(f"💾 Clean Output: {output_path}")
    print(f"\n👀 You will see TWO windows:")
    print(f"   1. 'CCTV Video Feed - Clean Output' - Clean video with person tracking")
    print(f"   2. 'CCTV Control Panel' - System information and statistics")
    print(f"\n🎮 Controls:")
    print(f"   • Press 'q' to quit")
    print(f"   • Press 'SPACE' to pause/resume")
    print(f"   • Click on windows to focus")
    print("=" * 60)
    
    try:
        # Process video with dual window system
        results = system.process_video(
            video_path=test_video,
            output_path=output_path,
            display=True
        )
        
        print(f"\n🎉 DUAL WINDOW DEMO COMPLETED SUCCESSFULLY!")
        print(f"📊 Results Summary:")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Average FPS: {results['avg_fps']:.1f}")
        print(f"   Global persons: {results['reid_statistics']['total_global_persons']}")
        print(f"   ReID matches: {results['reid_statistics']['reid_matches']}")
        print(f"   Match rate: {results['reid_statistics']['reid_match_rate']:.2%}")
        
        print(f"\n💾 Clean output video saved: {output_path}")
        print(f"🔍 ReID data saved: reid_data_demo_cam.pkl")
        print(f"\n✨ The output video contains ONLY the clean tracking visualization")
        print(f"   No system information overlays - perfect for presentations!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Window CCTV System Demo')
    parser.add_argument('--video', '-v', help='Specific video to use for demo')
    parser.add_argument('--camera-id', '-c', default='demo_cam', help='Camera ID')
    parser.add_argument('--output', '-o', help='Output video path')
    
    args = parser.parse_args()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Video not found: {args.video}")
            return
        
        output_path = args.output or f"clean_output_{args.camera_id}.mp4"
        
        print(f"🎬 Running dual window demo with: {args.video}")
        
        try:
            system = DualWindowCCTVSystem(camera_id=args.camera_id)
            
            results = system.process_video(
                video_path=args.video,
                output_path=output_path,
                display=True
            )
            
            print(f"✅ Demo completed! Clean output: {output_path}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        run_dual_window_demo()

if __name__ == "__main__":
    main()