#!/usr/bin/env python3
"""
CCTV Anomaly Detection System Demo
Demonstrates the complete workflow from training to detection
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files and directories exist"""
    required_files = [
        "vae_anomaly_detector.py",
        "train_vae_model.py", 
        "anomaly_detection_tracker.py",
        "batch_anomaly_detection.py"
    ]
    
    required_dirs = [
        "working/normal_shop",
        "working/test_anomaly"
    ]
    
    print("🔍 Checking system requirements...")
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
        print(f"✅ Found: {file}")
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            print(f"❌ Missing required directory: {dir}")
            return False
        
        # Check if directory has videos
        video_count = len([f for f in os.listdir(dir) if f.endswith('.mp4')])
        print(f"✅ Found: {dir} ({video_count} videos)")
    
    return True

def check_model_status():
    """Check if the VAE model is trained"""
    model_path = "models/vae_anomaly_detector.pth"
    
    if os.path.exists(model_path):
        print(f"✅ Trained model found: {model_path}")
        return True
    else:
        print(f"❌ No trained model found at: {model_path}")
        print("   Run 'python train_vae_model.py' to train the model first")
        return False

def demo_single_video():
    """Demo processing a single video"""
    print("\n🎬 Demo: Single Video Processing")
    print("=" * 50)
    
    # Find a test video
    test_dir = "working/test_anomaly"
    videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4')]
    
    if not videos:
        print("❌ No test videos found!")
        return
    
    test_video = videos[0]
    input_path = os.path.join(test_dir, test_video)
    output_path = f"demo_output_{test_video}"
    
    print(f"Processing: {test_video}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Run anomaly detection
    cmd = f"python anomaly_detection_tracker.py --input {input_path} --output {output_path} --no-display"
    print(f"\nRunning: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Successfully processed video!")
        print(f"📹 Output saved to: {output_path}")
    else:
        print(f"❌ Error processing video (exit code: {result})")

def demo_batch_processing():
    """Demo batch processing multiple videos"""
    print("\n📦 Demo: Batch Processing")
    print("=" * 50)
    
    input_dir = "working/test_anomaly"
    output_dir = "demo_batch_results"
    
    print(f"Processing all videos in: {input_dir}")
    print(f"Results will be saved to: {output_dir}")
    
    # Run batch processing
    cmd = f"python batch_anomaly_detection.py --input-dir {input_dir} --output-dir {output_dir}"
    print(f"\nRunning: {cmd}")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"✅ Successfully processed all videos!")
        print(f"📁 Results saved to: {output_dir}")
        
        # Show results summary
        results_file = os.path.join(output_dir, "anomaly_detection_results.json")
        if os.path.exists(results_file):
            print(f"📊 Results summary: {results_file}")
    else:
        print(f"❌ Error in batch processing (exit code: {result})")

def show_system_info():
    """Display system information and capabilities"""
    print("\n🤖 CCTV Anomaly Detection System")
    print("=" * 50)
    print("This system uses AI to detect suspicious behavior in surveillance videos.")
    print()
    print("🔧 Components:")
    print("  • YOLOv8: Person detection")
    print("  • BotSORT: Multi-object tracking") 
    print("  • VAE: Behavioral anomaly detection")
    print()
    print("🎯 Features:")
    print("  • Real-time anomaly detection")
    print("  • Color-coded visualization (Green=Normal, Orange=Warning, Red=Anomaly)")
    print("  • Batch processing capabilities")
    print("  • Detailed anomaly reports")
    print()
    print("📁 Directory Structure:")
    print("  • working/normal_shop/    - Training videos (normal behavior)")
    print("  • working/test_anomaly/   - Test videos (with anomalies)")
    print("  • models/                 - Trained AI models")
    print("  • results/                - Processing results")

def main():
    """Main demo function"""
    show_system_info()
    
    print("\n🔍 System Check")
    print("=" * 50)
    
    if not check_requirements():
        print("\n❌ System check failed! Please ensure all required files are present.")
        sys.exit(1)
    
    if not check_model_status():
        print("\n⚠️  Model not trained. Training is required before processing videos.")
        print("Run: python train_vae_model.py")
        return
    
    print("\n✅ System ready!")
    
    # Ask user what demo to run
    print("\n🎮 Available Demos:")
    print("1. Single video processing")
    print("2. Batch processing")
    print("3. Both demos")
    print("0. Exit")
    
    try:
        choice = input("\nSelect demo (0-3): ").strip()
        
        if choice == "1":
            demo_single_video()
        elif choice == "2":
            demo_batch_processing()
        elif choice == "3":
            demo_single_video()
            demo_batch_processing()
        elif choice == "0":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()