#!/usr/bin/env python3
"""
Demo script for CCTV Anomaly Detection System
This script demonstrates the complete workflow from training to detection
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files and directories exist"""
    print("🔍 Checking system requirements...")
    
    required_files = [
        "vae_anomaly_detector.py",
        "train_vae_model.py", 
        "anomaly_detection_tracker.py",
        "botsort.yaml"
    ]
    
    required_dirs = [
        "working/normal_shop",
        "working/test_anomaly"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # Check if we have training videos
    normal_videos = list(Path("working/normal_shop").glob("*.mp4"))
    test_videos = list(Path("working/test_anomaly").glob("*.mp4"))
    
    print(f"✓ Found {len(normal_videos)} normal training videos")
    print(f"✓ Found {len(test_videos)} test videos")
    
    if len(normal_videos) == 0:
        print("❌ No training videos found in working/normal_shop/")
        return False
    
    if len(test_videos) == 0:
        print("❌ No test videos found in working/test_anomaly/")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def run_training():
    """Run the training process"""
    print("\n🚀 Starting VAE model training...")
    print("This may take several minutes depending on your hardware...")
    
    os.system("python train_vae_model.py")
    
    # Check if model was created
    if os.path.exists("models/vae_anomaly_detector.pth"):
        print("✅ Model training completed successfully!")
        return True
    else:
        print("❌ Model training failed!")
        return False

def run_demo_detection():
    """Run anomaly detection on a test video"""
    print("\n🎯 Running anomaly detection demo...")
    
    # Find first test video
    test_videos = list(Path("working/test_anomaly").glob("*.mp4"))
    if not test_videos:
        print("❌ No test videos found!")
        return False
    
    test_video = test_videos[0]
    output_video = f"demo_output_{test_video.name}"
    
    print(f"Processing: {test_video}")
    print(f"Output will be saved as: {output_video}")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'space' to pause/resume")
    print("\nColor coding:")
    print("- Green: Normal behavior")
    print("- Orange: Warning/suspicious")
    print("- Red: Confirmed anomaly")
    
    cmd = f"python anomaly_detection_tracker.py -i {test_video} -o {output_video}"
    os.system(cmd)
    
    if os.path.exists(output_video):
        print(f"✅ Demo completed! Output saved as: {output_video}")
        return True
    else:
        print("❌ Demo failed!")
        return False

def main():
    """Main demo function"""
    print("=" * 60)
    print("🎬 CCTV Anomaly Detection System - Demo")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please ensure you have:")
        print("1. Video files in working/normal_shop/ (for training)")
        print("2. Video files in working/test_anomaly/ (for testing)")
        print("3. All required Python files")
        return
    
    # Check if model already exists
    model_exists = os.path.exists("models/vae_anomaly_detector.pth")
    
    if model_exists:
        print("\n✅ Trained model found!")
        choice = input("Do you want to retrain the model? (y/N): ").lower().strip()
        if choice == 'y':
            if not run_training():
                return
    else:
        print("\n📚 No trained model found. Training is required.")
        if not run_training():
            return
    
    # Run demo detection
    print("\n" + "=" * 60)
    choice = input("Run anomaly detection demo? (Y/n): ").lower().strip()
    if choice != 'n':
        run_demo_detection()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Process more videos: python batch_anomaly_detection.py -i working/test_anomaly -o results/")
    print("2. Process single video: python anomaly_detection_tracker.py -i your_video.mp4 -o output.mp4")
    print("3. Retrain with more data: python train_vae_model.py")
    print("=" * 60)

if __name__ == "__main__":
    main()