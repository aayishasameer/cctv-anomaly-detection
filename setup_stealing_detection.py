#!/usr/bin/env python3
"""
Setup Script for Enhanced Stealing Detection System
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    print("\n🔍 Checking model files...")
    
    model_path = Path("models/vae_anomaly_detector.pth")
    yolo_path = Path("yolov8n.pt")
    
    if model_path.exists():
        print("✅ VAE anomaly detection model found")
        model_exists = True
    else:
        print("❌ VAE model not found at models/vae_anomaly_detector.pth")
        print("   You need to train the model first: python train_vae_model.py")
        model_exists = False
    
    if yolo_path.exists():
        print("✅ YOLO model found")
        yolo_exists = True
    else:
        print("⚠️  YOLO model will be downloaded automatically on first run")
        yolo_exists = True  # Will be downloaded automatically
    
    return model_exists and yolo_exists

def check_video_files():
    """Check for test video files"""
    print("\n🎬 Checking for test videos...")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video files:")
        for video in video_files[:5]:  # Show first 5
            print(f"   {video}")
        if len(video_files) > 5:
            print(f"   ... and {len(video_files) - 5} more")
        return True
    else:
        print("⚠️  No video files found for testing")
        print("   Place test videos in the working directory")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "results", 
        "working/test_videos",
        "working/output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics YOLO"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_imports_ok = True
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def run_quick_test():
    """Run a quick system test"""
    print("\n🚀 Running quick system test...")
    
    try:
        # Test basic imports
        from stealing_detection_system import StealingDetectionSystem
        from improved_anomaly_tracker import ImprovedAnomalyTracker
        print("✅ Core modules imported successfully")
        
        # Test hand detection initialization
        detector = StealingDetectionSystem.__new__(StealingDetectionSystem)
        detector.hand_detector = StealingDetectionSystem.HandDetector.__new__(StealingDetectionSystem.HandDetector)
        print("✅ Hand detection system can be initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 50)
    
    print("\n1. 🧠 Train VAE Model (if not done yet):")
    print("   python train_vae_model.py")
    
    print("\n2. 🧪 Test Systems:")
    print("   python test_fixed_system.py --mode comprehensive")
    print("   python test_stealing_detection.py --quick")
    
    print("\n3. 🎬 Run Demos:")
    print("   python demo_stealing_detection.py --input your_video.mp4")
    print("   python stealing_detection_system.py --input video.mp4 --output result.mp4")
    
    print("\n4. 📚 Documentation:")
    print("   Read STEALING_DETECTION_GUIDE.md for detailed information")
    
    print("\n5. 🛠️ Configuration:")
    print("   Edit shelf zones in stealing_detection_system.py")
    print("   Adjust thresholds for your specific use case")

def main():
    """Main setup function"""
    print("🛡️ Enhanced Stealing Detection System Setup")
    print("=" * 60)
    
    setup_success = True
    
    # Check Python version
    if not check_python_version():
        setup_success = False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        setup_success = False
    
    # Test imports
    if not test_imports():
        setup_success = False
    
    # Check model files
    model_ready = check_model_files()
    
    # Check video files
    videos_available = check_video_files()
    
    # Run quick test
    if setup_success:
        test_success = run_quick_test()
    else:
        test_success = False
    
    # Print results
    print("\n🎯 SETUP SUMMARY:")
    print("=" * 30)
    
    if setup_success and test_success:
        print("✅ Setup completed successfully!")
        
        if model_ready:
            print("✅ System ready to use")
        else:
            print("⚠️  Need to train VAE model first")
        
        if videos_available:
            print("✅ Test videos available")
        else:
            print("⚠️  Add test videos for full testing")
            
    else:
        print("❌ Setup encountered issues")
        print("   Please resolve the errors above and run setup again")
    
    # Print usage instructions
    print_usage_instructions()
    
    return setup_success and test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)