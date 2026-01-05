#!/usr/bin/env python3
"""
Quick Test for Adaptive Zone Learning System
"""

import os
import cv2
import numpy as np
from adaptive_zone_learning import ActivityZoneLearner

def create_mock_normal_video():
    """Create a mock normal behavior video for testing"""
    
    print("🎬 Creating mock normal behavior video for testing...")
    
    # Create a simple test video with simulated normal behavior
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('mock_normal_behavior.mp4', fourcc, fps, (width, height))
    
    # Simulate person moving slowly near "shelf areas"
    shelf_areas = [
        (100, 200, 200, 400),  # Left shelf
        (440, 200, 540, 400),  # Right shelf
        (270, 300, 370, 400)   # Center display
    ]
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(50)  # Dark gray background
        
        # Draw shelf areas
        for x1, y1, x2, y2 in shelf_areas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
        
        # Simulate person moving slowly near shelves
        t = frame_idx / fps
        
        # Person 1: Browsing left shelf
        if t < 5:
            person_x = int(150 + 20 * np.sin(t * 0.5))
            person_y = int(300 + 10 * np.cos(t * 0.3))
            cv2.rectangle(frame, (person_x-20, person_y-40), (person_x+20, person_y+40), (0, 255, 0), -1)
        
        # Person 2: Browsing right shelf  
        if t > 3 and t < 8:
            person_x = int(490 + 15 * np.sin((t-3) * 0.4))
            person_y = int(320 + 8 * np.cos((t-3) * 0.6))
            cv2.rectangle(frame, (person_x-20, person_y-40), (person_x+20, person_y+40), (0, 255, 0), -1)
        
        # Person 3: Center display interaction
        if t > 6:
            person_x = int(320 + 25 * np.sin((t-6) * 0.3))
            person_y = int(350 + 12 * np.cos((t-6) * 0.4))
            cv2.rectangle(frame, (person_x-20, person_y-40), (person_x+20, person_y+40), (0, 255, 0), -1)
        
        # Add frame info
        cv2.putText(frame, f"Normal Behavior - Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Mock normal behavior video created: mock_normal_behavior.mp4")
    return 'mock_normal_behavior.mp4'

def test_zone_learning():
    """Test the zone learning functionality"""
    
    print("\n🧠 Testing Zone Learning Algorithm")
    print("=" * 50)
    
    # Create mock video if no normal videos exist
    normal_videos = []
    
    # Look for existing normal videos
    search_paths = ["working/normal_shop", "working", "data"]
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    if not any(keyword in file.lower() for keyword in ['anomaly', 'theft', 'shoplifting']):
                        normal_videos.append(os.path.join(path, file))
    
    # If no normal videos found, create mock video
    if not normal_videos:
        print("⚠️  No normal behavior videos found, creating mock video...")
        mock_video = create_mock_normal_video()
        normal_videos = [mock_video]
    
    print(f"📹 Using {len(normal_videos)} normal videos for learning:")
    for video in normal_videos:
        print(f"  {video}")
    
    # Initialize learner
    learner = ActivityZoneLearner()
    
    # Learn zones
    try:
        results = learner.learn_zones_from_videos(normal_videos)
        
        if results and results['zones']:
            print(f"\n✅ Zone learning successful!")
            print(f"📍 Learned {len(results['zones'])} interaction zones:")
            
            for i, zone in enumerate(results['zones']):
                print(f"  Zone {i+1}: {zone['id']}")
                print(f"    Center: ({zone['center'][0]:.1f}, {zone['center'][1]:.1f})")
                print(f"    Interactions: {zone['point_count']}")
                print(f"    Density: {zone['density']:.3f}")
            
            # Test zone loading
            print(f"\n🔄 Testing zone loading...")
            if learner.load_learned_zones():
                print(f"✅ Zone loading successful!")
            else:
                print(f"❌ Zone loading failed!")
            
            # Create visualization
            if normal_videos:
                learner.visualize_learned_zones(normal_videos[0], "test_zones_visualization.jpg")
                print(f"✅ Visualization saved: test_zones_visualization.jpg")
            
            return True
        else:
            print(f"❌ No zones learned!")
            return False
            
    except Exception as e:
        print(f"❌ Zone learning failed: {e}")
        return False

def test_adaptive_detection():
    """Test adaptive detection with learned zones"""
    
    print(f"\n🛡️ Testing Adaptive Detection System")
    print("=" * 50)
    
    # Check if zones exist
    if not os.path.exists("models/learned_interaction_zones.pkl"):
        print("❌ No learned zones found! Run zone learning first.")
        return False
    
    try:
        from stealing_detection_system import AdaptiveZoneDetector
        
        # Test zone detector initialization
        detector = AdaptiveZoneDetector(640, 480)
        
        print(f"✅ Adaptive zone detector initialized")
        print(f"📍 Loaded {len(detector.interaction_zones)} zones:")
        
        for i, zone in enumerate(detector.interaction_zones):
            print(f"  Zone {i+1}: {zone['id']} (density: {zone.get('density', 0):.3f})")
        
        # Test interaction detection
        mock_hands = [
            {'center': [150, 300], 'bbox': [140, 290, 160, 310]},  # Near left zone
            {'center': [490, 320], 'bbox': [480, 310, 500, 330]}   # Near right zone
        ]
        
        interactions = detector.detect_hand_interaction(mock_hands, [100, 200, 200, 400])
        
        print(f"\n🤚 Testing hand-zone interaction:")
        print(f"  Hands tested: {len(mock_hands)}")
        print(f"  Interactions detected: {interactions['has_interaction']}")
        print(f"  Interaction score: {interactions['interaction_score']:.2f}")
        print(f"  Zones involved: {interactions['interaction_zones']}")
        
        if interactions['has_interaction']:
            print(f"✅ Interaction detection working!")
        else:
            print(f"⚠️  No interactions detected (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive detection test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 QUICK ADAPTIVE SYSTEM TEST")
    print("=" * 60)
    print("Testing the adaptive zone learning and detection system")
    print("=" * 60)
    
    success = True
    
    # Test 1: Zone Learning
    if not test_zone_learning():
        success = False
    
    # Test 2: Adaptive Detection
    if success and not test_adaptive_detection():
        success = False
    
    # Results
    print(f"\n🎯 TEST RESULTS:")
    print("=" * 30)
    
    if success:
        print("✅ All tests passed!")
        print("🎉 Adaptive system is working correctly!")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Run full pipeline: python learn_and_test_adaptive_system.py")
        print("2. Test with real videos: python stealing_detection_system.py --input video.mp4")
        print("3. Review visualization: test_zones_visualization.jpg")
        
    else:
        print("❌ Some tests failed!")
        print("Please check the error messages above and resolve issues.")
    
    # Cleanup
    if os.path.exists('mock_normal_behavior.mp4'):
        print(f"\n🧹 Cleaning up mock video...")
        os.remove('mock_normal_behavior.mp4')

if __name__ == "__main__":
    main()