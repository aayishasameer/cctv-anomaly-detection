#!/usr/bin/env python3
"""
Test Person Re-Identification System
Comprehensive testing of ReID capabilities
"""

import cv2
import numpy as np
from person_reid_system import GlobalPersonTracker, PersonReIDExtractor
from stealing_detection_system import StealingDetectionSystem
import os
import time

def test_reid_feature_extraction():
    """Test ReID feature extraction"""
    
    print("🔍 Testing ReID Feature Extraction")
    print("=" * 50)
    
    # Initialize feature extractor
    try:
        extractor = PersonReIDExtractor()
        print("✅ ReID feature extractor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        return False
    
    # Create test person crops
    test_crops = []
    
    # Create synthetic person crops with different characteristics
    for i in range(3):
        # Create a person-like rectangle (height > width)
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Add some structure to make it more person-like
        # Simulate head (darker at top)
        crop[:50, :, :] = crop[:50, :, :] * 0.7
        
        # Simulate torso (middle section)
        crop[50:150, :, :] = crop[50:150, :, :] * 0.9
        
        # Simulate legs (bottom section)
        crop[150:, :, :] = crop[150:, :, :] * 0.8
        
        test_crops.append(crop)
    
    print(f"📸 Created {len(test_crops)} test person crops")
    
    # Extract features
    features = []
    for i, crop in enumerate(test_crops):
        start_time = time.time()
        feature_vector = extractor.extract_features(crop)
        extraction_time = time.time() - start_time
        
        features.append(feature_vector)
        print(f"  Person {i+1}: Feature dim {len(feature_vector)}, Time: {extraction_time:.3f}s")
    
    # Test feature similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n🔗 Feature Similarity Matrix:")
    similarity_matrix = cosine_similarity(features)
    
    for i in range(len(features)):
        for j in range(len(features)):
            print(f"  P{i+1}-P{j+1}: {similarity_matrix[i][j]:.3f}", end="")
        print()
    
    print(f"✅ ReID feature extraction test completed")
    return True

def test_global_tracking():
    """Test global person tracking with ReID"""
    
    print(f"\n🌍 Testing Global Person Tracking")
    print("=" * 50)
    
    # Initialize global tracker
    try:
        tracker = GlobalPersonTracker()
        print("✅ Global person tracker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Simulate multi-camera detections
    test_detections = [
        # Camera 1 detections
        ("cam1", 1, [100, 100, 200, 300], 0.8, 1.0),
        ("cam1", 2, [300, 150, 400, 350], 0.9, 2.0),
        ("cam1", 1, [105, 105, 205, 305], 0.7, 3.0),  # Same person, slight movement
        
        # Camera 2 detections (should match some from cam1)
        ("cam2", 1, [120, 110, 220, 310], 0.7, 4.0),  # Should match cam1_1
        ("cam2", 2, [350, 200, 450, 400], 0.8, 5.0),  # Should match cam1_2
        ("cam2", 3, [50, 50, 150, 250], 0.6, 6.0),    # New person
        
        # Camera 3 detections
        ("cam3", 1, [80, 90, 180, 290], 0.6, 7.0),    # Should match cam1_1
        ("cam3", 2, [320, 180, 420, 380], 0.7, 8.0),  # Should match cam1_2
    ]
    
    print(f"🎬 Processing {len(test_detections)} test detections...")
    
    detection_results = []
    for camera_id, local_id, bbox, conf, timestamp in test_detections:
        global_id = tracker.update_global_tracking(
            camera_id, local_id, dummy_frame, bbox, conf, timestamp
        )
        
        result = {
            'camera_id': camera_id,
            'local_id': local_id,
            'global_id': global_id,
            'timestamp': timestamp
        }
        detection_results.append(result)
        
        print(f"  📹 {camera_id} | Local: {local_id} | Global: {global_id} | Time: {timestamp:.1f}s")
    
    # Analyze results
    print(f"\n📊 Global Tracking Analysis:")
    
    # Get statistics
    stats = tracker.get_tracking_statistics()
    print(f"  🌍 Global persons: {stats['total_global_persons']}")
    print(f"  🔄 Total tracks: {stats['total_tracks_processed']}")
    print(f"  ✅ ReID matches: {stats['reid_matches']}")
    print(f"  🆕 New persons: {stats['new_persons_created']}")
    print(f"  📈 Match rate: {stats['reid_match_rate']:.2%}")
    
    # Analyze cross-camera tracking
    global_person_cameras = {}
    for result in detection_results:
        global_id = result['global_id']
        camera_id = result['camera_id']
        
        if global_id not in global_person_cameras:
            global_person_cameras[global_id] = set()
        global_person_cameras[global_id].add(camera_id)
    
    multi_camera_persons = {gid: cams for gid, cams in global_person_cameras.items() if len(cams) > 1}
    
    print(f"\n🔗 Cross-Camera Tracking:")
    print(f"  Multi-camera persons: {len(multi_camera_persons)}")
    
    for global_id, cameras in multi_camera_persons.items():
        person_info = tracker.get_person_info(global_id)
        duration = person_info.get('last_seen', 0) - person_info.get('first_seen', 0)
        print(f"    Global ID {global_id}: {sorted(cameras)} ({duration:.1f}s)")
    
    print(f"✅ Global tracking test completed")
    return True

def test_integrated_stealing_detection():
    """Test stealing detection with ReID integration"""
    
    print(f"\n🛡️ Testing Integrated Stealing Detection with ReID")
    print("=" * 50)
    
    # Find a test video
    test_video = None
    search_paths = ["working/test_anomaly", "working", "data"]
    
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    test_video = os.path.join(path, file)
                    break
        if test_video:
            break
    
    if not test_video:
        print("⚠️  No test video found, creating mock test...")
        return test_mock_integrated_system()
    
    print(f"📹 Testing with: {os.path.basename(test_video)}")
    
    # Initialize system with ReID
    try:
        detector = StealingDetectionSystem(enable_reid=True, camera_id="test_cam")
        print("✅ Integrated system initialized with ReID")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False
    
    # Process limited frames for testing
    cap = cv2.VideoCapture(test_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_count = 0
    max_frames = 100  # Limit for testing
    reid_detections = 0
    
    print(f"🎬 Processing {max_frames} frames for ReID testing...")
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Run YOLO tracking
            results = detector.yolo_model.track(
                source=frame,
                tracker="botsort.yaml",
                persist=True,
                classes=[0],
                conf=0.4,
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if conf < 0.4:
                        continue
                    
                    # Test ReID integration
                    global_id = detector.reid_tracker.update_global_tracking(
                        "test_cam", track_id, frame, box.tolist(), conf, timestamp
                    )
                    
                    reid_detections += 1
                    
                    if frame_count % 20 == 0:  # Print every 20th frame
                        print(f"  Frame {frame_count}: Local {track_id} → Global {global_id}")
            
            frame_count += 1
    
    finally:
        cap.release()
    
    # Get final statistics
    reid_stats = detector.reid_tracker.get_tracking_statistics()
    
    print(f"\n📊 Integrated System Results:")
    print(f"  🎬 Frames processed: {frame_count}")
    print(f"  🎭 ReID detections: {reid_detections}")
    print(f"  🌍 Global persons: {reid_stats['total_global_persons']}")
    print(f"  🔄 ReID matches: {reid_stats['reid_matches']}")
    print(f"  📈 Match rate: {reid_stats['reid_match_rate']:.2%}")
    
    print(f"✅ Integrated stealing detection with ReID test completed")
    return True

def test_mock_integrated_system():
    """Test with mock data when no video is available"""
    
    print("🎭 Running mock integrated system test...")
    
    try:
        detector = StealingDetectionSystem(enable_reid=True, camera_id="mock_cam")
        
        # Create mock frame
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate detections
        mock_detections = [
            (1, [100, 100, 200, 300], 0.8),
            (2, [300, 150, 400, 350], 0.9),
            (1, [105, 105, 205, 305], 0.7),  # Same person
        ]
        
        for i, (track_id, bbox, conf) in enumerate(mock_detections):
            global_id = detector.reid_tracker.update_global_tracking(
                "mock_cam", track_id, mock_frame, bbox, conf, i * 0.5
            )
            print(f"  Mock detection {i+1}: Local {track_id} → Global {global_id}")
        
        stats = detector.reid_tracker.get_tracking_statistics()
        print(f"  Mock results: {stats['total_global_persons']} global persons")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 PERSON RE-IDENTIFICATION SYSTEM TESTS")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Feature extraction
    test_results.append(("Feature Extraction", test_reid_feature_extraction()))
    
    # Test 2: Global tracking
    test_results.append(("Global Tracking", test_global_tracking()))
    
    # Test 3: Integrated system
    test_results.append(("Integrated System", test_integrated_stealing_detection()))
    
    # Print results summary
    print(f"\n🎯 TEST RESULTS SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All ReID system tests passed!")
        print("\n🚀 NEXT STEPS:")
        print("1. Run multi-camera demo: python multi_camera_reid_demo.py --videos video1.mp4 video2.mp4")
        print("2. Test with real videos: python stealing_detection_system.py --input video.mp4 --camera-id cam1")
        print("3. Review ReID documentation in STEALING_DETECTION_GUIDE.md")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()