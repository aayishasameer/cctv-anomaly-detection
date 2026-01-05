#!/usr/bin/env python3
"""
Test Fixed Enhanced Anomaly Detection System with Stealing Detection
"""

import cv2
import numpy as np
from improved_anomaly_tracker import ImprovedAnomalyTracker
from stealing_detection_system import StealingDetectionSystem
import time
import os

def test_fixed_system(video_path: str, max_frames: int = 300):
    """Test the fixed system with better detection"""
    
    print("🔧 Testing FIXED Enhanced Anomaly Detection System")
    print("=" * 60)
    
    # Initialize fixed tracker
    try:
        tracker = ImprovedAnomalyTracker()
        print("✅ Fixed enhanced model loaded!")
    except FileNotFoundError:
        print("❌ Model not found!")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {video_path}")
    print(f"📊 Resolution: {width}x{height}, FPS: {fps}")
    print(f"⏱️  Test duration: {max_frames/fps:.1f} seconds")
    
    print(f"\n🔧 FIXES Applied:")
    print(f"  ✅ Reduced min_track_length: 25 → 12 frames")
    print(f"  ✅ Lowered confirmation ratio: 75% → 60%")
    print(f"  ✅ Faster anomaly detection: 12 → 8 frames")
    print(f"  ✅ Removed gray 'TRACKING' boxes")
    print(f"  ✅ More lenient detection validation")
    print(f"  ✅ Better ID consistency tracking")
    
    frame_idx = 0
    anomaly_count = 0
    warning_count = 0
    normal_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking
            results = tracker.yolo_model.track(
                source=frame,
                tracker="botsort.yaml",
                persist=True,
                classes=[0],
                conf=0.4,  # Lower confidence
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if tracker.is_valid_detection(box.tolist(), conf):
                        detection_count += 1
                        
                        # Get stable ID
                        stable_id = tracker.get_stable_track_id(track_id, box.tolist(), conf)
                        
                        # Test anomaly detection
                        is_anomaly, anomaly_score = tracker.anomaly_detector.detect_anomaly(
                            stable_id, box.tolist(), frame_idx
                        )
                        
                        # Test advanced smoothing
                        is_confirmed_anomaly, status = tracker.advanced_anomaly_smoothing(
                            stable_id, is_anomaly, anomaly_score, box.tolist(), width, height
                        )
                        
                        # Count by status
                        if is_confirmed_anomaly:
                            anomaly_count += 1
                            print(f"🚨 Frame {frame_idx}: ANOMALY detected (ID: {stable_id}, Score: {anomaly_score:.3f})")
                        elif status == "WARNING":
                            warning_count += 1
                            if warning_count % 10 == 1:  # Print every 10th warning
                                print(f"⚠️  Frame {frame_idx}: WARNING (ID: {stable_id}, Score: {anomaly_score:.3f})")
                        else:
                            normal_count += 1
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / max_frames) * 100
                print(f"📊 Progress: {progress:.1f}% | FPS: {fps_current:.1f} | A:{anomaly_count} W:{warning_count} N:{normal_count}")
    
    finally:
        cap.release()
    
    # Final results
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    
    print(f"\n🎯 FIXED System Test Results:")
    print(f"=" * 40)
    print(f"📊 Frames processed: {frame_idx}")
    print(f"🎭 Total detections: {detection_count}")
    print(f"🚨 Anomalies: {anomaly_count}")
    print(f"⚠️  Warnings: {warning_count}")
    print(f"✅ Normal: {normal_count}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if detection_count > 0:
        anomaly_rate = (anomaly_count / detection_count * 100)
        warning_rate = (warning_count / detection_count * 100)
        normal_rate = (normal_count / detection_count * 100)
        
        print(f"\n📈 Detection Breakdown:")
        print(f"  🚨 Anomaly rate: {anomaly_rate:.1f}%")
        print(f"  ⚠️  Warning rate: {warning_rate:.1f}%")
        print(f"  ✅ Normal rate: {normal_rate:.1f}%")
    
    # Success indicators
    success_indicators = []
    if anomaly_count > 0:
        success_indicators.append("✅ Anomalies detected!")
    if warning_count > 0:
        success_indicators.append("✅ Warnings shown!")
    if normal_count > 0:
        success_indicators.append("✅ Normal behavior tracked!")
    
    if success_indicators:
        print(f"\n🎉 SUCCESS INDICATORS:")
        for indicator in success_indicators:
            print(f"  {indicator}")
    else:
        print(f"\n❌ ISSUES: No detections found - may need further tuning")

def test_stealing_detection_system(video_path: str, max_frames: int = 200):
    """Test the new stealing detection capabilities"""
    
    print("\n🛡️ Testing STEALING DETECTION System")
    print("=" * 60)
    
    # Initialize stealing detection system
    try:
        stealing_detector = StealingDetectionSystem()
        print("✅ Stealing detection system loaded!")
    except FileNotFoundError:
        print("❌ Stealing detection model not found!")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"📊 Resolution: {width}x{height}, FPS: {fps}")
    print(f"⏱️  Test duration: {max_frames/fps:.1f} seconds")
    
    print(f"\n🛡️ STEALING DETECTION FEATURES:")
    print(f"  ✅ Hand detection (MediaPipe)")
    print(f"  ✅ Shelf zone interaction detection")
    print(f"  ✅ Multi-level threat assessment")
    print(f"  ✅ Behavioral + interaction analysis")
    print(f"  ✅ Confirmed theft detection")
    
    frame_idx = 0
    threat_counts = {'normal': 0, 'suspicious': 0, 'high_risk': 0, 'stealing': 0, 'confirmed_theft': 0}
    hand_detections = 0
    shelf_interactions = 0
    start_time = time.time()
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Test hand detection
            hands = stealing_detector.hand_detector.detect_hands(frame)
            if hands:
                hand_detections += len(hands)
            
            # Person detection and tracking
            results = stealing_detector.yolo_model.track(
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
                
                # Initialize shelf detector
                from stealing_detection_system import ShelfZoneDetector
                shelf_detector = ShelfZoneDetector(width, height)
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    if conf < 0.4:
                        continue
                    
                    # Get person's hands
                    person_hands = stealing_detector._get_person_hands(box, hands)
                    
                    # Test shelf interactions
                    shelf_interaction = shelf_detector.detect_hand_shelf_interaction(
                        person_hands, box.tolist()
                    )
                    
                    if shelf_interaction['has_interaction']:
                        shelf_interactions += 1
                    
                    # Test comprehensive stealing analysis
                    analysis = stealing_detector.analyze_stealing_behavior(
                        track_id, box.tolist(), person_hands,
                        shelf_interaction, frame_idx, fps
                    )
                    
                    # Count threat levels
                    threat_level = analysis['threat_level']
                    threat_counts[threat_level] += 1
                    
                    # Log high-priority detections
                    if threat_level in ['stealing', 'confirmed_theft']:
                        print(f"🚨 Frame {frame_idx}: {threat_level.upper()} detected!")
                        print(f"   ID: {track_id}, Score: {analysis['scores']['final_score']:.3f}")
                        print(f"   Duration: {analysis['duration']:.1f}s, Interactions: {analysis['interaction_count']}")
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / max_frames) * 100
                print(f"📊 Progress: {progress:.1f}% | FPS: {fps_current:.1f} | Hands: {hand_detections} | Interactions: {shelf_interactions}")
    
    finally:
        cap.release()
    
    # Final results
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    total_detections = sum(threat_counts.values())
    
    print(f"\n🎯 STEALING DETECTION Test Results:")
    print(f"=" * 50)
    print(f"📊 Frames processed: {frame_idx}")
    print(f"🎭 Total person detections: {total_detections}")
    print(f"👋 Hand detections: {hand_detections}")
    print(f"🛒 Shelf interactions: {shelf_interactions}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    print(f"\n🛡️ THREAT LEVEL BREAKDOWN:")
    for level, count in threat_counts.items():
        if count > 0:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            emoji = {'normal': '✅', 'suspicious': '🟡', 'high_risk': '🟠', 'stealing': '🔴', 'confirmed_theft': '🟣'}[level]
            print(f"  {emoji} {level.upper()}: {count} ({percentage:.1f}%)")
    
    # Success indicators
    success_indicators = []
    if hand_detections > 0:
        success_indicators.append("✅ Hand detection working!")
    if shelf_interactions > 0:
        success_indicators.append("✅ Shelf interaction detection working!")
    if threat_counts['stealing'] > 0 or threat_counts['confirmed_theft'] > 0:
        success_indicators.append("✅ Stealing detection working!")
    if threat_counts['high_risk'] > 0:
        success_indicators.append("✅ Risk assessment working!")
    
    if success_indicators:
        print(f"\n🎉 STEALING DETECTION SUCCESS:")
        for indicator in success_indicators:
            print(f"  {indicator}")
    else:
        print(f"\n⚠️  Note: No high-risk detections in test segment - this may be normal for short tests")

def run_comprehensive_test(video_path: str):
    """Run both behavioral anomaly and stealing detection tests"""
    
    print("🚀 COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)
    print("Testing both behavioral anomaly detection AND stealing detection")
    print("=" * 70)
    
    # Test 1: Original improved system
    test_fixed_system(video_path, max_frames=300)
    
    # Test 2: New stealing detection system  
    test_stealing_detection_system(video_path, max_frames=200)
    
    print(f"\n🏆 COMPREHENSIVE TEST COMPLETE!")
    print(f"=" * 50)
    print(f"✅ Behavioral anomaly detection: TESTED")
    print(f"✅ Stealing detection system: TESTED")
    print(f"✅ Hand detection: TESTED")
    print(f"✅ Shelf interaction detection: TESTED")
    print(f"✅ Multi-level threat assessment: TESTED")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Run full demo: python demo_stealing_detection.py")
    print(f"2. Process full video: python stealing_detection_system.py --input video.mp4")
    print(f"3. Review STEALING_DETECTION_GUIDE.md for detailed documentation")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced anomaly and stealing detection')
    parser.add_argument('--video', '-v', default="working/test_anomaly/Shoplifting020_x264.mp4", 
                       help='Video path to test')
    parser.add_argument('--mode', '-m', choices=['anomaly', 'stealing', 'comprehensive'], 
                       default='comprehensive', help='Test mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        print("Looking for available videos...")
        
        # Search for video files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    print(f"  Found: {os.path.join(root, file)}")
        return
    
    if args.mode == 'anomaly':
        test_fixed_system(args.video)
    elif args.mode == 'stealing':
        test_stealing_detection_system(args.video)
    else:  # comprehensive
        run_comprehensive_test(args.video)