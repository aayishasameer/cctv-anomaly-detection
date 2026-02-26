#!/usr/bin/env python3
"""
Test Improved Tracking Configuration
Verify that ID switching issue is fixed
"""

import os
import cv2
import time
from complete_cctv_system import CompleteCCTVSystem

def test_tracking():
    """Test the improved tracking configuration"""
    
    print("🧪 TESTING IMPROVED TRACKING CONFIGURATION")
    print("=" * 70)
    
    # Test video
    video_path = "cctv-anomaly-detection/cctv-anomaly-detection-1/working/test_anomaly/Shoplifting020_x264.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"📹 Test Video: {os.path.basename(video_path)}")
    print(f"   Frames: {total_frames}")
    print(f"   FPS: {fps}")
    
    print(f"\n⚙️  Initializing system with improved tracking...")
    try:
        system = CompleteCCTVSystem(camera_id="tracking_test")
        print("✅ System initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Output path
    output_path = "improved_tracking_test_output.mp4"
    
    print(f"\n🎬 Processing with improved tracking configuration...")
    print(f"💾 Output: {output_path}")
    print(f"\n{'='*70}")
    print("📊 Monitoring for ID stability...")
    print("{'='*70}\n")
    
    start_time = time.time()
    
    try:
        results = system.process_video(
            video_path=video_path,
            output_path=output_path,
            display=False
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ TEST COMPLETED!")
        print(f"{'='*70}")
        
        # Analyze results
        print(f"\n📊 TRACKING PERFORMANCE:")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Processing FPS: {results['avg_fps']:.1f}")
        
        if 'reid_statistics' in results:
            stats = results['reid_statistics']
            match_rate = stats['reid_match_rate']
            
            print(f"\n🎯 ID STABILITY METRICS:")
            print(f"   Total persons: {stats['total_global_persons']}")
            print(f"   ReID matches: {stats['reid_matches']}")
            print(f"   Match rate: {match_rate:.1%}")
            
            # Evaluate improvement
            print(f"\n📈 EVALUATION:")
            if match_rate > 0.80:
                print(f"   ✅ EXCELLENT - Match rate >80%")
                print(f"   ✅ ID switching issue is FIXED!")
            elif match_rate > 0.60:
                print(f"   ✅ GOOD - Match rate >60%")
                print(f"   ✅ Significant improvement achieved")
            elif match_rate > 0.40:
                print(f"   ⚠️  MODERATE - Match rate >40%")
                print(f"   ⚠️  Some improvement, may need fine-tuning")
            else:
                print(f"   ❌ LOW - Match rate <40%")
                print(f"   ❌ Further optimization needed")
            
            # Compare with previous
            previous_rate = 0.004  # 0.4% from previous run
            improvement = ((match_rate - previous_rate) / previous_rate) * 100
            print(f"\n📊 IMPROVEMENT vs PREVIOUS:")
            print(f"   Previous: {previous_rate:.1%}")
            print(f"   Current: {match_rate:.1%}")
            print(f"   Improvement: {improvement:.0f}x better")
        
        print(f"\n💾 OUTPUT:")
        print(f"   Video: {output_path}")
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024*1024)
            print(f"   Size: {size:.1f} MB")
            print(f"   ✅ Saved successfully")
        
        print(f"\n{'='*70}")
        print(f"🎉 TRACKING TEST COMPLETE!")
        print(f"{'='*70}")
        
        # Recommendations
        print(f"\n💡 NEXT STEPS:")
        if match_rate > 0.80:
            print(f"   ✅ Tracking is stable - ready for production")
            print(f"   ✅ No further tuning needed")
        elif match_rate > 0.60:
            print(f"   ✅ Good results - consider testing on more videos")
            print(f"   💡 May fine-tune parameters for specific scenarios")
        else:
            print(f"   ⚠️  Consider further optimization:")
            print(f"      - Lower conf to 0.20")
            print(f"      - Increase imgsz to 1280")
            print(f"      - Adjust IoU threshold")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tracking()
