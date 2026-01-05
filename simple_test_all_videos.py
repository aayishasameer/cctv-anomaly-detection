#!/usr/bin/env python3
"""
Simple Test: Process All 5 Videos and Generate Clean Outputs
"""

import os
import time
from dual_window_cctv_system import DualWindowCCTVSystem

def process_all_videos_simple():
    """Process all test videos and generate clean outputs"""
    
    print("🚀 PROCESSING ALL 5 TEST VIDEOS")
    print("=" * 60)
    
    # Find test videos
    test_videos = []
    test_dir = "working/test_anomaly"
    
    if os.path.exists(test_dir):
        for file in sorted(os.listdir(test_dir)):
            if file.lower().endswith('.mp4'):
                test_videos.append(os.path.join(test_dir, file))
    
    if not test_videos:
        print("❌ No test videos found!")
        return
    
    print(f"📹 Found {len(test_videos)} videos:")
    for i, video in enumerate(test_videos, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # Create results directory
    results_dir = "clean_outputs"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each video
    results_summary = []
    total_start = time.time()
    
    for i, video_path in enumerate(test_videos, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n🎬 Processing {i}/{len(test_videos)}: {video_name}")
        print("-" * 50)
        
        try:
            # Initialize system
            camera_id = f"cam_{i}"
            system = DualWindowCCTVSystem(camera_id=camera_id)
            
            # Set output path
            clean_output = os.path.join(results_dir, f"clean_{video_name}.mp4")
            
            print(f"📹 Input: {video_path}")
            print(f"💾 Output: {clean_output}")
            
            # Process video
            start_time = time.time()
            
            results = system.process_video(
                video_path=video_path,
                output_path=clean_output,
                display=False
            )
            
            processing_time = time.time() - start_time
            
            # Get statistics
            reid_stats = system.reid_tracker.get_tracking_statistics()
            anomaly_stats = system.anomaly_detector.get_system_statistics()
            
            # Summary for this video
            video_summary = {
                'video': video_name,
                'status': 'SUCCESS',
                'processing_time': processing_time,
                'frames': results['frames_processed'],
                'avg_fps': results['avg_fps'],
                'global_persons': reid_stats['total_global_persons'],
                'reid_matches': reid_stats['reid_matches'],
                'anomaly_rate': anomaly_stats['anomaly_rate'],
                'output_file': clean_output,
                'file_size_mb': os.path.getsize(clean_output) / (1024*1024) if os.path.exists(clean_output) else 0
            }
            
            results_summary.append(video_summary)
            
            print(f"✅ SUCCESS!")
            print(f"   Time: {processing_time:.1f}s")
            print(f"   Frames: {results['frames_processed']:,}")
            print(f"   FPS: {results['avg_fps']:.1f}")
            print(f"   Persons: {reid_stats['total_global_persons']}")
            print(f"   Anomaly Rate: {anomaly_stats['anomaly_rate']:.2%}")
            print(f"   Output: {os.path.getsize(clean_output) / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            
            video_summary = {
                'video': video_name,
                'status': 'FAILED',
                'error': str(e)
            }
            results_summary.append(video_summary)
    
    # Final summary
    total_time = time.time() - total_start
    successful = [r for r in results_summary if r['status'] == 'SUCCESS']
    failed = [r for r in results_summary if r['status'] == 'FAILED']
    
    print(f"\n🏆 PROCESSING COMPLETE")
    print("=" * 60)
    print(f"📊 Results:")
    print(f"   Total videos: {len(test_videos)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Total time: {total_time:.1f}s")
    
    if successful:
        print(f"\n📈 Aggregate Stats:")
        total_frames = sum(r['frames'] for r in successful)
        avg_fps = sum(r['avg_fps'] for r in successful) / len(successful)
        total_persons = sum(r['global_persons'] for r in successful)
        avg_anomaly_rate = sum(r['anomaly_rate'] for r in successful) / len(successful)
        total_size = sum(r['file_size_mb'] for r in successful)
        
        print(f"   Total frames: {total_frames:,}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total persons: {total_persons}")
        print(f"   Avg anomaly rate: {avg_anomaly_rate:.2%}")
        print(f"   Total output size: {total_size:.1f} MB")
    
    print(f"\n💾 Clean output videos saved in: {results_dir}/")
    
    # List output files
    if os.path.exists(results_dir):
        print(f"\n📁 Output Files:")
        for file in sorted(os.listdir(results_dir)):
            if file.endswith('.mp4'):
                file_path = os.path.join(results_dir, file)
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"   {file} ({size_mb:.1f} MB)")
    
    return results_summary

if __name__ == "__main__":
    try:
        results = process_all_videos_simple()
        print(f"\n🎉 All videos processed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()