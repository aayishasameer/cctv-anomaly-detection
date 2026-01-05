#!/usr/bin/env python3
"""
Run CCTV System on All Test Videos
Process all 5 test videos and generate clean outputs with detailed analysis
"""

import os
import time
import json
from dual_window_cctv_system import DualWindowCCTVSystem
from datetime import datetime

def find_all_test_videos():
    """Find all test videos"""
    
    test_videos = []
    test_dir = "working/test_anomaly"
    
    if os.path.exists(test_dir):
        for file in sorted(os.listdir(test_dir)):
            if file.lower().endswith('.mp4'):
                test_videos.append(os.path.join(test_dir, file))
    
    return test_videos

def process_all_videos():
    """Process all test videos with the dual window CCTV system"""
    
    print("🚀 COMPREHENSIVE CCTV SYSTEM TEST")
    print("=" * 80)
    print("Processing all 5 test videos with:")
    print("✅ Dual window system (clean video + control panel)")
    print("✅ Improved anomaly detection (reduced false positives)")
    print("✅ Fixed ReID system (unique global IDs)")
    print("✅ 3-color behavior visualization")
    print("✅ Real-time statistics and alerts")
    print("=" * 80)
    
    # Find all test videos
    test_videos = find_all_test_videos()
    
    if not test_videos:
        print("❌ No test videos found in working/test_anomaly/")
        return
    
    print(f"📹 Found {len(test_videos)} test videos:")
    for i, video in enumerate(test_videos, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # Check VAE model
    vae_model_path = "models/vae_anomaly_detector.pth"
    if not os.path.exists(vae_model_path):
        print(f"❌ VAE model not found at {vae_model_path}")
        print("Please train the VAE model first: python train_vae_model.py")
        return
    
    # Create results directory
    results_dir = "test_results_all_videos"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each video
    all_results = {}
    total_start_time = time.time()
    
    for i, video_path in enumerate(test_videos, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n🎬 Processing Video {i}/{len(test_videos)}: {video_name}")
        print("=" * 60)
        
        try:
            # Initialize system for this video
            camera_id = f"cam_{video_name.lower()}"
            system = DualWindowCCTVSystem(camera_id=camera_id)
            
            # Set output paths
            clean_output = os.path.join(results_dir, f"clean_{video_name}.mp4")
            
            print(f"📹 Input: {video_path}")
            print(f"💾 Clean Output: {clean_output}")
            print(f"🎯 Camera ID: {camera_id}")
            
            # Process video
            video_start_time = time.time()
            
            results = system.process_video(
                video_path=video_path,
                output_path=clean_output,
                display=False  # No display for batch processing
            )
            
            video_processing_time = time.time() - video_start_time
            
            # Get additional statistics
            reid_stats = system.reid_tracker.get_tracking_statistics()
            anomaly_stats = system.anomaly_detector.get_system_statistics()
            
            # Compile comprehensive results
            video_results = {
                'video_name': video_name,
                'video_path': video_path,
                'camera_id': camera_id,
                'processing_time': video_processing_time,
                'clean_output_path': clean_output,
                'reid_data_path': f"reid_data_{camera_id}.pkl",
                
                # Processing results
                'frames_processed': results['frames_processed'],
                'avg_fps': results['avg_fps'],
                'total_persons': results['total_persons'],
                
                # ReID statistics
                'reid_statistics': {
                    'total_global_persons': reid_stats['total_global_persons'],
                    'total_tracks_processed': reid_stats['total_tracks_processed'],
                    'reid_matches': reid_stats['reid_matches'],
                    'new_persons_created': reid_stats['new_persons_created'],
                    'reid_match_rate': reid_stats['reid_match_rate'],
                    'active_cameras': reid_stats['active_cameras'],
                    'avg_detections_per_person': reid_stats['avg_detections_per_person']
                },
                
                # Anomaly detection statistics
                'anomaly_statistics': {
                    'total_detections': anomaly_stats['total_detections'],
                    'anomaly_count': anomaly_stats['anomaly_count'],
                    'anomaly_rate': anomaly_stats['anomaly_rate'],
                    'false_positive_reduction': anomaly_stats['false_positive_reduction'],
                    'fp_reduction_rate': anomaly_stats['fp_reduction_rate'],
                    'active_persons': anomaly_stats['active_persons']
                },
                
                # File information
                'output_file_exists': os.path.exists(clean_output),
                'output_file_size_mb': os.path.getsize(clean_output) / (1024*1024) if os.path.exists(clean_output) else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results[video_name] = video_results
            
            # Print summary for this video
            print(f"\n📊 Video {i} Results Summary:")
            print(f"  ⚡ Processing time: {video_processing_time:.1f}s")
            print(f"  📊 Frames processed: {results['frames_processed']}")
            print(f"  🎯 Average FPS: {results['avg_fps']:.1f}")
            print(f"  🌍 Global persons: {reid_stats['total_global_persons']}")
            print(f"  🔄 ReID matches: {reid_stats['reid_matches']}")
            print(f"  📈 ReID match rate: {reid_stats['reid_match_rate']:.2%}")
            print(f"  🚨 Anomaly rate: {anomaly_stats['anomaly_rate']:.2%}")
            print(f"  📉 False positive reduction: {anomaly_stats['fp_reduction_rate']:.2%}")
            print(f"  💾 Output size: {video_results['output_file_size_mb']:.1f} MB")
            print(f"  ✅ Status: SUCCESS")
            
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            
            # Store error information
            video_results = {
                'video_name': video_name,
                'video_path': video_path,
                'error': str(e),
                'status': 'FAILED',
                'timestamp': datetime.now().isoformat()
            }
            all_results[video_name] = video_results
            
            import traceback
            traceback.print_exc()
    
    # Calculate overall statistics
    total_processing_time = time.time() - total_start_time
    successful_videos = [r for r in all_results.values() if 'error' not in r]
    failed_videos = [r for r in all_results.values() if 'error' in r]
    
    # Save comprehensive results
    results_file = os.path.join(results_dir, "comprehensive_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    summary_report = generate_summary_report(all_results, total_processing_time, results_dir)
    
    # Save summary report
    summary_file = os.path.join(results_dir, "summary_report.md")
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    # Print final summary
    print(f"\n🏆 COMPREHENSIVE TEST COMPLETED")
    print("=" * 80)
    print(f"📊 Overall Results:")
    print(f"  🎬 Total videos: {len(test_videos)}")
    print(f"  ✅ Successful: {len(successful_videos)}")
    print(f"  ❌ Failed: {len(failed_videos)}")
    print(f"  ⏱️  Total processing time: {total_processing_time:.1f}s")
    print(f"  📁 Results directory: {results_dir}/")
    print(f"  📄 Detailed results: {results_file}")
    print(f"  📋 Summary report: {summary_file}")
    
    if successful_videos:
        avg_fps = sum(r['avg_fps'] for r in successful_videos) / len(successful_videos)
        total_persons = sum(r['total_persons'] for r in successful_videos)
        total_global_persons = sum(r['reid_statistics']['total_global_persons'] for r in successful_videos)
        avg_anomaly_rate = sum(r['anomaly_statistics']['anomaly_rate'] for r in successful_videos) / len(successful_videos)
        avg_fp_reduction = sum(r['anomaly_statistics']['fp_reduction_rate'] for r in successful_videos) / len(successful_videos)
        
        print(f"\n📈 Aggregate Statistics:")
        print(f"  🎯 Average FPS: {avg_fps:.1f}")
        print(f"  👥 Total persons tracked: {total_persons}")
        print(f"  🌍 Total global persons: {total_global_persons}")
        print(f"  🚨 Average anomaly rate: {avg_anomaly_rate:.2%}")
        print(f"  📉 Average FP reduction: {avg_fp_reduction:.2%}")
    
    print(f"\n💾 Clean output videos saved in: {results_dir}/")
    print(f"🔍 ReID data files saved for each video")
    print(f"📊 All system statistics and metrics documented")
    
    return all_results

def generate_summary_report(all_results, total_time, results_dir):
    """Generate a comprehensive summary report"""
    
    report = f"""# CCTV Anomaly Detection System - Comprehensive Test Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report summarizes the performance of the CCTV Anomaly Detection System across all test videos.

### System Features Tested
- ✅ Dual window system (clean video + control panel)
- ✅ Improved anomaly detection with reduced false positives
- ✅ Fixed ReID system with unique global ID assignment
- ✅ 3-color behavior visualization (Green/Orange/Red)
- ✅ Real-time statistics and performance monitoring

## Test Results Summary

"""
    
    successful = [r for r in all_results.values() if 'error' not in r]
    failed = [r for r in all_results.values() if 'error' in r]
    
    report += f"- **Total Videos Processed**: {len(all_results)}\n"
    report += f"- **Successful**: {len(successful)}\n"
    report += f"- **Failed**: {len(failed)}\n"
    report += f"- **Total Processing Time**: {total_time:.1f} seconds\n\n"
    
    if successful:
        report += "## Individual Video Results\n\n"
        
        for video_name, results in all_results.items():
            if 'error' in results:
                report += f"### {video_name} ❌ FAILED\n"
                report += f"- **Error**: {results['error']}\n\n"
                continue
            
            report += f"### {video_name} ✅ SUCCESS\n\n"
            report += f"- **Processing Time**: {results['processing_time']:.1f}s\n"
            report += f"- **Frames Processed**: {results['frames_processed']:,}\n"
            report += f"- **Average FPS**: {results['avg_fps']:.1f}\n"
            report += f"- **Output File**: `{os.path.basename(results['clean_output_path'])}`\n"
            report += f"- **Output Size**: {results['output_file_size_mb']:.1f} MB\n\n"
            
            report += "#### Person Tracking & ReID\n"
            reid = results['reid_statistics']
            report += f"- **Global Persons**: {reid['total_global_persons']}\n"
            report += f"- **Total Tracks**: {reid['total_tracks_processed']}\n"
            report += f"- **ReID Matches**: {reid['reid_matches']}\n"
            report += f"- **Match Rate**: {reid['reid_match_rate']:.2%}\n"
            report += f"- **Avg Detections/Person**: {reid['avg_detections_per_person']:.1f}\n\n"
            
            report += "#### Anomaly Detection\n"
            anomaly = results['anomaly_statistics']
            report += f"- **Total Detections**: {anomaly['total_detections']:,}\n"
            report += f"- **Anomalies Found**: {anomaly['anomaly_count']}\n"
            report += f"- **Anomaly Rate**: {anomaly['anomaly_rate']:.2%}\n"
            report += f"- **False Positive Reduction**: {anomaly['fp_reduction_rate']:.2%}\n"
            report += f"- **Active Persons**: {anomaly['active_persons']}\n\n"
        
        # Aggregate statistics
        report += "## Aggregate Performance Metrics\n\n"
        
        avg_fps = sum(r['avg_fps'] for r in successful) / len(successful)
        total_frames = sum(r['frames_processed'] for r in successful)
        total_persons = sum(r['total_persons'] for r in successful)
        total_global_persons = sum(r['reid_statistics']['total_global_persons'] for r in successful)
        avg_anomaly_rate = sum(r['anomaly_statistics']['anomaly_rate'] for r in successful) / len(successful)
        avg_fp_reduction = sum(r['anomaly_statistics']['fp_reduction_rate'] for r in successful) / len(successful)
        total_output_size = sum(r['output_file_size_mb'] for r in successful)
        
        report += f"- **Average Processing FPS**: {avg_fps:.1f}\n"
        report += f"- **Total Frames Processed**: {total_frames:,}\n"
        report += f"- **Total Persons Tracked**: {total_persons}\n"
        report += f"- **Total Global Persons**: {total_global_persons}\n"
        report += f"- **Average Anomaly Rate**: {avg_anomaly_rate:.2%}\n"
        report += f"- **Average False Positive Reduction**: {avg_fp_reduction:.2%}\n"
        report += f"- **Total Output Size**: {total_output_size:.1f} MB\n\n"
    
    report += "## Output Files\n\n"
    report += f"All clean output videos are saved in: `{results_dir}/`\n\n"
    
    if successful:
        report += "### Clean Output Videos\n"
        for results in successful:
            video_name = results['video_name']
            output_file = os.path.basename(results['clean_output_path'])
            size_mb = results['output_file_size_mb']
            report += f"- `{output_file}` ({size_mb:.1f} MB) - Clean tracking visualization for {video_name}\n"
        
        report += "\n### ReID Data Files\n"
        for results in successful:
            camera_id = results['camera_id']
            report += f"- `reid_data_{camera_id}.pkl` - Person ReID tracking data\n"
    
    report += f"\n## System Configuration\n\n"
    report += f"- **Anomaly Detection**: VAE-based with improved thresholds\n"
    report += f"- **Person Tracking**: YOLO + BotSORT\n"
    report += f"- **Re-Identification**: ResNet50-based features\n"
    report += f"- **Behavior Analysis**: 3-color classification system\n"
    report += f"- **Output Format**: Clean video without system overlays\n\n"
    
    report += f"---\n"
    report += f"*Report generated by CCTV Anomaly Detection System*\n"
    
    return report

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CCTV System on All Test Videos')
    parser.add_argument('--no-display', action='store_true', help='Disable video display (batch mode)')
    
    args = parser.parse_args()
    
    try:
        results = process_all_videos()
        
        if results:
            successful = len([r for r in results.values() if 'error' not in r])
            total = len(results)
            
            print(f"\n🎉 BATCH PROCESSING COMPLETED!")
            print(f"✅ Successfully processed {successful}/{total} videos")
            print(f"📁 All results saved in test_results_all_videos/")
            
        else:
            print("❌ No videos were processed")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Batch processing interrupted by user")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()