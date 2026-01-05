#!/usr/bin/env python3
"""
Demo: Complete Stealing Detection Pipeline
Shows all levels of detection from behavioral anomalies to confirmed theft
"""

import cv2
import numpy as np
from stealing_detection_system import StealingDetectionSystem
import os
import argparse

def run_stealing_detection_demo(video_path: str, output_path: str = None):
    """Run complete stealing detection demo"""
    
    print("🛡️ COMPLETE STEALING DETECTION DEMO")
    print("=" * 60)
    print("This demo shows a multi-level stealing detection system:")
    print("🟢 Level 1: Normal behavior tracking")
    print("🟡 Level 2: Suspicious behavior detection") 
    print("🟠 Level 3: High-risk behavior identification")
    print("🔴 Level 4: Stealing behavior detection")
    print("🟣 Level 5: Confirmed theft detection")
    print("=" * 60)
    
    # Initialize system
    try:
        detector = StealingDetectionSystem()
        print("✅ Stealing detection system initialized")
    except FileNotFoundError:
        print("❌ VAE model not found! Please train the model first.")
        return
    
    # Set output path if not provided
    if not output_path:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"stealing_detection_demo_{base_name}.mp4"
    
    print(f"📹 Processing: {video_path}")
    print(f"💾 Output: {output_path}")
    print(f"👋 Hand detection: MediaPipe")
    print(f"🧠 Behavioral analysis: VAE + Multi-criteria")
    print(f"🛒 Shelf interaction: Zone-based detection")
    print("\nPress 'q' to quit, 'SPACE' to pause")
    print("=" * 60)
    
    # Process video with full stealing detection
    results = detector.process_video(
        video_path=video_path,
        output_path=output_path,
        display=True
    )
    
    # Print comprehensive results
    print(f"\n🎯 STEALING DETECTION DEMO RESULTS")
    print(f"=" * 50)
    
    stealing_detections = results['stealing_detections']
    threat_counts = results['threat_counts']
    total_tracks = results['total_tracks']
    
    print(f"📊 Total persons tracked: {total_tracks}")
    print(f"🚨 Stealing alerts generated: {len(stealing_detections)}")
    
    if threat_counts:
        print(f"\n🛡️ Threat Level Summary:")
        for level, count in threat_counts.items():
            emoji = {
                'stealing': '🔴',
                'confirmed_theft': '🟣',
                'high_risk': '🟠'
            }.get(level, '⚠️')
            print(f"  {emoji} {level.upper()}: {count} detections")
    
    # Show top stealing incidents
    if stealing_detections:
        print(f"\n🚨 TOP STEALING INCIDENTS:")
        print(f"-" * 40)
        
        # Sort by threat level and score
        priority_order = {'confirmed_theft': 5, 'stealing': 4, 'high_risk': 3}
        sorted_detections = sorted(
            stealing_detections, 
            key=lambda x: (priority_order.get(x['threat_level'], 0), x['scores']['final_score']),
            reverse=True
        )
        
        for i, detection in enumerate(sorted_detections[:5]):
            threat = detection['threat_level'].upper()
            score = detection['scores']['final_score']
            timestamp = detection['timestamp']
            track_id = detection['track_id']
            
            print(f"{i+1}. {threat} - ID:{track_id} at {timestamp:.1f}s (Score: {score:.3f})")
            
            # Show details
            details = detection['details']
            detail_items = []
            if details['is_loitering']:
                detail_items.append("Loitering")
            if details['has_interactions']:
                detail_items.append(f"{details['recent_interactions']} interactions")
            if details['is_behaviorally_anomalous']:
                detail_items.append("Behavioral anomaly")
            
            if detail_items:
                print(f"   Details: {', '.join(detail_items)}")
    
    print(f"\n💾 Demo video saved to: {output_path}")
    print(f"🎬 You can review the annotated video to see:")
    print(f"   • Person tracking with stable IDs")
    print(f"   • Hand detection and shelf interaction zones")
    print(f"   • Color-coded threat levels")
    print(f"   • Real-time stealing risk assessment")
    
    print(f"\n🏆 DEMO COMPLETE!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Stealing Detection Demo')
    parser.add_argument('--input', '-i', 
                       default='working/test_anomaly/Shoplifting020_x264.mp4',
                       help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"❌ Input video not found: {args.input}")
        print("\nLooking for available videos...")
        
        # Search for video files
        video_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
        
        if video_files:
            print("📹 Available videos:")
            for i, video in enumerate(video_files[:10]):  # Show first 10
                print(f"  {i+1}. {video}")
            
            print(f"\nTo run demo with a specific video:")
            print(f"python demo_stealing_detection.py --input <video_path>")
        else:
            print("❌ No video files found in current directory")
        
        return
    
    # Run the demo
    try:
        results = run_stealing_detection_demo(args.input, args.output)
        
        # Additional analysis
        if results and results['stealing_detections']:
            print(f"\n📈 ADDITIONAL ANALYSIS:")
            detections = results['stealing_detections']
            
            # Time distribution
            timestamps = [d['timestamp'] for d in detections]
            if timestamps:
                print(f"   ⏰ First incident: {min(timestamps):.1f}s")
                print(f"   ⏰ Last incident: {max(timestamps):.1f}s")
                print(f"   ⏰ Incident duration: {max(timestamps) - min(timestamps):.1f}s")
            
            # Score distribution
            scores = [d['scores']['final_score'] for d in detections]
            if scores:
                print(f"   📊 Average risk score: {np.mean(scores):.3f}")
                print(f"   📊 Maximum risk score: {max(scores):.3f}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error during demo: {e}")

if __name__ == "__main__":
    main()