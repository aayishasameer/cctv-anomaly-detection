import os
import json
from anomaly_detection_tracker import AnomalyTracker
from tqdm import tqdm
import argparse

def process_video_batch(input_dir: str, output_dir: str, model_path: str = "models/vae_anomaly_detector.pth"):
    """Process multiple videos for anomaly detection"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracker
    try:
        tracker = AnomalyTracker(model_path)
    except FileNotFoundError:
        print("❌ Please train the model first using: python train_vae_model.py")
        return
    
    # Get video files
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    all_results = {}
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_dir, video_file)
        output_video_path = os.path.join(output_dir, f"anomaly_{video_file}")
        
        print(f"\nProcessing: {video_file}")
        
        # Process video (no display for batch processing)
        anomalies = tracker.process_video(
            video_path, 
            output_video_path, 
            display=False
        )
        
        # Store results (convert numpy types to Python types for JSON serialization)
        serializable_anomalies = []
        for anomaly in anomalies:
            serializable_anomaly = {}
            for key, value in anomaly.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_anomaly[key] = value.item()
                else:
                    serializable_anomaly[key] = value
            serializable_anomalies.append(serializable_anomaly)
        
        all_results[video_file] = {
            'total_anomalies': len(anomalies),
            'anomalies': serializable_anomalies
        }
        
        print(f"✓ Completed {video_file}: {len(anomalies)} anomalies detected")
    
    # Save results summary
    results_path = os.path.join(output_dir, "anomaly_detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n=== Batch Processing Complete ===")
    print(f"Processed {len(video_files)} videos")
    print(f"Results saved to: {results_path}")
    
    total_anomalies = sum(result['total_anomalies'] for result in all_results.values())
    print(f"Total anomalies detected across all videos: {total_anomalies}")
    
    # Show top anomalous videos
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['total_anomalies'], 
                          reverse=True)
    
    print(f"\nTop 5 videos with most anomalies:")
    for i, (video, result) in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {video}: {result['total_anomalies']} anomalies")

def main():
    parser = argparse.ArgumentParser(description='Batch process videos for anomaly detection')
    parser.add_argument('--input-dir', '-i', required=True, 
                       help='Directory containing input videos')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Directory to save processed videos and results')
    parser.add_argument('--model', '-m', default='models/vae_anomaly_detector.pth',
                       help='Path to trained VAE model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found!")
        return
    
    process_video_batch(args.input_dir, args.output_dir, args.model)

if __name__ == "__main__":
    main()