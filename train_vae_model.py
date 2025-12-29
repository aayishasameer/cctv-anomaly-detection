import os
import numpy as np
from vae_anomaly_detector import AnomalyDetector, FeatureExtractor
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def extract_features_from_videos(video_dir: str, output_dir: str = "data/features"):
    """Extract features from normal behavior videos for training"""
    
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    
    feature_extractor = FeatureExtractor()
    all_features = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        track_features = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking
            results = model.track(
                source=frame,
                tracker="botsort.yaml",
                persist=True,
                classes=[0],  # person only
                conf=0.3,
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    features = feature_extractor.extract_features(track_id, box.tolist(), frame_idx)
                    if features is not None:
                        if track_id not in track_features:
                            track_features[track_id] = []
                        track_features[track_id].append(features)
            
            frame_idx += 1
        
        cap.release()
        
        # Collect features from this video
        for track_id, features_list in track_features.items():
            all_features.extend(features_list)
        
        print(f"Processed {video_file}: {len(track_features)} tracks")
    
    # Convert to numpy array
    if all_features:
        features_array = np.array(all_features)
        
        # Save features
        features_path = os.path.join(output_dir, "normal_features.npy")
        np.save(features_path, features_array)
        
        print(f"Extracted {len(features_array)} feature vectors")
        print(f"Features saved to {features_path}")
        
        return features_array
    else:
        print("No features extracted!")
        return None

def main():
    # Extract features from normal videos
    normal_video_dir = "working/normal_shop"
    
    print("Step 1: Extracting features from normal behavior videos...")
    features = extract_features_from_videos(normal_video_dir)
    
    if features is None:
        print("Failed to extract features. Exiting.")
        return
    
    print(f"\nStep 2: Training VAE on {len(features)} samples...")
    
    # Initialize and train anomaly detector
    detector = AnomalyDetector()
    detector.train(features, epochs=150, batch_size=64)
    
    print("\nTraining completed! Model saved.")
    print("You can now use the trained model for real-time anomaly detection.")

if __name__ == "__main__":
    main()