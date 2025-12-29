#!/usr/bin/env python3
"""
Multi-Camera Person Re-Identification System
Integrates with CCTV Anomaly Detection for cross-camera tracking
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from collections import defaultdict
import pickle

class ReIDFeatureExtractor(nn.Module):
    """Person Re-Identification Feature Extractor using ResNet backbone"""
    
    def __init__(self, num_classes: int = 1000, feature_dim: int = 2048):
        super(ReIDFeatureExtractor, self).__init__()
        
        # Use ResNet50 as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add ReID-specific layers
        self.feature_dim = feature_dim
        self.bn = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(0.5)
        
        # Classification head for training
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Feature normalization
        self.l2_norm = nn.functional.normalize
        
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone(x)
        features = self.bn(features)
        
        if return_features:
            # Return normalized features for ReID
            return self.l2_norm(features, p=2, dim=1)
        
        # For training
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits, features

class MultiCameraReID:
    """Multi-Camera Person Re-Identification System"""
    
    def __init__(self, model_path: str = "models/reid_model.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load or initialize model
        self.model = None
        self.load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cross-camera tracking
        self.global_tracks = {}  # Global track ID -> features
        self.camera_tracks = {}  # camera_id -> {local_track_id -> global_track_id}
        self.next_global_id = 1
        
        # ReID parameters
        self.similarity_threshold = 0.7
        self.feature_buffer_size = 10
        
    def load_model(self):
        """Load pre-trained ReID model"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = ReIDFeatureExtractor(
                    num_classes=checkpoint.get('num_classes', 1000),
                    feature_dim=checkpoint.get('feature_dim', 2048)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"ReID model loaded from {self.model_path}")
            else:
                print("No pre-trained ReID model found. Using ImageNet pretrained ResNet50.")
                self.model = ReIDFeatureExtractor()
                self.model.to(self.device)
                self.model.eval()
        except Exception as e:
            print(f"Error loading ReID model: {e}")
            self.model = ReIDFeatureExtractor()
            self.model.to(self.device)
            self.model.eval()
    
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract ReID features from person crop"""
        if person_crop is None or person_crop.size == 0:
            return None
        
        try:
            # Preprocess image
            if len(person_crop.shape) == 3:
                person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Transform and add batch dimension
            input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor, return_features=True)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"Error extracting ReID features: {e}")
            return None
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(features1, features2)
        return float(similarity)
    
    def update_global_tracking(self, camera_id: str, local_track_id: int, 
                             person_crop: np.ndarray, bbox: List[float]) -> int:
        """Update global tracking with new detection"""
        
        # Extract ReID features
        features = self.extract_features(person_crop)
        if features is None:
            return -1
        
        # Initialize camera tracking if needed
        if camera_id not in self.camera_tracks:
            self.camera_tracks[camera_id] = {}
        
        # Check if this local track already has a global ID
        if local_track_id in self.camera_tracks[camera_id]:
            global_id = self.camera_tracks[camera_id][local_track_id]
            
            # Update features for existing global track
            if global_id in self.global_tracks:
                self.global_tracks[global_id]['features'].append(features)
                # Keep only recent features
                if len(self.global_tracks[global_id]['features']) > self.feature_buffer_size:
                    self.global_tracks[global_id]['features'] = \
                        self.global_tracks[global_id]['features'][-self.feature_buffer_size:]
                
                # Update last seen info
                self.global_tracks[global_id]['last_camera'] = camera_id
                self.global_tracks[global_id]['last_bbox'] = bbox
            
            return global_id
        
        # Try to match with existing global tracks
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, track_info in self.global_tracks.items():
            # Calculate average similarity with stored features
            similarities = []
            for stored_features in track_info['features']:
                sim = self.calculate_similarity(features, stored_features)
                similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity and avg_similarity > self.similarity_threshold:
                    best_similarity = avg_similarity
                    best_match_id = global_id
        
        if best_match_id is not None:
            # Match found - assign existing global ID
            self.camera_tracks[camera_id][local_track_id] = best_match_id
            
            # Update global track
            self.global_tracks[best_match_id]['features'].append(features)
            self.global_tracks[best_match_id]['cameras'].add(camera_id)
            self.global_tracks[best_match_id]['last_camera'] = camera_id
            self.global_tracks[best_match_id]['last_bbox'] = bbox
            
            return best_match_id
        
        else:
            # No match found - create new global track
            new_global_id = self.next_global_id
            self.next_global_id += 1
            
            self.camera_tracks[camera_id][local_track_id] = new_global_id
            self.global_tracks[new_global_id] = {
                'features': [features],
                'cameras': {camera_id},
                'first_camera': camera_id,
                'last_camera': camera_id,
                'last_bbox': bbox,
                'created_frame': 0  # You can pass frame number
            }
            
            return new_global_id
    
    def get_cross_camera_matches(self) -> Dict:
        """Get tracks that appear in multiple cameras"""
        cross_camera_tracks = {}
        
        for global_id, track_info in self.global_tracks.items():
            if len(track_info['cameras']) > 1:
                cross_camera_tracks[global_id] = {
                    'cameras': list(track_info['cameras']),
                    'first_camera': track_info['first_camera'],
                    'last_camera': track_info['last_camera']
                }
        
        return cross_camera_tracks
    
    def save_reid_database(self, save_path: str):
        """Save ReID database for persistence"""
        reid_data = {
            'global_tracks': self.global_tracks,
            'camera_tracks': self.camera_tracks,
            'next_global_id': self.next_global_id
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(reid_data, f)
        
        print(f"ReID database saved to {save_path}")
    
    def load_reid_database(self, load_path: str):
        """Load ReID database from file"""
        try:
            with open(load_path, 'rb') as f:
                reid_data = pickle.load(f)
            
            self.global_tracks = reid_data['global_tracks']
            self.camera_tracks = reid_data['camera_tracks']
            self.next_global_id = reid_data['next_global_id']
            
            print(f"ReID database loaded from {load_path}")
        except Exception as e:
            print(f"Error loading ReID database: {e}")

class MultiCameraAnomalySystem:
    """Integrated multi-camera anomaly detection with ReID"""
    
    def __init__(self, anomaly_model_path: str = "models/vae_anomaly_detector.pth",
                 reid_model_path: str = "models/reid_model.pth"):
        
        # Import anomaly detector
        from vae_anomaly_detector import AnomalyDetector
        
        # Initialize components
        self.anomaly_detector = AnomalyDetector(anomaly_model_path)
        self.reid_system = MultiCameraReID(reid_model_path)
        
        # Load models
        try:
            self.anomaly_detector.load_model()
            print("✓ Anomaly detection model loaded")
        except Exception as e:
            print(f"❌ Error loading anomaly model: {e}")
        
        # Camera management
        self.active_cameras = {}
        self.camera_configs = {}
        
    def add_camera(self, camera_id: str, camera_source, config: Dict = None):
        """Add a camera to the system"""
        self.active_cameras[camera_id] = {
            'source': camera_source,
            'cap': cv2.VideoCapture(camera_source),
            'frame_count': 0
        }
        
        self.camera_configs[camera_id] = config or {}
        print(f"Camera {camera_id} added")
    
    def process_multi_camera_frame(self, camera_frames: Dict[str, np.ndarray]) -> Dict:
        """Process frames from multiple cameras simultaneously"""
        results = {}
        
        for camera_id, frame in camera_frames.items():
            if frame is None:
                continue
            
            # Run YOLO detection and tracking (you'll need to integrate with your existing tracker)
            # This is a placeholder - integrate with your anomaly_detection_tracker.py
            
            # For each detected person:
            # 1. Run anomaly detection
            # 2. Extract person crop
            # 3. Update ReID system
            # 4. Get global track ID
            
            results[camera_id] = {
                'detections': [],  # List of detections with anomaly scores
                'global_tracks': [],  # Global track IDs
                'cross_camera_matches': []  # Tracks seen in multiple cameras
            }
        
        return results
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        cross_camera_tracks = self.reid_system.get_cross_camera_matches()
        
        return {
            'total_global_tracks': len(self.reid_system.global_tracks),
            'cross_camera_tracks': len(cross_camera_tracks),
            'active_cameras': len(self.active_cameras),
            'reid_similarity_threshold': self.reid_system.similarity_threshold
        }

# Example usage and testing functions
def test_reid_system():
    """Test the ReID system with sample data"""
    reid_system = MultiCameraReID()
    
    # Create dummy person crops for testing
    dummy_crop1 = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    dummy_crop2 = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    
    # Test feature extraction
    features1 = reid_system.extract_features(dummy_crop1)
    features2 = reid_system.extract_features(dummy_crop2)
    
    if features1 is not None and features2 is not None:
        similarity = reid_system.calculate_similarity(features1, features2)
        print(f"Feature similarity: {similarity:.3f}")
    
    # Test global tracking
    global_id1 = reid_system.update_global_tracking("cam1", 1, dummy_crop1, [100, 100, 200, 300])
    global_id2 = reid_system.update_global_tracking("cam2", 1, dummy_crop1, [150, 120, 250, 320])
    
    print(f"Global IDs: {global_id1}, {global_id2}")
    print(f"Cross-camera matches: {reid_system.get_cross_camera_matches()}")

if __name__ == "__main__":
    test_reid_system()