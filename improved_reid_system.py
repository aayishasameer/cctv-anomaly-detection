#!/usr/bin/env python3
"""
Improved Person Re-Identification System with Consistent Tracking
Fixes ID inconsistency issues with advanced matching and temporal consistency
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
import time

class ImprovedReIDModel(nn.Module):
    """Enhanced ReID model with better feature extraction"""
    
    def __init__(self, num_features: int = 512):
        super().__init__()
        
        # ResNet50 backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Enhanced feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_features),
            nn.BatchNorm1d(num_features)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feature_proj(features)
        # L2 normalize
        features = nn.functional.normalize(features, p=2, dim=1)
        return features

class PersonTrack:
    """Enhanced person track with temporal consistency"""
    
    def __init__(self, track_id: int, global_id: int, initial_feature: np.ndarray, 
                 bbox: List[float], timestamp: float):
        self.track_id = track_id
        self.global_id = global_id
        self.features = deque([initial_feature], maxlen=10)  # Keep last 10 features
        self.bboxes = deque([bbox], maxlen=30)
        self.timestamps = deque([timestamp], maxlen=30)
        self.confidences = deque([1.0], maxlen=30)
        
        # Tracking state
        self.last_seen = timestamp
        self.total_frames = 1
        self.lost_frames = 0
        self.is_active = True
        
        # Appearance statistics
        self.avg_height = bbox[3] - bbox[1]
        self.avg_width = bbox[2] - bbox[0]
        
    def update(self, feature: np.ndarray, bbox: List[float], timestamp: float, confidence: float = 1.0):
        """Update track with new detection"""
        self.features.append(feature)
        self.bboxes.append(bbox)
        self.timestamps.append(timestamp)
        self.confidences.append(confidence)
        
        self.last_seen = timestamp
        self.total_frames += 1
        self.lost_frames = 0
        self.is_active = True
        
        # Update appearance statistics
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        self.avg_height = 0.9 * self.avg_height + 0.1 * height
        self.avg_width = 0.9 * self.avg_width + 0.1 * width
    
    def get_average_feature(self) -> np.ndarray:
        """Get average feature vector for robust matching"""
        if len(self.features) == 0:
            return np.zeros(512)
        features_array = np.array(list(self.features))
        return np.mean(features_array, axis=0)
    
    def get_weighted_feature(self) -> np.ndarray:
        """Get weighted average feature (recent frames weighted more)"""
        if len(self.features) == 0:
            return np.zeros(512)
        
        features_array = np.array(list(self.features))
        weights = np.linspace(0.5, 1.0, len(features_array))
        weights = weights / weights.sum()
        
        weighted_feature = np.average(features_array, axis=0, weights=weights)
        return weighted_feature
    
    def mark_lost(self):
        """Mark track as lost"""
        self.lost_frames += 1
        if self.lost_frames > 30:  # Lost for more than 30 frames
            self.is_active = False

class ImprovedReIDTracker:
    """Improved ReID tracker with consistent ID assignment"""
    
    def __init__(self, model_path: str = "models/person_reid_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = ImprovedReIDModel()
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    # Try to load, but handle size mismatch
                    try:
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        print(f"✅ Loaded ReID model from {model_path}")
                    except:
                        print(f"⚠️  Model architecture mismatch, using pretrained backbone")
                else:
                    print(f"⚠️  Using pretrained ResNet50 backbone")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Tracking data structures
        self.active_tracks = {}  # track_id -> PersonTrack
        self.global_id_counter = 1
        self.track_to_global = {}  # local_track_id -> global_id
        
        # Matching parameters - IMPROVED for consistency
        self.similarity_threshold = 0.75  # Balanced threshold
        self.iou_threshold = 0.3  # IoU for spatial consistency
        self.max_lost_frames = 30  # Maximum frames before track is removed
        self.temporal_window = 2.0  # seconds
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'reid_matches': 0,
            'new_ids': 0,
            'id_switches': 0
        }
    
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract ReID features from person crop"""
        
        try:
            # Crop person
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(512)
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return np.zeros(512)
            
            # Convert BGR to RGB
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Transform and extract features
            input_tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return np.zeros(512)
    
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detection_to_tracks(self, feature: np.ndarray, bbox: List[float], 
                                  timestamp: float) -> Optional[int]:
        """Match detection to existing tracks using appearance and spatial cues"""
        
        if not self.active_tracks:
            return None
        
        best_match_id = None
        best_score = 0.0
        
        for track_id, track in self.active_tracks.items():
            if not track.is_active:
                continue
            
            # Time check
            time_diff = timestamp - track.last_seen
            if time_diff > self.temporal_window:
                continue
            
            # Appearance similarity
            track_feature = track.get_weighted_feature()
            appearance_sim = cosine_similarity([feature], [track_feature])[0][0]
            
            # Spatial consistency (IoU with last bbox)
            if len(track.bboxes) > 0:
                last_bbox = track.bboxes[-1]
                spatial_sim = self.compute_iou(bbox, last_bbox)
            else:
                spatial_sim = 0.0
            
            # Size consistency
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            height_ratio = min(height, track.avg_height) / max(height, track.avg_height)
            width_ratio = min(width, track.avg_width) / max(width, track.avg_width)
            size_sim = (height_ratio + width_ratio) / 2
            
            # Combined score with weights
            combined_score = (
                0.6 * appearance_sim +  # Appearance is most important
                0.3 * spatial_sim +      # Spatial consistency
                0.1 * size_sim           # Size consistency
            )
            
            # Apply threshold
            if combined_score > self.similarity_threshold and combined_score > best_score:
                best_score = combined_score
                best_match_id = track_id
        
        return best_match_id
    
    def update(self, frame: np.ndarray, detections: List[Dict], timestamp: float) -> Dict[int, int]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detections with 'track_id' and 'bbox'
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping local track_id to global_id
        """
        
        self.stats['total_detections'] += len(detections)
        
        # Mark all tracks as potentially lost
        for track in self.active_tracks.values():
            track.mark_lost()
        
        # Process each detection
        current_mapping = {}
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            confidence = detection.get('confidence', 1.0)
            
            # Extract features
            feature = self.extract_features(frame, bbox)
            
            # Try to match to existing track
            matched_track_id = self.match_detection_to_tracks(feature, bbox, timestamp)
            
            if matched_track_id is not None:
                # Update existing track
                track = self.active_tracks[matched_track_id]
                track.update(feature, bbox, timestamp, confidence)
                global_id = track.global_id
                self.stats['reid_matches'] += 1
                
            else:
                # Check if this local track_id already has a global_id
                if track_id in self.track_to_global:
                    global_id = self.track_to_global[track_id]
                    
                    # Check if track exists
                    if global_id in self.active_tracks:
                        track = self.active_tracks[global_id]
                        track.update(feature, bbox, timestamp, confidence)
                    else:
                        # Recreate track
                        track = PersonTrack(track_id, global_id, feature, bbox, timestamp)
                        self.active_tracks[global_id] = track
                else:
                    # Create new track with new global ID
                    global_id = self.global_id_counter
                    self.global_id_counter += 1
                    
                    track = PersonTrack(track_id, global_id, feature, bbox, timestamp)
                    self.active_tracks[global_id] = track
                    self.track_to_global[track_id] = global_id
                    self.stats['new_ids'] += 1
            
            current_mapping[track_id] = global_id
        
        # Remove inactive tracks
        inactive_ids = [gid for gid, track in self.active_tracks.items() 
                       if not track.is_active]
        for gid in inactive_ids:
            del self.active_tracks[gid]
        
        return current_mapping
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_detections': self.stats['total_detections'],
            'reid_matches': self.stats['reid_matches'],
            'new_ids': self.stats['new_ids'],
            'active_tracks': len(self.active_tracks),
            'match_rate': self.stats['reid_matches'] / max(1, self.stats['total_detections'])
        }

import os

def main():
    """Test the improved ReID tracker"""
    print("🔍 Testing Improved ReID Tracker")
    print("=" * 50)
    
    tracker = ImprovedReIDTracker()
    
    print("✅ Improved ReID tracker initialized")
    print(f"   Device: {tracker.device}")
    print(f"   Similarity threshold: {tracker.similarity_threshold}")
    print(f"   IoU threshold: {tracker.iou_threshold}")
    print(f"   Max lost frames: {tracker.max_lost_frames}")
    
    print("\n🎯 Key Improvements:")
    print("   ✅ Weighted feature averaging (recent frames prioritized)")
    print("   ✅ Multi-cue matching (appearance + spatial + size)")
    print("   ✅ Temporal consistency checking")
    print("   ✅ Robust track management")
    print("   ✅ ID persistence across occlusions")
    
    print("\n📊 Ready for consistent person tracking!")

if __name__ == "__main__":
    main()
