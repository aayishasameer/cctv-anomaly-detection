#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for CCTV Anomaly Detection System
Includes both anomaly detection and multi-object tracking evaluation
"""

import numpy as np
import json
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

class AnomalyEvaluator:
    """Evaluate anomaly detection performance"""
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.timestamps = []
        
    def add_prediction(self, frame_idx: int, track_id: int, is_anomaly: bool, 
                      anomaly_score: float, timestamp: float):
        """Add a prediction for evaluation"""
        self.predictions.append({
            'frame': frame_idx,
            'track_id': track_id,
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'timestamp': timestamp
        })
    
    def load_ground_truth(self, gt_file: str):
        """Load ground truth annotations
        Expected format: JSON with frame-level anomaly labels
        {
            "video_name": {
                "anomaly_frames": [frame1, frame2, ...],
                "normal_frames": [frame1, frame2, ...],
                "anomaly_tracks": {
                    "track_id": [start_frame, end_frame]
                }
            }
        }
        """
        with open(gt_file, 'r') as f:
            self.ground_truth_data = json.load(f)
    
    def evaluate_frame_level(self, video_name: str) -> Dict:
        """Evaluate frame-level anomaly detection"""
        if video_name not in self.ground_truth_data:
            raise ValueError(f"No ground truth for video: {video_name}")
        
        gt_data = self.ground_truth_data[video_name]
        anomaly_frames = set(gt_data.get('anomaly_frames', []))
        normal_frames = set(gt_data.get('normal_frames', []))
        
        # Create frame-level predictions
        pred_frames = defaultdict(list)
        for pred in self.predictions:
            pred_frames[pred['frame']].append(pred['is_anomaly'])
        
        y_true = []
        y_pred = []
        y_scores = []
        
        all_frames = anomaly_frames.union(normal_frames)
        
        for frame in sorted(all_frames):
            # Ground truth
            gt_label = 1 if frame in anomaly_frames else 0
            y_true.append(gt_label)
            
            # Prediction (any anomaly in frame = anomalous frame)
            frame_preds = pred_frames.get(frame, [False])
            pred_label = 1 if any(frame_preds) else 0
            y_pred.append(pred_label)
            
            # Score (max score in frame)
            frame_scores = [p['score'] for p in self.predictions if p['frame'] == frame]
            max_score = max(frame_scores) if frame_scores else 0.0
            y_scores.append(max_score)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc_roc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc_roc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_frames': len(y_true)
        }
    
    def evaluate_temporal_consistency(self, video_name: str, window_size: int = 30) -> Dict:
        """Evaluate temporal consistency of anomaly detection"""
        gt_data = self.ground_truth_data[video_name]
        anomaly_tracks = gt_data.get('anomaly_tracks', {})
        
        consistency_scores = []
        
        for track_id_str, (start_frame, end_frame) in anomaly_tracks.items():
            track_id = int(track_id_str)
            
            # Get predictions for this track in the anomaly period
            track_preds = [
                p for p in self.predictions 
                if p['track_id'] == track_id and start_frame <= p['frame'] <= end_frame
            ]
            
            if not track_preds:
                continue
            
            # Calculate consistency within sliding windows
            for i in range(0, len(track_preds) - window_size + 1, window_size // 2):
                window_preds = track_preds[i:i + window_size]
                anomaly_ratio = sum(p['is_anomaly'] for p in window_preds) / len(window_preds)
                consistency_scores.append(anomaly_ratio)
        
        return {
            'mean_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'std_consistency': np.std(consistency_scores) if consistency_scores else 0,
            'min_consistency': np.min(consistency_scores) if consistency_scores else 0,
            'max_consistency': np.max(consistency_scores) if consistency_scores else 0
        }
    
    def plot_roc_curve(self, video_name: str, save_path: str = None):
        """Plot ROC curve for anomaly detection"""
        from sklearn.metrics import roc_curve
        
        gt_data = self.ground_truth_data[video_name]
        anomaly_frames = set(gt_data.get('anomaly_frames', []))
        normal_frames = set(gt_data.get('normal_frames', []))
        
        y_true = []
        y_scores = []
        
        all_frames = anomaly_frames.union(normal_frames)
        
        for frame in sorted(all_frames):
            gt_label = 1 if frame in anomaly_frames else 0
            y_true.append(gt_label)
            
            frame_scores = [p['score'] for p in self.predictions if p['frame'] == frame]
            max_score = max(frame_scores) if frame_scores else 0.0
            y_scores.append(max_score)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {video_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

class MOTEvaluator:
    """Evaluate Multi-Object Tracking performance"""
    
    def __init__(self):
        self.predicted_tracks = {}
        self.ground_truth_tracks = {}
        
    def add_predicted_track(self, frame_idx: int, track_id: int, bbox: List[float], 
                          confidence: float = 1.0):
        """Add predicted tracking result"""
        if frame_idx not in self.predicted_tracks:
            self.predicted_tracks[frame_idx] = []
        
        self.predicted_tracks[frame_idx].append({
            'track_id': track_id,
            'bbox': bbox,  # [x1, y1, x2, y2]
            'confidence': confidence
        })
    
    def load_ground_truth_tracks(self, gt_file: str):
        """Load ground truth tracking data
        Expected format: MOT Challenge format or similar
        """
        # Implementation depends on ground truth format
        # Common format: frame, track_id, x1, y1, width, height, conf, class, visibility
        pass
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_mota_motp(self, iou_threshold: float = 0.5) -> Dict:
        """Calculate MOTA and MOTP metrics"""
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_distance = 0
        total_matches = 0
        
        # Track ID mapping for ID switch detection
        id_mapping = {}
        
        for frame_idx in sorted(self.predicted_tracks.keys()):
            if frame_idx not in self.ground_truth_tracks:
                continue
            
            pred_tracks = self.predicted_tracks[frame_idx]
            gt_tracks = self.ground_truth_tracks[frame_idx]
            
            total_gt += len(gt_tracks)
            
            # Hungarian algorithm for optimal matching (simplified version)
            matches, fp, fn = self._match_tracks(pred_tracks, gt_tracks, iou_threshold)
            
            total_fp += fp
            total_fn += fn
            total_matches += len(matches)
            
            # Calculate distances for matched pairs
            for pred_idx, gt_idx in matches:
                pred_bbox = pred_tracks[pred_idx]['bbox']
                gt_bbox = gt_tracks[gt_idx]['bbox']
                
                # Center distance
                pred_center = [(pred_bbox[0] + pred_bbox[2])/2, (pred_bbox[1] + pred_bbox[3])/2]
                gt_center = [(gt_bbox[0] + gt_bbox[2])/2, (gt_bbox[1] + gt_bbox[3])/2]
                
                distance = np.sqrt((pred_center[0] - gt_center[0])**2 + 
                                 (pred_center[1] - gt_center[1])**2)
                total_distance += distance
                
                # Check for ID switches
                pred_id = pred_tracks[pred_idx]['track_id']
                gt_id = gt_tracks[gt_idx]['track_id']
                
                if gt_id in id_mapping:
                    if id_mapping[gt_id] != pred_id:
                        total_id_switches += 1
                        id_mapping[gt_id] = pred_id
                else:
                    id_mapping[gt_id] = pred_id
        
        # Calculate MOTA and MOTP
        mota = 1 - (total_fn + total_fp + total_id_switches) / total_gt if total_gt > 0 else 0
        motp = total_distance / total_matches if total_matches > 0 else 0
        
        return {
            'MOTA': mota,
            'MOTP': motp,
            'total_gt': total_gt,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_id_switches': total_id_switches,
            'precision': total_matches / (total_matches + total_fp) if (total_matches + total_fp) > 0 else 0,
            'recall': total_matches / total_gt if total_gt > 0 else 0
        }
    
    def _match_tracks(self, pred_tracks: List, gt_tracks: List, 
                     iou_threshold: float) -> Tuple[List, int, int]:
        """Simple greedy matching (can be improved with Hungarian algorithm)"""
        matches = []
        used_pred = set()
        used_gt = set()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_tracks), len(gt_tracks)))
        for i, pred in enumerate(pred_tracks):
            for j, gt in enumerate(gt_tracks):
                iou_matrix[i, j] = self.calculate_iou(pred['bbox'], gt['bbox'])
        
        # Greedy matching
        while True:
            max_iou = 0
            max_i, max_j = -1, -1
            
            for i in range(len(pred_tracks)):
                if i in used_pred:
                    continue
                for j in range(len(gt_tracks)):
                    if j in used_gt:
                        continue
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= iou_threshold:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j
            
            if max_i == -1:
                break
            
            matches.append((max_i, max_j))
            used_pred.add(max_i)
            used_gt.add(max_j)
        
        fp = len(pred_tracks) - len(matches)
        fn = len(gt_tracks) - len(matches)
        
        return matches, fp, fn

class PerformanceProfiler:
    """Profile system performance (FPS, memory, etc.)"""
    
    def __init__(self):
        self.frame_times = []
        self.memory_usage = []
        self.detection_times = []
        self.tracking_times = []
        
    def start_frame(self):
        """Start timing a frame"""
        import time
        self.frame_start = time.time()
        
    def end_frame(self):
        """End timing a frame"""
        import time
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        
    def add_detection_time(self, detection_time: float):
        """Add detection processing time"""
        self.detection_times.append(detection_time)
        
    def add_tracking_time(self, tracking_time: float):
        """Add tracking processing time"""
        self.tracking_times.append(tracking_time)
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {}
        
        fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        
        return {
            'average_fps': fps,
            'min_fps': 1.0 / np.max(self.frame_times) if self.frame_times else 0,
            'max_fps': 1.0 / np.min(self.frame_times) if self.frame_times else 0,
            'average_frame_time': np.mean(self.frame_times),
            'std_frame_time': np.std(self.frame_times),
            'average_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'average_tracking_time': np.mean(self.tracking_times) if self.tracking_times else 0,
            'total_frames': len(self.frame_times)
        }

def create_sample_ground_truth():
    """Create sample ground truth file for testing"""
    sample_gt = {
        "Shoplifting055_x264.mp4": {
            "anomaly_frames": list(range(500, 600)) + list(range(800, 900)),
            "normal_frames": list(range(0, 500)) + list(range(600, 800)) + list(range(900, 1200)),
            "anomaly_tracks": {
                "1": [500, 600],
                "3": [800, 900]
            }
        }
    }
    
    with open('sample_ground_truth.json', 'w') as f:
        json.dump(sample_gt, f, indent=2)
    
    print("Sample ground truth created: sample_ground_truth.json")

if __name__ == "__main__":
    # Create sample ground truth for testing
    create_sample_ground_truth()
    print("Evaluation framework ready!")