#!/usr/bin/env python3
"""
Improved Anomaly Detection with Better Thresholds
Fixes the issue of normal walking being flagged as anomalies
"""

import numpy as np
import torch
from vae_anomaly_detector import AnomalyDetector
from typing import Dict, List, Tuple

class ImprovedAnomalyDetector:
    """Improved anomaly detector with better threshold management"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth"):
        # Load the base VAE detector
        self.base_detector = AnomalyDetector(model_path)
        self.base_detector.load_model()
        
        # Improved thresholds - much more conservative
        self.conservative_threshold = self.base_detector.threshold * 2.0  # Double the threshold
        self.moderate_threshold = self.base_detector.threshold * 1.5      # 1.5x threshold
        self.original_threshold = self.base_detector.threshold            # Original threshold
        
        # Track history for better decision making
        self.person_histories = {}
        self.global_anomaly_stats = {
            'total_detections': 0,
            'anomaly_count': 0,
            'false_positive_reduction': 0
        }
        
        print(f"🔧 Improved Anomaly Detector Initialized")
        print(f"   Original threshold: {self.original_threshold:.4f}")
        print(f"   Moderate threshold: {self.moderate_threshold:.4f}")
        print(f"   Conservative threshold: {self.conservative_threshold:.4f}")
    
    def detect_anomaly_improved(self, track_id: int, bbox: List[float], 
                               frame_idx: int, use_conservative: bool = True) -> Tuple[bool, float]:
        """
        Improved anomaly detection with better thresholds
        
        Args:
            track_id: Person track ID
            bbox: Bounding box coordinates
            frame_idx: Current frame index
            use_conservative: Use conservative thresholds to reduce false positives
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        
        # Get base VAE detection
        is_base_anomaly, raw_score = self.base_detector.detect_anomaly(track_id, bbox, frame_idx)
        
        # Initialize person history if new
        if track_id not in self.person_histories:
            self.person_histories[track_id] = {
                'scores': [],
                'detections': [],
                'confirmed_anomalies': 0,
                'total_detections': 0,
                'avg_score': 0.0
            }
        
        history = self.person_histories[track_id]
        history['scores'].append(raw_score)
        history['total_detections'] += 1
        
        # Keep only recent history
        if len(history['scores']) > 30:
            history['scores'] = history['scores'][-30:]
        
        # Calculate running average
        history['avg_score'] = np.mean(history['scores'])
        
        # Improved anomaly decision logic
        if use_conservative:
            # Use much more conservative thresholds
            
            # 1. Check if score is significantly above conservative threshold
            is_high_anomaly = raw_score > self.conservative_threshold
            
            # 2. Check if consistently anomalous (not just a spike)
            if len(history['scores']) >= 10:
                recent_scores = history['scores'][-10:]
                high_score_ratio = sum(1 for s in recent_scores if s > self.moderate_threshold) / len(recent_scores)
                is_consistent_anomaly = high_score_ratio > 0.6  # 60% of recent scores are high
            else:
                is_consistent_anomaly = False
            
            # 3. Final decision - require both high score AND consistency
            is_anomaly = is_high_anomaly and (is_consistent_anomaly or raw_score > self.conservative_threshold * 1.5)
            
            # 4. Normalize score for display (0-1 range)
            normalized_score = min(raw_score / self.conservative_threshold, 1.0)
            
        else:
            # Use moderate thresholds
            is_anomaly = raw_score > self.moderate_threshold
            normalized_score = min(raw_score / self.moderate_threshold, 1.0)
        
        # Update statistics
        self.global_anomaly_stats['total_detections'] += 1
        if is_anomaly:
            history['confirmed_anomalies'] += 1
            self.global_anomaly_stats['anomaly_count'] += 1
        
        # Track false positive reduction
        if is_base_anomaly and not is_anomaly:
            self.global_anomaly_stats['false_positive_reduction'] += 1
        
        return is_anomaly, normalized_score
    
    def get_person_anomaly_profile(self, track_id: int) -> Dict:
        """Get detailed anomaly profile for a person"""
        
        if track_id not in self.person_histories:
            return {'status': 'unknown', 'confidence': 0.0}
        
        history = self.person_histories[track_id]
        
        if history['total_detections'] < 5:
            return {'status': 'insufficient_data', 'confidence': 0.0}
        
        # Calculate anomaly ratio
        anomaly_ratio = history['confirmed_anomalies'] / history['total_detections']
        avg_score = history['avg_score']
        
        # Classify person based on their history
        if anomaly_ratio > 0.7 and avg_score > self.conservative_threshold:
            status = 'highly_anomalous'
            confidence = 0.9
        elif anomaly_ratio > 0.4 and avg_score > self.moderate_threshold:
            status = 'moderately_anomalous'
            confidence = 0.7
        elif anomaly_ratio > 0.2:
            status = 'occasionally_anomalous'
            confidence = 0.5
        else:
            status = 'normal'
            confidence = 0.8
        
        return {
            'status': status,
            'confidence': confidence,
            'anomaly_ratio': anomaly_ratio,
            'avg_score': avg_score,
            'total_detections': history['total_detections'],
            'confirmed_anomalies': history['confirmed_anomalies']
        }
    
    def get_system_statistics(self) -> Dict:
        """Get system-wide anomaly detection statistics"""
        
        total = self.global_anomaly_stats['total_detections']
        anomalies = self.global_anomaly_stats['anomaly_count']
        fp_reduction = self.global_anomaly_stats['false_positive_reduction']
        
        return {
            'total_detections': total,
            'anomaly_count': anomalies,
            'anomaly_rate': anomalies / max(1, total),
            'false_positive_reduction': fp_reduction,
            'fp_reduction_rate': fp_reduction / max(1, total),
            'active_persons': len(self.person_histories)
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.person_histories.clear()
        self.global_anomaly_stats = {
            'total_detections': 0,
            'anomaly_count': 0,
            'false_positive_reduction': 0
        }

class BalancedBehaviorAnalyzer:
    """Balanced behavior analyzer that considers multiple factors"""
    
    def __init__(self, anomaly_detector: ImprovedAnomalyDetector):
        self.anomaly_detector = anomaly_detector
        
        # More balanced thresholds for 3-color system
        self.thresholds = {
            'normal_max': 0.2,      # Below this = definitely normal
            'suspicious_min': 0.2,  # Above this = potentially suspicious
            'suspicious_max': 0.6,  # Below this = suspicious, above = anomaly
            'anomaly_min': 0.6      # Above this = definitely anomaly
        }
        
        # Behavior smoothing
        self.behavior_histories = {}
        self.smoothing_window = 20  # Frames to smooth over
    
    def analyze_behavior(self, track_id: int, bbox: List[float], frame_idx: int) -> Dict:
        """
        Analyze behavior with balanced thresholds
        
        Returns:
            Dict with behavior category, confidence, and details
        """
        
        # Get improved anomaly detection
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly_improved(
            track_id, bbox, frame_idx, use_conservative=True
        )
        
        # Initialize behavior history
        if track_id not in self.behavior_histories:
            self.behavior_histories[track_id] = {
                'scores': [],
                'categories': [],
                'frame_count': 0
            }
        
        history = self.behavior_histories[track_id]
        history['scores'].append(anomaly_score)
        history['frame_count'] += 1
        
        # Keep only recent history
        if len(history['scores']) > self.smoothing_window:
            history['scores'] = history['scores'][-self.smoothing_window:]
        
        # Calculate smoothed score
        if len(history['scores']) >= 5:  # Need at least 5 frames
            smoothed_score = np.mean(history['scores'])
            score_std = np.std(history['scores'])
        else:
            smoothed_score = 0.0  # Not enough data, assume normal
            score_std = 0.0
        
        # Determine behavior category with balanced thresholds
        if smoothed_score <= self.thresholds['normal_max']:
            category = 'normal'
            confidence = 0.9
            color_code = 'green'
        elif smoothed_score <= self.thresholds['suspicious_max']:
            # Check if consistently suspicious or just occasional spikes
            if score_std > 0.2:  # High variance = occasional spikes = probably normal
                category = 'normal'
                confidence = 0.7
                color_code = 'green'
            else:  # Low variance = consistently suspicious
                category = 'suspicious'
                confidence = 0.8
                color_code = 'orange'
        else:
            # High score - check consistency
            if len(history['scores']) >= 10:
                recent_high_scores = sum(1 for s in history['scores'][-10:] if s > self.thresholds['suspicious_max'])
                if recent_high_scores >= 7:  # 70% of recent scores are high
                    category = 'anomaly'
                    confidence = 0.9
                    color_code = 'red'
                else:
                    category = 'suspicious'
                    confidence = 0.7
                    color_code = 'orange'
            else:
                category = 'suspicious'  # Not enough data for anomaly confirmation
                confidence = 0.6
                color_code = 'orange'
        
        # Store category in history
        history['categories'].append(category)
        if len(history['categories']) > self.smoothing_window:
            history['categories'] = history['categories'][-self.smoothing_window:]
        
        # Get person profile from anomaly detector
        person_profile = self.anomaly_detector.get_person_anomaly_profile(track_id)
        
        return {
            'category': category,
            'confidence': confidence,
            'color_code': color_code,
            'anomaly_score': smoothed_score,
            'raw_score': anomaly_score,
            'score_stability': 1.0 - min(score_std, 1.0),  # Higher = more stable
            'frame_count': history['frame_count'],
            'person_profile': person_profile,
            'details': {
                'is_anomaly': is_anomaly,
                'smoothed_score': smoothed_score,
                'score_std': score_std,
                'recent_scores': len(history['scores'])
            }
        }

def test_improved_detection():
    """Test the improved anomaly detection"""
    
    print("🧪 Testing Improved Anomaly Detection")
    print("=" * 50)
    
    try:
        # Initialize improved detector
        improved_detector = ImprovedAnomalyDetector()
        behavior_analyzer = BalancedBehaviorAnalyzer(improved_detector)
        
        print("✅ Improved anomaly detection system initialized")
        
        # Simulate some detections
        test_cases = [
            # Normal walking person
            {'track_id': 1, 'bbox': [100, 100, 200, 300], 'frames': 50, 'description': 'Normal walking'},
            # Slightly erratic person
            {'track_id': 2, 'bbox': [300, 150, 400, 350], 'frames': 50, 'description': 'Slightly erratic'},
            # Truly anomalous person
            {'track_id': 3, 'bbox': [500, 200, 600, 400], 'frames': 50, 'description': 'Anomalous behavior'}
        ]
        
        print(f"\n🎬 Simulating detections...")
        
        for test_case in test_cases:
            track_id = test_case['track_id']
            bbox = test_case['bbox']
            description = test_case['description']
            
            print(f"\n📊 Testing {description} (ID: {track_id}):")
            
            anomaly_detections = 0
            for frame_idx in range(test_case['frames']):
                analysis = behavior_analyzer.analyze_behavior(track_id, bbox, frame_idx)
                
                if analysis['category'] != 'normal':
                    anomaly_detections += 1
                
                # Print periodic updates
                if frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx}: {analysis['category']} ({analysis['anomaly_score']:.3f})")
            
            # Final analysis
            final_analysis = behavior_analyzer.analyze_behavior(track_id, bbox, test_case['frames'])
            print(f"  Final: {final_analysis['category']} | Score: {final_analysis['anomaly_score']:.3f} | Confidence: {final_analysis['confidence']:.2f}")
            print(f"  Anomaly rate: {anomaly_detections/test_case['frames']:.2%}")
        
        # System statistics
        stats = improved_detector.get_system_statistics()
        print(f"\n📈 System Statistics:")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Anomaly rate: {stats['anomaly_rate']:.2%}")
        print(f"  False positive reduction: {stats['fp_reduction_rate']:.2%}")
        
        print(f"\n✅ Improved detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_improved_detection()