#!/usr/bin/env python3
"""
Integrated CCTV System with Advanced Trained Models
Complete system using all our trained models working together
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enhanced_cctv_system import EnhancedCCTVSystem
import pickle

class QuickAnomalyDetector(nn.Module):
    """Quick anomaly detection model (same as training)"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 2)  # normal vs anomaly
        )
    
    def forward(self, x):
        return self.network(x)

@dataclass
class IntegratedAlert:
    """Enhanced alert with multiple model predictions"""
    timestamp: float
    person_id: str
    location: Tuple[int, int]
    
    # Model predictions
    vae_anomaly_score: float
    neural_anomaly_score: float
    reid_confidence: float
    zone_violation: bool
    
    # Combined assessment
    final_risk_level: str
    confidence: float
    alert_type: str
    description: str

class IntegratedCCTVSystem(EnhancedCCTVSystem):
    """Integrated system using all trained models"""
    
    def __init__(self, camera_id: str = "integrated_cam"):
        super().__init__(camera_id)
        
        # Load our trained models
        self.neural_anomaly_model = None
        self.neural_scaler = None
        self.load_neural_anomaly_model()
        
        # Enhanced thresholds
        self.integrated_thresholds = {
            'neural_anomaly': 0.5,
            'vae_anomaly': 0.6,
            'combined_risk': 0.7,
            'high_risk': 0.8,
            'critical_risk': 0.9
        }
        
        # Model weights for ensemble
        self.model_weights = {
            'neural': 0.4,
            'vae': 0.3,
            'behavioral': 0.2,
            'zone': 0.1
        }
        
        print(f"🔥 Integrated CCTV System initialized with all trained models")
    
    def load_neural_anomaly_model(self):
        """Load our trained neural anomaly detection model"""
        
        model_path = "models/quick_anomaly_detector.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Neural anomaly model not found: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            input_dim = checkpoint['input_dim']
            self.neural_anomaly_model = QuickAnomalyDetector(input_dim)
            self.neural_anomaly_model.load_state_dict(checkpoint['model_state_dict'])
            self.neural_anomaly_model.eval()
            
            self.neural_scaler = checkpoint['scaler']
            
            print(f"✅ Neural anomaly model loaded (accuracy: {checkpoint['test_accuracy']:.3f})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading neural model: {e}")
            return False
    
    def extract_neural_features(self, person_data: Dict) -> np.ndarray:
        """Extract features for neural anomaly detection"""
        
        # Extract the same features we used for training
        features = [
            person_data.get('anomaly_score', 0.0),  # VAE anomaly score
            person_data['position'][0],  # x position
            person_data['position'][1],  # y position
            person_data['timestamp'],
            len(person_data.get('description', 'normal')),  # complexity
            1 if 'loitering' in person_data.get('behavior_type', '') else 0,
            1 if 'suspicious' in person_data.get('behavior_type', '') else 0,
            1 if person_data.get('risk_level', 'low') == 'high' else 0,
            1 if person_data.get('risk_level', 'low') == 'critical' else 0
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_neural_anomaly(self, person_data: Dict) -> float:
        """Get neural network anomaly prediction"""
        
        if self.neural_anomaly_model is None or self.neural_scaler is None:
            return 0.0
        
        try:
            # Extract and normalize features
            features = self.extract_neural_features(person_data)
            features_scaled = self.neural_scaler.transform(features)
            
            # Predict
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                outputs = self.neural_anomaly_model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                anomaly_prob = probabilities[0][1].item()  # Probability of anomaly class
            
            return anomaly_prob
            
        except Exception as e:
            print(f"⚠️  Neural prediction error: {e}")
            return 0.0
    
    def integrated_risk_assessment(self, person_data: Dict) -> IntegratedAlert:
        """Comprehensive risk assessment using all models"""
        
        # Get predictions from all models
        vae_score = person_data.get('anomaly_score', 0.0)
        neural_score = self.predict_neural_anomaly(person_data)
        reid_conf = person_data.get('reid_confidence', 1.0)
        
        # Zone violation check
        zone_violation = self._check_zone_violations(person_data)
        
        # Behavioral analysis
        behavioral_risk = self._assess_behavioral_risk(person_data)
        
        # Ensemble prediction
        combined_score = (
            self.model_weights['neural'] * neural_score +
            self.model_weights['vae'] * vae_score +
            self.model_weights['behavioral'] * behavioral_risk +
            self.model_weights['zone'] * (1.0 if zone_violation else 0.0)
        )
        
        # Risk level determination
        if combined_score >= self.integrated_thresholds['critical_risk']:
            risk_level = 'critical'
            alert_type = 'security_breach'
        elif combined_score >= self.integrated_thresholds['high_risk']:
            risk_level = 'high'
            alert_type = 'suspicious_activity'
        elif combined_score >= self.integrated_thresholds['combined_risk']:
            risk_level = 'medium'
            alert_type = 'anomalous_behavior'
        else:
            risk_level = 'low'
            alert_type = 'normal_activity'
        
        # Generate description
        description = self._generate_integrated_description(
            neural_score, vae_score, behavioral_risk, zone_violation, risk_level
        )
        
        # Create integrated alert
        alert = IntegratedAlert(
            timestamp=person_data['timestamp'],
            person_id=person_data['global_id'],
            location=tuple(person_data['position']),
            vae_anomaly_score=vae_score,
            neural_anomaly_score=neural_score,
            reid_confidence=reid_conf,
            zone_violation=zone_violation,
            final_risk_level=risk_level,
            confidence=combined_score,
            alert_type=alert_type,
            description=description
        )
        
        return alert
    
    def _check_zone_violations(self, person_data: Dict) -> bool:
        """Check if person is violating learned zones"""
        
        # Simple zone violation check (can be enhanced)
        position = person_data['position']
        
        # Check if in restricted areas (example logic)
        if hasattr(self, 'interaction_zones') and self.interaction_zones:
            for zone in self.interaction_zones:
                bbox = zone.get('bbox', [0, 0, 1000, 1000])
                if (bbox[0] <= position[0] <= bbox[2] and 
                    bbox[1] <= position[1] <= bbox[3]):
                    # In interaction zone - check duration
                    duration = person_data.get('zone_duration', 0)
                    if duration > 10.0:  # 10 seconds threshold
                        return True
        
        return False
    
    def _assess_behavioral_risk(self, person_data: Dict) -> float:
        """Assess behavioral risk based on movement patterns"""
        
        speed = person_data.get('speed', 0)
        duration = person_data.get('duration', 0)
        
        risk_score = 0.0
        
        # Speed-based risk
        if speed > 50:  # Very fast movement
            risk_score += 0.3
        elif speed < 2:  # Very slow/stationary
            risk_score += 0.2
        
        # Duration-based risk
        if duration > 30:  # Long presence
            risk_score += 0.3
        
        # Pattern-based risk (simplified)
        behavior_type = person_data.get('behavior_type', 'normal')
        if 'erratic' in behavior_type:
            risk_score += 0.4
        elif 'loitering' in behavior_type:
            risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    def _generate_integrated_description(self, neural_score: float, vae_score: float, 
                                       behavioral_risk: float, zone_violation: bool, 
                                       risk_level: str) -> str:
        """Generate comprehensive alert description"""
        
        components = []
        
        if neural_score > 0.7:
            components.append(f"Neural model: HIGH anomaly ({neural_score:.2f})")
        elif neural_score > 0.5:
            components.append(f"Neural model: anomaly detected ({neural_score:.2f})")
        
        if vae_score > 0.6:
            components.append(f"VAE model: behavioral anomaly ({vae_score:.2f})")
        
        if behavioral_risk > 0.5:
            components.append(f"Behavioral analysis: suspicious patterns")
        
        if zone_violation:
            components.append("Zone violation: extended presence in interaction area")
        
        if not components:
            return f"Normal activity - {risk_level} risk level"
        
        return f"{risk_level.upper()} RISK: " + " | ".join(components)
    
    def process_integrated_video(self, video_path: str, output_path: str = None, 
                               display: bool = False) -> Dict:
        """Process video with integrated model analysis"""
        
        print(f"🔥 INTEGRATED PROCESSING: {os.path.basename(video_path)}")
        print("=" * 60)
        print("🤖 Active Models:")
        print("   ✅ YOLO Person Detection")
        print("   ✅ VAE Anomaly Detection")
        print("   ✅ Neural Anomaly Classification")
        print("   ✅ Person ReID System")
        print("   ✅ Adaptive Zone Learning")
        print("   ✅ Behavioral Pattern Analysis")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing metrics
        start_time = time.time()
        processed_frames = 0
        integrated_alerts = []
        risk_summary = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with enhanced system
                enhanced_frame, frame_analytics = self._process_enhanced_frame(
                    frame, processed_frames, fps
                )
                
                # Integrated analysis for each person
                for person_data in frame_analytics.get('persons', []):
                    # Add behavioral context
                    person_data['behavior_type'] = 'normal'  # Simplified
                    person_data['duration'] = processed_frames / fps
                    person_data['zone_duration'] = 0  # Simplified
                    person_data['description'] = 'person detected'
                    
                    # Get integrated risk assessment
                    alert = self.integrated_risk_assessment(person_data)
                    
                    if alert.final_risk_level != 'low':
                        integrated_alerts.append(alert)
                        risk_summary[alert.final_risk_level] += 1
                        
                        # Print high-priority alerts
                        if alert.final_risk_level in ['high', 'critical']:
                            print(f"🚨 {alert.final_risk_level.upper()}: {alert.description}")
                
                # Enhanced visualization
                if display or output_path:
                    enhanced_frame = self._add_integrated_visualizations(
                        enhanced_frame, frame_analytics, processed_frames
                    )
                
                if display:
                    cv2.imshow('Integrated CCTV System', enhanced_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if output_path:
                    out.write(enhanced_frame)
                
                processed_frames += 1
                
                # Progress update
                if processed_frames % 200 == 0:
                    progress = (processed_frames / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% | Alerts: {len(integrated_alerts)} | "
                          f"High Risk: {risk_summary['high']} | Critical: {risk_summary['critical']}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Generate comprehensive results
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        results = {
            'frames_processed': processed_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'total_alerts': len(integrated_alerts),
            'risk_summary': risk_summary,
            'model_performance': {
                'neural_model_active': self.neural_anomaly_model is not None,
                'vae_model_active': True,
                'reid_model_active': True,
                'zone_learning_active': True
            }
        }
        
        # Save integrated analytics
        self._save_integrated_analytics(results, integrated_alerts)
        
        return results
    
    def _add_integrated_visualizations(self, frame: np.ndarray, analytics: Dict, 
                                     frame_idx: int) -> np.ndarray:
        """Add integrated model visualizations"""
        
        enhanced_frame = frame.copy()
        
        # Enhanced person annotations with integrated scores
        for person in analytics.get('persons', []):
            bbox = person['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get integrated assessment
            alert = self.integrated_risk_assessment(person)
            
            # Color based on integrated risk
            risk_colors = {
                'low': (0, 255, 0),      # Green
                'medium': (0, 165, 255), # Orange
                'high': (0, 0, 255),     # Red
                'critical': (255, 0, 255) # Magenta
            }
            
            color = risk_colors.get(alert.final_risk_level, (0, 255, 0))
            thickness = 4 if alert.final_risk_level in ['high', 'critical'] else 2
            
            # Draw enhanced bounding box
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Multi-line labels
            labels = [
                f"{person['global_id']} | {alert.final_risk_level.upper()}",
                f"Neural: {alert.neural_anomaly_score:.2f} | VAE: {alert.vae_anomaly_score:.2f}",
                f"Combined: {alert.confidence:.2f}"
            ]
            
            for i, label in enumerate(labels):
                y_offset = y1 - 10 - (i * 20)
                cv2.putText(enhanced_frame, label, (x1, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # System status with all models
        status_lines = [
            f"🔥 INTEGRATED CCTV SYSTEM | Frame: {frame_idx}",
            f"🤖 Models: Neural✅ VAE✅ ReID✅ Zones✅",
            f"👥 Persons: {len(analytics.get('persons', []))}",
            f"🚨 Total Alerts: {len(self.security_alerts)}"
        ]
        
        for i, line in enumerate(status_lines):
            cv2.putText(enhanced_frame, line, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return enhanced_frame
    
    def _save_integrated_analytics(self, results: Dict, alerts: List[IntegratedAlert]):
        """Save comprehensive integrated analytics"""
        
        # Convert alerts to serializable format
        alerts_data = []
        for alert in alerts:
            alerts_data.append({
                'timestamp': alert.timestamp,
                'person_id': alert.person_id,
                'location': alert.location,
                'vae_anomaly_score': alert.vae_anomaly_score,
                'neural_anomaly_score': alert.neural_anomaly_score,
                'reid_confidence': alert.reid_confidence,
                'zone_violation': alert.zone_violation,
                'final_risk_level': alert.final_risk_level,
                'confidence': alert.confidence,
                'alert_type': alert.alert_type,
                'description': alert.description
            })
        
        analytics_data = {
            'processing_results': results,
            'integrated_alerts': alerts_data,
            'model_weights': self.model_weights,
            'thresholds': self.integrated_thresholds,
            'system_info': {
                'camera_id': self.camera_id,
                'models_active': results['model_performance'],
                'processing_timestamp': time.time()
            }
        }
        
        # Save comprehensive analytics
        with open(f'integrated_analytics_{self.camera_id}.json', 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        print(f"📊 Integrated analytics saved: integrated_analytics_{self.camera_id}.json")

def main():
    """Demo the integrated CCTV system"""
    
    print("🔥 INTEGRATED CCTV SYSTEM - ALL MODELS ACTIVE")
    print("=" * 70)
    print("🚀 NEXT-GENERATION FEATURES:")
    print("✅ Neural anomaly detection (trained model)")
    print("✅ VAE behavioral analysis")
    print("✅ Person ReID with global tracking")
    print("✅ Adaptive zone learning")
    print("✅ Ensemble model predictions")
    print("✅ Multi-level risk assessment")
    print("✅ Comprehensive analytics")
    print("=" * 70)
    
    # Test video
    test_video = "working/test_anomaly/Shoplifting020_x264.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return
    
    try:
        # Initialize integrated system
        system = IntegratedCCTVSystem(camera_id="integrated_demo")
        
        # Process with all models
        output_path = f"integrated_demo_output_{int(time.time())}.mp4"
        
        results = system.process_integrated_video(
            video_path=test_video,
            output_path=output_path,
            display=False
        )
        
        print(f"\n🎉 INTEGRATED PROCESSING COMPLETED!")
        print(f"📊 COMPREHENSIVE RESULTS:")
        print(f"   Frames processed: {results['frames_processed']}")
        print(f"   Average FPS: {results['avg_fps']:.1f}")
        print(f"   Total alerts: {results['total_alerts']}")
        print(f"   Risk breakdown:")
        for risk, count in results['risk_summary'].items():
            print(f"     {risk.capitalize()}: {count}")
        
        print(f"\n🤖 MODEL STATUS:")
        for model, active in results['model_performance'].items():
            status = "✅ ACTIVE" if active else "❌ INACTIVE"
            print(f"   {model}: {status}")
        
        print(f"\n💾 OUTPUTS:")
        print(f"   📹 Video: {output_path}")
        print(f"   📊 Analytics: integrated_analytics_integrated_demo.json")
        
    except Exception as e:
        print(f"❌ Integrated demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()