#!/usr/bin/env python3
"""
Advanced Anomaly Detection Model Training
Train sophisticated models using collected behavioral data and video analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pickle
from ultralytics import YOLO
from collections import defaultdict

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 200
    validation_split: float = 0.2
    early_stopping_patience: int = 20
    feature_dim: int = 128
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

class AdvancedAnomalyDetector(nn.Module):
    """Advanced neural network for anomaly detection"""
    
    def __init__(self, input_dim: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Feature extraction layers
        layers.extend([
            nn.Linear(prev_dim, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Anomaly classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # normal, suspicious, anomaly
        )
        
        # Reconstruction head for unsupervised learning
        decoder_layers = []
        prev_dim = config.feature_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        features = self.encoder(x)
        classification = self.classifier(features)
        reconstruction = self.decoder(features)
        
        return {
            'features': features,
            'classification': classification,
            'reconstruction': reconstruction
        }

class BehaviorFeatureExtractor:
    """Extract comprehensive behavioral features from video data"""
    
    def __init__(self):
        self.yolo_model = YOLO("yolov8n.pt")
        self.feature_history = defaultdict(list)
        
    def extract_video_features(self, video_path: str, label: str = "normal") -> List[Dict]:
        """Extract behavioral features from entire video"""
        
        print(f"🎬 Extracting features from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        track_histories = defaultdict(list)
        video_features = []
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Track people
                    results = self.yolo_model.track(
                        source=frame,
                        tracker="botsort.yaml",
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        verbose=False
                    )
                    
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        confidences = results[0].boxes.conf.cpu().numpy()
                        
                        for box, track_id, conf in zip(boxes, track_ids, confidences):
                            # Calculate person metrics
                            center_x = (box[0] + box[2]) / 2
                            center_y = (box[1] + box[3]) / 2
                            width = box[2] - box[0]
                            height = box[3] - box[1]
                            area = width * height
                            
                            person_data = {
                                'frame': frame_idx,
                                'timestamp': frame_idx / fps,
                                'center': [center_x, center_y],
                                'bbox': box.tolist(),
                                'area': area,
                                'confidence': conf,
                                'aspect_ratio': width / height if height > 0 else 0
                            }
                            
                            track_histories[track_id].append(person_data)
                    
                    frame_idx += 1
                    pbar.update(1)
        
        finally:
            cap.release()
        
        # Extract behavioral features for each track
        for track_id, history in track_histories.items():
            if len(history) < 10:  # Skip short tracks
                continue
            
            features = self._extract_track_features(history, label)
            if features is not None:
                video_features.append(features)
        
        print(f"✅ Extracted {len(video_features)} behavioral feature sets")
        return video_features
    
    def _extract_track_features(self, track_history: List[Dict], label: str) -> Optional[Dict]:
        """Extract comprehensive features from a person's track"""
        
        if len(track_history) < 10:
            return None
        
        # Position and movement features
        positions = np.array([p['center'] for p in track_history])
        timestamps = np.array([p['timestamp'] for p in track_history])
        areas = np.array([p['area'] for p in track_history])
        confidences = np.array([p['confidence'] for p in track_history])
        
        # Calculate velocities and accelerations
        velocities = np.diff(positions, axis=0) / np.diff(timestamps).reshape(-1, 1)
        speeds = np.linalg.norm(velocities, axis=1)
        accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[1:]).reshape(-1, 1)
        
        # Movement statistics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        displacement = np.linalg.norm(positions[-1] - positions[0])
        path_efficiency = displacement / total_distance if total_distance > 0 else 0
        
        # Direction changes
        direction_changes = 0
        if len(velocities) > 1:
            for i in range(1, len(velocities)):
                if len(velocities[i-1]) > 0 and len(velocities[i]) > 0:
                    dot_product = np.dot(velocities[i-1], velocities[i])
                    norms = np.linalg.norm(velocities[i-1]) * np.linalg.norm(velocities[i])
                    if norms > 0:
                        angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                        if angle > np.pi / 4:  # 45 degrees
                            direction_changes += 1
        
        # Temporal features
        duration = timestamps[-1] - timestamps[0]
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
        max_speed = np.max(speeds) if len(speeds) > 0 else 0
        speed_variance = np.var(speeds) if len(speeds) > 0 else 0
        
        # Area and size features
        avg_area = np.mean(areas)
        area_variance = np.var(areas)
        
        # Stopping behavior
        stop_threshold = 2.0  # pixels per frame
        stops = np.sum(speeds < stop_threshold) if len(speeds) > 0 else 0
        stop_ratio = stops / len(speeds) if len(speeds) > 0 else 0
        
        # Erratic movement detection
        if len(accelerations) > 0:
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            avg_acceleration = np.mean(acceleration_magnitudes)
            max_acceleration = np.max(acceleration_magnitudes)
        else:
            avg_acceleration = 0
            max_acceleration = 0
        
        # Spatial features
        position_variance = np.var(positions, axis=0)
        spatial_spread = np.sqrt(np.sum(position_variance))
        
        # Compile feature vector
        feature_vector = [
            # Movement features
            avg_speed,
            max_speed,
            speed_variance,
            total_distance,
            displacement,
            path_efficiency,
            direction_changes / duration if duration > 0 else 0,
            
            # Temporal features
            duration,
            len(track_history),  # track length
            stop_ratio,
            
            # Acceleration features
            avg_acceleration,
            max_acceleration,
            
            # Spatial features
            spatial_spread,
            position_variance[0],  # x variance
            position_variance[1],  # y variance
            
            # Size features
            avg_area,
            area_variance,
            np.mean([p['aspect_ratio'] for p in track_history]),
            
            # Confidence features
            np.mean(confidences),
            np.var(confidences)
        ]
        
        return {
            'features': feature_vector,
            'label': label,
            'track_length': len(track_history),
            'duration': duration,
            'metadata': {
                'avg_speed': avg_speed,
                'total_distance': total_distance,
                'stops': stops,
                'direction_changes': direction_changes
            }
        }

class AdvancedAnomalyTrainer:
    """Advanced trainer for anomaly detection models"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = BehaviorFeatureExtractor()
        self.scaler = StandardScaler()
        
        print(f"🔧 Advanced Anomaly Trainer initialized on {self.device}")
    
    def collect_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect and prepare training data from video datasets"""
        
        print("📊 Collecting Training Data")
        print("=" * 50)
        
        all_features = []
        all_labels = []
        
        # Define video datasets
        datasets = [
            {
                'path': 'working/normal_shop',
                'label': 'normal',
                'class_id': 0
            },
            {
                'path': 'working/test_anomaly',
                'label': 'anomaly',
                'class_id': 2
            }
        ]
        
        for dataset in datasets:
            if not os.path.exists(dataset['path']):
                print(f"⚠️  Dataset not found: {dataset['path']}")
                continue
            
            print(f"\n📁 Processing {dataset['label']} videos from {dataset['path']}")
            
            video_files = [f for f in os.listdir(dataset['path']) if f.endswith('.mp4')]
            
            for video_file in video_files[:3]:  # Limit for demo
                video_path = os.path.join(dataset['path'], video_file)
                
                try:
                    features = self.feature_extractor.extract_video_features(
                        video_path, dataset['label']
                    )
                    
                    for feature_data in features:
                        all_features.append(feature_data['features'])
                        all_labels.append(dataset['class_id'])
                    
                    print(f"  ✅ {video_file}: {len(features)} feature sets")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {video_file}: {e}")
        
        if not all_features:
            print("❌ No training data collected!")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n📊 Training Data Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Feature dimensions: {X.shape[1]}")
        print(f"   Normal samples: {np.sum(y == 0)}")
        print(f"   Anomaly samples: {np.sum(y == 2)}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the advanced anomaly detection model"""
        
        print(f"\n🚀 Training Advanced Anomaly Detection Model")
        print("=" * 60)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.config.validation_split, 
            random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Initialize model
        input_dim = X_scaled.shape[1]
        model = AdvancedAnomalyDetector(input_dim, self.config).to(self.device)
        
        # Loss functions and optimizer
        classification_loss = nn.CrossEntropyLoss()
        reconstruction_loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🎯 Training Configuration:")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Input dimensions: {input_dim}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), self.config.batch_size):
                batch_X = X_train_tensor[i:i+self.config.batch_size]
                batch_y = y_train_tensor[i:i+self.config.batch_size]
                
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                
                # Combined loss: classification + reconstruction
                cls_loss = classification_loss(outputs['classification'], batch_y)
                rec_loss = reconstruction_loss(outputs['reconstruction'], batch_X)
                total_loss = cls_loss + 0.1 * rec_loss  # Weight reconstruction loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['classification'].data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                outputs = model(X_val_tensor)
                cls_loss = classification_loss(outputs['classification'], y_val_tensor)
                rec_loss = reconstruction_loss(outputs['reconstruction'], X_val_tensor)
                val_loss = cls_loss + 0.1 * rec_loss
                
                _, predicted = torch.max(outputs['classification'].data, 1)
                val_total += y_val_tensor.size(0)
                val_correct += (predicted == y_val_tensor).sum().item()
            
            # Update history
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss / (len(X_train_tensor) // self.config.batch_size))
            history['val_loss'].append(val_loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': self.scaler,
                    'config': self.config,
                    'input_dim': input_dim
                }, 'models/advanced_anomaly_detector.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Loss: {history['train_loss'][-1]:.4f} | "
                      f"Val Loss: {history['val_loss'][-1]:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            _, predicted = torch.max(outputs['classification'].data, 1)
            
            # Generate classification report
            y_val_np = y_val_tensor.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            # Get unique classes present in data
            unique_classes = np.unique(np.concatenate([y_val_np, predicted_np]))
            class_names = ['Normal', 'Suspicious', 'Anomaly']
            present_class_names = [class_names[i] for i in unique_classes]
            
            report = classification_report(y_val_np, predicted_np, 
                                         labels=unique_classes,
                                         target_names=present_class_names, 
                                         output_dict=True)
        
        results = {
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'classification_report': report,
            'history': history
        }
        
        return results
    
    def save_training_plots(self, history: Dict):
        """Save training visualization plots"""
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            ax1.plot(history['train_loss'], label='Training Loss')
            ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            ax2.plot(history['train_acc'], label='Training Accuracy')
            ax2.plot(history['val_acc'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Training plots saved to: models/training_history.png")
            
        except Exception as e:
            print(f"⚠️  Could not save plots: {e}")
            # Save training history as JSON instead
            with open('models/training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            print("📊 Training history saved to: models/training_history.json")

def main():
    """Main training function"""
    
    print("🚀 ADVANCED ANOMALY DETECTION TRAINING")
    print("=" * 70)
    print("🔥 FEATURES:")
    print("✅ Multi-class classification (Normal/Suspicious/Anomaly)")
    print("✅ Advanced behavioral feature extraction")
    print("✅ Deep neural network with reconstruction loss")
    print("✅ Real video data training")
    print("✅ Comprehensive evaluation metrics")
    print("=" * 70)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize trainer
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=100,
        validation_split=0.2,
        early_stopping_patience=15
    )
    
    trainer = AdvancedAnomalyTrainer(config)
    
    try:
        # Collect training data
        X, y = trainer.collect_training_data()
        
        if X is None or len(X) == 0:
            print("❌ No training data available!")
            return
        
        # Train model
        results = trainer.train_model(X, y)
        
        # Save training plots
        trainer.save_training_plots(results['history'])
        
        # Print final results
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"=" * 50)
        print(f"📊 FINAL RESULTS:")
        print(f"   Training Accuracy: {results['final_train_acc']:.2f}%")
        print(f"   Validation Accuracy: {results['final_val_acc']:.2f}%")
        print(f"   Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"   Epochs Trained: {results['epochs_trained']}")
        
        print(f"\n📋 CLASSIFICATION REPORT:")
        report = results['classification_report']
        class_names = ['Normal', 'Suspicious', 'Anomaly']
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                metrics = report[str(i)]
                print(f"   {class_name:10s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            elif class_name in report:
                metrics = report[class_name]
                print(f"   {class_name:10s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        print(f"\n💾 SAVED FILES:")
        print(f"   🤖 Model: models/advanced_anomaly_detector.pth")
        print(f"   📊 Plots: models/training_history.png")
        print(f"\n🔄 Model ready for enhanced anomaly detection!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()