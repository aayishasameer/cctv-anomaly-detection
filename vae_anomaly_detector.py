import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Tuple, Optional

class VariationalAutoEncoder(nn.Module):
    """Variational Autoencoder for anomaly detection in human behavior"""
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, latent_dim: int = 16):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class FeatureExtractor:
    """Enhanced behavioral feature extractor with multi-scale temporal analysis"""
    
    def __init__(self, sequence_length: int = 60):  # Increased from 30
        self.sequence_length = sequence_length
        self.track_histories = {}
        
        # Multi-scale temporal windows
        self.short_window = 10   # Short-term patterns
        self.medium_window = 30  # Medium-term patterns  
        self.long_window = 60    # Long-term patterns
        
    def extract_features(self, track_id: int, bbox: List[float], frame_idx: int) -> Optional[np.ndarray]:
        """Extract features from a single detection"""
        x1, y1, x2, y2 = bbox
        
        # Basic geometric features
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        # Initialize track history if new
        if track_id not in self.track_histories:
            self.track_histories[track_id] = {
                'positions': [],
                'sizes': [],
                'frame_indices': []
            }
        
        history = self.track_histories[track_id]
        history['positions'].append([center_x, center_y])
        history['sizes'].append([width, height, area])
        history['frame_indices'].append(frame_idx)
        
        # Keep only recent history
        if len(history['positions']) > self.sequence_length:
            history['positions'] = history['positions'][-self.sequence_length:]
            history['sizes'] = history['sizes'][-self.sequence_length:]
            history['frame_indices'] = history['frame_indices'][-self.sequence_length:]
        
        # Need minimum sequence length for feature extraction (reduced for faster response)
        if len(history['positions']) < 15:  # Reduced from 20
            return None
            
        return self._compute_enhanced_behavioral_features(history)
    
    def _compute_enhanced_behavioral_features(self, history: Dict) -> np.ndarray:
        """Compute enhanced behavioral features with multi-scale temporal analysis"""
        positions = np.array(history['positions'])
        sizes = np.array(history['sizes'])
        frame_indices = np.array(history['frame_indices'])
        
        features = []
        
        # === ENHANCED MOTION FEATURES ===
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Multi-scale speed analysis
            for window_size in [self.short_window, self.medium_window, min(len(speeds), self.long_window)]:
                if len(speeds) >= window_size:
                    recent_speeds = speeds[-window_size:]
                    features.extend([
                        np.mean(recent_speeds),
                        np.std(recent_speeds),
                        np.max(recent_speeds),
                        np.min(recent_speeds),
                        np.percentile(recent_speeds, 75) - np.percentile(recent_speeds, 25)  # IQR
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            
            # === ACCELERATION FEATURES ===
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0)
                acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
                
                features.extend([
                    np.mean(acceleration_magnitudes),
                    np.std(acceleration_magnitudes),
                    np.max(acceleration_magnitudes),
                    len([x for x in acceleration_magnitudes if x > np.mean(acceleration_magnitudes) + 2*np.std(acceleration_magnitudes)])  # Sudden accelerations
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # === ENHANCED DIRECTION ANALYSIS ===
            if len(velocities) > 1:
                direction_changes = []
                velocity_angles = []
                
                for i in range(len(velocities)):
                    if np.linalg.norm(velocities[i]) > 0:
                        angle = np.arctan2(velocities[i][1], velocities[i][0])
                        velocity_angles.append(angle)
                
                # Direction change analysis
                for i in range(1, len(velocities)):
                    if np.linalg.norm(velocities[i-1]) > 0 and np.linalg.norm(velocities[i]) > 0:
                        cos_angle = np.dot(velocities[i-1], velocities[i]) / (
                            np.linalg.norm(velocities[i-1]) * np.linalg.norm(velocities[i])
                        )
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        direction_changes.append(angle)
                
                if direction_changes:
                    features.extend([
                        np.mean(direction_changes),
                        np.std(direction_changes),
                        len([x for x in direction_changes if x > np.pi/2]),  # Sharp turns
                        len([x for x in direction_changes if x > np.pi/4]),  # Moderate turns
                        np.max(direction_changes)  # Maximum direction change
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
                
                # Directional consistency
                if len(velocity_angles) > 1:
                    angle_std = np.std(velocity_angles)
                    features.append(angle_std)
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 0, 0, 0, 0])
        else:
            # No motion data available
            features.extend([0] * 21)  # 15 + 4 + 6 motion features
        
        # === ENHANCED SIZE VARIATION FEATURES ===
        size_areas = sizes[:, 2]
        size_widths = sizes[:, 0]
        size_heights = sizes[:, 1]
        
        for size_data, name in [(size_areas, 'area'), (size_widths, 'width'), (size_heights, 'height')]:
            features.extend([
                np.mean(size_data),
                np.std(size_data),
                np.max(size_data) - np.min(size_data),
                np.percentile(size_data, 75) - np.percentile(size_data, 25)  # IQR
            ])
        
        # === ENHANCED POSITION FEATURES ===
        # Multi-scale position analysis
        for window_size in [self.short_window, self.medium_window, min(len(positions), self.long_window)]:
            if len(positions) >= window_size:
                recent_positions = positions[-window_size:]
                features.extend([
                    np.mean(recent_positions[:, 0]),  # avg x
                    np.mean(recent_positions[:, 1]),  # avg y
                    np.std(recent_positions[:, 0]),   # x variance
                    np.std(recent_positions[:, 1]),   # y variance
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # === ENHANCED TRAJECTORY FEATURES ===
        if len(positions) > 2:
            # Path analysis
            path_segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            path_length = np.sum(path_segments)
            displacement = np.linalg.norm(positions[-1] - positions[0])
            
            # Tortuosity (path efficiency)
            tortuosity = path_length / displacement if displacement > 0 else 0
            
            # Path complexity metrics
            segment_variations = np.std(path_segments) if len(path_segments) > 1 else 0
            
            # Bounding box of trajectory
            min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
            min_y, max_y = np.min(positions[:, 1]), np.max(positions[:, 1])
            trajectory_area = (max_x - min_x) * (max_y - min_y)
            
            features.extend([
                path_length,
                displacement, 
                tortuosity,
                segment_variations,
                trajectory_area,
                (max_x - min_x),  # trajectory width
                (max_y - min_y)   # trajectory height
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # === TEMPORAL FEATURES ===
        if len(frame_indices) > 1:
            frame_diffs = np.diff(frame_indices)
            features.extend([
                np.mean(frame_diffs),  # Average frame gap
                np.std(frame_diffs),   # Frame gap consistency
                np.max(frame_diffs)    # Maximum gap (tracking interruptions)
            ])
        else:
            features.extend([0, 0, 0])
        
        # === BEHAVIORAL PATTERN FEATURES ===
        # Stopping behavior (low speed periods)
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            
            low_speed_threshold = np.percentile(speeds, 25) if len(speeds) > 4 else 0
            stop_periods = len([s for s in speeds if s <= low_speed_threshold])
            
            # Loitering detection (staying in small area)
            if len(positions) >= 10:
                recent_positions = positions[-10:]
                position_spread = np.std(recent_positions, axis=0)
                loitering_score = 1.0 / (1.0 + np.mean(position_spread))
            else:
                loitering_score = 0
            
            features.extend([
                stop_periods / len(speeds) if len(speeds) > 0 else 0,  # Stop ratio
                loitering_score
            ])
        else:
            features.extend([0, 0])
        
        # Pad or truncate to fixed size (increased to 256D)
        target_size = 256
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)

class AnomalyDetector:
    """VAE-based anomaly detector for suspicious behavior"""
    
    def __init__(self, model_path: str = "models/vae_anomaly_detector.pth"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_extractor = FeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, normal_features: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train VAE on normal behavior patterns"""
        print(f"Training VAE on {len(normal_features)} normal samples...")
        
        # Normalize features
        self.scaler = StandardScaler()
        normalized_features = self.scaler.fit_transform(normal_features)
        
        # Create model
        input_dim = normalized_features.shape[1]
        self.model = VariationalAutoEncoder(input_dim=input_dim).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(normalized_features)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch_data)
                
                # VAE loss
                recon_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kld_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Calculate threshold on training data
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                recon_batch, _, _ = self.model(batch_data)
                errors = torch.mean((batch_data - recon_batch) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        # Set threshold as 95th percentile of reconstruction errors (5% detection rate)
        # This is more balanced - was 99.99th (too strict, low recall)
        self.threshold = np.percentile(reconstruction_errors, 95.0)
        print(f"Anomaly threshold set to: {self.threshold:.4f} (95th percentile)")
        
        # Also store additional thresholds for ensemble approach
        self.threshold_90 = np.percentile(reconstruction_errors, 90.0)
        self.threshold_98 = np.percentile(reconstruction_errors, 98.0)
        print(f"Additional thresholds - 90th: {self.threshold_90:.4f}, 98th: {self.threshold_98:.4f}")
        
        # Save model
        self.save_model()
        
    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'threshold_90': self.threshold_90,
            'threshold_98': self.threshold_98,
            'input_dim': self.model.encoder[0].in_features
        }, self.model_path)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        input_dim = checkpoint['input_dim']
        self.model = VariationalAutoEncoder(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']
        
        # Load additional thresholds if available (for ensemble approach)
        self.threshold_90 = checkpoint.get('threshold_90', self.threshold * 0.8)
        self.threshold_98 = checkpoint.get('threshold_98', self.threshold * 1.2)
        
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def detect_anomaly(self, track_id: int, bbox: List[float], frame_idx: int) -> Tuple[bool, float]:
        """Detect if current behavior is anomalous using ensemble approach"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.feature_extractor.extract_features(track_id, bbox, frame_idx)
        if features is None:
            return False, 0.0
        
        # Normalize features
        features_normalized = self.scaler.transform(features.reshape(1, -1))
        
        # Get reconstruction error
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_normalized).to(self.device)
            recon_features, _, _ = self.model(features_tensor)
            reconstruction_error = torch.mean((features_tensor - recon_features) ** 2).item()
        
        # Ensemble approach with multiple thresholds
        is_anomaly_95 = reconstruction_error > self.threshold        # 95th percentile (main)
        is_anomaly_90 = reconstruction_error > self.threshold_90     # 90th percentile (sensitive)
        is_anomaly_98 = reconstruction_error > self.threshold_98     # 98th percentile (conservative)
        
        # Weighted ensemble decision
        ensemble_score = (
            0.6 * float(is_anomaly_95) +    # Main threshold (60% weight)
            0.3 * float(is_anomaly_90) +    # Sensitive threshold (30% weight)  
            0.1 * float(is_anomaly_98)      # Conservative threshold (10% weight)
        )
        
        # Final decision based on ensemble score
        is_anomaly = ensemble_score > 0.5
        
        # Normalize reconstruction error for consistent scoring
        normalized_score = min(reconstruction_error / self.threshold, 2.0)
        
        return is_anomaly, normalized_score