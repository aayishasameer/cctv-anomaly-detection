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
    """Extract behavioral features from tracked persons"""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.track_histories = {}
        
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
        
        # Need minimum sequence length for feature extraction (more frames for stability)
        if len(history['positions']) < 20:
            return None
            
        return self._compute_behavioral_features(history)
    
    def _compute_behavioral_features(self, history: Dict) -> np.ndarray:
        """Compute behavioral features from track history"""
        positions = np.array(history['positions'])
        sizes = np.array(history['sizes'])
        
        features = []
        
        # Motion features
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Speed statistics
            features.extend([
                np.mean(speeds),
                np.std(speeds),
                np.max(speeds),
                np.min(speeds)
            ])
            
            # Direction changes
            if len(velocities) > 1:
                direction_changes = []
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
                        len([x for x in direction_changes if x > np.pi/2])  # Sharp turns
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Size variation features
        size_areas = sizes[:, 2]
        features.extend([
            np.mean(size_areas),
            np.std(size_areas),
            np.max(size_areas) - np.min(size_areas)
        ])
        
        # Position features
        features.extend([
            np.mean(positions[:, 0]),  # avg x
            np.mean(positions[:, 1]),  # avg y
            np.std(positions[:, 0]),   # x variance
            np.std(positions[:, 1]),   # y variance
        ])
        
        # Trajectory features
        if len(positions) > 2:
            # Path length
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            # Displacement
            displacement = np.linalg.norm(positions[-1] - positions[0])
            # Tortuosity (path efficiency)
            tortuosity = path_length / displacement if displacement > 0 else 0
            
            features.extend([path_length, displacement, tortuosity])
        else:
            features.extend([0, 0, 0])
        
        # Pad or truncate to fixed size
        target_size = 128
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
        
        # Set threshold as 99.99th percentile of reconstruction errors (0.01% detection rate)
        self.threshold = np.percentile(reconstruction_errors, 99.99)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")
        
        # Save model
        self.save_model()
        
    def save_model(self):
        """Save trained model and scaler"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
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
        
        self.model.eval()
        print(f"Model loaded from {self.model_path}")
    
    def detect_anomaly(self, track_id: int, bbox: List[float], frame_idx: int) -> Tuple[bool, float]:
        """Detect if current behavior is anomalous"""
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
        
        # Check if anomalous
        is_anomaly = reconstruction_error > self.threshold
        
        return is_anomaly, reconstruction_error