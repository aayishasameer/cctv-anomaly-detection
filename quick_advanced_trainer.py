#!/usr/bin/env python3
"""
Quick Advanced Anomaly Detection Training
Simplified version focused on training without GUI dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

class QuickAnomalyDetector(nn.Module):
    """Simplified but effective anomaly detection model"""
    
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

def extract_simple_features():
    """Extract features from our enhanced analytics data"""
    
    print("📊 Loading features from enhanced analytics...")
    
    # Load the enhanced analytics we generated earlier
    analytics_file = "enhanced_analytics_enhanced_demo.json"
    
    if not os.path.exists(analytics_file):
        print(f"❌ Analytics file not found: {analytics_file}")
        return None, None
    
    with open(analytics_file, 'r') as f:
        data = json.load(f)
    
    # Extract features from security alerts
    alerts = data.get('security_alerts', [])
    
    if not alerts:
        print("❌ No alerts found in analytics data")
        return None, None
    
    features = []
    labels = []
    
    for alert in alerts:
        # Extract numerical features from each alert
        feature_vector = [
            alert['confidence'],
            alert['location'][0],  # x position
            alert['location'][1],  # y position
            alert['timestamp'],
            len(alert['description']),  # description length as complexity measure
            1 if 'loitering' in alert['type'] else 0,
            1 if 'suspicious' in alert['type'] else 0,
            1 if alert['severity'] == 'high' else 0,
            1 if alert['severity'] == 'critical' else 0
        ]
        
        features.append(feature_vector)
        
        # Label: 0 = normal/low risk, 1 = anomaly/high risk
        if alert['severity'] in ['high', 'critical'] or alert['confidence'] > 0.6:
            labels.append(1)  # anomaly
        else:
            labels.append(0)  # normal
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"✅ Extracted {len(X)} feature samples")
    print(f"   Feature dimensions: {X.shape[1]}")
    print(f"   Normal samples: {np.sum(y == 0)}")
    print(f"   Anomaly samples: {np.sum(y == 1)}")
    
    return X, y

def create_synthetic_data(n_samples=1000):
    """Create synthetic training data for demonstration"""
    
    print("🎲 Creating synthetic training data...")
    
    np.random.seed(42)
    
    # Normal behavior features
    normal_samples = n_samples // 2
    normal_features = []
    
    for _ in range(normal_samples):
        # Normal: low confidence, regular movement, short duration
        feature = [
            np.random.normal(0.3, 0.1),  # low confidence
            np.random.normal(160, 50),   # x position (center area)
            np.random.normal(120, 30),   # y position (center area)
            np.random.uniform(0, 100),   # timestamp
            np.random.randint(20, 40),   # description length
            np.random.choice([0, 1], p=[0.8, 0.2]),  # loitering (rare)
            np.random.choice([0, 1], p=[0.9, 0.1]),  # suspicious (rare)
            0,  # not high severity
            0   # not critical
        ]
        normal_features.append(feature)
    
    # Anomaly behavior features
    anomaly_samples = n_samples - normal_samples
    anomaly_features = []
    
    for _ in range(anomaly_samples):
        # Anomaly: high confidence, erratic movement, long duration
        feature = [
            np.random.normal(0.7, 0.15),  # high confidence
            np.random.normal(160, 80),    # x position (wider spread)
            np.random.normal(120, 60),    # y position (wider spread)
            np.random.uniform(0, 100),    # timestamp
            np.random.randint(40, 80),    # longer description
            np.random.choice([0, 1], p=[0.3, 0.7]),  # loitering (common)
            np.random.choice([0, 1], p=[0.2, 0.8]),  # suspicious (common)
            np.random.choice([0, 1], p=[0.6, 0.4]),  # high severity
            np.random.choice([0, 1], p=[0.9, 0.1])   # critical (rare)
        ]
        anomaly_features.append(feature)
    
    # Combine data
    X = np.array(normal_features + anomaly_features)
    y = np.array([0] * normal_samples + [1] * anomaly_samples)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Created {len(X)} synthetic samples")
    print(f"   Normal samples: {np.sum(y == 0)}")
    print(f"   Anomaly samples: {np.sum(y == 1)}")
    
    return X, y

def train_quick_model():
    """Train the quick anomaly detection model"""
    
    print("🚀 QUICK ADVANCED ANOMALY TRAINING")
    print("=" * 50)
    
    # Try to load real data first, fallback to synthetic
    X, y = extract_simple_features()
    
    if X is None or len(X) < 100 or len(np.unique(y)) < 2:
        print("⚠️  Real data insufficient, using synthetic data for training...")
        X, y = create_synthetic_data(2000)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Initialize model
    input_dim = X_scaled.shape[1]
    model = QuickAnomalyDetector(input_dim).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🎯 Training on {device}")
    print(f"   Input dimensions: {input_dim}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Training loop
    epochs = 50
    batch_size = 32
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            accuracy = 100 * correct / total
            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), test_predicted.cpu().numpy())
        
        # Classification report
        report = classification_report(
            y_test_tensor.cpu().numpy(), 
            test_predicted.cpu().numpy(),
            target_names=['Normal', 'Anomaly'],
            output_dict=True
        )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'test_accuracy': test_accuracy,
        'classification_report': report
    }
    
    torch.save(model_data, 'models/quick_anomaly_detector.pth')
    
    # Save training summary
    summary = {
        'model_type': 'QuickAnomalyDetector',
        'input_dimensions': input_dim,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': float(test_accuracy),
        'epochs_trained': epochs,
        'classification_report': report
    }
    
    with open('models/quick_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"📊 RESULTS:")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Normal Precision: {report['Normal']['precision']:.3f}")
    print(f"   Normal Recall: {report['Normal']['recall']:.3f}")
    print(f"   Anomaly Precision: {report['Anomaly']['precision']:.3f}")
    print(f"   Anomaly Recall: {report['Anomaly']['recall']:.3f}")
    
    print(f"\n💾 SAVED FILES:")
    print(f"   🤖 Model: models/quick_anomaly_detector.pth")
    print(f"   📊 Summary: models/quick_training_summary.json")
    
    return model_data

if __name__ == "__main__":
    train_quick_model()