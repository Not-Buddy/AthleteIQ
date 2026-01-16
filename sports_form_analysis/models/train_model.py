"""
Model Training Script
Generates synthetic training data and trains the ML classifier
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_engineering import FeatureEngineer
from models.ml_model import FormClassifier


def generate_synthetic_data(n_samples: int = 200, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on biomechanical rules
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    feature_engineer = FeatureEngineer()
    
    features_list = []
    labels = []
    
    # Generate correct form samples (label = 1)
    n_correct = n_samples // 2
    for _ in range(n_correct):
        # Generate keypoints that represent correct form
        keypoints = generate_correct_keypoints()
        features = feature_engineer.extract_frame_features(keypoints)
        features_list.append(list(features.values()))
        labels.append(1)
    
    # Generate incorrect form samples (label = 0)
    n_incorrect = n_samples - n_correct
    for _ in range(n_incorrect):
        # Generate keypoints that represent incorrect form
        keypoints = generate_incorrect_keypoints()
        features = feature_engineer.extract_frame_features(keypoints)
        features_list.append(list(features.values()))
        labels.append(0)
    
    # Shuffle
    indices = np.random.permutation(len(features_list))
    features_array = np.array(features_list)[indices]
    labels_array = np.array(labels)[indices]
    
    return features_array, labels_array


def generate_correct_keypoints() -> np.ndarray:
    """Generate keypoints representing correct form"""
    keypoints = np.zeros((33, 3))
    
    # Base positions (normalized 0-1)
    # Shoulders
    keypoints[11] = [0.4, 0.3, 1.0]  # Left shoulder
    keypoints[12] = [0.6, 0.3, 1.0]  # Right shoulder
    
    # Hips
    keypoints[23] = [0.45, 0.5, 1.0]  # Left hip
    keypoints[24] = [0.55, 0.5, 1.0]  # Right hip
    
    # Elbows (good extension)
    keypoints[13] = [0.3, 0.4, 1.0]   # Left elbow (extended)
    keypoints[14] = [0.7, 0.4, 1.0]   # Right elbow (extended)
    
    # Wrists
    keypoints[15] = [0.2, 0.5, 1.0]   # Left wrist
    keypoints[16] = [0.8, 0.5, 1.0]   # Right wrist
    
    # Knees (stable angles > 90°)
    keypoints[25] = [0.45, 0.65, 1.0]  # Left knee
    keypoints[26] = [0.55, 0.65, 1.0]  # Right knee
    
    # Ankles
    keypoints[27] = [0.45, 0.85, 1.0]  # Left ankle
    keypoints[28] = [0.55, 0.85, 1.0]  # Right ankle
    
    # Head
    keypoints[0] = [0.5, 0.1, 1.0]     # Nose
    
    # Set visibility for all keypoints
    for i in range(33):
        if keypoints[i][2] == 0:
            keypoints[i][2] = 0.5  # Medium visibility for unset points
    
    return keypoints


def generate_incorrect_keypoints() -> np.ndarray:
    """Generate keypoints representing incorrect form"""
    keypoints = np.zeros((33, 3))
    
    # Base positions with deviations
    # Shoulders (uneven)
    keypoints[11] = [0.35, 0.3, 1.0]  # Left shoulder (lower)
    keypoints[12] = [0.65, 0.35, 1.0]  # Right shoulder (higher, misaligned)
    
    # Hips (tilted)
    keypoints[23] = [0.4, 0.5, 1.0]   # Left hip
    keypoints[24] = [0.6, 0.52, 1.0]  # Right hip (slightly higher)
    
    # Elbows (over-flexed)
    keypoints[13] = [0.35, 0.35, 1.0]  # Left elbow (too close to shoulder)
    keypoints[14] = [0.65, 0.35, 1.0]  # Right elbow (too close to shoulder)
    
    # Wrists
    keypoints[15] = [0.3, 0.4, 1.0]    # Left wrist
    keypoints[16] = [0.7, 0.4, 1.0]    # Right wrist
    
    # Knees (unstable angles < 90°)
    keypoints[25] = [0.4, 0.6, 1.0]    # Left knee (too bent)
    keypoints[26] = [0.6, 0.6, 1.0]    # Right knee (too bent)
    
    # Ankles (misaligned)
    keypoints[27] = [0.4, 0.85, 1.0]   # Left ankle
    keypoints[28] = [0.6, 0.88, 1.0]   # Right ankle (misaligned)
    
    # Head (misaligned)
    keypoints[0] = [0.45, 0.1, 1.0]    # Nose (off-center)
    
    # Set visibility
    for i in range(33):
        if keypoints[i][2] == 0:
            keypoints[i][2] = 0.5
    
    return keypoints


def generate_sequence_data(n_sequences: int = 50, frames_per_sequence: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sequence data for training
    
    Args:
        n_sequences: Number of sequences
        frames_per_sequence: Frames per sequence
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)
    feature_engineer = FeatureEngineer()
    
    features_list = []
    labels = []
    
    # Generate correct sequences
    for _ in range(n_sequences // 2):
        keypoints_sequence = []
        for _ in range(frames_per_sequence):
            keypoints = generate_correct_keypoints()
            # Add small temporal variations
            keypoints[:, :2] += np.random.normal(0, 0.01, (33, 2))
            keypoints_sequence.append(keypoints)
        
        features = feature_engineer.extract_all_features(keypoints_sequence)
        features_list.append(features)
        labels.append(1)
    
    # Generate incorrect sequences
    for _ in range(n_sequences - n_sequences // 2):
        keypoints_sequence = []
        for _ in range(frames_per_sequence):
            keypoints = generate_incorrect_keypoints()
            # Add larger temporal variations (jerky movement)
            keypoints[:, :2] += np.random.normal(0, 0.03, (33, 2))
            keypoints_sequence.append(keypoints)
        
        features = feature_engineer.extract_all_features(keypoints_sequence)
        features_list.append(features)
        labels.append(0)
    
    # Shuffle
    indices = np.random.permutation(len(features_list))
    features_array = np.array(features_list)[indices]
    labels_array = np.array(labels)[indices]
    
    return features_array, labels_array


def train_model():
    """Main training function"""
    print("Generating synthetic training data...")
    
    # Generate sequence-based training data
    X, y = generate_sequence_data(n_sequences=100, frames_per_sequence=30)
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features each")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest classifier...")
    classifier = FormClassifier(n_estimators=100, random_state=42)
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_proba = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'form_classifier.pkl')
    classifier.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature importance
    feature_importance = classifier.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    return classifier


if __name__ == "__main__":
    train_model()

