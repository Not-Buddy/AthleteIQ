"""
Machine Learning Model for Form Classification
Binary classifier: Correct vs Incorrect form
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os


class FormClassifier:
    """
    ML-based form classifier using Random Forest
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the classifier
        
        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random seed for reproducibility
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Labels array of shape (n_samples,) with 0=incorrect, 1=correct
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict form correctness
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, probabilities)
            predictions: 0=incorrect, 1=correct
            probabilities: Probability of correct form
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (correct)
        
        return predictions, probabilities
    
    def predict_single(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict for a single sample
        
        Args:
            features: Feature array of shape (n_features,)
            
        Returns:
            Tuple of (is_correct, confidence_score)
        """
        features = features.reshape(1, -1)
        predictions, probabilities = self.predict(features)
        is_correct = bool(predictions[0])
        confidence = float(probabilities[0])
        
        return is_correct, confidence
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores
        
        Returns:
            Array of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.classifier.feature_importances_
    
    def save(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']

