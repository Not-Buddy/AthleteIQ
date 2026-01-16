"""
Biometric Feature Engineering Module
Extracts biomechanical features from pose keypoints
"""

import numpy as np
from typing import List, Dict


class FeatureEngineer:
    """
    Extracts biomechanical features from pose keypoints
    """
    
    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, 
                       point3: np.ndarray) -> float:
        """
        Calculate angle between three points (point2 is the vertex)
        
        Args:
            point1: First point (x, y)
            point2: Vertex point (x, y)
            point3: Third point (x, y)
            
        Returns:
            Angle in degrees
        """
        # Convert to vectors
        vec1 = point1[:2] - point2[:2]
        vec2 = point3[:2] - point2[:2]
        
        # Calculate angle using dot product
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def extract_frame_features(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Extract all biomechanical features from a single frame
        
        Args:
            keypoints: Keypoints array (33, 3)
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Elbow angles
        features['left_elbow_angle'] = self.calculate_angle(
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.LEFT_ELBOW],
            keypoints[self.LEFT_WRIST]
        )
        features['right_elbow_angle'] = self.calculate_angle(
            keypoints[self.RIGHT_SHOULDER],
            keypoints[self.RIGHT_ELBOW],
            keypoints[self.RIGHT_WRIST]
        )
        
        # Knee angles
        features['left_knee_angle'] = self.calculate_angle(
            keypoints[self.LEFT_HIP],
            keypoints[self.LEFT_KNEE],
            keypoints[self.LEFT_ANKLE]
        )
        features['right_knee_angle'] = self.calculate_angle(
            keypoints[self.RIGHT_HIP],
            keypoints[self.RIGHT_KNEE],
            keypoints[self.RIGHT_ANKLE]
        )
        
        # Hip angles
        features['left_hip_angle'] = self.calculate_angle(
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.LEFT_HIP],
            keypoints[self.LEFT_KNEE]
        )
        features['right_hip_angle'] = self.calculate_angle(
            keypoints[self.RIGHT_SHOULDER],
            keypoints[self.RIGHT_HIP],
            keypoints[self.RIGHT_KNEE]
        )
        
        # Spine inclination (angle between vertical and shoulder-hip line)
        shoulder_center = (keypoints[self.LEFT_SHOULDER][:2] + 
                          keypoints[self.RIGHT_SHOULDER][:2]) / 2
        hip_center = (keypoints[self.LEFT_HIP][:2] + 
                     keypoints[self.RIGHT_HIP][:2]) / 2
        
        # Vertical vector (pointing down)
        vertical = np.array([0, 1])
        spine_vector = hip_center - shoulder_center
        
        if np.linalg.norm(spine_vector) > 0:
            cos_angle = np.dot(vertical, spine_vector) / (
                np.linalg.norm(vertical) * np.linalg.norm(spine_vector) + 1e-8
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            spine_angle = np.degrees(np.arccos(cos_angle))
            features['spine_inclination'] = spine_angle
        else:
            features['spine_inclination'] = 0.0
        
        # Head-to-foot alignment (vertical alignment of head and feet)
        head_center = keypoints[self.NOSE][:2]
        left_foot = keypoints[self.LEFT_ANKLE][:2]
        right_foot = keypoints[self.RIGHT_ANKLE][:2]
        foot_center = (left_foot + right_foot) / 2
        
        # Horizontal distance between head and foot centers
        horizontal_offset = abs(head_center[0] - foot_center[0])
        features['head_foot_alignment'] = horizontal_offset
        
        # Center of mass approximation (weighted average of key body points)
        body_points = np.array([
            keypoints[self.LEFT_SHOULDER][:2],
            keypoints[self.RIGHT_SHOULDER][:2],
            keypoints[self.LEFT_HIP][:2],
            keypoints[self.RIGHT_HIP][:2],
            keypoints[self.LEFT_KNEE][:2],
            keypoints[self.RIGHT_KNEE][:2]
        ])
        com = np.mean(body_points, axis=0)
        features['com_x'] = com[0]
        features['com_y'] = com[1]
        
        # Shoulder width (normalized)
        shoulder_width = np.linalg.norm(
            keypoints[self.LEFT_SHOULDER][:2] - 
            keypoints[self.RIGHT_SHOULDER][:2]
        )
        features['shoulder_width'] = shoulder_width
        
        # Hip width (normalized)
        hip_width = np.linalg.norm(
            keypoints[self.LEFT_HIP][:2] - 
            keypoints[self.RIGHT_HIP][:2]
        )
        features['hip_width'] = hip_width
        
        return features
    
    def extract_temporal_features(self, keypoints_sequence: List[np.ndarray]) -> Dict[str, float]:
        """
        Extract temporal consistency features across frames
        
        Args:
            keypoints_sequence: List of keypoint arrays
            
        Returns:
            Dictionary of temporal feature names and values
        """
        if len(keypoints_sequence) < 2:
            return {'movement_smoothness': 0.0, 'velocity_variance': 0.0}
        
        # Calculate velocities for key joints
        velocities = []
        for i in range(1, len(keypoints_sequence)):
            # Calculate velocity for center of mass
            prev_com = self._calculate_com(keypoints_sequence[i-1])
            curr_com = self._calculate_com(keypoints_sequence[i])
            velocity = np.linalg.norm(curr_com - prev_com)
            velocities.append(velocity)
        
        velocities = np.array(velocities)
        
        # Movement smoothness (inverse of velocity variance)
        velocity_variance = np.var(velocities) if len(velocities) > 0 else 0.0
        movement_smoothness = 1.0 / (1.0 + velocity_variance) if velocity_variance > 0 else 1.0
        
        return {
            'movement_smoothness': movement_smoothness,
            'velocity_variance': velocity_variance
        }
    
    def _calculate_com(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate center of mass for a single frame"""
        body_points = np.array([
            keypoints[self.LEFT_SHOULDER][:2],
            keypoints[self.RIGHT_SHOULDER][:2],
            keypoints[self.LEFT_HIP][:2],
            keypoints[self.RIGHT_HIP][:2]
        ])
        return np.mean(body_points, axis=0)
    
    def extract_all_features(self, keypoints_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Extract all features from a sequence of keypoints
        
        Args:
            keypoints_sequence: List of keypoint arrays
            
        Returns:
            Feature array of shape (n_frames, n_features)
        """
        frame_features = []
        
        for keypoints in keypoints_sequence:
            features = self.extract_frame_features(keypoints)
            frame_features.append(list(features.values()))
        
        # Add temporal features
        temporal_features = self.extract_temporal_features(keypoints_sequence)
        
        # Aggregate frame features (mean, std, min, max)
        frame_features_array = np.array(frame_features)
        aggregated = []
        
        # Mean features
        aggregated.extend(np.mean(frame_features_array, axis=0))
        # Std features
        aggregated.extend(np.std(frame_features_array, axis=0))
        # Min features
        aggregated.extend(np.min(frame_features_array, axis=0))
        # Max features
        aggregated.extend(np.max(frame_features_array, axis=0))
        
        # Add temporal features
        aggregated.extend(list(temporal_features.values()))
        
        return np.array(aggregated)

