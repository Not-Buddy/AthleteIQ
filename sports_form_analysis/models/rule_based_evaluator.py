"""
Rule-Based Form Evaluator
Deterministic biomechanical threshold-based evaluation
"""

import numpy as np
from typing import Dict, List, Tuple


class RuleBasedEvaluator:
    """
    Evaluates form using biomechanical rules and thresholds
    """
    
    def __init__(self):
        """Initialize rule-based evaluator with biomechanical thresholds"""
        # Biomechanical thresholds (in degrees for angles)
        self.thresholds = {
            'elbow_angle_min': 30.0,  # Minimum elbow flexion
            'elbow_angle_max': 180.0,  # Maximum elbow extension
            'knee_angle_min': 90.0,   # Minimum knee flexion for stability
            'knee_angle_max': 180.0,  # Maximum knee extension
            'hip_angle_min': 120.0,   # Minimum hip angle for proper posture
            'hip_angle_max': 180.0,   # Maximum hip angle
            'spine_inclination_max': 30.0,  # Maximum spine deviation from vertical
            'head_foot_alignment_max': 0.15,  # Maximum horizontal offset (normalized)
            'movement_smoothness_min': 0.5,  # Minimum movement smoothness
        }
    
    def evaluate_frame(self, features: Dict[str, float]) -> Tuple[bool, List[str], float]:
        """
        Evaluate form for a single frame
        
        Args:
            features: Dictionary of biomechanical features
            
        Returns:
            Tuple of (is_correct, list of issues, confidence_score)
        """
        issues = []
        confidence = 1.0
        
        # Check elbow angles
        if features.get('left_elbow_angle', 180) < self.thresholds['elbow_angle_min']:
            issues.append("Left elbow over-flexed (< 30°)")
            confidence -= 0.1
        if features.get('right_elbow_angle', 180) < self.thresholds['elbow_angle_min']:
            issues.append("Right elbow over-flexed (< 30°)")
            confidence -= 0.1
        
        # Check knee angles (stability)
        left_knee = features.get('left_knee_angle', 180)
        right_knee = features.get('right_knee_angle', 180)
        
        if left_knee < self.thresholds['knee_angle_min']:
            issues.append("Left knee angle too acute (< 90°) - unstable base")
            confidence -= 0.15
        if right_knee < self.thresholds['knee_angle_min']:
            issues.append("Right knee angle too acute (< 90°) - unstable base")
            confidence -= 0.15
        
        # Check hip angles
        left_hip = features.get('left_hip_angle', 180)
        right_hip = features.get('right_hip_angle', 180)
        
        if left_hip < self.thresholds['hip_angle_min']:
            issues.append("Left hip angle indicates poor posture (< 120°)")
            confidence -= 0.1
        if right_hip < self.thresholds['hip_angle_min']:
            issues.append("Right hip angle indicates poor posture (< 120°)")
            confidence -= 0.1
        
        # Check spine inclination
        spine_inclination = features.get('spine_inclination', 0)
        if spine_inclination > self.thresholds['spine_inclination_max']:
            issues.append(f"Spine deviation from vertical ({spine_inclination:.1f}° > {self.thresholds['spine_inclination_max']}°)")
            confidence -= 0.15
        
        # Check head-foot alignment
        head_foot_alignment = features.get('head_foot_alignment', 0)
        if head_foot_alignment > self.thresholds['head_foot_alignment_max']:
            issues.append("Poor head-to-foot alignment - balance issue")
            confidence -= 0.1
        
        # Check movement smoothness (if available)
        movement_smoothness = features.get('movement_smoothness', 1.0)
        if movement_smoothness < self.thresholds['movement_smoothness_min']:
            issues.append("Jerkiness detected in movement")
            confidence -= 0.1
        
        # Ensure confidence is in [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Form is correct if no issues found
        is_correct = len(issues) == 0
        
        return is_correct, issues, confidence
    
    def evaluate_sequence(self, features_sequence: List[Dict[str, float]]) -> Tuple[bool, List[str], float, Dict[str, List[int]]]:
        """
        Evaluate form across entire sequence
        
        Args:
            features_sequence: List of feature dictionaries for each frame
            
        Returns:
            Tuple of (is_correct, list of issues, confidence_score, problematic_frames)
        """
        all_issues = []
        frame_issues = {}
        confidences = []
        problematic_frames = {
            'elbow': [],
            'knee': [],
            'hip': [],
            'spine': [],
            'alignment': []
        }
        
        for frame_idx, features in enumerate(features_sequence):
            is_correct, issues, confidence = self.evaluate_frame(features)
            confidences.append(confidence)
            
            if issues:
                all_issues.extend(issues)
                frame_issues[frame_idx] = issues
                
                # Track problematic frames by category
                for issue in issues:
                    if 'elbow' in issue.lower():
                        problematic_frames['elbow'].append(frame_idx)
                    elif 'knee' in issue.lower():
                        problematic_frames['knee'].append(frame_idx)
                    elif 'hip' in issue.lower():
                        problematic_frames['hip'].append(frame_idx)
                    elif 'spine' in issue.lower():
                        problematic_frames['spine'].append(frame_idx)
                    elif 'alignment' in issue.lower() or 'balance' in issue.lower():
                        problematic_frames['alignment'].append(frame_idx)
        
        # Aggregate confidence (average across frames)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Overall verdict: correct if > 70% of frames are correct
        correct_frames = sum(1 for c in confidences if c >= 0.7)
        overall_correct = (correct_frames / len(confidences)) >= 0.7 if confidences else False
        
        # Get unique issues
        unique_issues = list(set(all_issues))
        
        return overall_correct, unique_issues, avg_confidence, problematic_frames
    
    def get_joint_feedback(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Get detailed feedback for each joint
        
        Args:
            features: Dictionary of biomechanical features
            
        Returns:
            Dictionary mapping joint names to feedback messages
        """
        feedback = {}
        
        # Elbow feedback
        left_elbow = features.get('left_elbow_angle', 180)
        right_elbow = features.get('right_elbow_angle', 180)
        
        if left_elbow < self.thresholds['elbow_angle_min']:
            feedback['left_elbow'] = f"Over-flexed ({left_elbow:.1f}°)"
        elif left_elbow > 160:
            feedback['left_elbow'] = f"Good extension ({left_elbow:.1f}°)"
        else:
            feedback['left_elbow'] = f"Normal flexion ({left_elbow:.1f}°)"
        
        if right_elbow < self.thresholds['elbow_angle_min']:
            feedback['right_elbow'] = f"Over-flexed ({right_elbow:.1f}°)"
        elif right_elbow > 160:
            feedback['right_elbow'] = f"Good extension ({right_elbow:.1f}°)"
        else:
            feedback['right_elbow'] = f"Normal flexion ({right_elbow:.1f}°)"
        
        # Knee feedback
        left_knee = features.get('left_knee_angle', 180)
        right_knee = features.get('right_knee_angle', 180)
        
        if left_knee < self.thresholds['knee_angle_min']:
            feedback['left_knee'] = f"Unstable angle ({left_knee:.1f}°)"
        else:
            feedback['left_knee'] = f"Stable angle ({left_knee:.1f}°)"
        
        if right_knee < self.thresholds['knee_angle_min']:
            feedback['right_knee'] = f"Unstable angle ({right_knee:.1f}°)"
        else:
            feedback['right_knee'] = f"Stable angle ({right_knee:.1f}°)"
        
        # Hip feedback
        left_hip = features.get('left_hip_angle', 180)
        right_hip = features.get('right_hip_angle', 180)
        
        if left_hip < self.thresholds['hip_angle_min']:
            feedback['left_hip'] = f"Poor posture ({left_hip:.1f}°)"
        else:
            feedback['left_hip'] = f"Good posture ({left_hip:.1f}°)"
        
        if right_hip < self.thresholds['hip_angle_min']:
            feedback['right_hip'] = f"Poor posture ({right_hip:.1f}°)"
        else:
            feedback['right_hip'] = f"Good posture ({right_hip:.1f}°)"
        
        # Spine feedback
        spine_inclination = features.get('spine_inclination', 0)
        if spine_inclination > self.thresholds['spine_inclination_max']:
            feedback['spine'] = f"Deviated ({spine_inclination:.1f}°)"
        else:
            feedback['spine'] = f"Aligned ({spine_inclination:.1f}°)"
        
        # Alignment feedback
        head_foot_alignment = features.get('head_foot_alignment', 0)
        if head_foot_alignment > self.thresholds['head_foot_alignment_max']:
            feedback['alignment'] = "Poor balance"
        else:
            feedback['alignment'] = "Good balance"
        
        return feedback

