"""
Visualization Utilities
Functions for visualizing pose, features, and results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


def draw_keypoints_on_frame(frame: np.ndarray, keypoints: np.ndarray, 
                           problematic_joints: Optional[List[int]] = None) -> np.ndarray:
    """
    Draw keypoints on frame with highlighting for problematic joints
    
    Args:
        frame: Input frame
        keypoints: Keypoints array (33, 3)
        problematic_joints: List of joint indices to highlight in red
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Key joint indices to draw
    important_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    # Draw connections
    connections = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        (0, 11), (0, 12),  # Head to shoulders
    ]
    
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = (int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h))
            pt2 = (int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h))
            
            # Check if either joint is problematic
            is_problematic = (problematic_joints and 
                            (start_idx in problematic_joints or end_idx in problematic_joints))
            
            color = (0, 0, 255) if is_problematic else (0, 255, 0)
            thickness = 3 if is_problematic else 2
            
            cv2.line(annotated, pt1, pt2, color, thickness)
    
    # Draw joints
    for idx in important_joints:
        if idx < len(keypoints):
            x = int(keypoints[idx][0] * w)
            y = int(keypoints[idx][1] * h)
            
            is_problematic = problematic_joints and idx in problematic_joints
            color = (0, 0, 255) if is_problematic else (0, 255, 0)
            radius = 5 if is_problematic else 3
            
            cv2.circle(annotated, (x, y), radius, color, -1)
    
    return annotated


def create_analysis_report(features_sequence: List[Dict], 
                          rule_verdict: bool,
                          rule_confidence: float,
                          rule_issues: List[str],
                          ml_verdict: bool,
                          ml_confidence: float,
                          problematic_frames: Dict[str, List[int]],
                          output_path: str):
    """
    Create a text report of the analysis
    
    Args:
        features_sequence: List of feature dictionaries
        rule_verdict: Rule-based verdict (True=correct, False=incorrect)
        rule_confidence: Rule-based confidence score
        rule_issues: List of issues found
        ml_verdict: ML-based verdict
        ml_confidence: ML-based confidence score
        problematic_frames: Dictionary mapping issue types to frame indices
        output_path: Path to save report
    """
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("SPORTS FORM ANALYSIS REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Overall verdict
    report_lines.append("OVERALL VERDICT")
    report_lines.append("-"*60)
    rule_status = "CORRECT" if rule_verdict else "INCORRECT"
    ml_status = "CORRECT" if ml_verdict else "INCORRECT"
    
    report_lines.append(f"Rule-Based Evaluation: {rule_status}")
    report_lines.append(f"  Confidence: {rule_confidence:.2%}")
    report_lines.append("")
    report_lines.append(f"ML-Based Evaluation: {ml_status}")
    report_lines.append(f"  Confidence: {ml_confidence:.2%}")
    report_lines.append("")
    
    # Issues
    if rule_issues:
        report_lines.append("IDENTIFIED ISSUES")
        report_lines.append("-"*60)
        for issue in rule_issues:
            report_lines.append(f"  • {issue}")
        report_lines.append("")
    
    # Problematic frames
    if problematic_frames:
        report_lines.append("PROBLEMATIC FRAMES BY CATEGORY")
        report_lines.append("-"*60)
        for category, frames in problematic_frames.items():
            if frames:
                unique_frames = sorted(set(frames))
                report_lines.append(f"  {category.capitalize()}: Frames {unique_frames[:10]}")
                if len(unique_frames) > 10:
                    report_lines.append(f"    ... and {len(unique_frames) - 10} more")
        report_lines.append("")
    
    # Feature statistics
    if features_sequence:
        report_lines.append("BIOMETRIC FEATURE STATISTICS")
        report_lines.append("-"*60)
        
        # Aggregate features
        all_left_elbow = [f.get('left_elbow_angle', 0) for f in features_sequence]
        all_right_elbow = [f.get('right_elbow_angle', 0) for f in features_sequence]
        all_left_knee = [f.get('left_knee_angle', 0) for f in features_sequence]
        all_right_knee = [f.get('right_knee_angle', 0) for f in features_sequence]
        
        report_lines.append(f"Left Elbow Angle:  Mean={np.mean(all_left_elbow):.1f}°, "
                           f"Min={np.min(all_left_elbow):.1f}°, Max={np.max(all_left_elbow):.1f}°")
        report_lines.append(f"Right Elbow Angle: Mean={np.mean(all_right_elbow):.1f}°, "
                           f"Min={np.min(all_right_elbow):.1f}°, Max={np.max(all_right_elbow):.1f}°")
        report_lines.append(f"Left Knee Angle:   Mean={np.mean(all_left_knee):.1f}°, "
                           f"Min={np.min(all_left_knee):.1f}°, Max={np.max(all_left_knee):.1f}°")
        report_lines.append(f"Right Knee Angle:  Mean={np.mean(all_right_knee):.1f}°, "
                           f"Min={np.min(all_right_knee):.1f}°, Max={np.max(all_right_knee):.1f}°")
        report_lines.append("")
    
    report_lines.append("="*60)
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {output_path}")


def plot_feature_timeline(features_sequence: List[Dict], output_path: Optional[str] = None):
    """
    Plot feature values over time
    
    Args:
        features_sequence: List of feature dictionaries
        output_path: Optional path to save plot
    """
    if not features_sequence:
        return
    
    frames = range(len(features_sequence))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biometric Features Over Time', fontsize=14)
    
    # Elbow angles
    left_elbow = [f.get('left_elbow_angle', 0) for f in features_sequence]
    right_elbow = [f.get('right_elbow_angle', 0) for f in features_sequence]
    axes[0, 0].plot(frames, left_elbow, label='Left', color='blue')
    axes[0, 0].plot(frames, right_elbow, label='Right', color='red')
    axes[0, 0].set_title('Elbow Angles')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Knee angles
    left_knee = [f.get('left_knee_angle', 0) for f in features_sequence]
    right_knee = [f.get('right_knee_angle', 0) for f in features_sequence]
    axes[0, 1].plot(frames, left_knee, label='Left', color='blue')
    axes[0, 1].plot(frames, right_knee, label='Right', color='red')
    axes[0, 1].set_title('Knee Angles')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spine inclination
    spine = [f.get('spine_inclination', 0) for f in features_sequence]
    axes[1, 0].plot(frames, spine, color='green')
    axes[1, 0].set_title('Spine Inclination')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Head-foot alignment
    alignment = [f.get('head_foot_alignment', 0) for f in features_sequence]
    axes[1, 1].plot(frames, alignment, color='purple')
    axes[1, 1].set_title('Head-Foot Alignment')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Offset (normalized)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

