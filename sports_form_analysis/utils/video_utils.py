"""
Video Utility Functions
Helper functions for video processing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return width, height, fps, frame_count


def save_video(frames: list, output_path: str, fps: float = 30.0):
    """
    Save list of frames as video
    
    Args:
        frames: List of frame arrays
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a specific frame from video
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract (0-indexed)
        
    Returns:
        Frame array or None if frame doesn't exist
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None


def resize_video(video_path: str, max_width: int = 640, max_height: int = 480) -> str:
    """
    Resize video to fit within max dimensions (maintains aspect ratio)
    
    Args:
        video_path: Path to input video
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Path to resized video (saves to temporary location)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Create output path
    import os
    output_path = video_path.replace('.mp4', '_resized.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)
    
    cap.release()
    out.release()
    
    return output_path

