"""FPS (Frames Per Second) meter module."""

import time
from collections import deque
from typing import Optional
import numpy as np

class FPSMeter:
    """FPS meter for measuring frame rate."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS meter.
        
        Args:
            window_size: Window size for moving average
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self):
        """Update FPS meter with new frame."""
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
        
        self.last_time = current_time
        self.frame_count += 1
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current FPS
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_avg_fps(self) -> float:
        """
        Get average FPS since start.
        
        Returns:
            Average FPS
        """
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def get_min_fps(self) -> float:
        """
        Get minimum FPS in window.
        
        Returns:
            Minimum FPS
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        max_frame_time = np.max(self.frame_times)
        return 1.0 / max_frame_time if max_frame_time > 0 else 0.0
    
    def get_max_fps(self) -> float:
        """
        Get maximum FPS in window.
        
        Returns:
            Maximum FPS
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        min_frame_time = np.min(self.frame_times)
        return 1.0 / min_frame_time if min_frame_time > 0 else 0.0
    
    def get_stats(self) -> dict:
        """
        Get FPS statistics.
        
        Returns:
            Dictionary with FPS statistics
        """
        return {
            "current_fps": self.get_fps(),
            "avg_fps": self.get_avg_fps(),
            "min_fps": self.get_min_fps(),
            "max_fps": self.get_max_fps(),
            "frame_count": self.frame_count,
            "window_size": len(self.frame_times),
        }
    
    def reset(self):
        """Reset FPS meter."""
        self.frame_times.clear()
        self.last_time = None
        self.frame_count = 0
        self.start_time = time.time()

