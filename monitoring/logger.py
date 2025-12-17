"""Performance logging module."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import numpy as np

class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, log_dir: str = "logs", window_size: int = 100):
        """
        Initialize performance logger.
        
        Args:
            log_dir: Directory to save logs
            window_size: Window size for moving average
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        self.latency_history = deque(maxlen=window_size)
        self.gpu_memory_history = deque(maxlen=window_size)
        self.gpu_util_history = deque(maxlen=window_size)
        self.cpu_usage_history = deque(maxlen=window_size)
        
        # Setup file logger
        self.logger = logging.getLogger('performance')
        self.logger.setLevel(logging.INFO)
        
        log_file = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Initialize log file with JSON array
        self.log_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.metrics_log = []
    
    def log_inference(self, 
                     latency_ms: float,
                     gpu_memory_mb: Optional[float] = None,
                     gpu_util: Optional[float] = None,
                     cpu_usage: Optional[float] = None):
        """
        Log inference metrics.
        
        Args:
            latency_ms: Inference latency in milliseconds
            gpu_memory_mb: GPU memory usage in MB
            gpu_util: GPU utilization percentage
            cpu_usage: CPU usage percentage
        """
        timestamp = datetime.now().isoformat()
        
        # Update history
        self.latency_history.append(latency_ms)
        if gpu_memory_mb is not None:
            self.gpu_memory_history.append(gpu_memory_mb)
        if gpu_util is not None:
            self.gpu_util_history.append(gpu_util)
        if cpu_usage is not None:
            self.cpu_usage_history.append(cpu_usage)
        
        # Create log entry
        entry = {
            "timestamp": timestamp,
            "latency_ms": latency_ms,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_utilization": gpu_util,
            "cpu_usage": cpu_usage,
            "moving_avg_latency_ms": self.get_moving_avg_latency(),
            "latency_p50": self.get_percentile_latency(50),
            "latency_p90": self.get_percentile_latency(90),
            "latency_p95": self.get_percentile_latency(95),
        }
        
        # Log to file
        self.logger.info(json.dumps(entry))
        self.metrics_log.append(entry)
        
        # Periodically save to JSON file
        if len(self.metrics_log) % 100 == 0:
            self.save_metrics()
    
    def get_moving_avg_latency(self) -> float:
        """Get moving average latency."""
        if len(self.latency_history) == 0:
            return 0.0
        return np.mean(self.latency_history)
    
    def get_percentile_latency(self, percentile: int) -> float:
        """
        Get percentile latency.
        
        Args:
            percentile: Percentile (50, 90, 95, etc.)
        
        Returns:
            Percentile latency in milliseconds
        """
        if len(self.latency_history) == 0:
            return 0.0
        return np.percentile(self.latency_history, percentile)
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "latency_ms": {
                "mean": self.get_moving_avg_latency(),
                "p50": self.get_percentile_latency(50),
                "p90": self.get_percentile_latency(90),
                "p95": self.get_percentile_latency(95),
                "min": float(np.min(self.latency_history)) if len(self.latency_history) > 0 else 0.0,
                "max": float(np.max(self.latency_history)) if len(self.latency_history) > 0 else 0.0,
            },
            "gpu_memory_mb": {
                "mean": float(np.mean(self.gpu_memory_history)) if len(self.gpu_memory_history) > 0 else 0.0,
                "max": float(np.max(self.gpu_memory_history)) if len(self.gpu_memory_history) > 0 else 0.0,
            },
            "gpu_utilization": {
                "mean": float(np.mean(self.gpu_util_history)) if len(self.gpu_util_history) > 0 else 0.0,
                "max": float(np.max(self.gpu_util_history)) if len(self.gpu_util_history) > 0 else 0.0,
            },
            "cpu_usage": {
                "mean": float(np.mean(self.cpu_usage_history)) if len(self.cpu_usage_history) > 0 else 0.0,
                "max": float(np.max(self.cpu_usage_history)) if len(self.cpu_usage_history) > 0 else 0.0,
            },
        }
        
        return stats
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def reset(self):
        """Reset logger state."""
        self.latency_history.clear()
        self.gpu_memory_history.clear()
        self.gpu_util_history.clear()
        self.cpu_usage_history.clear()
        self.metrics_log = []

