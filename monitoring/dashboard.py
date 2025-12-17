"""Performance dashboard module."""

import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import psutil
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from .logger import PerformanceLogger
from .fps_meter import FPSMeter

class Dashboard:
    """Performance dashboard for monitoring system metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize dashboard.
        
        Args:
            log_dir: Directory to save logs
        """
        self.logger = PerformanceLogger(log_dir=log_dir)
        self.fps_meter = FPSMeter()
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_available = False
    
    def update(self, latency_ms: float):
        """
        Update dashboard with new metrics.
        
        Args:
            latency_ms: Inference latency in milliseconds
        """
        # Update FPS meter
        self.fps_meter.update()
        
        # Get GPU metrics
        gpu_memory_mb = None
        gpu_util = None
        if self.gpu_available:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = mem_info.used / (1024 ** 2)
                
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util.gpu
            except:
                pass
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Log metrics
        self.logger.log_inference(
            latency_ms=latency_ms,
            gpu_memory_mb=gpu_memory_mb,
            gpu_util=gpu_util,
            cpu_usage=cpu_usage
        )
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics.
        
        Returns:
            Dictionary with current metrics
        """
        stats = self.logger.get_statistics()
        fps_stats = self.fps_meter.get_stats()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "latency": stats["latency_ms"],
            "fps": fps_stats,
            "gpu": {
                "available": self.gpu_available,
                "memory_mb": stats["gpu_memory_mb"],
                "utilization": stats["gpu_utilization"],
            },
            "cpu": stats["cpu_usage"],
        }
        
        return metrics
    
    def get_summary(self) -> str:
        """
        Get formatted summary string.
        
        Returns:
            Formatted summary string
        """
        metrics = self.get_metrics()
        
        summary = f"""
Performance Dashboard Summary
==============================
Timestamp: {metrics['timestamp']}

Latency (ms):
  Mean: {metrics['latency']['mean']:.2f}
  P50:   {metrics['latency']['p50']:.2f}
  P90:   {metrics['latency']['p90']:.2f}
  P95:   {metrics['latency']['p95']:.2f}

FPS:
  Current: {metrics['fps']['current_fps']:.2f}
  Average: {metrics['fps']['avg_fps']:.2f}
  Min:     {metrics['fps']['min_fps']:.2f}
  Max:     {metrics['fps']['max_fps']:.2f}

GPU:
  Available: {metrics['gpu']['available']}
  Memory:    {metrics['gpu']['memory_mb']['mean']:.2f} MB (avg), {metrics['gpu']['memory_mb']['max']:.2f} MB (max)
  Utilization: {metrics['gpu']['utilization']['mean']:.2f}% (avg), {metrics['gpu']['utilization']['max']:.2f}% (max)

CPU:
  Usage: {metrics['cpu']['mean']:.2f}% (avg), {metrics['cpu']['max']:.2f}% (max)
"""
        return summary
    
    def save_report(self, output_path: Optional[str] = None):
        """
        Save performance report to file.
        
        Args:
            output_path: Path to save report (optional)
        """
        if output_path is None:
            output_path = f"logs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "summary": self.get_summary(),
            "metrics": self.get_metrics(),
            "statistics": self.logger.get_statistics(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def reset(self):
        """Reset dashboard state."""
        self.logger.reset()
        self.fps_meter.reset()

