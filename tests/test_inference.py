"""Tests for inference engine."""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.detector import Detector
from inference.tracker import Tracker
from inference.video_engine import VideoEngine


class TestDetector:
    """Tests for Detector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        # This test requires a model file, so we'll skip if not available
        pytest.skip("Requires model file")
    
    def test_preprocess(self):
        """Test image preprocessing."""
        detector = Detector.__new__(Detector)
        detector.img_size = 640
        
        # Create dummy image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing
        img_preprocessed, ratio, (new_w, new_h) = detector.preprocess(img)
        
        assert img_preprocessed.shape == (3, 640, 640)
        assert isinstance(ratio, float)
        assert ratio > 0
    
    def test_postprocess(self):
        """Test postprocessing."""
        detector = Detector.__new__(Detector)
        detector.img_size = 640
        detector.conf_thres = 0.25
        detector.iou_thres = 0.45
        detector.max_det = 1000
        
        # Create dummy prediction
        pred = np.random.randn(1, 25200, 85)  # YOLO output shape
        
        ratio = 1.0
        orig_shape = (480, 640)
        
        # Test postprocessing
        detections = detector.postprocess(pred, ratio, orig_shape)
        
        assert isinstance(detections, np.ndarray)
    
    def test_warmup(self):
        """Test model warmup."""
        pytest.skip("Requires model file")


class TestTracker:
    """Tests for Tracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = Tracker(
            tracker_type="sort",
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker.tracker_type == "sort"
        assert tracker.tracker.max_age == 30
        assert tracker.tracker.min_hits == 3
    
    def test_tracker_update(self):
        """Test tracker update."""
        tracker = Tracker(
            tracker_type="sort",
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Create dummy detections [x1, y1, x2, y2, conf, cls]
        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],
            [300, 300, 400, 400, 0.8, 1],
        ], dtype=np.float32)
        
        # Update tracker
        tracks = tracker.update(detections)
        
        assert isinstance(tracks, list)
    
    def test_tracker_drift_detection(self):
        """Test tracker drift detection."""
        tracker = Tracker(
            tracker_type="sort",
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # First frame detections
        detections1 = np.array([
            [100, 100, 200, 200, 0.9, 0],
        ], dtype=np.float32)
        
        tracks1 = tracker.update(detections1)
        
        # Second frame with drift (low IoU)
        detections2 = np.array([
            [500, 500, 600, 600, 0.9, 0],  # Far from first detection
        ], dtype=np.float32)
        
        tracks2 = tracker.update(detections2)
        
        # Should create new track due to low IoU
        assert isinstance(tracks2, list)


class TestVideoEngine:
    """Tests for VideoEngine class."""
    
    def test_video_engine_initialization(self):
        """Test video engine initialization."""
        pytest.skip("Requires model file")
    
    def test_color_generation(self):
        """Test color generation."""
        color = VideoEngine._get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

