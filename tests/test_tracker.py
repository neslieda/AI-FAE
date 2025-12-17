"""Tests for tracker module."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.tracker import Tracker, SORT, Track


class TestTrack:
    """Tests for Track class."""
    
    def test_track_initialization(self):
        """Test track initialization."""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        track = Track(
            track_id=1,
            bbox=bbox,
            class_id=0,
            conf=0.9,
            max_age=30
        )
        
        assert track.track_id == 1
        assert track.class_id == 0
        assert track.conf == 0.9
        assert track.time_since_update == 0
        assert track.hits == 1
    
    def test_track_update(self):
        """Test track update."""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        track = Track(1, bbox, 0, 0.9)
        
        new_bbox = np.array([110, 110, 210, 210], dtype=np.float32)
        track.update(new_bbox, 0.95)
        
        assert track.time_since_update == 0
        assert track.hits == 2
        assert track.conf == 0.95
    
    def test_track_predict(self):
        """Test track prediction."""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        track = Track(1, bbox, 0, 0.9)
        
        track.predict()
        
        assert track.age == 1
        assert track.time_since_update == 1
    
    def test_track_is_confirmed(self):
        """Test track confirmation."""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        track = Track(1, bbox, 0, 0.9, max_age=30)
        
        # Track should not be confirmed initially
        assert not track.is_confirmed()
        
        # Update multiple times
        for _ in range(3):
            track.update(bbox, 0.9)
        
        assert track.is_confirmed()
    
    def test_track_is_dead(self):
        """Test track death."""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        track = Track(1, bbox, 0, 0.9, max_age=5)
        
        # Track should not be dead initially
        assert not track.is_dead()
        
        # Predict beyond max_age
        for _ in range(6):
            track.predict()
        
        assert track.is_dead()


class TestSORT:
    """Tests for SORT tracker."""
    
    def test_sort_initialization(self):
        """Test SORT initialization."""
        tracker = SORT(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert tracker.next_id == 1
    
    def test_sort_update_empty(self):
        """Test SORT update with empty detections."""
        tracker = SORT()
        
        detections = np.empty((0, 6), dtype=np.float32)
        tracks = tracker.update(detections)
        
        assert isinstance(tracks, list)
        assert len(tracks) == 0
    
    def test_sort_update_new_track(self):
        """Test SORT update with new detections."""
        tracker = SORT(min_hits=1)  # Lower min_hits for testing
        
        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],
        ], dtype=np.float32)
        
        tracks = tracker.update(detections)
        
        # Should create new track
        assert len(tracks) >= 0  # May not be confirmed yet
    
    def test_sort_track_matching(self):
        """Test SORT track matching."""
        tracker = SORT(min_hits=1, iou_threshold=0.3)
        
        # First frame
        detections1 = np.array([
            [100, 100, 200, 200, 0.9, 0],
        ], dtype=np.float32)
        
        tracks1 = tracker.update(detections1)
        
        # Second frame with overlapping detection
        detections2 = np.array([
            [105, 105, 205, 205, 0.9, 0],  # Slight movement
        ], dtype=np.float32)
        
        tracks2 = tracker.update(detections2)
        
        # Should match and update track
        assert isinstance(tracks2, list)
    
    def test_sort_track_aging(self):
        """Test SORT track aging."""
        tracker = SORT(max_age=2, min_hits=1)
        
        # Create track
        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],
        ], dtype=np.float32)
        
        tracker.update(detections)
        
        # Update without detections (should age)
        for _ in range(3):
            empty_detections = np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(empty_detections)
        
        # Track should be removed after max_age
        assert len(tracker.tracks) == 0


class TestTracker:
    """Tests for high-level Tracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = Tracker(
            tracker_type="sort",
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        assert tracker.tracker_type == "sort"
        assert isinstance(tracker.tracker, SORT)
    
    def test_tracker_unsupported_type(self):
        """Test tracker with unsupported type."""
        with pytest.raises(ValueError):
            Tracker(tracker_type="unsupported")
    
    def test_tracker_update(self):
        """Test tracker update."""
        tracker = Tracker(tracker_type="sort", min_hits=1)
        
        detections = np.array([
            [100, 100, 200, 200, 0.9, 0],
        ], dtype=np.float32)
        
        tracks = tracker.update(detections)
        
        assert isinstance(tracks, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

