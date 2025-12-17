import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Union
import time

class Track:
    """Track class for tracking individual objects."""
    
    def __init__(self, 
                 track_id: int, 
                 bbox: np.ndarray, 
                 class_id: int, 
                 conf: float, 
                 max_age: int = 30):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique track ID
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID
            conf: Detection confidence
            max_age: Maximum number of frames to keep a track alive without updates
        """
        self.track_id = track_id
        self.bbox = bbox.astype(np.float32)  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.conf = conf
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.max_age = max_age
        self.history = [bbox.astype(np.float32)]
        
    def predict(self) -> None:
        """Predict the next state of the track."""
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox: np.ndarray, conf: float) -> None:
        """Update the track with new detection.
        
        Args:
            bbox: New bounding box [x1, y1, x2, y2]
            conf: Detection confidence
        """
        self.bbox = bbox.astype(np.float32)
        self.conf = conf
        self.history.append(bbox.astype(np.float32))
        if len(self.history) > self.max_age:
            self.history.pop(0)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
    def mark_missed(self) -> None:
        """Mark the track as missed in the current frame."""
        self.hit_streak = 0
        
    def is_confirmed(self) -> bool:
        """Check if the track is confirmed (has enough hits)."""
        return self.hits >= 3
    
    def is_dead(self) -> bool:
        """Check if the track is dead (too old)."""
        return self.time_since_update > self.max_age
    
    def get_state(self) -> np.ndarray:
        """Get the current state of the track."""
        return self.bbox
    
    def get_class_id(self) -> int:
        """Get the class ID of the track."""
        return self.class_id
    
    def get_confidence(self) -> float:
        """Get the confidence of the track."""
        return self.conf


class SORT:
    """SORT (Simple Online and Realtime Tracker) implementation."""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track alive without updates
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IOU threshold for matching detections to tracks
        """
        self.next_id = 1
        self.tracks: List[Track] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
    def update(self, detections: np.ndarray) -> List[Dict]:
        """
        Update the tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            List of tracks with format [x1, y1, x2, y2, track_id, class_id, conf]
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            bbox = detections[det_idx, :4]
            conf = detections[det_idx, 4]
            self.tracks[track_idx].update(bbox, conf)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = detections[det_idx, :4]
            conf = detections[det_idx, 4]
            class_id = int(detections[det_idx, 5])
            self.tracks.append(Track(
                track_id=self.next_id,
                bbox=bbox,
                class_id=class_id,
                conf=conf,
                max_age=self.max_age
            ))
            self.next_id += 1
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        # Return confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = track.get_state()
                track_id = track.track_id
                class_id = track.get_class_id()
                conf = track.get_confidence()
                confirmed_tracks.append([x1, y1, x2, y2, track_id, class_id, conf])
        
        return confirmed_tracks
    
    def _match_detections_to_tracks(self, detections: np.ndarray) -> Tuple[list, list, list]:
        """
        Match detections to existing tracks using IOU.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Tuple of (matched, unmatched_dets, unmatched_tracks)
        """
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate IOU between all tracks and detections
        iou_matrix = self._calculate_iou(detections)
        
        # Match using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T
        
        # Filter out matches with low IOU
        matched = []
        unmatched_dets = []
        unmatched_tracks = []
        
        # Check for unmatched detections
        for d in range(len(detections)):
            if d not in matched_indices[:, 1]:
                unmatched_dets.append(d)
        
        # Check for unmatched tracks
        for t in range(len(self.tracks)):
            if t not in matched_indices[:, 0]:
                unmatched_tracks.append(t)
        
        # Filter out matches with low IOU
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[1])
                unmatched_tracks.append(m[0])
            else:
                matched.append((m[0], m[1]))
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _calculate_iou(self, detections: np.ndarray) -> np.ndarray:
        """
        Calculate IOU between all tracks and detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            IOU matrix of shape (num_tracks, num_detections)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.zeros((0, 0))
        
        # Get track and detection boxes
        track_boxes = np.array([t.get_state() for t in self.tracks])
        det_boxes = detections[:, :4]
        
        # Calculate intersection areas
        x1 = np.maximum(track_boxes[:, None, 0], det_boxes[:, 0])  # [M, N]
        y1 = np.maximum(track_boxes[:, None, 1], det_boxes[:, 1])
        x2 = np.minimum(track_boxes[:, None, 2], det_boxes[:, 2])
        y2 = np.minimum(track_boxes[:, None, 3], det_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union areas
        track_areas = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        det_areas = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        
        union = track_areas[:, None] + det_areas - intersection
        
        # Calculate IOU
        iou = intersection / (union + 1e-6)
        
        # Apply class matching (only match detections with tracks of the same class)
        track_classes = np.array([t.get_class_id() for t in self.tracks])
        det_classes = detections[:, 5].astype(int)
        class_match = track_classes[:, None] == det_classes
        
        # Zero out IOUs for different classes
        iou = iou * class_match.astype(float)
        
        return iou


class Tracker:
    """High-level tracker interface that can be used with different tracking algorithms."""
    
    def __init__(self, 
                 tracker_type: str = 'sort',
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize the tracker.
        
        Args:
            tracker_type: Type of tracker to use ('sort', 'deepsort', 'bytetrack')
            max_age: Maximum number of frames to keep a track alive without updates
            min_hits: Minimum number of hits to confirm a track
            iou_threshold: IOU threshold for matching detections to tracks
        """
        self.tracker_type = tracker_type.lower()
        
        if self.tracker_type == 'sort':
            self.tracker = SORT(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold
            )
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update the tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Array of tracks [x1, y1, x2, y2, track_id, class_id, conf]
        """
        return self.tracker.update(detections)
