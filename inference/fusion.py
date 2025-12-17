"""Detector + Tracker fusion module."""

import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from .detector import Detector
from .tracker import Tracker

class Fusion:
    """Fusion module for combining detector and tracker outputs."""
    
    def __init__(self, 
                 detector: Detector,
                 tracker: Tracker,
                 iou_threshold: float = 0.5,
                 detect_every_n: int = 5,
                 confidence_threshold: float = 0.25):
        """
        Initialize fusion module.
        
        Args:
            detector: Detector instance
            tracker: Tracker instance
            iou_threshold: IoU threshold for drift detection
            detect_every_n: Run detection every N frames
            confidence_threshold: Confidence threshold for detections
        """
        self.detector = detector
        self.tracker = tracker
        self.iou_threshold = iou_threshold
        self.detect_every_n = detect_every_n
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.last_detections = None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame with detector and tracker fusion.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Dictionary containing detections and tracks
        """
        self.frame_count += 1
        
        # Run detector every N frames
        if self.frame_count % self.detect_every_n == 0:
            # Get detections from detector
            detections = self.detector.detect(frame)
            
            # Filter by confidence
            detections = [d for d in detections if len(d) > 4 and d[4] >= self.confidence_threshold]
            
            if len(detections) > 0:
                # Convert to numpy array format [x1, y1, x2, y2, conf, cls]
                det_array = np.array([[d[0], d[1], d[2], d[3], d[4], d[5] if len(d) > 5 else 0] 
                                      for d in detections], dtype=np.float32)
                
                # Update tracker with new detections
                tracks = self.tracker.update(det_array)
                
                # Check for drift and reinitialize if needed
                tracks = self._check_drift(det_array, tracks)
                
                self.last_detections = det_array
            else:
                # No detections, update tracker with empty array
                tracks = self.tracker.update(np.empty((0, 6), dtype=np.float32))
        else:
            # Use tracker only (no detection)
            if self.last_detections is not None:
                # Predict tracks without new detections
                tracks = self.tracker.update(np.empty((0, 6), dtype=np.float32))
            else:
                tracks = []
        
        return {
            'detections': self.last_detections if self.last_detections is not None else np.empty((0, 6)),
            'tracks': tracks,
            'frame_count': self.frame_count
        }
    
    def _check_drift(self, detections: np.ndarray, tracks: List) -> List:
        """
        Check for drift between detections and tracks.
        
        Args:
            detections: Current detections
            tracks: Current tracks
        
        Returns:
            Updated tracks with drift correction
        """
        if len(detections) == 0 or len(tracks) == 0:
            return tracks
        
        # Convert tracks to numpy array
        track_array = np.array([[t[0], t[1], t[2], t[3], t[4], t[5], t[6]] 
                               for t in tracks if len(t) >= 7], dtype=np.float32)
        
        if len(track_array) == 0:
            return tracks
        
        # Calculate IoU between detections and tracks
        iou_matrix = self._calculate_iou_matrix(detections[:, :4], track_array[:, :4])
        
        # Find matches
        matched_tracks = []
        matched_detections = set()
        
        for track_idx, track in enumerate(tracks):
            if len(track) < 7:
                matched_tracks.append(track)
                continue
            
            # Find best matching detection
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in range(len(detections)):
                if det_idx in matched_detections:
                    continue
                
                iou = iou_matrix[det_idx, track_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Check for drift
            if best_iou < self.iou_threshold:
                # Drift detected - reinitialize with detection if available
                if best_det_idx >= 0:
                    det = detections[best_det_idx]
                    # Update track with new detection
                    track = [det[0], det[1], det[2], det[3], track[4], det[5], det[4]]
                    matched_detections.add(best_det_idx)
            
            matched_tracks.append(track)
        
        return matched_tracks
    
    @staticmethod
    def _calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU matrix between two sets of boxes.
        
        Args:
            boxes1: First set of boxes [N, 4] (x1, y1, x2, y2)
            boxes2: Second set of boxes [M, 4] (x1, y1, x2, y2)
        
        Returns:
            IoU matrix [N, M]
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # Calculate intersection
        x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def reset(self):
        """Reset fusion state."""
        self.frame_count = 0
        self.last_detections = None
        self.tracker = Tracker(
            tracker_type=self.tracker.tracker_type,
            max_age=self.tracker.tracker.max_age,
            min_hits=self.tracker.tracker.min_hits,
            iou_threshold=self.tracker.tracker.iou_threshold
        )

