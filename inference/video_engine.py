import os
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from dataclasses import dataclass
import threading
import queue
from collections import deque

from .detector import Detector
from .tracker import Tracker

@dataclass
class DetectionResult:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    track_id: Optional[int] = None

class VideoEngine:
    """Video processing engine for object detection and tracking."""
    
    def __init__(self,
                 model_path: str,
                 tracker_type: str = 'sort',
                 backend: str = 'onnx',
                 device: str = 'cuda:0',
                 img_size: int = 640,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_det: int = 1000,
                 tracker_max_age: int = 30,
                 tracker_min_hits: int = 3,
                 tracker_iou_threshold: float = 0.3,
                 detect_every_n: int = 1,
                 show_fps: bool = True,
                 show_labels: bool = True,
                 show_conf: bool = True,
                 show_tracks: bool = True):
        """
        Initialize the video engine.
        
        Args:
            model_path: Path to the model file
            tracker_type: Type of tracker to use ('sort', 'deepsort', 'bytetrack')
            backend: Backend to use for detection ('pytorch', 'onnx', 'tensorrt')
            device: Device to run inference on ('cuda:0', 'cpu')
            img_size: Input image size for the model
            conf_thres: Confidence threshold for detections
            iou_thres: IOU threshold for NMS
            max_det: Maximum number of detections per image
            tracker_max_age: Maximum number of frames to keep a track alive without updates
            tracker_min_hits: Minimum number of hits to confirm a track
            tracker_iou_threshold: IOU threshold for matching detections to tracks
            detect_every_n: Run detection every N frames (tracking runs on intermediate frames)
            show_fps: Whether to show FPS on the output
            show_labels: Whether to show class labels on detections
            show_conf: Whether to show confidence scores on detections
            show_tracks: Whether to show tracking IDs
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.detect_every_n = max(1, detect_every_n)
        self.show_fps = show_fps
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.show_tracks = show_tracks
        
        # Initialize detector
        self.detector = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )
        
        # Initialize tracker
        self.tracker = Tracker(
            tracker_type=tracker_type,
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
            iou_threshold=tracker_iou_threshold
        )
        
        # Performance metrics
        self.fps = 0.0
        self.avg_fps = 0.0
        self.frame_count = 0
        self.avg_inference_time = 0.0
        self.avg_tracking_time = 0.0
        self.fps_history = deque(maxlen=100)
        
        # Threading
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detection_thread = None
        self.tracking_thread = None
        
    def process_video(self,
                     source: Union[str, int],
                     output_path: Optional[str] = None,
                     show: bool = True) -> None:
        """
        Process a video file or webcam stream.
        
        Args:
            source: Path to video file or camera index
            output_path: Path to save the output video (optional)
            show: Whether to display the output
        """
        # Open video source
        if isinstance(source, str) and not source.isdigit():
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Warmup
        print("Warming up...")
        self.detector.warmup()
        
        # Start processing
        print("Starting video processing...")
        self.running = True
        
        # Start detection and tracking threads
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.detection_thread.start()
        self.tracking_thread.start()
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame if needed
                if frame.shape[:2] != (self.img_size, self.img_size):
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                
                # Update frame queue for detection thread
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_idx, frame.copy()))
                
                # Get results from tracking thread
                if not self.result_queue.empty():
                    result_frame, detections = self.result_queue.get()
                    
                    # Draw detections and tracks
                    result_frame = self._draw_detections(result_frame, detections)
                    
                    # Calculate and display FPS
                    if self.show_fps:
                        self._display_fps(result_frame)
                    
                    # Write frame to output
                    if writer is not None:
                        writer.write(result_frame)
                    
                    # Show frame
                    if show:
                        cv2.imshow('Video Analytics', result_frame)
                        
                        # Check for exit key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                frame_idx += 1
                
                # Print progress
                if total_frames > 0 and frame_idx % 10 == 0:
                    progress = frame_idx / total_frames * 100
                    print(f"\rProgress: {progress:.1f}% ({frame_idx}/{total_frames})", end='')
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Clean up
            self.running = False
            if self.detection_thread.is_alive():
                self.detection_thread.join()
            if self.tracking_thread.is_alive():
                self.tracking_thread.join()
            
            # Release resources
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            
            # Print final stats
            total_time = time.time() - start_time
            avg_fps = frame_idx / total_time if total_time > 0 else 0
            print(f"\nProcessing complete. Average FPS: {avg_fps:.2f}")
    
    def _detection_worker(self):
        """Worker thread for running detection."""
        while self.running:
            if not self.frame_queue.empty():
                frame_idx, frame = self.frame_queue.get()
                
                # Run detection every N frames
                if frame_idx % self.detect_every_n == 0:
                    # Run detection
                    start_time = time.time()
                    detections = self.detector.detect(frame)
                    inference_time = time.time() - start_time
                    
                    # Update metrics
                    self.avg_inference_time = 0.9 * self.avg_inference_time + 0.1 * inference_time
                    
                    # Pass to tracking
                    self.result_queue.put((frame, detections))
    
    def _tracking_worker(self):
        """Worker thread for running tracking."""
        while self.running:
            if not self.result_queue.empty():
                frame, detections = self.result_queue.get()
                
                # Run tracking
                start_time = time.time()
                tracks = self.tracker.update(detections)
                tracking_time = time.time() - start_time
                
                # Update metrics
                self.avg_tracking_time = 0.9 * self.avg_tracking_time + 0.1 * tracking_time
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count > 10:  # Skip first few frames for warmup
                    current_fps = 1.0 / (self.avg_inference_time + self.avg_tracking_time + 1e-6)
                    self.fps_history.append(current_fps)
                    self.avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Draw results
                result_frame = frame.copy()
                self._draw_tracks(result_frame, tracks)
                
                # Put result back in queue for main thread
                self.result_queue.put((result_frame, tracks))
    
    def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detections on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            conf = det.confidence
            class_id = det.class_id
            track_id = getattr(det, 'track_id', None)
            
            # Draw bounding box
            color = self._get_color(track_id if track_id is not None else class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = []
            if self.show_labels:
                label.append(f'Class: {class_id}')
            if self.show_conf:
                label.append(f'{conf:.2f}')
            if self.show_tracks and track_id is not None:
                label.insert(0, f'ID: {track_id}')
            
            if label:
                label = ' '.join(label)
                self._draw_label(frame, label, (x1, y1 - 10), color)
        
        return frame
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[np.ndarray]) -> None:
        """Draw tracks on the frame."""
        for track in tracks:
            if len(track) < 7:  # [x1, y1, x2, y2, track_id, class_id, conf]
                continue
                
            x1, y1, x2, y2, track_id, class_id, conf = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            color = self._get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = []
            if self.show_labels:
                label.append(f'Class: {int(class_id)}')
            if self.show_conf:
                label.append(f'{conf:.2f}')
            if self.show_tracks:
                label.insert(0, f'ID: {int(track_id)}')
            
            if label:
                label = ' '.join(label)
                self._draw_label(frame, label, (x1, y1 - 10), color)
    
    def _draw_label(self, frame: np.ndarray, label: str, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """Draw a label with background on the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = pos
        cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width + 5, y + 5),
            color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x + 2, y - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )
    
    def _display_fps(self, frame: np.ndarray) -> None:
        """Display FPS on the frame."""
        fps_text = f'FPS: {self.avg_fps:.1f}'
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),  # Red
            2,
            cv2.LINE_AA
        )
    
    @staticmethod
    def _get_color(idx: int) -> Tuple[int, int, int]:
        """Get a consistent color for a given index."""
        # List of distinct colors
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
            (64, 0, 64), (0, 64, 64)
        ]
        return colors[idx % len(colors)]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Edge AI Video Analytics System')
    parser.add_argument('--source', type=str, default='0', help='Video source (file path or camera index)')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--output', type=str, default='', help='Output video path (optional)')
    parser.add_argument('--backend', type=str, default='onnx', choices=['pytorch', 'onnx', 'tensorrt'], 
                       help='Inference backend')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (e.g., cuda:0, cpu)')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort', 'deepsort', 'bytetrack'], 
                       help='Tracker type')
    parser.add_argument('--detect-every', type=int, default=1, 
                       help='Run detection every N frames (tracking runs on intermediate frames)')
    parser.add_argument('--no-show', action='store_true', help='Do not display output')
    
    args = parser.parse_args()
    
    # Initialize video engine
    engine = VideoEngine(
        model_path=args.model,
        tracker_type=args.tracker,
        backend=args.backend,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        detect_every_n=args.detect_every
    )
    
    # Process video
    engine.process_video(
        source=args.source,
        output_path=args.output if args.output else None,
        show=not args.no_show
    )

if __name__ == '__main__':
    main()
