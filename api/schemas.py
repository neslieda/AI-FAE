"""API request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BBox(BaseModel):
    """Bounding box schema."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

class Detection(BaseModel):
    """Detection result schema."""
    bbox: BBox = Field(..., description="Bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class ID")
    class_name: Optional[str] = Field(None, description="Class name")
    track_id: Optional[int] = Field(None, description="Track ID if tracking enabled")

class DetectionRequest(BaseModel):
    """Detection request schema."""
    image: Optional[str] = Field(None, description="Base64 encoded image")
    confidence: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    tracker_enabled: bool = Field(True, description="Enable tracking")

class DetectionResponse(BaseModel):
    """Detection response schema."""
    detections: List[Detection] = Field(..., description="List of detections")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    num_detections: int = Field(..., description="Number of detections")
    image_shape: Dict[str, int] = Field(..., description="Image shape (width, height)")

class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU name")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="GPU memory usage in GB")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")

class Metrics(BaseModel):
    """Performance metrics schema."""
    latency_ms: Dict[str, float] = Field(..., description="Latency statistics (mean, p50, p95)")
    throughput_fps: float = Field(..., description="Throughput in FPS")
    gpu_utilization: float = Field(..., ge=0.0, le=100.0, description="GPU utilization percentage")
    gpu_memory_mb: float = Field(..., description="GPU memory usage in MB")
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")

class VideoProcessRequest(BaseModel):
    """Video processing request schema."""
    confidence: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    tracker_enabled: bool = Field(True, description="Enable tracking")
    output_format: str = Field("json", description="Output format (json, video)")

class VideoProcessResponse(BaseModel):
    """Video processing response schema."""
    success: bool = Field(..., description="Whether processing succeeded")
    message: str = Field(..., description="Status message")
    processing_time: float = Field(..., description="Total processing time in seconds")
    num_detections: int = Field(..., description="Total number of detections")
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    fps: float = Field(..., description="Video FPS")
    output_path: Optional[str] = Field(None, description="Path to output video if saved")

