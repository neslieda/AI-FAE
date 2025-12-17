from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import io
import time
import json
import os
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize FastAPI app
app = FastAPI(
    title="Edge AI Video Analytics API",
    description="REST API for real-time video analytics with object detection and tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: Optional[str] = None
    track_id: Optional[int] = None

class VideoProcessResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    num_detections: int
    width: int
    height: int
    fps: float

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    memory_usage: Optional[dict] = None

# Global variables
model = None
tracker = None
executor = ThreadPoolExecutor()

# Initialize model and tracker
def init_model():
    global model, tracker
    try:
        from inference.detector import Detector
        from inference.tracker import Tracker
        
        model_path = os.getenv("MODEL_PATH", "models/yolov8n.pt")
        backend = os.getenv("BACKEND", "onnx")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing model from {model_path} on {device} with {backend} backend")
        model = Detector(
            model_path=model_path,
            backend=backend,
            device=device,
            img_size=640
        )
        
        logger.info("Initializing tracker")
        tracker = Tracker(
            tracker_type="sort",
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Warm up
        logger.info("Warming up model...")
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        model.detect(dummy_input)
        
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    try:
        gpu_available = torch.cuda.is_available()
        health_data = {
            "status": "healthy" if model is not None else "unhealthy",
            "model_loaded": model is not None,
            "gpu_available": gpu_available,
        }
        
        if gpu_available:
            try:
                health_data["gpu_name"] = torch.cuda.get_device_name(0)
                health_data["memory_usage"] = {
                    "total": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                    "reserved": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                    "allocated": round(torch.cuda.memory_allocated(0) / 1e9, 2)
                }
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")
                health_data["gpu_name"] = None
                health_data["memory_usage"] = None
        
        return health_data
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "error",
            "model_loaded": False,
            "gpu_available": False,
            "gpu_name": None,
            "memory_usage": None
        }

# Process video endpoint (GET - serves the upload form)
@app.get("/process", response_class=HTMLResponse)
async def process_video_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Process video endpoint
@app.post("/process", response_model=VideoProcessResponse)
async def process_video(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    tracker_enabled: bool = True
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Read video file
        video_bytes = await file.read()
        video_np = np.frombuffer(video_bytes, np.uint8)
        
        # Process video
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            process_video_frame,
            video_np,
            confidence,
            iou_threshold,
            tracker_enabled
        )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": "Video processed successfully",
            "processing_time": processing_time,
            "num_detections": len(result["detections"]),
            "width": result["width"],
            "height": result["height"],
            "fps": result.get("fps", 0)
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Process video endpoint (POST - handles file upload)
@app.post("/process/upload", response_class=JSONResponse)
async def process_video_upload(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Only video files (mp4, avi, mov) are allowed")

        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the video (this is a placeholder - replace with your actual processing logic)
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Release the video capture object
        cap.release()

        return {
            "status": "success",
            "filename": file.filename,
            "file_size": os.path.getsize(file_path),
            "video_properties": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": frame_count / fps if fps > 0 else 0
            },
            "message": "Video processing started. This is a placeholder response."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Process video frames
def process_video_frame(video_np, confidence, iou_threshold, tracker_enabled):
    # Convert bytes to numpy array
    frame = cv2.imdecode(np.frombuffer(video_np, np.uint8), cv2.IMREAD_COLOR)
    
    if frame is None:
        raise ValueError("Could not decode video frame")
    
    # Process frame
    detections = model.detect(frame)
    
    # Apply confidence threshold
    detections = [d for d in detections if d[4] >= confidence]
    
    # Convert to list of DetectionResult
    result = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        result.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "class_id": int(cls_id),
            "class_name": "object",  # Replace with actual class names
            "track_id": None
        })
    
    return {
        "detections": result,
        "width": frame.shape[1],
        "height": frame.shape[0]
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    # Add your metrics collection logic here
    return {
        "inference_time": 0.0,  # Add actual metrics
        "fps": 0.0,
        "gpu_usage": 0.0,
        "memory_usage": 0.0
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API server...")
    if not init_model():
        logger.error("Failed to initialize model. Check logs for details.")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
