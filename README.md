# Edge AI Video Analytics System

A high-performance computer vision system for real-time object detection and tracking, optimized for edge deployment.

## Features

- 🚀 Multi-backend inference (PyTorch, ONNX Runtime, TensorRT)
- 🎯 Object detection and tracking pipeline
- ⚡ Optimized for edge devices with TensorRT
- 📊 Performance monitoring and metrics
- 🐳 Docker deployment ready
- 🧪 Comprehensive test suite

## Project Structure

```
cv-advanced-assessment/
├── training/          # Model training scripts
├── optimization/      # Model optimization (ONNX, TensorRT, INT8)
├── inference/         # Core detection and tracking logic
├── api/               # FastAPI server and Docker config
├── monitoring/        # Performance monitoring tools
├── tests/             # Unit and integration tests
└── models/            # Model weights and configs
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset in the `data/` directory

## Training

```bash
python training/train.py --data dataset.yaml --weights yolov8n.pt --img 640 --batch 16
```

## Inference

```bash
python inference/video_engine.py --source 0  # webcam
```

## API Server

```bash
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

## Testing

```bash
pytest tests/
```
