import os
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2

# Optional torchvision NMS (CPU build may miss compiled ops)
try:
    from torchvision.ops import nms as tv_nms  # type: ignore
except Exception:
    tv_nms = None

# Optional TensorRT / PyCUDA (only needed when backend == 'tensorrt')
try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # type: ignore
except ImportError:
    trt = None
    cuda = None

class Detector:
    """Base detector class for object detection with multiple backend support."""
    
    def __init__(self, 
                 model_path: str, 
                 backend: str = 'onnx',
                 device: str = 'cuda:0',
                 img_size: int = 640,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_det: int = 1000):
        """
        Initialize the detector with the specified backend.
        
        Args:
            model_path: Path to the model file (.pt, .onnx, .engine)
            backend: Backend to use ('pytorch', 'onnx', 'tensorrt')
            device: Device to run inference on ('cuda:0', 'cpu')
            img_size: Input image size (square)
            conf_thres: Confidence threshold
            iou_thres: IOU threshold for NMS
            max_det: Maximum number of detections per image
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.model = None
        self.input_name = None
        self.output_name = None
        self.context = None
        self.bindings = None
        self.stream = None
        
        # Load the model based on backend
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model based on the specified backend."""
        if self.backend == 'pytorch':
            self._load_pytorch_model()
        elif self.backend == 'onnx':
            self._load_onnx_model()
        elif self.backend == 'tensorrt':
            self._load_tensorrt_engine()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_pytorch_model(self) -> None:
        """Load a PyTorch model."""
        if not self.model_path.endswith(('.pt', '.pth')):
            raise ValueError("PyTorch model must be a .pt or .pth file")
            
        # Load the model here (implementation depends on your model architecture)
        # self.model = YourModel()
        # self.model.load_state_dict(torch.load(self.model_path))
        # self.model.to(self.device).eval()
        raise NotImplementedError("PyTorch backend not implemented yet")
    
    def _load_onnx_model(self) -> None:
        """Load an ONNX model."""
        if not self.model_path.endswith('.onnx'):
            raise ValueError("ONNX model must be a .onnx file")
            
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in str(self.device) else ['CPUExecutionProvider']
        self.model = ort.InferenceSession(
            self.model_path, 
            providers=providers,
            provider_options=[{'device_id': int(str(self.device).split(':')[-1])} if 'cuda' in str(self.device) else {}]
        )
        
        # Get input/output names
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
    
    def _load_tensorrt_engine(self) -> None:
        """Load a TensorRT engine."""
        if trt is None or cuda is None:
            raise ImportError("TensorRT/PyCUDA not installed. Install tensorrt and pycuda or use BACKEND=onnx/pytorch.")
        if not self.model_path.endswith(('.engine', '.plan')):
            raise ValueError("TensorRT engine must be a .engine or .plan file")
        
        logger = trt.Logger(trt.Logger.INFO)
        
        # Load engine
        with open(self.model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        self.context = self.model.create_execution_context()
        
        # Allocate buffers
        self.bindings = []
        for binding in self.model:
            size = trt.volume(self.model.get_binding_shape(binding)) * self.model.max_batch_size
            dtype = trt.nptype(self.model.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.model.binding_is_input(binding):
                self.input_name = binding
                input_shape = self.model.get_binding_shape(binding)
                input_dtype = dtype
            else:
                self.output_name = binding
                output_shape = self.model.get_binding_shape(binding)
                output_dtype = dtype
        
        # Create stream
        self.stream = cuda.Stream()
    
    def preprocess(self, 
                  img: np.ndarray, 
                  target_size: int = None) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess the input image for inference.
        
        Args:
            img: Input image (BGR format)
            target_size: Target size for resizing (square)
            
        Returns:
            Tuple of (preprocessed_image, ratio, (new_width, new_height))
        """
        target_size = target_size or self.img_size
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image shape
        h, w = img.shape[:2]
        
        # Calculate padding
        r = min(target_size / h, target_size / w)
        new_w, new_h = int(w * r), int(h * r)
        pad_w = (target_size - new_w) / 2
        pad_h = (target_size - new_h) / 2
        
        # Resize and pad
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        img_padded[int(pad_h):int(pad_h + new_h), int(pad_w):int(pad_w + new_w)] = img
        
        # Normalize
        img_norm = img_padded.astype(np.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1)  # HWC to CHW
        img_norm = np.ascontiguousarray(img_norm)
        
        return img_norm, r, (new_w, new_h)
    
    def postprocess(self, 
                   pred: np.ndarray, 
                   ratio: float, 
                   orig_shape: Tuple[int, int],
                   conf_thres: float = None,
                   iou_thres: float = None,
                   max_det: int = None) -> np.ndarray:
        """
        Postprocess the model output.
        
        Args:
            pred: Raw model output
            ratio: Ratio used in preprocessing
            orig_shape: Original image shape (h, w)
            conf_thres: Confidence threshold
            iou_thres: IOU threshold for NMS
            max_det: Maximum number of detections
            
        Returns:
            Array of detections [x1, y1, x2, y2, conf, cls]
        """
        conf_thres = conf_thres or self.conf_thres
        iou_thres = iou_thres or self.iou_thres
        max_det = max_det or self.max_det
        
        # Process predictions
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        
        # Rescale boxes from img_size to original image size
        det = pred[0]  # Get detections for first image in batch
        if len(det):
            # Rescale boxes from img_size to original image size
            det[:, :4] = scale_coords((self.img_size, self.img_size), det[:, :4], orig_shape).round()
        
        return det.numpy()
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Array of detections [x1, y1, x2, y2, conf, cls]
        """
        # Preprocess
        img_preprocessed, ratio, (new_w, new_h) = self.preprocess(img)
        img_tensor = torch.from_numpy(img_preprocessed).to(self.device).unsqueeze(0)
        orig_shape = img.shape[:2]  # (h, w)
        
        # Run inference
        if self.backend == 'pytorch':
            with torch.no_grad():
                pred = self.model(img_tensor)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]  # Get the first output if multiple outputs
        
        elif self.backend == 'onnx':
            pred = self.model.run([self.output_name], {self.input_name: img_preprocessed[None]})[0]
            
        elif self.backend == 'tensorrt':
            # Get output shape from engine
            output_shape = self.model.get_binding_shape(self.output_name)
            output_size = trt.volume(output_shape) * self.model.max_batch_size
            output_dtype = trt.nptype(self.model.get_binding_dtype(self.output_name))
            
            # Allocate host and device buffers
            d_input = cuda.mem_alloc(img_tensor.nbytes)
            host_output = cuda.pagelocked_empty(output_size, output_dtype)
            d_output = cuda.mem_alloc(host_output.nbytes)
            
            # Transfer input data to device
            cuda.memcpy_htod_async(d_input, img_tensor.numpy(), self.stream)
            
            # Execute model
            self.context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=self.stream.handle)
            
            # Transfer predictions back from device
            cuda.memcpy_dtoh_async(host_output, d_output, self.stream)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            # Reshape output
            pred = host_output.reshape(output_shape)
        
        # Postprocess
        detections = self.postprocess(pred, ratio, orig_shape)
        
        return detections
    
    def warmup(self, batch_size: int = 1, img_size: int = 640) -> None:
        """
        Warmup the model with dummy data.
        
        Args:
            batch_size: Batch size for warmup
            img_size: Image size for warmup
        """
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        for _ in range(10):  # Warmup iterations
            self.detect(img)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=1000):
    """Non-Maximum Suppression (NMS) on inference results."""
    # Implementation of NMS
    # This is a simplified version, you may need to adapt it to your specific model output
    
    # Filter out confidence scores below threshold
    x = prediction[prediction[..., 4] > conf_thres]
    
    # If no boxes remain, return empty array
    if not x.shape[0]:
        return torch.zeros((0, 6))
    
    # Compute conf
    x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf
    
    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[..., :4])
    
    # Detections matrix nx6 (xyxy, conf, cls)
    conf, j = x[..., 5:].max(1, keepdim=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    
    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return torch.zeros((0, 6))
    
    # Batched NMS
    c = x[:, 5:6] * 0  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

    if tv_nms is not None:
        i = tv_nms(boxes, scores, iou_thres)
    else:
        # Fallback NMS implementation (CPU-only, slower)
        # Sort by score descending
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            idx = order[0]
            keep.append(idx)
            if order.numel() == 1:
                break
            ious = box_iou(boxes[idx].unsqueeze(0), boxes[order[1:]])[0]
            mask = ious <= iou_thres
            order = order[1:][mask]
        i = torch.tensor(keep, device=boxes.device, dtype=torch.long)

    i = i[:max_det]  # limit detections
    
    return x[i]

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    
    return coords
