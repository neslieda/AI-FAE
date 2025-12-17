import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import glob
import argparse
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator for TensorRT."""
    
    def __init__(self, calibration_data, cache_file, batch_size=1, input_shape=(3, 640, 640)):
        """
        Initialize the calibrator.
        
        Args:
            calibration_data (list): List of image paths for calibration
            cache_file (str): Path to save/load calibration cache
            batch_size (int): Batch size for calibration
            input_shape (tuple): Input shape in CHW format
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = calibration_data
        self.shape = input_shape
        self.current_index = 0
        
        # Allocate device memory for batch
        self.device_input = cuda.mem_alloc(self.batch_size * np.prod(self.shape) * np.dtype(np.float32).itemsize)
        
        # Preprocess calibration data
        self.preprocessed_images = []
        for img_path in tqdm(self.data, desc="Preprocessing calibration images"):
            img = self.preprocess_image(img_path)
            self.preprocessed_images.append(img)
        
        logger.info(f"Calibration data prepared with {len(self.preprocessed_images)} images")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input."""
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Resize and normalize
        img = cv2.resize(img, (self.shape[2], self.shape[1]))
        img = img.transpose(2, 0, 1).astype(np.float32)  # HWC to CHW
        img = img / 255.0  # Normalize to [0, 1]
        
        return img
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names, p_str=None):
        """Get next batch for calibration."""
        if self.current_index + self.batch_size > len(self.preprocessed_images):
            return None
            
        batch = []
        for _ in range(self.batch_size):
            if self.current_index < len(self.preprocessed_images):
                batch.append(self.preprocessed_images[self.current_index])
                self.current_index += 1
        
        if not batch:
            return None
            
        # Stack and copy to device
        batch_data = np.stack(batch, axis=0).astype(np.float32)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch_data))
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """Read calibration cache if exists."""
        if os.path.exists(self.cache_file):
            logger.info(f"Using calibration cache file: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache to file."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        logger.info(f"Calibration cache saved to {self.cache_file}")

def collect_calibration_images(calib_dir, num_images=100):
    """Collect calibration images from directory."""
    calib_dir = Path(calib_dir)
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(calib_dir.glob(f'**/{ext}'))
        image_paths.extend(calib_dir.glob(f'**/{ext.upper()}'))
    
    # Limit number of images
    if len(image_paths) > num_images:
        image_paths = image_paths[:num_images]
    
    if not image_paths:
        raise ValueError(f"No images found in {calib_dir}")
    
    logger.info(f"Found {len(image_paths)} calibration images")
    return [str(p) for p in image_paths]

def parse_args():
    parser = argparse.ArgumentParser(description='INT8 Calibration for TensorRT')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--calib-dir', type=str, required=True, help='Directory containing calibration images')
    parser.add_argument('--output', type=str, default='model_int8.engine', help='Output engine path')
    parser.add_argument('--cache-file', type=str, default='calibration.cache', help='Calibration cache file')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for calibration')
    parser.add_argument('--num-images', type=int, default=100, help='Number of calibration images to use')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Collect calibration images
    try:
        calib_images = collect_calibration_images(args.calib_dir, args.num_images)
    except Exception as e:
        logger.error(f"Error collecting calibration images: {e}")
        return
    
    # Create calibrator
    calibrator = Calibrator(
        calibration_data=calib_images,
        cache_file=args.cache_file,
        batch_size=args.batch_size,
        input_shape=(3, 640, 640)  # Adjust based on your model input shape
    )
    
    # Build TensorRT engine with INT8 calibration
    logger.info("Building INT8 TensorRT engine...")
    
    # Initialize TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace * 1 << 30  # Convert GB to bytes
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    # Parse ONNX model
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(args.onnx, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise ValueError('ONNX parse failed')
    
    # Build and save engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Failed to build TensorRT engine')
    
    with open(args.output, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"Successfully built INT8 TensorRT engine: {args.output}")

if __name__ == '__main__':
    main()
