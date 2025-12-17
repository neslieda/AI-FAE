import os
import sys
import cv2
import torch
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import zipfile
import requests
import json

# Configuration
CONFIG = {
    'data_dir': 'data',
    'models_dir': 'models',
    'calib_dir': 'data/calibration/calibration_images',
    'num_calib_images': 100,
    'batch_size': 8,
    'img_size': [640, 640],
    'onnx_opset': 12,
    'trt_workspace': 4  # GB
}

class SetupAndCalibrate:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config['data_dir']).mkdir(exist_ok=True)
        Path(self.config['models_dir']).mkdir(exist_ok=True)
        Path(self.config['calib_dir']).parent.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, destination):
        """Download a file with progress bar."""
        if Path(destination).exists():
            return True
            
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=Path(destination).name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        return True
    
    def prepare_calibration_dataset(self):
        """Download and prepare calibration dataset."""
        calib_dir = Path(self.config['calib_dir'])
        
        # Skip if we already have enough images
        existing_images = list(calib_dir.glob("*.jpg"))
        if len(existing_images) >= self.config['num_calib_images']:
            print(f"Found {len(existing_images)} existing calibration images")
            return str(calib_dir)
        
        # Download COCO validation set
        url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_path = Path(self.config['data_dir']) / "val2017.zip"
        
        if not zip_path.exists():
            self.download_file(url, zip_path)
        
        # Extract the dataset
        extract_dir = Path(self.config['data_dir']) / "val2017"
        if not extract_dir.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config['data_dir'])
        
        # Process and save calibration images
        calib_dir.mkdir(exist_ok=True)
        image_files = list(extract_dir.glob("*.jpg"))[:self.config['num_calib_images']]
        
        print(f"Preprocessing {len(image_files)} images for calibration...")
        for img_path in tqdm(image_files, desc="Processing images"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Resize to model input size
            img = cv2.resize(img, tuple(self.config['img_size']))
            output_path = calib_dir / img_path.name
            cv2.imwrite(str(output_path), img)
        
        print(f"\nCalibration dataset prepared at: {calib_dir}")
        return str(calib_dir)
    
    def export_to_onnx(self, weights_path, output_path=None):
        """Export PyTorch model to ONNX format."""
        if output_path is None:
            output_path = Path(weights_path).with_suffix('.onnx')
        
        print(f"Exporting {weights_path} to ONNX...")
        
        # This is a simplified version - in practice, you'd use your model's export function
        try:
            import torch
            from models.yolo import Model  # Adjust import based on your model
            
            # Load model
            model = Model(weights_path)
            model.eval()
            
            # Dummy input
            dummy_input = torch.randn(1, 3, *self.config['img_size'])
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.config['onnx_opset'],
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output0'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output0': {0: 'batch_size'}
                }
            )
            print(f"Model exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error exporting to ONNX: {e}")
            print("Please export your model to ONNX manually and provide the path.")
            return None
    
    def build_trt_engine(self, onnx_path, output_path=None, precision='int8'):
        """Build TensorRT engine from ONNX model."""
        if output_path is None:
            output_path = Path(onnx_path).with_suffix(f'.{precision}.engine')
        
        print(f"Building {precision.upper()} TensorRT engine...")
        
        try:
            cmd = [
                'python', 'optimization/build_trt_engine.py',
                '--onnx', str(onnx_path),
                '--output', str(output_path),
                '--precision', precision,
                '--workspace', str(self.config['trt_workspace']),
                '--batch-size', str(self.config['batch_size'])
            ]
            
            if precision == 'int8':
                cmd.extend(['--calib-dir', self.config['calib_dir']])
            
            subprocess.run(cmd, check=True)
            print(f"TensorRT engine saved to {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            print(f"Error building TensorRT engine: {e}")
            return None
    
    def run_benchmark(self, model_path, model_type):
        """Run benchmark on the model."""
        print(f"\nBenchmarking {model_type.upper()} model...")
        
        try:
            output_file = f"benchmarks/{Path(model_path).stem}_benchmark.json"
            Path('benchmarks').mkdir(exist_ok=True)
            
            cmd = [
                'python', 'optimization/benchmarks.py',
                '--model-path', str(model_path),
                '--model-type', model_type,
                '--output', output_file
            ]
            
            subprocess.run(cmd, check=True)
            
            # Print benchmark results
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                
                print("\nBenchmark Results:")
                print(f"Model: {Path(model_path).name}")
                print(f"Type: {model_type.upper()}")
                print(f"Latency (ms): {results['latency_ms']['mean']:.2f} (mean), {results['latency_ms']['p95']:.2f} (p95)")
                print(f"Throughput (FPS): {results['throughput_fps']['mean']:.2f} (mean), {results['throughput_fps']['max']:.2f} (max)")
                print(f"GPU Memory Usage: {results['gpu_memory_usage_mb']['mean']:.2f} MB (mean), {results['gpu_memory_usage_mb']['max']:.2f} MB (max)")
            
        except Exception as e:
            print(f"Error running benchmark: {e}")
    
    def run(self, weights_path):
        """Run the complete setup and calibration pipeline."""
        print("="*50)
        print("Starting Model Optimization Pipeline")
        print("="*50)
        
        # 1. Prepare calibration dataset
        print("\n[1/4] Preparing calibration dataset...")
        self.prepare_calibration_dataset()
        
        # 2. Export to ONNX
        print("\n[2/4] Exporting model to ONNX...")
        onnx_path = self.export_to_onnx(weights_path)
        if not onnx_path or not Path(onnx_path).exists():
            print("ONNX export failed. Please provide the path to your ONNX model:")
            onnx_path = input("Path to ONNX model: ").strip('"')
        
        # 3. Build FP16 TensorRT engine
        print("\n[3/4] Building FP16 TensorRT engine...")
        fp16_engine_path = self.build_trt_engine(onnx_path, precision='fp16')
        
        # 4. Build INT8 TensorRT engine with calibration
        print("\n[4/4] Building INT8 TensorRT engine with calibration...")
        int8_engine_path = self.build_trt_engine(onnx_path, precision='int8')
        
        # 5. Run benchmarks
        print("\nRunning benchmarks...")
        if fp16_engine_path and Path(fp16_engine_path).exists():
            self.run_benchmark(fp16_engine_path, 'tensorrt')
        
        if int8_engine_path and Path(int8_engine_path).exists():
            self.run_benchmark(int8_engine_path, 'tensorrt')
        
        print("\n" + "="*50)
        print("Optimization Complete!")
        print("="*50)
        print("\nOutput files:")
        if 'onnx_path' in locals() and onnx_path:
            print(f"- ONNX Model: {onnx_path}")
        if 'fp16_engine_path' in locals() and fp16_engine_path:
            print(f"- FP16 TensorRT Engine: {fp16_engine_path}")
        if 'int8_engine_path' in locals() and int8_engine_path:
            print(f"- INT8 TensorRT Engine: {int8_engine_path}")
        print("\nYou can now use the optimized models for inference.")

def main():
    parser = argparse.ArgumentParser(description='Setup and run model optimization pipeline')
    parser.add_argument('--weights', type=str, required=True, help='Path to PyTorch model weights (.pt file)')
    parser.add_argument('--num-calib', type=int, default=100, help='Number of calibration images to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for calibration')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG.update({
        'num_calib_images': args.num_calib,
        'batch_size': args.batch_size
    })
    
    # Run the pipeline
    pipeline = SetupAndCalibrate(CONFIG)
    pipeline.run(args.weights)

if __name__ == "__main__":
    main()
