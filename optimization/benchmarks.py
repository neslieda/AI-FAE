import time
import argparse
import json
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from tqdm import tqdm
import logging
import psutil
import GPUtil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseBenchmark:
    """Base class for model benchmarking."""
    
    def __init__(self, warmup=10, num_runs=100):
        """
        Initialize benchmark.
        
        Args:
            warmup (int): Number of warmup runs
            num_runs (int): Number of benchmark runs
        """
        self.warmup = warmup
        self.num_runs = num_runs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize CUDA events for precise timing
        self.start_event = cuda.Event(enable_timing=True)
        self.end_event = cuda.Event(enable_timing=True)
    
    def preprocess(self, input_data):
        """Preprocess input data."""
        return input_data
    
    def get_input_data(self, batch_size=1, shape=(3, 640, 640)):
        """Generate random input data."""
        return np.random.randn(batch_size, *shape).astype(np.float32)
    
    def get_system_info(self):
        """Get system information."""
        gpu_info = {}
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info[f'gpu_{i}'] = {
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'load': gpu.load * 100
            }
        
        return {
            'cpu': {
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available
            },
            'gpu': gpu_info
        }
    
    def run_benchmark(self):
        """Run benchmark and collect metrics."""
        # Warmup
        for _ in range(self.warmup):
            self.run_inference()
        
        # Benchmark
        latencies = []
        gpu_mem_usage = []
        cpu_usage = []
        
        for _ in tqdm(range(self.num_runs), desc="Running benchmark"):
            # Measure CPU usage before inference
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Measure GPU memory before inference
            gpu_before = self.get_gpu_memory()
            
            # Run inference and measure latency
            self.start_event.record()
            output = self.run_inference()
            self.end_event.record()
            
            # Wait for GPU to finish
            self.end_event.synchronize()
            
            # Calculate latency in milliseconds
            latency = cuda.event_elapsed_time(self.start_event, self.end_event)
            
            # Measure CPU and GPU usage after inference
            cpu_after = psutil.cpu_percent(interval=None)
            gpu_after = self.get_gpu_memory()
            
            # Store metrics
            latencies.append(latency)
            cpu_usage.append((cpu_before + cpu_after) / 2)
            gpu_mem_usage.append(max(0, gpu_after - gpu_before))
        
        # Calculate statistics
        latencies = np.array(latencies)
        cpu_usage = np.array(cpu_usage)
        gpu_mem_usage = np.array(gpu_mem_usage)
        
        results = {
            'latency_ms': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p90': float(np.percentile(latencies, 90)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'std': float(np.std(latencies)),
            },
            'throughput_fps': {
                'mean': 1000.0 / float(np.mean(latencies)),
                'max': 1000.0 / float(np.min(latencies))
            },
            'cpu_usage_percent': {
                'mean': float(np.mean(cpu_usage)),
                'max': float(np.max(cpu_usage))
            },
            'gpu_memory_usage_mb': {
                'mean': float(np.mean(gpu_mem_usage)),
                'max': float(np.max(gpu_mem_usage))
            },
            'system': self.get_system_info()
        }
        
        return results
    
    def get_gpu_memory(self):
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    
    def run_inference(self):
        """Run a single inference. To be implemented by subclasses."""
        raise NotImplementedError


class PyTorchBenchmark(BaseBenchmark):
    """Benchmark PyTorch model."""
    
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
    
    def run_inference(self):
        input_tensor = torch.randn(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output


class ONNXBenchmark(BaseBenchmark):
    """Benchmark ONNX model."""
    
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.io_binding = self.session.io_binding()
    
    def run_inference(self):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        return self.session.run([output_name], {input_name: input_data})


class TensorRTBenchmark(BaseBenchmark):
    """Benchmark TensorRT engine."""
    
    def __init__(self, engine_path, **kwargs):
        super().__init__(**kwargs)
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def run_inference(self):
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], np.random.randn(1, 3, 640, 640).astype(np.float32).ravel())
        
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()
        
        return self.outputs[0]['host']

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model file')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['pytorch', 'onnx', 'tensorrt'],
                       help='Type of model to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize appropriate benchmark class
    if args.model_type == 'pytorch':
        benchmark = PyTorchBenchmark(
            model_path=args.model_path,
            warmup=args.warmup,
            num_runs=args.num_runs
        )
    elif args.model_type == 'onnx':
        benchmark = ONNXBenchmark(
            model_path=args.model_path,
            warmup=args.warmup,
            num_runs=args.num_runs
        )
    elif args.model_type == 'tensorrt':
        benchmark = TensorRTBenchmark(
            engine_path=args.model_path,
            warmup=args.warmup,
            num_runs=args.num_runs
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Run benchmark
    logger.info(f"Running {args.model_type.upper()} benchmark...")
    results = benchmark.run_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {args.output}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Model: {args.model_path}")
    print(f"Type: {args.model_type.upper()}")
    print(f"Latency (ms): {results['latency_ms']['mean']:.2f} (mean), {results['latency_ms']['p95']:.2f} (p95)")
    print(f"Throughput (FPS): {results['throughput_fps']['mean']:.2f} (mean), {results['throughput_fps']['max']:.2f} (max)")
    print(f"GPU Memory Usage: {results['gpu_memory_usage_mb']['mean']:.2f} MB (mean), {results['gpu_memory_usage_mb']['max']:.2f} MB (max)")

if __name__ == '__main__':
    main()
