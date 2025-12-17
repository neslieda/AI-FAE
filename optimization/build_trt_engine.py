import os
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_engine(onnx_path, engine_path, precision='fp16', workspace=4, batch_size=1, dynamic_shape=None):
    """
    Build a TensorRT engine from an ONNX model.
    
    Args:
        onnx_path (str): Path to the ONNX model
        engine_path (str): Path to save the TensorRT engine
        precision (str): Precision mode ('fp32', 'fp16', 'int8')
        workspace (int): Maximum workspace size in GB
        batch_size (int): Batch size for static shapes
        dynamic_shape (dict): Dictionary containing dynamic shape information
    """
    logger.info(f"Building TensorRT engine from {onnx_path}")
    logger.info(f"Precision: {precision}, Workspace: {workspace}GB")
    
    # Initialize TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30  # Convert GB to bytes
    
    # Set precision flags
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: Add INT8 calibration
        
    # Parse the ONNX model
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise ValueError('ONNX parse failed')
    
    # Set optimization profiles for dynamic shapes
    if dynamic_shape:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        
        # Set min/opt/max shapes
        min_shape = dynamic_shape.get('min', (1, 3, 320, 320))
        opt_shape = dynamic_shape.get('opt', (batch_size, 3, 640, 640))
        max_shape = dynamic_shape.get('max', (batch_size, 3, 1024, 1024))
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        logger.info(f"Dynamic shapes - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")
    
    # Build engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Failed to build TensorRT engine')
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"Successfully built TensorRT engine: {engine_path}")
    return engine_path

def parse_args():
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX model')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='', help='Output engine path')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], 
                       help='Precision mode (fp32, fp16, int8)')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for static shapes')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic shapes')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set output path if not provided
    if not args.output:
        output_path = Path(args.onnx).with_suffix(f'.{args.precision}.engine')
    else:
        output_path = args.output
    
    # Set dynamic shapes if enabled
    dynamic_shape = None
    if args.dynamic:
        dynamic_shape = {
            'min': (1, 3, 320, 320),
            'opt': (args.batch_size, 3, 640, 640),
            'max': (args.batch_size, 3, 1024, 1024)
        }
    
    # Build the engine
    try:
        build_engine(
            onnx_path=args.onnx,
            engine_path=output_path,
            precision=args.precision,
            workspace=args.workspace,
            batch_size=args.batch_size,
            dynamic_shape=dynamic_shape
        )
    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        raise

if __name__ == '__main__':
    main()
