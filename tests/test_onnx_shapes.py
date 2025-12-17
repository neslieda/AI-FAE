"""Tests for ONNX model shapes and dynamic axes."""

import pytest
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestONNXShapes:
    """Tests for ONNX model shape validation."""
    
    def test_onnx_model_exists(self):
        """Test if ONNX model file exists."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        # Load ONNX model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        assert model is not None
    
    def test_onnx_input_shape(self):
        """Test ONNX model input shape."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        # Load model
        session = ort.InferenceSession(str(model_path))
        
        # Get input shape
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        
        # Should support dynamic batch size
        assert input_shape[0] == 'batch_size' or isinstance(input_shape[0], int)
        assert input_shape[1] == 3  # RGB channels
        assert len(input_shape) == 4  # NCHW format
    
    def test_onnx_dynamic_batch(self):
        """Test ONNX model with different batch sizes."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            input_data = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
            
            try:
                output = session.run([output_name], {input_name: input_data})
                assert output[0].shape[0] == batch_size
            except Exception as e:
                pytest.fail(f"Failed with batch size {batch_size}: {e}")
    
    def test_onnx_dynamic_resolution(self):
        """Test ONNX model with different resolutions."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Test different resolutions
        resolutions = [(320, 320), (640, 640), (1024, 1024)]
        
        for h, w in resolutions:
            input_data = np.random.randn(1, 3, h, w).astype(np.float32)
            
            try:
                output = session.run([output_name], {input_name: input_data})
                assert output[0] is not None
            except Exception as e:
                pytest.fail(f"Failed with resolution {h}x{w}: {e}")
    
    def test_onnx_output_shape(self):
        """Test ONNX model output shape."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        session = ort.InferenceSession(str(model_path))
        output_meta = session.get_outputs()[0]
        output_shape = output_meta.shape
        
        # Output should have dynamic batch dimension
        assert output_shape[0] == 'batch_size' or isinstance(output_shape[0], int)
        assert len(output_shape) >= 2
    
    def test_onnx_opset_version(self):
        """Test ONNX opset version."""
        model_path = Path("models/model.onnx")
        if not model_path.exists():
            pytest.skip(f"ONNX model not found at {model_path}")
        
        model = onnx.load(str(model_path))
        opset_version = model.opset_import[0].version
        
        # Should be >= 12 as per requirements
        assert opset_version >= 12, f"ONNX opset version {opset_version} < 12"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

