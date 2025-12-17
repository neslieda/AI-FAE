import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent))

from training.train import parse_opt
from models.yolo import Model

def export_onnx(weights='yolov8n.pt',
               imgsz=(640, 640),
               batch_size=1,
               device='cpu',
               include_nms=True,
               opset=12,
               dynamic=False,
               simplify=False,
               output=None):
    """Export YOLO model to ONNX format.
    
    Args:
        weights (str): Path to PyTorch model weights
        imgsz (tuple): Image size (height, width)
        batch_size (int): Batch size
        device (str): Device to use ('cpu' or 'cuda')
        include_nms (bool): Whether to include NMS in the exported model
        opset (int): ONNX opset version
        dynamic (bool): Whether to use dynamic axes
        simplify (bool): Whether to simplify the ONNX model
        output (str, optional): Output ONNX model path
    """
    # Check device
    device = torch.device(device)
    
    # Load PyTorch model
    model = Model(weights).to(device)
    model.eval()
    
    # Input tensor
    im = torch.zeros(batch_size, 3, *imgsz).to(device)
    
    # Input and output names
    input_names = ['images']
    output_names = ['output0']
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch_size'},  # batch size
            'output0': {0: 'batch_size'}
        }
    
    # Export the model
    output = output or str(Path(weights).with_suffix('.onnx'))
    torch.onnx.export(
        model,
        im,
        output,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Simplify ONNX model if requested
    if simplify:
        try:
            import onnx
            import onnxsim
            
            print('Starting to simplify ONNX...')
            model_onnx = onnx.load(output)
            model_simp, check = onnxsim.simplify(model_onnx)
            assert check, 'Simplified ONNX model could not be validated'
            onnx.save(model_simp, output)
            print(f'ONNX simplified and saved to {output}')
        except Exception as e:
            print(f'Simplification failure: {e}')
    
    # Verify the exported model
    import onnx
    model_onnx = onnx.load(output)
    onnx.checker.check_model(model_onnx)
    print(f'ONNX export success, saved as {output}')
    return output

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model.pt path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include-nms', action='store_true', help='include NMS in the exported model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--output', type=str, default='', help='output file')
    return parser.parse_args()

def main(opt):
    export_onnx(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    main(opt)
