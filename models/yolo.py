"""YOLO model definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path

class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize convolution layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            act: Activation function
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        Initialize bottleneck layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Whether to use shortcut connection
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        """Forward pass."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """C3 module."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottlenecks
            shortcut: Whether to use shortcut connection
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    
    def forward(self, x):
        """Forward pass."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, c1, c2, k=5):
        """
        Initialize SPPF layer.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        """Forward pass."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """Detection head."""
    
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """
        Initialize detection head.
        
        Args:
            nc: Number of classes
            anchors: Anchor boxes
            ch: Channels
            inplace: Whether to use inplace operations
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2 if isinstance(anchors[0], (list, tuple)) else len(anchors[0])  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    
    def forward(self, x):
        """Forward pass."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    def _make_grid(self, nx=20, ny=20, i=0):
        """Create grid for anchor boxes."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if 1:  # torch>=1.8.0
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    """YOLO model."""
    
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize YOLO model.
        
        Args:
            cfg: Model configuration file or dict
            ch: Input channels
            nc: Number of classes
            verbose: Whether to print model info
        """
        super().__init__()
        
        # Load configuration
        if isinstance(cfg, str):
            if Path(cfg).exists():
                import yaml
                with open(cfg, 'r') as f:
                    cfg = yaml.safe_load(f)
            else:
                # Default YOLOv8n configuration
                cfg = {
                    'nc': nc or 80,
                    'scale': 'n',
                    'depth_multiple': 0.33,
                    'width_multiple': 0.25,
                    'backbone': [
                        [-1, 1, Conv, [64, 3, 2]],
                        [-1, 1, Conv, [128, 3, 2]],
                        [-1, 2, C3, [128]],
                        [-1, 1, Conv, [256, 3, 2]],
                        [-1, 2, C3, [256]],
                        [-1, 1, Conv, [512, 3, 2]],
                        [-1, 2, C3, [512]],
                        [-1, 1, Conv, [1024, 3, 2]],
                        [-1, 1, C3, [1024]],
                        [-1, 1, SPPF, [1024, 5]],
                    ],
                    'head': [
                        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                        [[-1, 6], 1, Concat, [1]],
                        [-1, 1, C3, [512, False]],
                        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                        [[-1, 4], 1, Concat, [1]],
                        [-1, 1, C3, [256, False]],
                        [-1, 1, Conv, [256, 3, 2]],
                        [[-1, -4], 1, Concat, [1]],
                        [-1, 1, C3, [512, False]],
                        [-1, 1, Conv, [512, 3, 2]],
                        [[-1, -6], 1, Concat, [1]],
                        [-1, 1, C3, [1024, False]],
                        [[17, 20, 23], 1, Detect, [nc, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]]],
                    ]
                }
        
        self.nc = nc or cfg.get('nc', 80)
        self.yaml = cfg
        
        # Build model
        self.model, self.save = self._build_model(cfg, ch, verbose)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_model(self, cfg, ch, verbose):
        """Build model from configuration."""
        layers, save = [], []
        depth_multiple = cfg.get('depth_multiple', 1.0)
        width_multiple = cfg.get('width_multiple', 1.0)
        
        # Build backbone and head
        backbone = cfg.get('backbone', [])
        head = cfg.get('head', [])
        
        # Simplified model structure
        # In practice, you would parse the full YAML configuration
        # For now, we'll create a basic structure
        
        # Backbone
        layers.append(Conv(ch, 64, 3, 2))
        layers.append(Conv(64, 128, 3, 2))
        layers.append(C3(128, 128, n=int(3 * depth_multiple)))
        layers.append(Conv(128, 256, 3, 2))
        layers.append(C3(256, 256, n=int(6 * depth_multiple)))
        layers.append(Conv(256, 512, 3, 2))
        layers.append(C3(512, 512, n=int(6 * depth_multiple)))
        layers.append(Conv(512, 1024, 3, 2))
        layers.append(C3(1024, 1024, n=int(3 * depth_multiple)))
        layers.append(SPPF(1024, 1024, 5))
        
        # Head (simplified)
        layers.append(Conv(1024, 512, 1, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(Conv(512, 256, 1, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(Conv(256, 256, 3, 1))
        layers.append(Detect(self.nc, anchors=[[10, 13], [16, 30], [33, 23]], ch=[256]))
        
        return nn.Sequential(*layers), save
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            augment: Whether to use augmentation
            profile: Whether to profile
            visualize: Whether to visualize
        
        Returns:
            Model output
        """
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)
    
    def _forward_once(self, x, profile=False, visualize=False):
        """Single forward pass."""
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x
    
    def _forward_augment(self, x):
        """Augmented forward pass."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = F.interpolate(x, size=(int(img_size[0] * si), int(img_size[1] * si)), mode='bilinear', align_corners=False)
            if fi == 2:
                xi = torch.flip(xi, dims=[2])
            elif fi == 3:
                xi = torch.flip(xi, dims=[3])
            y.append(self._forward_once(xi)[0])
        y = self._non_max_suppression(torch.cat(y, 1), 0, 0.5, classes=None, agnostic=False)
        return y
    
    def _non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
        """Non-maximum suppression."""
        # Simplified NMS implementation
        # In practice, you would use a more complete implementation
        return prediction
    
    def compute_loss(self, pred, targets):
        """Compute loss."""
        # This should be implemented with the actual loss computation
        # For now, return a dummy loss
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        return loss, torch.tensor([0.0, 0.0, 0.0, 0.0])


class Concat(nn.Module):
    """Concatenate layers."""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        """Forward pass."""
        return torch.cat(x, self.d)

