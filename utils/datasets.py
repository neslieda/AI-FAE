"""Dataset loading utilities."""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class LoadImages:
    """Load images from directory."""
    
    def __init__(self, path, img_size=640, stride=32):
        """
        Initialize image loader.
        
        Args:
            path: Path to image directory or single image file
            img_size: Target image size
            stride: Model stride
        """
        path = str(Path(path).resolve())
        if os.path.isdir(path):
            self.files = sorted([os.path.join(path, f) for f in os.listdir(path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        elif os.path.isfile(path):
            self.files = [path]
        else:
            raise FileNotFoundError(f"Path not found: {path}")
        
        self.img_size = img_size
        self.stride = stride
        self.nf = len(self.files)
        self.mode = 'image'
    
    def __iter__(self):
        """Iterate over images."""
        self.count = 0
        return self
    
    def __next__(self):
        """Get next image."""
        if self.count == self.nf:
            raise StopIteration
        
        path = self.files[self.count]
        self.count += 1
        
        # Read image
        img0 = cv2.imread(path)
        if img0 is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Letterbox resize
        img = self.letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        
        return path, img, img0, None
    
    def __len__(self):
        """Return number of images."""
        return self.nf
    
    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, don't scale up (for better test mAP)
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


class LoadStreams:
    """Load video streams."""
    
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        """
        Initialize stream loader.
        
        Args:
            sources: Path to file with stream URLs or list of URLs
            img_size: Target image size
            stride: Model stride
        """
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = threading.Thread(target=self.update, args=([i, cap]), daemon=True)
            print('success')
            thread.start()
        print('')
    
    def update(self, i, cap):
        """Update stream frames."""
        n, f = 0, self.fps
        while cap.isOpened():
            n += 1
            cap.grab()
            if n % 4 == 0:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
                time.sleep(1 / f)  # wait time
    
    def __iter__(self):
        """Iterate over streams."""
        self.count = -1
        return self
    
    def __next__(self):
        """Get next frame."""
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        # Letterbox
        img = [self.letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        
        return self.sources, img, img0, None
    
    def __len__(self):
        """Return number of streams."""
        return len(self.sources)
    
    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Resize and pad image."""
        return LoadImages.letterbox(img, new_shape, color, auto, scaleFill, scaleup, stride)


class YOLODataset(Dataset):
    """YOLO dataset class."""
    
    def __init__(self, img_paths, labels=None, img_size=640, augment=False, hyp=None):
        """
        Initialize YOLO dataset.
        
        Args:
            img_paths: List of image paths
            labels: List of label files (optional)
            img_size: Image size
            augment: Whether to apply augmentation
            hyp: Hyperparameters dict
        """
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        
        # Define augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomRotate90(p=0.1),
                A.Transpose(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        self.to_tensor = ToTensorV2()
    
    def __len__(self):
        """Return dataset size."""
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """Get item by index."""
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels if available
        boxes = []
        class_labels = []
        if self.labels and idx < len(self.labels):
            label_path = self.labels[idx]
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x, y, w, h = map(float, parts)
                            boxes.append([x, y, w, h])
                            class_labels.append(int(class_id))
        
        # Apply augmentation
        if boxes:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        # Convert to tensor
        img = self.to_tensor(image=img)['image']
        img = img.float() / 255.0
        
        # Format targets
        targets = []
        for i, (box, cls) in enumerate(zip(boxes, class_labels)):
            targets.append([cls, *box])
        
        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))
        
        return img, targets, img_path, idx


def create_dataloader(path, img_size, batch_size, augment=False, workers=8, hyp=None):
    """
    Create data loader for training/validation.
    
    Args:
        path: Path to dataset directory or YAML file
        img_size: Image size
        batch_size: Batch size
        augment: Whether to apply augmentation
        workers: Number of worker threads
        hyp: Hyperparameters dict
    
    Returns:
        DataLoader instance
    """
    # Load dataset configuration
    if isinstance(path, str) and path.endswith('.yaml'):
        from .general import check_dataset
        data_dict = check_dataset(path)
        path = data_dict.get('train', path)
    
    # Collect image paths
    if os.path.isdir(path):
        img_paths = sorted([os.path.join(path, f) for f in os.listdir(path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    else:
        img_paths = [path]
    
    # Collect label paths (if available)
    label_paths = []
    for img_path in img_paths:
        label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(label_path):
            label_paths.append(label_path)
        else:
            label_paths.append(None)
    
    # Create dataset
    dataset = YOLODataset(
        img_paths=img_paths,
        labels=label_paths if any(label_paths) else None,
        img_size=img_size,
        augment=augment,
        hyp=hyp
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=augment
    )
    
    return dataloader


def collate_fn(batch):
    """Collate function for batching."""
    imgs, targets, paths, indices = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, targets, paths, indices


import threading
import time

