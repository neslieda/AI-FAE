"""Strong augmentation pipeline for training."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Optional

def get_train_augmentation(img_size: int = 640, hsv_h: float = 0.015, hsv_s: float = 0.7, hsv_v: float = 0.4,
                           degrees: float = 0.0, translate: float = 0.1, scale: float = 0.5,
                           shear: float = 0.0, perspective: float = 0.0, flipud: float = 0.0,
                           fliplr: float = 0.5, mosaic: float = 1.0, mixup: float = 0.0,
                           copy_paste: float = 0.0):
    """
    Get training augmentation pipeline with strong augmentations.
    
    Args:
        img_size: Target image size
        hsv_h: HSV-Hue augmentation
        hsv_s: HSV-Saturation augmentation
        hsv_v: HSV-Value augmentation
        degrees: Rotation degrees
        translate: Translation fraction
        scale: Scale factor
        shear: Shear angle
        perspective: Perspective transform
        flipud: Vertical flip probability
        fliplr: Horizontal flip probability
        mosaic: Mosaic augmentation probability
        mixup: MixUp augmentation probability
        copy_paste: Copy-paste augmentation probability
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=fliplr),
        A.VerticalFlip(p=flipud),
        A.Rotate(limit=int(degrees), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=translate,
            scale_limit=scale,
            rotate_limit=int(degrees),
            shear_limit=shear,
            p=0.5
        ),
        A.Perspective(scale=(0, perspective), p=0.5),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=int(hsv_h * 360),
            sat_shift_limit=hsv_s,
            val_shift_limit=hsv_v,
            p=0.5
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Blur and noise
        A.MotionBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        
        # Advanced augmentations
        A.Cutout(
            num_holes=8,
            max_h_size=int(img_size * 0.1),
            max_w_size=int(img_size * 0.1),
            p=0.5
        ),
        A.RandomGridShuffle(grid=(2, 2), p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.1),
            max_width=int(img_size * 0.1),
            p=0.3
        ),
        
        # Resize and normalize
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def get_val_augmentation(img_size: int = 640):
    """
    Get validation augmentation pipeline (minimal augmentations).
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class Mosaic:
    """Mosaic augmentation - combines 4 images."""
    
    def __init__(self, img_size: int = 640, p: float = 1.0):
        """
        Initialize Mosaic augmentation.
        
        Args:
            img_size: Target image size
            p: Probability of applying mosaic
        """
        self.img_size = img_size
        self.p = p
    
    def __call__(self, images, labels):
        """
        Apply mosaic augmentation.
        
        Args:
            images: List of 4 images
            labels: List of 4 label arrays
        
        Returns:
            Combined image and labels
        """
        if np.random.random() > self.p or len(images) < 4:
            return images[0], labels[0]
        
        # Create output image
        s = self.img_size
        yc, xc = [int(np.random.uniform(s * 0.25, s * 0.75)) for _ in range(2)]
        
        # Place 4 images
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        labels4 = []
        
        for i, (img, label) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            
            # Place image
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust labels
            padw = x1a - x1b
            padh = y1a - y1b
            if len(label) > 0:
                label[:, 1] = (label[:, 1] * w + padw) / (s * 2)  # x center
                label[:, 2] = (label[:, 2] * h + padh) / (s * 2)  # y center
                label[:, 3] = label[:, 3] * w / (s * 2)  # width
                label[:, 4] = label[:, 4] * h / (s * 2)  # height
                labels4.append(label)
        
        # Concatenate labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 1, out=labels4[:, 1:])
        
        # Resize to target size
        img4 = cv2.resize(img4, (s, s), interpolation=cv2.INTER_LINEAR)
        
        return img4, labels4


class MixUp:
    """MixUp augmentation - blends two images."""
    
    def __init__(self, img_size: int = 640, p: float = 0.15, alpha: float = 0.2):
        """
        Initialize MixUp augmentation.
        
        Args:
            img_size: Target image size
            p: Probability of applying mixup
            alpha: Beta distribution parameter
        """
        self.img_size = img_size
        self.p = p
        self.alpha = alpha
    
    def __call__(self, img1, label1, img2, label2):
        """
        Apply MixUp augmentation.
        
        Args:
            img1: First image
            label1: First image labels
            img2: Second image
            label2: Second image labels
        
        Returns:
            Mixed image and labels
        """
        if np.random.random() > self.p:
            return img1, label1
        
        r = np.random.beta(self.alpha, self.alpha)
        r = max(r, 1 - r)
        
        # Mix images
        img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
        
        # Mix labels
        labels = np.concatenate((label1, label2), 0)
        
        return img, labels


import cv2

