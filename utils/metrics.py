"""Metrics computation utilities."""

import torch
import numpy as np
from typing import Tuple, List, Optional
from collections import Counter

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='', names=(), eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    
    Args:
        tp: True positives (nparray, nx1 or nx10).
        conf: Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
        plot: Plot precision-recall curve at mAP@0.5.
        save_dir: Directory to save plots.
        names: Names of classes.
        eps: Small epsilon to avoid division by zero.
    
    Returns:
        Tuple of (p, r, ap, f1, ap_class)
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = len(unique_classes)  # number of classes, number of detections
    
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions
        
        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            
            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, save_dir, names)
    return p, r, ap, f1, unique_classes


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    
    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).
    
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Look for points where x axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    
    Args:
        box1: Boxes 1 (tensor, shape: [N, 4])
        box2: Boxes 2 (tensor, shape: [M, 4])
        eps: Small epsilon to avoid division by zero.
    
    Returns:
        IoU values (tensor, shape: [N, M])
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    return iou


class ConfusionMatrix:
    """Confusion matrix for classification."""
    
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        Initialize confusion matrix.
        
        Args:
            nc: Number of classes
            conf: Confidence threshold
            iou_thres: IoU threshold
        """
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))
    
    def process_batch(self, detections, labels):
        """
        Process a batch of detections and labels.
        
        Args:
            detections: Detections (tensor, shape: [N, 6]) - [x1, y1, x2, y2, conf, cls]
            labels: Labels (tensor, shape: [M, 5]) - [cls, x1, y1, x2, y2]
        """
        if detections is None or len(detections) == 0:
            if labels is not None and len(labels) > 0:
                for *lb, in labels:
                    self.matrix[self.nc, int(lb[0])] += 1
            return
        
        if labels is None or len(labels) == 0:
            for *det, conf, cls in detections:
                self.matrix[int(cls), self.nc] += 1
            return
        
        # Convert to xyxy format
        detections_xyxy = detections[:, :4]
        labels_xyxy = labels[:, 1:5]
        
        # Compute IoU
        iou = box_iou(detections_xyxy, labels_xyxy)
        
        # Match detections to labels
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.stack((x[0], x[1], iou[x[0], x[1]]), 1).cpu().numpy()
            if len(matches) > 0:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2] > self.iou_thres]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))
        
        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, det in enumerate(detections):
            if i not in m0:
                self.matrix[int(det[5]), self.nc] += 1  # background FP
        
        for i, label in enumerate(labels):
            if i not in m1:
                self.matrix[self.nc, int(label[0])] += 1  # background FN
        
        if n:
            for i, j in zip(m0, m1):
                self.matrix[int(detections[i, 5]), int(labels[j, 0])] += 1  # correct
    
    def matrix(self):
        """Return confusion matrix."""
        return self.matrix
    
    def plot(self, save_dir='', names=(), on_plot=None):
        """Plot confusion matrix."""
        try:
            import seaborn as sn
            import matplotlib.pyplot as plt
            
            array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            ticklabels = (names + ['background']) if labels else "auto"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array,
                          ax=ax,
                          annot=nc < 30,
                          annot_kws={"size": 8},
                          cmap='Blues',
                          fmt='.2f',
                          square=True,
                          vmin=0.0,
                          xticklabels=ticklabels,
                          yticklabels=ticklabels).set_facecolor((1.0, 1.0, 1.0))
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title('Confusion Matrix')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')


def plot_pr_curve(px, py, ap, save_dir='', names=()):
    """Plot precision-recall curve."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)
        
        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)
        
        ax.plot(px, py.mean(1), linewidth=3, color='blue', label=f'all classes {ap.mean():.3f} mAP@0.5')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.set_title('Precision-Recall Curve')
        fig.savefig(Path(save_dir) / 'PR_curve.png', dpi=250)
        plt.close()
    except Exception as e:
        print(f'WARNING: PR curve plot failure: {e}')


import warnings
from pathlib import Path

