"""Loss computation utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ComputeLoss:
    """Compute YOLO loss."""
    
    def __init__(self, model, autobalance=False):
        """
        Initialize loss computer.
        
        Args:
            model: Model instance
            autobalance: Whether to use autobalance for loss weights
        """
        self.sort_obj_iou = False
        self.autobalance = autobalance
        
        # Loss weights
        self.hyp = {
            'box': 0.05,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
        }
        
        # BCE with logits loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_logits = nn.BCEWithLogitsLoss()
        
        # Focal loss
        self.focal_loss = FocalLoss()
        
        # Box loss
        self.box_loss = BoxLoss()
        
        # Class loss
        self.cls_loss = ClassLoss()
    
    def __call__(self, preds, targets):
        """
        Compute loss.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(preds, targets)
        
        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = self.bbox_iou(pbox.T, targets[:, 2:6], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.hyp['label_smoothing']) + self.hyp['label_smoothing'] * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                # Classification
                if ps.size(1) - 5 > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                    t[range(n), targets[:, 1].long()] = 1
                    lcls += self.bce_logits(ps[:, 5:], t)  # BCE
            
            obji = self.bce_logits(pi[..., 4], tobj)
            lobj += obji * self.hyp['box']  # obj loss
        
        lbox *= self.hyp['box']
        lobj *= self.hyp['cls']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
    
    def build_targets(self, p, targets):
        """Build targets for loss computation."""
        na, nt = 3, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        off = torch.tensor([[0, 0]], device=targets.device).float() * g  # offsets
        
        for i, yolo_layer in enumerate(p):
            anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=targets.device).float()  # anchors
            gain[2:6] = torch.tensor(yolo_layer.shape)[[3, 2, 3, 2]]  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare
                t = t[j]  # filter
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch
    
    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """Compute IoU between boxes."""
        box2 = box2.T
        
        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        
        # IoU
        iou = inter / union
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/utils.py
                    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou  # IoU


class FocalLoss(nn.Module):
    """Focal loss implementation."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """Compute focal loss."""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class BoxLoss(nn.Module):
    """Box regression loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """Compute box loss."""
        return F.smooth_l1_loss(pred, target)


class ClassLoss(nn.Module):
    """Classification loss."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """Compute class loss."""
        return self.bce(pred, target)

