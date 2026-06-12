"""
Utility functions for glacial lake segmentation
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, cohen_kappa_score, jaccard_score
)

def dice_score(pred, target, smooth=1e-6):
    pred   = (pred == 1).float()
    target = (target == 1).float()
    inter  = (pred * target).sum()
    return (2*inter + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred   = (pred == 1).float()
    target = (target == 1).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)

def compute_metrics(pred_mask, gt_mask):
    """
    Compute all competition metrics for a single image pair.
    Args:
        pred_mask: numpy array (H, W), values 0 or 255
        gt_mask  : numpy array (H, W), values 0 or 255
    Returns:
        dict of metrics
    """
    pred_bin = (pred_mask > 127).astype(int).flatten()
    gt_bin   = (gt_mask   > 127).astype(int).flatten()
    smooth   = 1e-6
    inter    = (pred_bin * gt_bin).sum()

    return {
        "IoU"      : float((inter + smooth) /
                     (pred_bin.sum() + gt_bin.sum() - inter + smooth)),
        "Dice"     : float((2*inter + smooth) /
                     (pred_bin.sum() + gt_bin.sum() + smooth)),
        "F1"       : f1_score(gt_bin, pred_bin, zero_division=0),
        "Precision": precision_score(gt_bin, pred_bin, zero_division=0),
        "Recall"   : recall_score(gt_bin, pred_bin, zero_division=0),
        "Accuracy" : accuracy_score(gt_bin, pred_bin),
        "Kappa"    : cohen_kappa_score(gt_bin, pred_bin)
    }

def postprocess(mask, min_area=150):
    """Morphological cleanup of raw model predictions."""
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_large)
    clean   = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  k_small)
    clean   = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_large)
    h, w    = clean.shape
    pad     = np.zeros((h+2, w+2), np.uint8)
    pad[1:h+1, 1:w+1] = 1 - clean
    cv2.floodFill(pad, None, (0,0), 0)
    clean   = np.clip(clean + pad[1:h+1,1:w+1], 0, 1).astype(np.uint8)
    from skimage.measure import label, regionprops
    labeled = label(clean)
    final   = np.zeros_like(clean)
    for region in regionprops(labeled):
        if region.area >= min_area:
            final[labeled == region.label] = 1
    smooth  = cv2.GaussianBlur(final.astype(float), (7,7), 0)
    return (smooth > 0.5).astype(np.uint8)
