"""
src/evaluation.py

Stage F — Pixel-level segmentation evaluation metrics.

Supported metrics:
    Dice coefficient  (primary overlap metric)
    IoU / Jaccard     (alternative overlap metric)
    Sensitivity       (recall / true positive rate)
    Specificity       (true negative rate)
    Precision         (positive predictive value)
    F1                (harmonic mean of precision and sensitivity)

Why not accuracy alone:
    Retinal vessel and lesion pixels are a minority of all image
    pixels. A trivial predictor that labels everything as background
    can achieve >95% accuracy while completely missing vessels or
    lesions. Overlap metrics (Dice, IoU) and class-specific rates
    (sensitivity, specificity) are therefore mandatory.

All functions accept boolean or uint8 (0/255) arrays of identical shape.
"""

import numpy as np


# ─────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────

def _to_bool(arr: np.ndarray) -> np.ndarray:
    """Convert uint8 mask or boolean array to boolean."""
    return arr.astype(bool)


def confusion_matrix(pred: np.ndarray, gt: np.ndarray,
                      fov_mask: np.ndarray | None = None) -> dict:
    """
    Compute pixel-level TP, FP, FN, TN counts.

    Args:
        pred:     Predicted binary mask (H, W) — uint8 or bool.
        gt:       Ground-truth binary mask (H, W) — same shape.
        fov_mask: Optional mask restricting evaluation to retinal area.

    Returns:
        dict with integer keys 'tp', 'fp', 'fn', 'tn'.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}. "
            "Resize one before evaluating."
        )

    p = _to_bool(pred)
    g = _to_bool(gt)

    if fov_mask is not None:
        fov = _to_bool(fov_mask)
        p = p & fov
        g = g & fov

    tp = int(np.logical_and(p,  g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ─────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray,
               fov_mask: np.ndarray | None = None) -> float:
    """Dice = 2·TP / (2·TP + FP + FN)"""
    cm = confusion_matrix(pred, gt, fov_mask)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    return (2.0 * tp) / (2 * tp + fp + fn + 1e-8)


def iou_score(pred: np.ndarray, gt: np.ndarray,
              fov_mask: np.ndarray | None = None) -> float:
    """IoU (Jaccard) = TP / (TP + FP + FN)"""
    cm = confusion_matrix(pred, gt, fov_mask)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    return tp / (tp + fp + fn + 1e-8)


# ─────────────────────────────────────────────
# Complete metric set
# ─────────────────────────────────────────────

def calculate_metrics(pred: np.ndarray,
                       gt: np.ndarray,
                       fov_mask: np.ndarray | None = None) -> dict:
    """
    Compute all segmentation metrics from a prediction and ground truth.

    Args:
        pred:     Predicted binary mask.
        gt:       Ground-truth mask.
        fov_mask: Optional region-of-interest mask.

    Returns:
        dict containing:
            tp, fp, fn, tn   (raw counts)
            dice, iou        (overlap metrics)
            sensitivity      (recall)
            specificity      (true negative rate)
            precision        (positive predictive value)
            f1               (harmonic mean of precision & recall)
    """
    cm = confusion_matrix(pred, gt, fov_mask)
    tp = cm["tp"]
    fp = cm["fp"]
    fn = cm["fn"]
    tn = cm["tn"]

    precision    = tp / (tp + fp + 1e-8)
    sensitivity  = tp / (tp + fn + 1e-8)   # recall
    specificity  = tn / (tn + fp + 1e-8)
    dice         = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou          = tp / (tp + fp + fn + 1e-8)
    f1           = (2 * precision * sensitivity
                    / (precision + sensitivity + 1e-8))

    return {
        "tp":          tp,
        "fp":          fp,
        "fn":          fn,
        "tn":          tn,
        "dice":        round(float(dice), 4),
        "iou":         round(float(iou), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision":   round(float(precision), 4),
        "f1":          round(float(f1), 4),
    }


# ─────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────

def sanity_check(pred: np.ndarray, gt: np.ndarray) -> list[str]:
    """
    Return a list of warning strings for common evaluation mistakes.

    Checks performed:
        - Shape mismatch
        - Prediction is all-zero or all-255 (degenerate)
        - Ground truth is all-zero (no positive pixels to match)
        - Metrics come out suspiciously perfect (data-leakage signal)
    """
    warnings = []
    if pred.shape != gt.shape:
        warnings.append(
            f"[SHAPE MISMATCH] pred={pred.shape}, gt={gt.shape}"
        )
    if np.all(pred == 0):
        warnings.append("[DEGENERATE PRED] Prediction is all zeros.")
    if np.all(pred == 255):
        warnings.append("[DEGENERATE PRED] Prediction is all 255.")
    if np.all(gt == 0):
        warnings.append("[EMPTY GT] Ground truth has no positive pixels.")
    if pred.shape == gt.shape:
        if np.array_equal(pred > 0, gt > 0):
            warnings.append(
                "[IDENTICAL ARRAYS] pred == gt; check for data leakage."
            )
    return warnings
