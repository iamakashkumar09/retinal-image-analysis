"""
src/preprocessing.py

Stage B — Preprocessing pipeline.

Two branches:
  - preprocess_for_vessels  : green channel → CLAHE → background-subtraction
  - preprocess_for_exudates : Lab L-channel → CLAHE

Every function takes and returns NumPy arrays in RGB order (H, W, 3) or
single-channel (H, W) uint8 unless stated otherwise.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _require_rgb(image: np.ndarray) -> None:
    """Raise ValueError if image is not 3-channel uint8."""
    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel (H, W, 3) RGB image.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}.")


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Why CLAHE: improves local contrast in regions with low intensity
    variation (e.g. faint vessels in dim retinal zones) while clipping
    prevents excessive amplification of noise.

    Args:
        gray:           Single-channel uint8 image.
        clip_limit:     Clipping limit — keep at 2.0 for baseline.
        tile_grid_size: Tile size — keep at (8, 8) for baseline.

    Returns:
        Enhanced single-channel uint8 image (same shape as input).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def subtract_background(enhanced: np.ndarray,
                        sigma: float = 15.0) -> np.ndarray:
    """
    Illumination / background correction via Gaussian background subtraction.

    Why: retinal images have a slowly-varying background brightness caused
    by the camera optics and eye curvature. Subtracting an estimated
    background (large-sigma blur) normalises this variation so Gabor
    filters can operate on a uniform signal.

    Args:
        enhanced: Single-channel uint8 image after CLAHE.
        sigma:    Gaussian sigma for background estimation.

    Returns:
        Illumination-corrected, normalised uint8 image (same shape).
    """
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    corrected = cv2.subtract(enhanced, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def preprocess_for_vessels(image: np.ndarray,
                            clahe_clip: float = 2.0,
                            clahe_tile: tuple = (8, 8),
                            bg_sigma: float = 15.0) -> dict:
    """
    Full preprocessing branch for blood-vessel segmentation.

    Pipeline:
        RGB fundus image
        → green channel   (best vessel/background contrast)
        → CLAHE           (local contrast enhancement)
        → background subtraction (illumination correction)

    Args:
        image:      RGB uint8 image (H, W, 3).
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile grid size.
        bg_sigma:   Gaussian sigma for background estimation.

    Returns:
        dict with keys:
            'green'     : raw green channel (H, W) uint8
            'clahe'     : CLAHE-enhanced green channel
            'corrected' : illumination-corrected image (used by Gabor)
    """
    _require_rgb(image)

    # Extract green channel
    # Why green: in RGB fundus images the green channel provides the
    # best contrast between vessels and the background.
    b, g, r = cv2.split(image)
    green = g

    clahe_img = apply_clahe(green, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    corrected = subtract_background(clahe_img, sigma=bg_sigma)

    return {
        "green": green,
        "clahe": clahe_img,
        "corrected": corrected,
    }


def preprocess_for_exudates(image: np.ndarray,
                              clahe_clip: float = 2.0,
                              clahe_tile: tuple = (8, 8)) -> dict:
    """
    Preprocessing branch for hard-exudate candidate detection.

    Pipeline:
        RGB → Lab color space → L channel → CLAHE

    Why Lab: the L channel captures luminance independently of colour.
    Hard exudates appear as bright (high-L) spots, so working in Lab
    isolates them cleanly.

    Args:
        image:      RGB uint8 image (H, W, 3).
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile grid size.

    Returns:
        dict with keys:
            'L'         : raw L channel (H, W) uint8
            'L_clahe'   : CLAHE-enhanced L channel (used by top-hat)
    """
    _require_rgb(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L_clahe = apply_clahe(L, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    return {
        "L": L,
        "L_clahe": L_clahe,
    }
