"""
src/vessels.py

Stage C — Blood-vessel segmentation using Gabor filters.

Pipeline:
    corrected_green (from preprocessing)
    → Gabor filter bank (12 orientations, max-response pooling)
    → Normalise response to [0, 255]
    → Otsu threshold → binary mask
    → Morphological cleanup (open + close)
    → FOV masking
    → vessel mask

Key DIP concepts:
    Gabor filter: an oriented band-pass filter. It responds strongly
    to elongated, line-like structures at a specific orientation and
    spatial frequency. Vessels are approximately elongated; using a
    BANK of 12 orientations (0° – 165°) covers all vessel directions.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from .morphology import (make_rect_kernel, morpho_open_close, apply_fov_mask)


# ─────────────────────────────────────────────
# Gabor bank
# ─────────────────────────────────────────────

def _gabor_single_orientation(image_f: np.ndarray,
                               ksize: int,
                               sigma: float,
                               theta: float,
                               lambd: float,
                               gamma: float,
                               psi: float) -> np.ndarray:
    """Apply a single oriented Gabor kernel and return its absolute response."""
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma=sigma,
        theta=theta,
        lambd=lambd,
        gamma=gamma,
        psi=psi,
        ktype=cv2.CV_32F,
    )
    response = cv2.filter2D(image_f, cv2.CV_32F, kernel)
    np.abs(response, out=response)                   # sign-invariant
    return response


def gabor_max_response(image: np.ndarray,
                        ksize: int = 21,
                        sigma: float = 4.0,
                        lambd: float = 10.0,
                        gamma: float = 0.5,
                        psi: float = 0.0,
                        n_orientations: int = 12) -> np.ndarray:
    """
    Apply a bank of Gabor filters at multiple orientations and return
    the per-pixel maximum absolute response.

    Why maximum pooling: vessels appear at many directions. Using the
    maximum response over all angles ensures that a vessel at any
    orientation generates a strong output regardless of which single
    filter is closest to its true angle.

    Args:
        image:         Single-channel float or uint8 (H, W).
        ksize:         Gabor kernel size (must be odd).
        sigma:         Gaussian envelope spread.
        lambd:         Sinusoidal wavelength (spatial frequency).
        gamma:         Aspect ratio (elongation).
        psi:           Phase offset.
        n_orientations: Number of orientations from 0 to π.

    Returns:
        Max-pooled response map as float32 (H, W).
    """
    image_f = image.astype(np.float32)
    max_response = np.zeros_like(image_f)

    # OpenCV filter2D releases the GIL, so parallelising orientations speeds up large images
    workers = min(n_orientations, max(1, os.cpu_count() or 2), 4)
    if workers <= 1 or image_f.size < 250_000:
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations      # 0, 15°, 30°, … 165°
            response = _gabor_single_orientation(
                image_f, ksize, sigma, theta, lambd, gamma, psi
            )
            np.maximum(max_response, response, out=max_response)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _gabor_single_orientation,
                    image_f, ksize, sigma, i * np.pi / n_orientations,
                    lambd, gamma, psi,
                )
                for i in range(n_orientations)
            ]
            for fut in as_completed(futures):
                np.maximum(max_response, fut.result(), out=max_response)

    return max_response


# ─────────────────────────────────────────────
# FOV estimation (fallback for user uploads)
# ─────────────────────────────────────────────

def estimate_fov(image: np.ndarray) -> np.ndarray:
    """
    Estimate a field-of-view mask when no dataset mask is provided.

    Strategy: threshold non-black pixels → morphological clean-up →
    retain the largest connected component (the retinal disc).

    This is a practical heuristic for arbitrary uploaded images; it
    should NOT be used as a benchmark-standard annotation.

    Args:
        image: RGB uint8 image (H, W, 3).

    Returns:
        Binary uint8 FOV mask (H, W), values 0 or 255.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray > 10, 255, 0).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask                                  # no components found

    # Keep the largest component (background is component 0)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def segment_vessels(corrected: np.ndarray,
                     fov_mask: np.ndarray | None = None,
                     gabor_ksize: int = 21,
                     gabor_sigma: float = 4.0,
                     gabor_lambda: float = 10.0,
                     gabor_gamma: float = 0.5,
                     gabor_psi: float = 0.0,
                     n_orientations: int = 12,
                     morph_kernel_size: int = 3) -> dict:
    """
    Full vessel-segmentation pipeline.

    Args:
        corrected:          Illumination-corrected green channel (H, W) uint8.
        fov_mask:           Binary mask (H, W) uint8. If None, estimated.
        gabor_*:            Gabor filter parameters (experiment variables).
        n_orientations:     Number of Gabor angles.
        morph_kernel_size:  Size of morphological cleanup kernel.

    Returns:
        dict with keys:
            'gabor_response' : float32 max-pooled Gabor response map
            'response_norm'  : uint8 normalised response (0-255)
            'vessel_binary'  : uint8 binary mask after Otsu threshold
            'vessel_mask'    : uint8 final mask after morphology + FOV
            'fov_mask'       : uint8 FOV mask used
    """
    # ── Step 1: Gabor filter bank ──────────────────────────────────
    gabor_response = gabor_max_response(
        corrected,
        ksize=gabor_ksize,
        sigma=gabor_sigma,
        lambd=gabor_lambda,
        gamma=gabor_gamma,
        psi=gabor_psi,
        n_orientations=n_orientations,
    )

    # ── Step 2: Normalise to uint8 for thresholding ────────────────
    response_norm = cv2.normalize(
        gabor_response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ── Step 3: Otsu threshold ──────────────────────────────────────
    # Why Otsu: automatically chooses a threshold based on the
    # bimodal histogram of response values (vessel vs. background).
    _, vessel_binary = cv2.threshold(
        response_norm, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Step 4: Morphological cleanup ──────────────────────────────
    kernel = make_rect_kernel(morph_kernel_size)
    vessel_clean = morpho_open_close(vessel_binary, kernel, kernel)

    # ── Step 5: FOV masking ─────────────────────────────────────────
    # Build an RGB proxy for FOV estimation if no mask provided
    if fov_mask is None:
        # Best effort: treat the corrected image as grayscale proxy
        gray_proxy = corrected
        fov_mask_bin = np.where(gray_proxy > 5, 255, 0).astype(np.uint8)
        k = np.ones((9, 9), np.uint8)
        fov_mask_bin = cv2.morphologyEx(fov_mask_bin, cv2.MORPH_CLOSE, k)
        fov_mask = fov_mask_bin

    vessel_mask = apply_fov_mask(vessel_clean, fov_mask)

    return {
        "gabor_response": gabor_response,
        "response_norm": response_norm,
        "vessel_binary": vessel_binary,
        "vessel_mask": vessel_mask,
        "fov_mask": fov_mask,
    }
