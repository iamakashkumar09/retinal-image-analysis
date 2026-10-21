"""
src/morphology.py

Reusable morphological operations and kernel helpers.

Keeping these functions here prevents each module from duplicating
kernel-creation boilerplate and makes experiments easier to trace.
"""

import cv2
import numpy as np


def make_ellipse_kernel(size: int) -> np.ndarray:
    """Return an elliptical structuring element of the given diameter."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def make_rect_kernel(size: int) -> np.ndarray:
    """Return a rectangular structuring element of the given side length."""
    return np.ones((size, size), np.uint8)


def morpho_open(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Morphological opening (erosion → dilation).

    Why: removes small isolated noise pixels and thin spurs from a
    binary mask without significantly altering larger structures.
    """
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morpho_close(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Morphological closing (dilation → erosion).

    Why: bridges small gaps within a binary mask and connects nearby
    vessel fragments that should be continuous.
    """
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def morpho_open_close(binary: np.ndarray,
                       open_kernel: np.ndarray,
                       close_kernel: np.ndarray) -> np.ndarray:
    """Open then close — the standard two-step cleanup sequence."""
    opened = morpho_open(binary, open_kernel)
    return morpho_close(opened, close_kernel)


def white_tophat(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    White top-hat transform: image − morphological_opening(image).

    Why: emphasises bright structures that are *smaller* than the
    structuring element. Used in the exudate branch to isolate
    small bright spots against a slowly-varying background.
    """
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def apply_fov_mask(binary: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """
    Zero out pixels outside the retinal field of view.

    Why: large black image borders are treated as non-retinal and their
    transitions can create false vessel or exudate detections. Masking
    restricts evaluation to only the illuminated retinal disc.
    """
    return cv2.bitwise_and(binary, binary, mask=fov_mask)
