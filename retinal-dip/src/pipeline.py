"""
src/pipeline.py

Stage H — Integrated analysis pipeline.

Single entry point: analyze_retina()
    Combines preprocessing → vessel segmentation → exudate detection
    → region analysis → vessel density → overlay generation.

The UI (app.py) and CLI (main.py) both call only this function.
They do not need to know about internal modules.
"""

import cv2
import numpy as np
from pathlib import Path

from .preprocessing import preprocess_for_vessels, preprocess_for_exudates
from .vessels import segment_vessels, estimate_fov
from .exudates import detect_exudates, compute_region_stats
from .visualization import make_overlay


def validate_image(image: np.ndarray) -> tuple[bool, str]:
    """
    Check that the image is a valid 3-channel colour fundus image.

    Returns:
        (True, "OK") if valid, (False, reason_string) otherwise.
    """
    if image is None:
        return False, "No image provided."
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False, "Expected a 3-channel colour image (H, W, 3)."
    if image.dtype != np.uint8:
        return False, f"Expected uint8 image, got {image.dtype}."
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False, f"Image too small ({w}×{h}). Minimum 100×100 px."
    return True, "OK"


def analyze_retina(image: np.ndarray,
                    fov_mask: np.ndarray | None = None,
                    optic_disc_mask: np.ndarray | None = None,
                    # Vessel parameters
                    gabor_ksize: int = 21,
                    gabor_sigma: float = 4.0,
                    gabor_lambda: float = 10.0,
                    gabor_gamma: float = 0.5,
                    n_orientations: int = 12,
                    vessel_morph_size: int = 3,
                    # Exudate parameters
                    clahe_clip: float = 2.0,
                    clahe_tile: tuple = (8, 8),
                    bg_sigma: float = 15.0,
                    tophat_kernel_size: int = 15,
                    exudate_morph_size: int = 3,
                    min_area: int = 20) -> dict:
    """
    Full retinal image analysis pipeline.

    Args:
        image:              RGB uint8 fundus image (H, W, 3).
        fov_mask:           Binary FOV mask (H, W) uint8. Auto-estimated
                            from image borders if None.
        optic_disc_mask:    Optional optic-disc mask for exudate suppression.
        gabor_*:            Gabor filter parameters (vessel branch).
        vessel_morph_size:  Morphological cleanup kernel size (vessel branch).
        clahe_*:            CLAHE parameters (both branches).
        bg_sigma:           Background subtraction Gaussian sigma.
        tophat_kernel_size: Top-hat structuring element size (exudate branch).
        exudate_morph_size: Cleanup kernel size (exudate branch).
        min_area:           Minimum exudate candidate area in pixels.

    Returns:
        dict with all intermediate and final results:

        Preprocessing:
            'original'      : input RGB image
            'green'         : green channel
            'clahe'         : CLAHE-enhanced green channel
            'corrected'     : illumination-corrected channel
            'L'             : Lab L channel
            'L_clahe'       : CLAHE-enhanced L channel

        Vessel branch:
            'gabor_response': float32 max Gabor response
            'vessel_mask'   : binary uint8 vessel mask

        Exudate branch:
            'tophat'        : uint8 top-hat response
            'exudate_mask'  : binary uint8 candidate mask
            'regions'       : list of region dicts

        Measurements:
            'vessel_density': float, % of FOV covered by vessels
            'region_stats'  : dict with count, total/largest/avg area

        Final:
            'overlay'       : RGB overlay image
            'fov_mask'      : binary FOV mask used
    """
    # ── Validate ────────────────────────────────────────────────────
    ok, reason = validate_image(image)
    if not ok:
        raise ValueError(f"Invalid image: {reason}")

    # ── FOV mask ────────────────────────────────────────────────────
    if fov_mask is None:
        fov_mask = estimate_fov(image)

    # ── Preprocessing ───────────────────────────────────────────────
    vessel_prep  = preprocess_for_vessels(
        image,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        bg_sigma=bg_sigma,
    )
    exudate_prep = preprocess_for_exudates(
        image,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
    )

    # ── Vessel segmentation ─────────────────────────────────────────
    vessel_result = segment_vessels(
        vessel_prep["corrected"],
        fov_mask=fov_mask,
        gabor_ksize=gabor_ksize,
        gabor_sigma=gabor_sigma,
        gabor_lambda=gabor_lambda,
        gabor_gamma=gabor_gamma,
        n_orientations=n_orientations,
        morph_kernel_size=vessel_morph_size,
    )

    # ── Exudate detection ───────────────────────────────────────────
    exudate_result = detect_exudates(
        exudate_prep["L_clahe"],
        fov_mask=fov_mask,
        optic_disc_mask=optic_disc_mask,
        tophat_kernel_size=tophat_kernel_size,
        morph_kernel_size=exudate_morph_size,
        min_area=min_area,
    )

    # ── Measurements ────────────────────────────────────────────────
    vessel_mask  = vessel_result["vessel_mask"]
    vessel_pixels = int(np.sum(vessel_mask > 0))
    fov_pixels    = int(np.sum(fov_mask > 0))
    vessel_density = 100.0 * vessel_pixels / (fov_pixels + 1e-8)

    region_stats = compute_region_stats(exudate_result["regions"])

    # ── Overlay ─────────────────────────────────────────────────────
    overlay = make_overlay(
        image,
        vessel_mask,
        exudate_result["candidate_mask"],
    )

    # ── Assemble result dict ─────────────────────────────────────────
    return {
        # Input
        "original":       image,
        # Preprocessing
        "green":          vessel_prep["green"],
        "clahe":          vessel_prep["clahe"],
        "corrected":      vessel_prep["corrected"],
        "L":              exudate_prep["L"],
        "L_clahe":        exudate_prep["L_clahe"],
        # Vessel branch
        "gabor_response": vessel_result["gabor_response"],
        "response_norm":  vessel_result["response_norm"],
        "vessel_mask":    vessel_mask,
        "fov_mask":       fov_mask,
        # Exudate branch
        "tophat":         exudate_result["tophat"],
        "exudate_mask":   exudate_result["candidate_mask"],
        "regions":        exudate_result["regions"],
        # Measurements
        "vessel_density": round(vessel_density, 4),
        "region_stats":   region_stats,
        # Final output
        "overlay":        overlay,
    }
