"""
src/exudates.py

Stage D — Candidate hard-exudate detection.

Pipeline:
    L_clahe (Lab luminance from preprocessing)
    → White top-hat morphology  (highlights bright local structures)
    → Otsu threshold            (binary candidate map)
    → Morphological cleanup     (open + close)
    → (optional) Optic-disc suppression
    → Connected-components      (individual candidate regions)
    → Area filtering            (remove noise-sized blobs)

Why "candidate" not "lesion":
    Top-hat detects bright local regions, not the semantic clinical
    concept of a hard exudate. The optic disc and bright reflections
    also satisfy this criterion. Clinical interpretation is required.
"""

import cv2
import numpy as np
import math
from .morphology import (make_ellipse_kernel, make_rect_kernel,
                          white_tophat, morpho_open_close, apply_fov_mask)


# ─────────────────────────────────────────────
# Candidate detection pipeline
# ─────────────────────────────────────────────

def detect_exudates(L_clahe: np.ndarray,
                     fov_mask: np.ndarray | None = None,
                     optic_disc_mask: np.ndarray | None = None,
                     tophat_kernel_size: int = 15,
                     morph_kernel_size: int = 3,
                     min_area: int = 20) -> dict:
    """
    Detect bright candidate exudate regions.

    Args:
        L_clahe:            CLAHE-enhanced L channel (H, W) uint8.
        fov_mask:           Optional FOV mask (H, W) uint8.
        optic_disc_mask:    Optional optic-disc mask (H, W) uint8.
                            When provided, disc pixels are suppressed.
                            (Evaluation-mode only — see docstring note.)
        tophat_kernel_size: Diameter of the structuring element for top-hat.
                            Larger → detects larger bright regions.
        morph_kernel_size:  Cleanup morphology kernel size.
        min_area:           Minimum connected-component area in pixels.
                            Components below this are discarded as noise.

    Returns:
        dict with keys:
            'tophat'          : float/uint8 top-hat response map
            'candidate_binary': uint8 binary map after threshold
            'candidate_mask'  : uint8 final mask (post-cleanup, post-disc)
            'regions'         : list of dicts, one per retained candidate
            'num_candidates'  : total retained candidate count
    """
    # ── Step 1: White top-hat ───────────────────────────────────────
    # Why elliptical kernel: rotationally symmetric → no orientation bias
    tophat_kernel = make_ellipse_kernel(tophat_kernel_size)
    tophat = white_tophat(L_clahe, tophat_kernel)

    # ── Step 2: Otsu threshold ──────────────────────────────────────
    _, candidate_binary = cv2.threshold(
        tophat, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Step 3: Morphological cleanup ──────────────────────────────
    cleanup_kernel = make_rect_kernel(morph_kernel_size)
    candidate_clean = morpho_open_close(
        candidate_binary, cleanup_kernel, cleanup_kernel
    )

    # ── Step 4: FOV masking ─────────────────────────────────────────
    if fov_mask is not None:
        candidate_clean = apply_fov_mask(candidate_clean, fov_mask)

    # ── Step 5: Optic-disc suppression (evaluation mode) ───────────
    # Why: the optic disc is a large, bright retinal structure that is
    # NOT a lesion. IDRiD provides optic-disc masks, enabling a
    # controlled experiment that removes disc false positives.
    # IMPORTANT: for live uploads without a GT disc mask, this step is
    # skipped. Do not claim the system automatically removes the disc
    # for arbitrary images unless an auto-localisation module is added.
    if optic_disc_mask is not None:
        candidate_clean[optic_disc_mask > 0] = 0

    # ── Step 6: Connected-components analysis ──────────────────────
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_clean, connectivity=8
    )

    # ── Step 7: Area filtering and region table ─────────────────────
    final_mask = np.zeros_like(candidate_clean)
    regions = []

    for i in range(1, num_labels):                   # skip background (0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue                                  # too small → noise

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        final_mask[labels == i] = 255
        regions.append({
            "id":       len(regions) + 1,
            "area":     area,
            "bbox":     [x, y, w, h],
            "centroid": [cx, cy],
        })

    return {
        "tophat":           tophat,
        "candidate_binary": candidate_binary,
        "candidate_mask":   final_mask,
        "regions":          regions,
        "num_candidates":   len(regions),
    }


# ─────────────────────────────────────────────
# Region statistics helpers
# ─────────────────────────────────────────────

def compute_region_stats(regions: list) -> dict:
    """
    Aggregate statistics over the candidate region list.

    Returns:
        dict with 'count', 'total_area', 'largest_area', 'average_area'.
    """
    if not regions:
        return {"count": 0, "total_area": 0,
                "largest_area": 0, "average_area": 0.0}

    areas = [r["area"] for r in regions]
    return {
        "count":        len(regions),
        "total_area":   sum(areas),
        "largest_area": max(areas),
        "average_area": sum(areas) / len(areas),
    }


def add_fovea_distance(regions: list, fovea_xy: tuple) -> list:
    """
    Add 'fovea_distance' (pixels) to each region dict.

    IDRiD provides fovea-centre coordinates separately. This spatial
    feature can support an optional analysis experiment.

    Args:
        regions:   List of region dicts from detect_exudates().
        fovea_xy: (x, y) pixel coordinates of the fovea centre.

    Returns:
        Same list with 'fovea_distance' field added in-place.
    """
    fx, fy = fovea_xy
    for r in regions:
        cx, cy = r["centroid"]
        r["fovea_distance"] = math.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
    return regions
