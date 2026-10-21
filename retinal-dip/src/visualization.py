"""
src/visualization.py

Stage visualisation utilities.

Functions:
    make_overlay           — combined vessel + exudate overlay on original
    draw_candidate_boxes   — draw bounding boxes on image
    save_pipeline_figure   — 4-panel preprocessing figure
    pipeline_grid          — multi-panel figure for the full pipeline
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")              # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Colour constants (RGB)
VESSEL_COLOR   = (0,   200, 0)     # green
EXUDATE_COLOR  = (255, 220, 0)     # yellow
BBOX_COLOR_BGR = (0,   220, 255)   # cyan (BGR for cv2.rectangle)


# ─────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────

def make_overlay(image: np.ndarray,
                  vessel_mask: np.ndarray,
                  exudate_mask: np.ndarray,
                  vessel_alpha: float = 0.6,
                  exudate_alpha: float = 0.7) -> np.ndarray:
    """
    Create a colour overlay showing vessels and candidate exudate regions.

    Vessels are tinted green; candidate exudate regions are tinted yellow.
    Pixels belonging to both are blended in order (vessel first, then
    exudate on top).

    Args:
        image:         Original RGB fundus image (H, W, 3) uint8.
        vessel_mask:   Binary vessel mask (H, W) uint8 (0/255).
        exudate_mask:  Binary exudate mask (H, W) uint8 (0/255).
        vessel_alpha:  Blend factor for vessel tint (0–1).
        exudate_alpha: Blend factor for exudate tint (0–1).

    Returns:
        RGB overlay image (H, W, 3) uint8.
    """
    overlay = image.copy().astype(np.float32)

    # Apply vessel tint
    vessel_px = vessel_mask > 0
    for c, col in enumerate(VESSEL_COLOR):
        overlay[vessel_px, c] = (
            (1 - vessel_alpha) * overlay[vessel_px, c] + vessel_alpha * col
        )

    # Apply exudate tint on top
    exudate_px = exudate_mask > 0
    for c, col in enumerate(EXUDATE_COLOR):
        overlay[exudate_px, c] = (
            (1 - exudate_alpha) * overlay[exudate_px, c] + exudate_alpha * col
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_candidate_boxes(image: np.ndarray,
                          regions: list,
                          color_bgr: tuple = BBOX_COLOR_BGR,
                          thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes around candidate exudate regions (OpenCV BGR).

    Args:
        image:     BGR image for drawing (modified in-place copy).
        regions:   List of region dicts from detect_exudates().
        color_bgr: Box colour in BGR.
        thickness: Line thickness.

    Returns:
        BGR image with drawn boxes.
    """
    out = image.copy()
    for r in regions:
        x, y, w, h = r["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color_bgr, thickness)
        label = f"#{r['id']}  {r['area']}px"
        cv2.putText(out, label, (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
    return out


# ─────────────────────────────────────────────
# Multi-panel figures
# ─────────────────────────────────────────────

def save_pipeline_figure(panels: list[tuple],
                          save_path: str | Path,
                          suptitle: str = "Preprocessing pipeline",
                          figsize: tuple = (16, 4)) -> None:
    """
    Save a multi-panel figure with labelled images.

    Args:
        panels:    List of (title, image_array) tuples.
                   Images may be grayscale (H, W) or RGB (H, W, 3).
        save_path: Output file path (.png recommended).
        suptitle:  Figure super-title.
        figsize:   Matplotlib figure size.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def pipeline_grid(result: dict, save_path: str | Path) -> None:
    """
    Save a comprehensive 8-panel figure summarising the full pipeline.

    Panels (row 1): original, green, CLAHE, corrected
    Panels (row 2): Gabor response, vessel mask, top-hat, exudate mask + overlay
    """
    def _get(key, default=None):
        return result.get(key, default)

    panels = [
        ("Original",        _get("original")),
        ("Green channel",   _get("green")),
        ("CLAHE",           _get("clahe")),
        ("Illum. corrected",_get("corrected")),
        ("Gabor response",  _get("gabor_response")),
        ("Vessel mask",     _get("vessel_mask")),
        ("Top-hat",         _get("tophat")),
        ("Exudate mask",    _get("exudate_mask")),
    ]

    # Filter out None panels
    panels = [(t, img) for t, img in panels if img is not None]

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax, (title, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(panels):]:
        ax.axis("off")

    if _get("overlay") is not None:
        # Add a legend patch
        legend_patches = [
            mpatches.Patch(color=[c / 255 for c in VESSEL_COLOR],  label="Vessels"),
            mpatches.Patch(color=[c / 255 for c in EXUDATE_COLOR], label="Exudate candidates"),
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=2, fontsize=10, framealpha=0.9)

    fig.suptitle("Retinal DIP Pipeline Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def figure_to_numpy(fig) -> np.ndarray:
    """Convert a Matplotlib figure to an RGB NumPy array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)
