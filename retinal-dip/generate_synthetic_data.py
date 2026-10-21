"""
generate_synthetic_data.py

Creates realistic synthetic retinal fundus images for development and testing
when DRIVE / IDRiD datasets are not yet downloaded.

Generates:
    data/DRIVE/training/images/  — 5 synthetic retinal TIF images
    data/DRIVE/training/fov_masks/   — circular FOV masks
    data/DRIVE/training/vessel_masks/ — synthetic vessel ground-truth masks
    data/IDRiD/images/               — 3 synthetic high-resolution fundus images
    data/IDRiD/exudate_masks/        — synthetic hard-exudate ground-truth masks
    data/IDRiD/optic_disc_masks/     — synthetic optic-disc masks
    data/IDRiD/metadata/fovea.csv    — fovea coordinates

Usage:
    python generate_synthetic_data.py
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd


RNG = np.random.default_rng(42)      # reproducible


# ─────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────

def _make_retinal_background(H: int, W: int) -> np.ndarray:
    """
    Create a realistic reddish-orange retinal background with radial
    vignetting and subtle illumination gradient.
    """
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Base colour: reddish-orange fundus
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]

    # Radial distance normalised to [0, 1]
    radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    fov_r  = min(H, W) * 0.47
    mask   = radius < fov_r

    # Background gradient (brighter in centre)
    grad = np.clip(1.0 - 0.5 * (radius / fov_r), 0.2, 1.0)

    img[:, :, 0] = mask * grad * 100   # R  (fundus is mostly red channel)
    img[:, :, 1] = mask * grad * 60    # G
    img[:, :, 2] = mask * grad * 20    # B

    # Add subtle noise
    noise = RNG.normal(0, 4, (H, W, 3)).astype(np.float32)
    img += noise * mask[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_vessel_tree(canvas: np.ndarray,
                       mask_canvas: np.ndarray,
                       start: tuple,
                       direction: float,
                       length: int,
                       width: float,
                       depth: int,
                       max_depth: int = 5) -> None:
    """
    Recursively draw a branching vessel tree on canvas and fill mask_canvas.

    Args:
        canvas:       RGB image to draw on.
        mask_canvas:  Single-channel binary mask.
        start:        (x, y) starting point.
        direction:    Angle in radians.
        length:       Segment length.
        width:        Line width.
        depth:        Current recursion depth.
        max_depth:    Maximum recursion depth.
    """
    if depth > max_depth or length < 4 or width < 0.5:
        return

    sx, sy = start
    ex = int(sx + length * np.cos(direction))
    ey = int(sy + length * np.sin(direction))

    # Vessel colour — dark reddish on green channel (visible contrast)
    color = (
        int(40 - depth * 3),   # R
        int(80 - depth * 5),   # G
        int(30 - depth * 2),   # B
    )
    color = tuple(max(0, c) for c in color)

    iw = max(1, int(width))
    cv2.line(canvas,      (sx, sy), (ex, ey), color, iw)
    cv2.line(mask_canvas, (sx, sy), (ex, ey), 255,   iw)

    # Random angle perturbation for natural look
    jitter = RNG.uniform(-0.25, 0.25)

    # Branch 1 (main continuation)
    _draw_vessel_tree(
        canvas, mask_canvas,
        (ex, ey),
        direction + jitter,
        int(length * 0.78),
        width * 0.82,
        depth + 1, max_depth,
    )

    # Branch 2 (side branch, not always)
    if depth < max_depth - 1 and RNG.random() > 0.35:
        branch_angle = direction + RNG.choice([-1, 1]) * RNG.uniform(0.4, 0.8)
        _draw_vessel_tree(
            canvas, mask_canvas,
            (ex, ey),
            branch_angle,
            int(length * 0.55),
            width * 0.65,
            depth + 1, max_depth,
        )


def _make_fov_mask(H: int, W: int) -> np.ndarray:
    """Circular FOV mask centred on the image."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H // 2, W // 2
    r = int(min(H, W) * 0.47)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _add_exudate_spots(img: np.ndarray,
                        mask: np.ndarray,
                        n_spots: int,
                        fov_mask: np.ndarray) -> None:
    """
    Add bright yellowish exudate-like spots to the image and fill mask.
    Spots are placed only inside the FOV, away from the disc.
    """
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    fov_r  = int(min(H, W) * 0.47)

    spots_added = 0
    attempts    = 0
    while spots_added < n_spots and attempts < n_spots * 20:
        attempts += 1
        angle = RNG.uniform(0, 2 * np.pi)
        dist  = RNG.uniform(0.15, 0.75) * fov_r
        x = int(cx + dist * np.cos(angle))
        y = int(cy + dist * np.sin(angle))

        if not (0 <= x < W and 0 <= y < H):
            continue
        if fov_mask[y, x] == 0:
            continue

        radius = int(RNG.integers(8, 25))
        # Cluster of small bright spots (exudate-like)
        brightness = int(RNG.integers(180, 255))
        cv2.circle(img,  (x, y), radius, (brightness, brightness, int(brightness * 0.8)), -1)
        cv2.circle(mask, (x, y), radius, 255, -1)
        spots_added += 1


def _add_optic_disc(img: np.ndarray, disc_mask: np.ndarray, fov_mask: np.ndarray) -> tuple:
    """
    Add a bright circular optic disc and return its (cx, cy).
    Disc placed to the nasal side (~30% from centre).
    """
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    disc_cx = int(cx + W * 0.28)
    disc_cy = cy
    disc_r  = int(min(H, W) * 0.07)

    # Bright yellowish-white disc
    cv2.circle(img,       (disc_cx, disc_cy), disc_r,     (240, 230, 180), -1)
    cv2.circle(disc_mask, (disc_cx, disc_cy), disc_r,     255, -1)

    return disc_cx, disc_cy


# ─────────────────────────────────────────────
# DRIVE image generator
# ─────────────────────────────────────────────

def generate_drive_image(idx: int, output_dir: Path) -> None:
    """Generate one DRIVE-style synthetic image + masks."""
    H, W = 584, 768

    img        = _make_retinal_background(H, W)
    vessel_mask = np.zeros((H, W), dtype=np.uint8)
    fov_mask    = _make_fov_mask(H, W)

    # Draw 4–6 main vessel trees from the disc region
    disc_x = int(W * 0.65)
    disc_y = H // 2

    for k in range(RNG.integers(4, 7)):
        angle  = RNG.uniform(0, 2 * np.pi)
        length = RNG.integers(80, 160)
        width  = RNG.uniform(4, 7)
        _draw_vessel_tree(img, vessel_mask,
                          (disc_x, disc_y), angle, length, width,
                          depth=0, max_depth=6)

    # Apply FOV mask to vessel mask
    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=fov_mask)

    # Slight Gaussian blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    name = f"0{idx:02d}_training"
    cv2.imwrite(str(output_dir / "images"      / f"{name}.tif"), img)
    cv2.imwrite(str(output_dir / "fov_masks"   / f"{name}_fov.png"), fov_mask)
    cv2.imwrite(str(output_dir / "vessel_masks"/ f"{name}_vessel.png"), vessel_mask)
    print(f"  [DRIVE train] {name}.tif — vessels: {np.sum(vessel_mask>0)} px")


def generate_drive_test_image(idx: int, output_dir: Path) -> None:
    """Generate one DRIVE-style test image (no vessel GT)."""
    H, W = 584, 768
    img      = _make_retinal_background(H, W)
    fov_mask = _make_fov_mask(H, W)
    vessel_mask = np.zeros((H, W), dtype=np.uint8)

    disc_x = int(W * 0.65)
    disc_y = H // 2

    for k in range(RNG.integers(4, 6)):
        angle  = RNG.uniform(0, 2 * np.pi)
        length = RNG.integers(80, 160)
        width  = RNG.uniform(3, 6)
        _draw_vessel_tree(img, vessel_mask,
                          (disc_x, disc_y), angle, length, width,
                          depth=0, max_depth=5)

    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    name = f"0{idx:02d}_test"
    cv2.imwrite(str(output_dir / "images"    / f"{name}.tif"), img)
    cv2.imwrite(str(output_dir / "fov_masks" / f"{name}_fov.png"), fov_mask)
    print(f"  [DRIVE test]  {name}.tif")


# ─────────────────────────────────────────────
# IDRiD image generator
# ─────────────────────────────────────────────

def generate_idrid_image(idx: int, output_dir: Path,
                          fovea_records: list) -> None:
    """Generate one IDRiD-style synthetic image + masks."""
    H, W = 1072, 1424        # scaled-down version of actual 4288×2848

    img          = _make_retinal_background(H, W)
    exudate_mask = np.zeros((H, W), dtype=np.uint8)
    disc_mask    = np.zeros((H, W), dtype=np.uint8)
    fov_mask     = _make_fov_mask(H, W)
    vessel_mask  = np.zeros((H, W), dtype=np.uint8)

    # Add optic disc
    disc_cx, disc_cy = _add_optic_disc(img, disc_mask, fov_mask)

    # Draw vessels from disc
    for k in range(RNG.integers(5, 8)):
        angle  = RNG.uniform(0, 2 * np.pi)
        length = RNG.integers(120, 220)
        width  = RNG.uniform(5, 9)
        _draw_vessel_tree(img, vessel_mask,
                          (disc_cx, disc_cy), angle, length, width,
                          depth=0, max_depth=7)

    # Add exudate spots
    n_spots = RNG.integers(10, 30)
    _add_exudate_spots(img, exudate_mask, n_spots, fov_mask)

    # Suppress disc from exudate mask (to mimic IDRiD GT)
    exudate_mask[disc_mask > 0] = 0

    img = cv2.GaussianBlur(img, (5, 5), 1.2)

    # Fovea (temporal side, opposite to disc)
    fovea_cx = int(W // 2 - W * 0.18)
    fovea_cy = H // 2

    name = f"IDRiD_{idx:03d}"
    cv2.imwrite(str(output_dir / "images"         / f"{name}.jpg"),  img)
    cv2.imwrite(str(output_dir / "exudate_masks"  / f"{name}_EX.png"), exudate_mask)
    cv2.imwrite(str(output_dir / "optic_disc_masks"/ f"{name}_OD.png"), disc_mask)

    fovea_records.append({
        "image_id":  name,
        "fovea_x":   fovea_cx,
        "fovea_y":   fovea_cy,
        "disc_x":    disc_cx,
        "disc_y":    disc_cy,
    })
    print(f"  [IDRiD]  {name}.jpg — exudates: {np.sum(exudate_mask>0)} px")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    base = Path(__file__).parent / "data"

    # ── DRIVE training (5 images) ───────────────────────────────────
    drive_train = base / "DRIVE" / "training"
    print("Generating DRIVE training images …")
    for i in range(1, 6):
        generate_drive_image(i, drive_train)

    # ── DRIVE test (3 images, no GT) ────────────────────────────────
    drive_test = base / "DRIVE" / "test"
    print("Generating DRIVE test images …")
    for i in range(1, 4):
        generate_drive_test_image(i, drive_test)

    # ── IDRiD (5 images) ────────────────────────────────────────────
    idrid = base / "IDRiD"
    fovea_records: list = []
    print("Generating IDRiD images …")
    for i in range(1, 6):
        generate_idrid_image(i, idrid, fovea_records)

    # Save fovea CSV
    df = pd.DataFrame(fovea_records)
    df.to_csv(str(idrid / "metadata" / "fovea.csv"), index=False)
    print(f"  Saved fovea.csv ({len(df)} rows)")

    print("\n✅  Synthetic data generation complete.")
    print("    Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
