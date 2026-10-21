"""
test_pipeline.py — Unit tests for all pipeline stages.

Run with:
    python test_pipeline.py

Tests cover:
    - Image loading and validation
    - Preprocessing (green channel, CLAHE, background subtraction)
    - Vessel segmentation (Gabor response, mask dtypes)
    - Exudate detection (top-hat, connected components)
    - Evaluation metrics (sanity cases)
    - Overlay (shape check)
    - Report generation (file creation)
"""

import sys
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_for_vessels, preprocess_for_exudates
from src.vessels import segment_vessels, estimate_fov
from src.exudates import detect_exudates
from src.evaluation import calculate_metrics, sanity_check
from src.visualization import make_overlay
from src.pipeline import analyze_retina, validate_image
from src.report import generate_report


# ─────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────

PASS = "✅  PASS"
FAIL = "❌  FAIL"


def assert_equal(a, b, msg=""):
    assert a == b, f"{msg}: expected {b}, got {a}"


def assert_true(cond, msg=""):
    assert cond, f"Assertion failed: {msg}"


def make_fake_fundus(H: int = 200, W: int = 256) -> np.ndarray:
    """Create a minimal synthetic RGB fundus image."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Orange-ish background (retinal colour)
    img[:, :, 0] = 100
    img[:, :, 1] = 60
    img[:, :, 2] = 20
    # Circular FOV
    cy, cx = H // 2, W // 2
    cv2.circle(img, (cx, cy), min(H, W) // 2 - 5, (0, 0, 0), -1)
    img[:, :, 0] = np.clip(img[:, :, 0] + 80, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + 40, 0, 255)
    # Add fake vessels
    cv2.line(img, (cx, 0), (cx, H), (40, 80, 30), 4)
    cv2.line(img, (0, cy), (W, cy), (35, 75, 28), 3)
    # Add fake bright spots (exudates)
    cv2.circle(img, (cx // 2, cy // 2), 12, (220, 220, 180), -1)
    cv2.circle(img, (cx + 30, cy - 20), 8, (200, 200, 160), -1)
    return img


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

tests_passed = 0
tests_failed = 0


def run_test(name, fn):
    global tests_passed, tests_failed
    try:
        fn()
        print(f"  {PASS}  {name}")
        tests_passed += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"          {e}")
        tests_failed += 1


# ── Test: validate_image ────────────────────────────────────────────

def test_validate_valid():
    img = make_fake_fundus()
    ok, reason = validate_image(img)
    assert_true(ok, reason)


def test_validate_none():
    ok, reason = validate_image(None)
    assert_true(not ok, "None image should fail validation")


def test_validate_too_small():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    ok, reason = validate_image(img)
    assert_true(not ok, "Too-small image should fail")


def test_validate_wrong_channels():
    img = np.zeros((200, 256), dtype=np.uint8)
    ok, reason = validate_image(img)
    assert_true(not ok, "Grayscale image should fail 3-channel check")


# ── Test: preprocessing ──────────────────────────────────────────────

def test_preprocess_vessels_output_keys():
    img = make_fake_fundus()
    out = preprocess_for_vessels(img)
    for key in ("green", "clahe", "corrected"):
        assert_true(key in out, f"Key '{key}' missing from vessel preprocess output")


def test_preprocess_vessels_shapes():
    img = make_fake_fundus(200, 256)
    out = preprocess_for_vessels(img)
    assert_equal(out["green"].shape,     (200, 256), "green channel shape")
    assert_equal(out["clahe"].shape,     (200, 256), "clahe shape")
    assert_equal(out["corrected"].shape, (200, 256), "corrected shape")


def test_preprocess_exudate_output():
    img = make_fake_fundus()
    out = preprocess_for_exudates(img)
    assert_true("L" in out and "L_clahe" in out, "L and L_clahe must be present")
    assert_equal(out["L"].dtype, np.uint8, "L channel must be uint8")


def test_preprocess_clahe_dtype():
    img = make_fake_fundus()
    out = preprocess_for_vessels(img)
    assert_equal(out["corrected"].dtype, np.uint8, "corrected must be uint8")


# ── Test: vessel segmentation ────────────────────────────────────────

def test_vessel_gabor_response_shape():
    img = make_fake_fundus()
    out_prep = preprocess_for_vessels(img)
    out_seg  = segment_vessels(out_prep["corrected"])
    assert_equal(out_seg["gabor_response"].shape, (200, 256), "Gabor shape")
    assert_equal(out_seg["gabor_response"].dtype, np.float32, "Gabor dtype")


def test_vessel_mask_values():
    img = make_fake_fundus()
    out_prep = preprocess_for_vessels(img)
    out_seg  = segment_vessels(out_prep["corrected"])
    unique   = set(np.unique(out_seg["vessel_mask"]).tolist())
    assert_true(unique.issubset({0, 255}),
                f"vessel_mask must be 0/255 only, got {unique}")


def test_vessel_mask_not_all_zero():
    img = make_fake_fundus()
    out_prep = preprocess_for_vessels(img)
    out_seg  = segment_vessels(out_prep["corrected"])
    assert_true(np.sum(out_seg["vessel_mask"] > 0) > 0,
                "Vessel mask must have some positive pixels")


def test_fov_estimate_shape():
    img = make_fake_fundus()
    fov = estimate_fov(img)
    assert_equal(fov.shape, (200, 256), "FOV mask shape")
    unique = set(np.unique(fov).tolist())
    assert_true(unique.issubset({0, 255}), f"FOV mask must be 0/255, got {unique}")


# ── Test: exudate detection ──────────────────────────────────────────

def test_exudate_mask_values():
    img = make_fake_fundus()
    out_prep = preprocess_for_exudates(img)
    out_ex   = detect_exudates(out_prep["L_clahe"])
    unique   = set(np.unique(out_ex["candidate_mask"]).tolist())
    assert_true(unique.issubset({0, 255}),
                f"candidate_mask must be 0/255, got {unique}")


def test_exudate_regions_have_area():
    img = make_fake_fundus()
    out_prep = preprocess_for_exudates(img)
    out_ex   = detect_exudates(out_prep["L_clahe"])
    for r in out_ex["regions"]:
        assert_true(r["area"] > 0, "All regions must have positive area")


def test_exudate_tophat_shape():
    img = make_fake_fundus()
    out_prep = preprocess_for_exudates(img)
    out_ex   = detect_exudates(out_prep["L_clahe"])
    assert_equal(out_ex["tophat"].shape, (200, 256), "Top-hat shape")


# ── Test: evaluation metrics ─────────────────────────────────────────

def test_metrics_perfect():
    """Identical prediction and ground truth → Dice=1."""
    gt   = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 255
    pred = gt.copy()
    m    = calculate_metrics(pred, gt)
    assert abs(m["dice"] - 1.0) < 1e-4, f"Expected Dice=1.0, got {m['dice']}"


def test_metrics_zero_pred():
    """All-zero prediction → Dice near 0."""
    gt   = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 255
    pred = np.zeros_like(gt)
    m    = calculate_metrics(pred, gt)
    assert m["dice"] < 0.01, f"Expected Dice≈0, got {m['dice']}"


def test_metrics_no_nan():
    """Metrics must not contain NaN or Inf."""
    img = make_fake_fundus()
    result = analyze_retina(img)
    gt     = np.zeros(img.shape[:2], dtype=np.uint8)
    gt[50:80, 50:80] = 255
    m = calculate_metrics(result["vessel_mask"], gt)
    for k, v in m.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Metric '{k}' is {v} (not finite)"


def test_sanity_check_identical():
    """Sanity check flags identical arrays."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[5:15, 5:15] = 255
    warns = sanity_check(gt, gt)
    assert len(warns) > 0, "Sanity check should warn about identical pred/gt"


def test_metrics_shape_mismatch():
    """calculate_metrics should raise on shape mismatch."""
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt   = np.zeros((60, 60), dtype=np.uint8)
    try:
        calculate_metrics(pred, gt)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Test: overlay ────────────────────────────────────────────────────

def test_overlay_shape():
    img         = make_fake_fundus()
    vessel_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    exudate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    vessel_mask[50:80, 50:80]   = 255
    exudate_mask[100:120, 80:110] = 255
    overlay = make_overlay(img, vessel_mask, exudate_mask)
    assert_equal(overlay.shape, img.shape, "Overlay must have same shape as input")


def test_overlay_dtype():
    img         = make_fake_fundus()
    vessel_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    exudate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    overlay = make_overlay(img, vessel_mask, exudate_mask)
    assert_equal(overlay.dtype, np.uint8, "Overlay must be uint8")


# ── Test: full pipeline ──────────────────────────────────────────────

def test_analyze_retina_keys():
    img    = make_fake_fundus()
    result = analyze_retina(img)
    for key in ("vessel_mask", "exudate_mask", "overlay", "regions",
                 "vessel_density", "region_stats", "fov_mask"):
        assert_true(key in result, f"Key '{key}' missing from analyze_retina output")


def test_analyze_retina_vessel_density_range():
    img    = make_fake_fundus()
    result = analyze_retina(img)
    vd = result["vessel_density"]
    assert_true(0.0 <= vd <= 100.0,
                f"vessel_density must be in [0, 100], got {vd}")


def test_analyze_retina_overlay_same_shape():
    img    = make_fake_fundus()
    result = analyze_retina(img)
    assert_equal(result["overlay"].shape, img.shape, "Overlay shape")


# ── Test: PDF report ─────────────────────────────────────────────────

def test_report_creates_file():
    img    = make_fake_fundus()
    result = analyze_retina(img)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_report.pdf"
        generate_report(result, out_path, image_id="test_001")
        assert_true(out_path.exists(), "PDF report file must exist after generation")
        assert_true(out_path.stat().st_size > 1000,
                    f"PDF file too small: {out_path.stat().st_size} bytes")


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Retinal DIP — Unit Test Suite")
    print("="*60 + "\n")

    test_cases = [
        # validate_image
        ("validate valid image",          test_validate_valid),
        ("validate None image",           test_validate_none),
        ("validate too-small image",      test_validate_too_small),
        ("validate wrong channels",       test_validate_wrong_channels),
        # preprocessing
        ("preprocess vessels output keys",test_preprocess_vessels_output_keys),
        ("preprocess vessels shapes",     test_preprocess_vessels_shapes),
        ("preprocess exudate output",     test_preprocess_exudate_output),
        ("preprocess CLAHE dtype",        test_preprocess_clahe_dtype),
        # vessels
        ("Gabor response shape",          test_vessel_gabor_response_shape),
        ("vessel mask values 0/255",      test_vessel_mask_values),
        ("vessel mask not all zero",      test_vessel_mask_not_all_zero),
        ("FOV estimate shape",            test_fov_estimate_shape),
        # exudates
        ("exudate mask values 0/255",     test_exudate_mask_values),
        ("exudate regions have area",     test_exudate_regions_have_area),
        ("top-hat shape",                 test_exudate_tophat_shape),
        # evaluation
        ("metrics perfect prediction",    test_metrics_perfect),
        ("metrics zero prediction",       test_metrics_zero_pred),
        ("metrics no NaN",                test_metrics_no_nan),
        ("sanity check identical",        test_sanity_check_identical),
        ("metrics shape mismatch error",  test_metrics_shape_mismatch),
        # overlay
        ("overlay shape",                 test_overlay_shape),
        ("overlay dtype",                 test_overlay_dtype),
        # full pipeline
        ("analyze_retina output keys",    test_analyze_retina_keys),
        ("vessel density in range",       test_analyze_retina_vessel_density_range),
        ("overlay same shape as input",   test_analyze_retina_overlay_same_shape),
        # report
        ("PDF report creates file",       test_report_creates_file),
    ]

    print("Running tests …\n")
    for name, fn in test_cases:
        run_test(name, fn)

    print("\n" + "="*60)
    print(f"  Results: {tests_passed} passed, {tests_failed} failed")
    print("="*60 + "\n")

    if tests_failed > 0:
        sys.exit(1)
