"""
api.py — FastAPI backend for the Retinal Image Analysis Platform.

Replaces the Streamlit app.py with a proper REST API.
Next.js frontend calls these endpoints.

Endpoints:
    POST /analyze   — full pipeline analysis
    POST /evaluate  — metric evaluation against GT mask
    POST /report    — generate PDF report (returns binary)
    GET  /health    — health check

Run with:
    uvicorn api:app --reload --port 8000
"""

import base64
import io
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ── Project modules ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import analyze_retina, validate_image
from src.evaluation import calculate_metrics, sanity_check
from src.report import generate_report

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Retinal Image Analysis API",
    description="Classical DIP pipeline: vessel segmentation + exudate detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────

def _decode_image_rgb(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded file to an RGB uint8 numpy array."""
    raw = np.frombuffer(upload.file.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {upload.filename}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_mask(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded mask to a grayscale uint8 numpy array."""
    raw = np.frombuffer(upload.file.read(), np.uint8)
    gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode mask: {upload.filename}")
    return (gray > 127).astype(np.uint8) * 255


def _arr_to_b64(arr: Optional[np.ndarray]) -> str:
    """Encode a numpy image (grayscale or RGB) to a PNG base64 string."""
    if arr is None:
        return ""
    if arr.dtype != np.uint8:
        # Normalise float arrays to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        encode_input = arr
    else:
        encode_input = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", encode_input)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_result_images_b64(image_map: dict[str, Optional[np.ndarray]]) -> dict[str, str]:
    """Encode multiple result images to PNG base64 strings in parallel."""
    keys = list(image_map.keys())
    workers = min(len(keys), max(1, os.cpu_count() or 2), 4)
    if workers <= 1:
        return {k: _arr_to_b64(image_map[k]) for k in keys}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        encoded = list(pool.map(_arr_to_b64, (image_map[k] for k in keys)))
    return dict(zip(keys, encoded))


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze")
def analyze(
    image: UploadFile = File(...),
    fov_mask: Optional[UploadFile] = File(None),
    disc_mask: Optional[UploadFile] = File(None),
    # Gabor params
    gabor_sigma: float = Form(4.0),
    gabor_lambda: float = Form(10.0),
    gabor_gamma: float = Form(0.5),
    n_orientations: int = Form(12),
    vessel_morph: int = Form(3),
    # Exudate params
    tophat_size: int = Form(15),
    exudate_morph: int = Form(3),
    min_area: int = Form(20),
    # Preprocessing params
    clahe_clip: float = Form(2.0),
    bg_sigma: float = Form(15.0),
):
    """
    Run the full retinal analysis pipeline on an uploaded fundus image.

    Returns all masks, overlay and intermediate images encoded as PNG base64,
    plus quantitative measurements.
    """
    img_rgb = _decode_image_rgb(image)
    ok, reason = validate_image(img_rgb)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Invalid image: {reason}")

    fov_arr = _decode_mask(fov_mask) if fov_mask and fov_mask.filename else None
    disc_arr = _decode_mask(disc_mask) if disc_mask and disc_mask.filename else None

    try:
        result = analyze_retina(
            img_rgb,
            fov_mask=fov_arr,
            optic_disc_mask=disc_arr,
            gabor_sigma=gabor_sigma,
            gabor_lambda=gabor_lambda,
            gabor_gamma=gabor_gamma,
            n_orientations=n_orientations,
            vessel_morph_size=vessel_morph,
            clahe_clip=clahe_clip,
            bg_sigma=bg_sigma,
            tophat_kernel_size=tophat_size,
            exudate_morph_size=exudate_morph,
            min_area=min_area,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Serialise regions (convert numpy types to Python native)
    regions_out = []
    for r in result["regions"]:
        regions_out.append({
            "id": int(r["id"]),
            "area": int(r["area"]),
            "centroid": [float(r["centroid"][0]), float(r["centroid"][1])],
            "bbox": [int(x) for x in r["bbox"]],
        })

    encoded_images = _encode_result_images_b64({
        "original_b64":      result["original"],
        "vessel_mask_b64":   result["vessel_mask"],
        "exudate_mask_b64":  result["exudate_mask"],
        "overlay_b64":       result["overlay"],
        "green_b64":         result.get("green"),
        "clahe_b64":         result.get("clahe"),
        "corrected_b64":     result.get("corrected"),
        "L_b64":             result.get("L"),
        "L_clahe_b64":       result.get("L_clahe"),
        "response_norm_b64": result.get("response_norm"),
        "tophat_b64":        result.get("tophat"),
    })

    return {
        "vessel_density": float(result["vessel_density"]),
        "region_stats": {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
            for k, v in result["region_stats"].items()
        },
        "regions": regions_out,
        # Images as base64 PNG
        **encoded_images,
    }


@app.post("/evaluate")
def evaluate(
    pred_b64: str = Form(...),
    gt_mask: UploadFile = File(...),
    mask_type: str = Form("vessel"),
):
    """
    Evaluate a predicted mask against a ground-truth mask.
    pred_b64 is a PNG base64 string (from /analyze response).
    """
    # Decode prediction from base64
    pred_bytes = base64.b64decode(pred_b64)
    pred_arr_raw = np.frombuffer(pred_bytes, np.uint8)
    pred_img = cv2.imdecode(pred_arr_raw, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        raise HTTPException(status_code=400, detail="Cannot decode predicted mask.")

    gt_arr = _decode_mask(gt_mask)

    # Resize GT to match pred if needed
    if gt_arr.shape != pred_img.shape:
        gt_arr = cv2.resize(gt_arr, (pred_img.shape[1], pred_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    warnings = sanity_check(pred_img, gt_arr)
    metrics = calculate_metrics(pred_img, gt_arr)
    metrics["warnings"] = warnings

    return metrics


@app.post("/report")
def report(payload: dict):
    """
    Generate a PDF report and return it as binary.
    Expects JSON body: { "result": {...analyze output...}, "image_id": "..." }
    """
    image_id = payload.get("image_id", "retinal_image")
    result_data = payload.get("result", {})

    # Decode base64 images back to numpy arrays for the report generator
    def b64_to_arr(b64: str, gray: bool = False) -> Optional[np.ndarray]:
        if not b64:
            return None
        raw = base64.b64decode(b64)
        arr_raw = np.frombuffer(raw, np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(arr_raw, flag)
        if img is not None and not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    reconstructed = {
        "original":     b64_to_arr(result_data.get("original_b64", "")),
        "vessel_mask":  b64_to_arr(result_data.get("vessel_mask_b64", ""), gray=True),
        "exudate_mask": b64_to_arr(result_data.get("exudate_mask_b64", ""), gray=True),
        "overlay":      b64_to_arr(result_data.get("overlay_b64", "")),
        "vessel_density": result_data.get("vessel_density", 0.0),
        "region_stats":   result_data.get("region_stats", {}),
        "regions":        result_data.get("regions", []),
    }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="retinal_") as tmp:
        tmp_path = tmp.name

    try:
        generate_report(reconstructed, output_path=tmp_path, image_id=image_id)
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="retinal_analysis_{image_id}.pdf"'},
    )
