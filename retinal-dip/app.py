"""
app.py — Retinal Image Analysis Platform (Streamlit UI)

Stage I: Interactive web interface.

Screens:
    1. Upload & Preview
    2. Analysis Results (vessel mask, exudate mask, overlay)
    3. Quantitative Measurements
    4. Evaluation (when ground truth is uploaded)
    5. Research / Debug view (intermediate pipeline images)
    6. PDF Report download

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Import project modules ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import analyze_retina, validate_image
from src.evaluation import calculate_metrics, sanity_check
from src.report import generate_report


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Retinal Image Analysis Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ──────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ───────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #1c2128 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Main header card ─────────────────────────────── */
.header-card {
    background: linear-gradient(135deg, #1a2a3a 0%, #0d2137 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(56, 189, 248, 0.1);
}

.header-card h1 {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
}

.header-card p {
    color: rgba(255,255,255,0.55);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Disclaimer banner ────────────────────────────── */
.disclaimer {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 20px;
    color: #fca5a5;
    font-size: 0.82rem;
    font-weight: 500;
    text-align: center;
}

/* ── Metric cards ─────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e2d3d, #162130);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.25s ease;
}

.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
    transform: translateY(-2px);
}

.metric-card .label {
    color: rgba(255,255,255,0.5);
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.metric-card .value {
    color: #38bdf8;
    font-size: 1.8rem;
    font-weight: 700;
}

/* ── Section headers ──────────────────────────────── */
.section-header {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 8px 0 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
}

/* ── Result image containers ──────────────────────── */
.img-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}

.img-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.80rem;
    font-weight: 500;
    margin-bottom: 8px;
    text-align: center;
}

/* ── Success / info pills ─────────────────────────── */
.pill-green {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #4ade80;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-yellow {
    display: inline-block;
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
    color: #facc15;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-blue {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.4);
    color: #38bdf8;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Stmetric overrides ───────────────────────────── */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.78rem !important;
}

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 28px;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(56, 189, 248, 0.4);
}

/* ── Tables ───────────────────────────────────────── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
}

/* ── Progress bar ─────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
}

/* ── Slider ───────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — Algorithm parameters
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Algorithm Parameters")
    st.markdown("---")

    st.markdown("### 🩸 Vessel Branch (Gabor)")
    gabor_sigma  = st.slider("Gabor σ (sigma)",   1.0, 8.0, 4.0, 0.5,
                              help="Gaussian envelope spread")
    gabor_lambda = st.slider("Gabor λ (lambda)",  4.0, 16.0, 10.0, 1.0,
                              help="Sinusoidal wavelength")
    gabor_gamma  = st.slider("Gabor γ (gamma)",   0.3, 1.2, 0.5, 0.1,
                              help="Aspect ratio (elongation)")
    n_orientations = st.select_slider("Orientations", [6, 8, 12, 16, 18], 12,
                                       help="Number of Gabor filter angles")
    vessel_morph = st.slider("Vessel morph kernel", 1, 7, 3, 2,
                              help="Morphological cleanup kernel size")

    st.markdown("---")
    st.markdown("### 🟡 Exudate Branch")
    tophat_size  = st.slider("Top-hat kernel", 7, 31, 15, 2,
                              help="Structuring element size for white top-hat")
    exudate_morph = st.slider("Exudate morph kernel", 1, 7, 3, 2)
    min_area     = st.slider("Min candidate area (px)", 5, 100, 20, 5,
                              help="Discard candidates smaller than this")

    st.markdown("---")
    st.markdown("### 🔧 Preprocessing")
    clahe_clip   = st.slider("CLAHE clip limit", 0.5, 5.0, 2.0, 0.5)
    bg_sigma     = st.slider("BG correction σ",  5.0, 30.0, 15.0, 1.0,
                              help="Gaussian sigma for background subtraction")

    st.markdown("---")
    st.caption("🔬 Research prototype | Not a medical diagnostic tool")


# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────

st.markdown("""
<div class="header-card">
    <h1>🩺 Retinal Image Analysis Platform</h1>
    <p>Classical Digital Image Processing · Blood Vessel Segmentation · Hard-Exudate Candidate Detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ RESEARCH / DECISION-SUPPORT PROTOTYPE — NOT A MEDICAL DIAGNOSIS.
    Candidate regions detected by classical DIP algorithms.
    Clinical review by a qualified ophthalmologist is required.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_upload, tab_results, tab_eval, tab_debug, tab_report = st.tabs([
    "📤 Upload & Analyze",
    "📊 Results",
    "🎯 Evaluation",
    "🔬 Debug / Research View",
    "📄 PDF Report",
])


# ─────────────────────────────────────────────
# Tab 1 — Upload & Analyze
# ─────────────────────────────────────────────

with tab_upload:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">Upload Retinal Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a retinal fundus photograph",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            key="main_image",
            help="Supports JPEG, PNG, TIFF formats",
        )

        if uploaded_file:
            raw  = np.frombuffer(uploaded_file.read(), np.uint8)
            bgr  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("❌ Could not decode the uploaded image.")
            else:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ok, reason = validate_image(image)
                if not ok:
                    st.error(f"❌ Invalid image: {reason}")
                else:
                    st.session_state["image"] = image
                    h, w = image.shape[:2]
                    st.markdown(f"""
                    <span class="pill-green">✓ Valid Image</span>
                    &nbsp; <span class="pill-blue">{w} × {h} px</span>
                    """, unsafe_allow_html=True)
                    st.image(image, caption="Uploaded Fundus Image",
                             use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Optional Masks</p>', unsafe_allow_html=True)

        st.markdown("**FOV Mask** *(optional — auto-estimated if omitted)*")
        fov_file = st.file_uploader(
            "FOV mask", type=["png", "bmp", "tif"],
            key="fov_mask_upload", label_visibility="collapsed",
        )

        st.markdown("**Optic-Disc Mask** *(optional — suppresses disc FPs in evaluation mode)*")
        disc_file = st.file_uploader(
            "Optic-disc mask", type=["png", "bmp", "tif"],
            key="disc_mask_upload", label_visibility="collapsed",
        )

        fov_mask_arr  = None
        disc_mask_arr = None

        if fov_file:
            fov_raw = np.frombuffer(fov_file.read(), np.uint8)
            fov_bgr = cv2.imdecode(fov_raw, cv2.IMREAD_GRAYSCALE)
            if fov_bgr is not None:
                fov_mask_arr = (fov_bgr > 127).astype(np.uint8) * 255
                st.image(fov_mask_arr, caption="FOV Mask", use_container_width=True)

        if disc_file:
            disc_raw = np.frombuffer(disc_file.read(), np.uint8)
            disc_bgr = cv2.imdecode(disc_raw, cv2.IMREAD_GRAYSCALE)
            if disc_bgr is not None:
                disc_mask_arr = (disc_bgr > 127).astype(np.uint8) * 255
                st.image(disc_mask_arr, caption="Optic-Disc Mask", use_container_width=True)

        st.session_state["fov_mask"]  = fov_mask_arr
        st.session_state["disc_mask"] = disc_mask_arr

        # ── Analyze button ────────────────────────────────────────
        st.markdown("---")
        analyze_btn = st.button("🔍 Analyze Retina", use_container_width=True,
                                 disabled="image" not in st.session_state)

        if analyze_btn and "image" in st.session_state:
            with st.status("🔄 Analysing retinal image …", expanded=True) as status:
                st.write("⚙️ Preprocessing (CLAHE + illumination correction) …")
                st.write("🔬 Running Gabor filter bank …")
                st.write("🩸 Segmenting blood vessels …")
                st.write("🟡 Running white top-hat exudate detector …")
                st.write("📐 Analysing candidate regions …")
                st.write("🖼️ Generating overlay …")

                try:
                    result = analyze_retina(
                        st.session_state["image"],
                        fov_mask=st.session_state.get("fov_mask"),
                        optic_disc_mask=st.session_state.get("disc_mask"),
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
                    st.session_state["result"] = result
                    status.update(label="✅ Analysis complete!", state="complete")
                    st.success("Analysis finished. View results in the **Results** tab.")
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(f"Error during analysis: {e}")


# ─────────────────────────────────────────────
# Tab 2 — Results
# ─────────────────────────────────────────────

with tab_results:
    if "result" not in st.session_state:
        st.info("👆 Upload an image and click **Analyze Retina** to see results.")
    else:
        result = st.session_state["result"]
        regions = result["regions"]

        # ── Quick stats row ───────────────────────────────────────
        st.markdown('<p class="section-header">Quantitative Measurements</p>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🩸 Vessel Density",
                       f"{result['vessel_density']:.2f}%",
                       help="Percentage of FOV pixels classified as vessel")
        with m2:
            st.metric("🟡 Candidates",
                       result["region_stats"]["count"],
                       help="Number of retained exudate candidate regions")
        with m3:
            st.metric("📐 Largest Region",
                       f"{result['region_stats']['largest_area']} px",
                       help="Area of the largest candidate region")
        with m4:
            st.metric("📊 Total Area",
                       f"{result['region_stats']['total_area']} px",
                       help="Combined area of all candidate regions")

        st.markdown("---")

        # ── Images row 1 ─────────────────────────────────────────
        st.markdown('<p class="section-header">Segmentation Results</p>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p class="img-label">🔴 Original Fundus</p>',
                        unsafe_allow_html=True)
            st.image(result["original"], use_container_width=True)
        with c2:
            st.markdown('<p class="img-label">🩸 Vessel Mask</p>',
                        unsafe_allow_html=True)
            st.image(result["vessel_mask"], use_container_width=True,
                     clamp=True)
        with c3:
            st.markdown('<p class="img-label">🟡 Exudate Candidate Mask</p>',
                        unsafe_allow_html=True)
            st.image(result["exudate_mask"], use_container_width=True,
                     clamp=True)

        st.markdown("---")

        # ── Combined overlay ──────────────────────────────────────
        st.markdown('<p class="section-header">Combined Overlay</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <p style="color:rgba(255,255,255,0.5); font-size:0.82rem; margin-bottom:10px;">
        🟢 <b>Green</b> = Detected vessels &nbsp;|&nbsp;
        🟡 <b>Yellow</b> = Candidate exudate regions
        </p>
        """, unsafe_allow_html=True)
        st.image(result["overlay"], use_container_width=True,
                 caption="Combined vessel + exudate candidate overlay")

        # ── Candidate region table ────────────────────────────────
        if regions:
            st.markdown("---")
            st.markdown('<p class="section-header">Candidate Region Table</p>',
                        unsafe_allow_html=True)
            df_regions = pd.DataFrame([{
                "ID":             r["id"],
                "Area (px)":      r["area"],
                "Centroid X":     f"{r['centroid'][0]:.1f}",
                "Centroid Y":     f"{r['centroid'][1]:.1f}",
                "BBox X":         r["bbox"][0],
                "BBox Y":         r["bbox"][1],
                "Width":          r["bbox"][2],
                "Height":         r["bbox"][3],
            } for r in regions])
            st.dataframe(df_regions, use_container_width=True, height=260)
            st.caption(
                "⚠️ Candidate regions only — not confirmed clinical lesions. "
                "Professional clinical review is required."
            )


# ─────────────────────────────────────────────
# Tab 3 — Evaluation
# ─────────────────────────────────────────────

with tab_eval:
    st.markdown('<p class="section-header">Ground-Truth Evaluation</p>',
                unsafe_allow_html=True)
    st.markdown("""
    Upload a ground-truth binary mask to evaluate prediction quality.
    Ground-truth masks are available in the **DRIVE** (vessel GT) and
    **IDRiD** (exudate GT) datasets.
    """)

    col_gt1, col_gt2 = st.columns(2)
    with col_gt1:
        st.markdown("**Vessel Ground Truth** (DRIVE training set)")
        vessel_gt_file = st.file_uploader(
            "Vessel GT mask", type=["png", "bmp", "tif"],
            key="vessel_gt",
        )
    with col_gt2:
        st.markdown("**Exudate Ground Truth** (IDRiD pixel-level subset)")
        exudate_gt_file = st.file_uploader(
            "Exudate GT mask", type=["png", "bmp", "tif"],
            key="exudate_gt",
        )

    if "result" not in st.session_state:
        st.warning("Run the analysis first (Upload & Analyze tab).")
    else:
        result = st.session_state["result"]

        if vessel_gt_file or exudate_gt_file:
            st.markdown("---")
            st.markdown("### 📊 Metric Results")

            for gt_file, mask_key, task_name in [
                (vessel_gt_file,  "vessel_mask",  "Vessel Segmentation"),
                (exudate_gt_file, "exudate_mask", "Exudate Candidate Detection"),
            ]:
                if gt_file is None:
                    continue

                gt_raw = np.frombuffer(gt_file.read(), np.uint8)
                gt_img = cv2.imdecode(gt_raw, cv2.IMREAD_GRAYSCALE)
                if gt_img is None:
                    st.error(f"Could not decode {task_name} GT mask.")
                    continue

                pred = result[mask_key]

                # Resize GT to match pred if needed
                if gt_img.shape != pred.shape:
                    gt_img = cv2.resize(gt_img, (pred.shape[1], pred.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    st.warning(f"GT mask resized to match prediction: {pred.shape}")

                # Sanity checks
                warns = sanity_check(pred, gt_img)
                for w in warns:
                    st.warning(w)

                fov = result.get("fov_mask") if mask_key == "vessel_mask" else None
                metrics = calculate_metrics(pred, gt_img, fov_mask=fov)

                st.markdown(f"#### {task_name}")
                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                for col, (key, label) in zip(
                    [mc1, mc2, mc3, mc4, mc5, mc6],
                    [("dice", "Dice"), ("iou", "IoU"),
                     ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                     ("precision", "Precision"), ("f1", "F1")],
                ):
                    with col:
                        val = metrics[key]
                        color = "#4ade80" if val > 0.7 else "#facc15" if val > 0.4 else "#f87171"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="label">{label}</div>
                            <div class="value" style="color:{color}">{val:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Store for report
                st.session_state[f"metrics_{mask_key}"] = metrics

                # Confusion matrix
                with st.expander("Confusion Matrix"):
                    cm_data = {
                        "":              ["Pred +", "Pred −"],
                        "GT +":          [metrics["tp"], metrics["fn"]],
                        "GT −":          [metrics["fp"], metrics["tn"]],
                    }
                    st.dataframe(pd.DataFrame(cm_data).set_index(""),
                                 use_container_width=True)

                st.markdown("")


# ─────────────────────────────────────────────
# Tab 4 — Debug / Research View
# ─────────────────────────────────────────────

with tab_debug:
    if "result" not in st.session_state:
        st.info("Run analysis first.")
    else:
        result = st.session_state["result"]
        st.markdown('<p class="section-header">Intermediate Pipeline Images</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        *These intermediate outputs are for research inspection only.
        They are not shown in the main results view.*
        """)

        debug_panels = [
            ("Green Channel",           result.get("green"),          "gray"),
            ("CLAHE-enhanced",          result.get("clahe"),          "gray"),
            ("Illumination-corrected",  result.get("corrected"),      "gray"),
            ("Lab L-channel",           result.get("L"),              "gray"),
            ("Lab L-channel + CLAHE",   result.get("L_clahe"),        "gray"),
            ("Gabor max-response",      result.get("response_norm"),  "gray"),
            ("Top-hat response",        result.get("tophat"),         "gray"),
            ("FOV mask",                result.get("fov_mask"),       "gray"),
        ]

        cols_per_row = 4
        for row_start in range(0, len(debug_panels), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, (title, img, cmap) in zip(
                cols, debug_panels[row_start:row_start + cols_per_row]
            ):
                with col:
                    if img is not None:
                        st.image(img, caption=title,
                                 use_container_width=True, clamp=True)
                    else:
                        st.markdown(f"*{title}: N/A*")

        # Gabor response colourised
        st.markdown("---")
        st.markdown('<p class="section-header">Gabor Response (Colorised)</p>',
                    unsafe_allow_html=True)
        if result.get("response_norm") is not None:
            colored = cv2.applyColorMap(result["response_norm"], cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            st.image(colored_rgb, use_container_width=True,
                     caption="Jet-colourised Gabor response (blue=low, red=high)")


# ─────────────────────────────────────────────
# Tab 5 — PDF Report
# ─────────────────────────────────────────────

with tab_report:
    st.markdown('<p class="section-header">PDF Report Generation</p>',
                unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("Run analysis first to generate a report.")
    else:
        result  = st.session_state["result"]
        img_id  = st.text_input("Image identifier (for report)", "retinal_image_001")
        include_vessel_metrics  = st.checkbox("Include vessel metrics in report",
                                               "metrics_vessel_mask" in st.session_state)
        include_exudate_metrics = st.checkbox("Include exudate metrics in report",
                                               "metrics_exudate_mask" in st.session_state)

        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF …"):
                metrics_to_report = None
                if include_vessel_metrics and "metrics_vessel_mask" in st.session_state:
                    metrics_to_report = st.session_state["metrics_vessel_mask"]
                elif include_exudate_metrics and "metrics_exudate_mask" in st.session_state:
                    metrics_to_report = st.session_state["metrics_exudate_mask"]

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False, prefix="retinal_report_"
                ) as tmp:
                    tmp_path = tmp.name

                try:
                    generate_report(
                        result,
                        output_path=tmp_path,
                        image_id=img_id,
                        metrics=metrics_to_report,
                    )
                    with open(tmp_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.success("✅ PDF report generated successfully!")
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"retinal_analysis_{img_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        st.markdown("---")
        st.markdown("### 📑 Report Contents")
        report_contents = [
            "✅ Project title and research disclaimer",
            "✅ Image identifier and dimensions",
            "✅ Original fundus photograph",
            "✅ Preprocessed image",
            "✅ Blood-vessel segmentation mask",
            "✅ Candidate exudate mask",
            "✅ Combined overlay image",
            "✅ Candidate-region table (area, centroid, bounding box)",
            "✅ Vessel density and region statistics",
            "✅ Evaluation metrics (when ground truth provided)",
            "✅ System limitations and non-diagnostic disclaimer",
        ]
        for item in report_contents:
            st.markdown(f"- {item}")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<p style="color:rgba(255,255,255,0.25); font-size:0.75rem; text-align:center;">
Retinal DIP Project · Classical Digital Image Processing ·
IPCV Academic Research Prototype · Not for clinical use.
</p>
""", unsafe_allow_html=True)
