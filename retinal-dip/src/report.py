"""
src/report.py

Stage J — PDF report generation using ReportLab.

The report includes:
    1. Title page with disclaimer
    2. Image ID and dimensions
    3. Original fundus image
    4. Preprocessing images
    5. Vessel segmentation mask
    6. Candidate exudate mask
    7. Combined overlay
    8. Candidate-region table
    9. Quantitative measurements (vessel density, region stats)
    10. Evaluation metrics (when ground truth is provided)
    11. Disclaimer footer
"""

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _np_to_rl_image(arr: np.ndarray, max_width: float, max_height: float) -> RLImage:
    """
    Convert a NumPy RGB/gray image to a ReportLab Image flowable.
    Preserves aspect ratio within the given bounds.
    """
    if arr is None:
        return Spacer(1, 1)

    if arr.ndim == 2:                       # grayscale → RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Encode as PNG in memory
    is_ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not is_ok:
        return Spacer(1, 1)

    img_bytes = io.BytesIO(buf.tobytes())
    h, w = arr.shape[:2]
    aspect = w / h

    # Scale to fit within the bounding box
    if (max_width / aspect) <= max_height:
        final_w = max_width
        final_h = max_width / aspect
    else:
        final_h = max_height
        final_w = max_height * aspect

    return RLImage(img_bytes, width=final_w, height=final_h)


def _style_table(data: list[list], col_widths: list) -> Table:
    """Build a styled Table flowable from a 2-D list of strings."""
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F2F3F4"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_report(result: dict,
                     output_path: str | Path,
                     image_id: str = "Unknown",
                     metrics: dict | None = None) -> Path:
    """
    Generate a PDF report from the analysis result dictionary.

    Args:
        result:      Output dict from analyze_retina().
        output_path: Where to save the PDF file.
        image_id:    Identifier string for the analysed image.
        metrics:     Optional evaluation metrics dict from calculate_metrics().

    Returns:
        Path to the saved PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor("#1A252F"),
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#7F8C8D"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#2C3E50"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#E74C3C"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    story = []
    page_w = A4[0] - 4 * cm   # usable width
    img_w  = page_w * 0.45
    img_h  = 6 * cm

    # ── Title ──────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("RETINAL IMAGE ANALYSIS REPORT", title_style))
    story.append(Paragraph(
        f"Image ID: <b>{image_id}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#2C3E50")))

    # Disclaimer
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "⚠ RESEARCH / DECISION-SUPPORT PROTOTYPE — NOT A MEDICAL DIAGNOSIS. "
        "Candidate regions detected by classical DIP algorithms. "
        "Clinical review by a qualified ophthalmologist is required.",
        disclaimer_style,
    ))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E74C3C")))

    # ── Image dimensions ───────────────────────────────────────────
    original = result.get("original")
    if original is not None:
        h, w = original.shape[:2]
        story.append(Paragraph(
            f"Image dimensions: <b>{w} × {h} pixels</b>", body_style
        ))

    # ── Section: Pipeline images ───────────────────────────────────
    story.append(Paragraph("1. Pipeline Images", heading_style))

    def _row(*keys_titles):
        """Build a 2-column image row."""
        cells = []
        for key, title in keys_titles:
            arr = result.get(key)
            if arr is not None:
                cells.append([
                    Paragraph(f"<b>{title}</b>", body_style),
                    _np_to_rl_image(arr, img_w, img_h),
                ])
        return cells

    img_pairs = [
        ("original",     "Original Fundus Image"),
        ("corrected",    "Preprocessed (Illumination-Corrected)"),
        ("vessel_mask",  "Blood-Vessel Mask"),
        ("exudate_mask", "Candidate Exudate Mask"),
        ("overlay",      "Combined Overlay"),
        ("tophat",       "Top-Hat Response"),
    ]

    for i in range(0, len(img_pairs), 2):
        row_data = []
        for key, title in img_pairs[i:i + 2]:
            arr = result.get(key)
            if arr is not None:
                row_data.append([
                    Paragraph(f"<b>{title}</b>", body_style),
                    _np_to_rl_image(arr, img_w, img_h),
                ])
            else:
                row_data.append(["", ""])

        if len(row_data) == 1:
            row_data.append(["", ""])

        tbl = Table(
            [[row_data[0][1], row_data[1][1]],
             [row_data[0][0], row_data[1][0]]],
            colWidths=[page_w / 2, page_w / 2],
        )
        tbl.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # ── Section: Quantitative measurements ────────────────────────
    story.append(Paragraph("2. Quantitative Measurements", heading_style))

    vessel_density = result.get("vessel_density", 0.0)
    regions        = result.get("regions", [])
    num_candidates = len(regions)
    total_area     = sum(r["area"] for r in regions)
    largest_area   = max((r["area"] for r in regions), default=0)

    meas_data = [
        ["Metric", "Value"],
        ["Vessel density (% of FOV)", f"{vessel_density:.2f}%"],
        ["Candidate regions detected", str(num_candidates)],
        ["Total candidate area (px)", str(total_area)],
        ["Largest candidate area (px)", str(largest_area)],
    ]
    story.append(_style_table(meas_data, [page_w * 0.6, page_w * 0.4]))
    story.append(Spacer(1, 0.4 * cm))

    # ── Section: Candidate region table ───────────────────────────
    if regions:
        story.append(Paragraph("3. Candidate Region Table", heading_style))
        reg_header = ["ID", "Area (px)", "Centroid (x, y)", "Bounding Box"]
        reg_rows   = [reg_header]
        for r in regions[:50]:                      # cap at 50 rows in PDF
            cx, cy = r["centroid"]
            x, y, w, h = r["bbox"]
            reg_rows.append([
                str(r["id"]),
                str(r["area"]),
                f"({cx:.1f}, {cy:.1f})",
                f"[{x}, {y}, {w}×{h}]",
            ])
        col_w = page_w / 4
        story.append(_style_table(reg_rows, [col_w] * 4))
        story.append(Spacer(1, 0.4 * cm))

    # ── Section: Evaluation metrics ────────────────────────────────
    if metrics:
        story.append(Paragraph("4. Evaluation Metrics", heading_style))
        met_data = [
            ["Metric", "Value", "Description"],
            ["Dice",        f"{metrics.get('dice', 0):.4f}",
             "2·TP / (2·TP + FP + FN)"],
            ["IoU",         f"{metrics.get('iou', 0):.4f}",
             "TP / (TP + FP + FN)"],
            ["Sensitivity", f"{metrics.get('sensitivity', 0):.4f}",
             "TP / (TP + FN) — recall"],
            ["Specificity", f"{metrics.get('specificity', 0):.4f}",
             "TN / (TN + FP)"],
            ["Precision",   f"{metrics.get('precision', 0):.4f}",
             "TP / (TP + FP)"],
            ["F1",          f"{metrics.get('f1', 0):.4f}",
             "Harmonic mean of precision & recall"],
        ]
        story.append(_style_table(
            met_data,
            [page_w * 0.25, page_w * 0.2, page_w * 0.55],
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ── Section: Limitations ──────────────────────────────────────
    story.append(Paragraph("5. Limitations", heading_style))
    limitations = [
        "• This system uses classical DIP algorithms; it is not trained "
        "or validated as a clinical tool.",
        "• Top-hat morphology detects bright regions, not semantic lesion "
        "categories. False positives (e.g., optic disc, image artefacts) are expected.",
        "• Results are sensitive to image quality, acquisition conditions, "
        "and algorithm parameter choices.",
        "• Gabor-based vessel segmentation may miss fine capillaries or "
        "produce false positives in textured regions.",
        "• Ground-truth evaluation is only available for standard dataset images.",
        "• This report does not constitute a medical diagnosis.",
    ]
    for line in limitations:
        story.append(Paragraph(line, body_style))

    # ── Footer disclaimer ──────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#E74C3C")))
    story.append(Paragraph(
        "RESEARCH / DECISION-SUPPORT PROTOTYPE | NOT A MEDICAL DIAGNOSIS | "
        "Retinal DIP Project — Classical Image Processing",
        disclaimer_style,
    ))

    doc.build(story)
    return output_path
