"""
src/__init__.py
Retinal Image Analysis Platform
"""
from .preprocessing import preprocess_for_vessels, preprocess_for_exudates
from .vessels import segment_vessels
from .exudates import detect_exudates
from .evaluation import calculate_metrics
from .visualization import make_overlay, save_pipeline_figure
from .report import generate_report
from .pipeline import analyze_retina

__all__ = [
    "preprocess_for_vessels",
    "preprocess_for_exudates",
    "segment_vessels",
    "detect_exudates",
    "calculate_metrics",
    "make_overlay",
    "save_pipeline_figure",
    "generate_report",
    "analyze_retina",
]
