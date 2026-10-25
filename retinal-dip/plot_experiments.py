"""
plot_experiments.py

Reads experiments/results.csv and generates publication-quality figures:
    experiments/figures/
        01_dice_comparison.png     — bar chart: mean Dice per experiment E1–E5
        02_sensitivity_specificity.png — sens/spec trade-off per experiment
        03_per_image_dice.png      — individual image Dice scores (grouped)
        04_precision_recall.png    — precision vs recall per experiment
        05_metric_heatmap.png      — heatmap of all metrics across experiments

Usage:
    python plot_experiments.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
})

EXP_COLORS = {
    "E1": "#ef4444",   # red
    "E2": "#f59e0b",   # amber
    "E3": "#3b82f6",   # blue
    "E4": "#8b5cf6",   # purple
    "E5": "#10b981",   # emerald (full pipeline)
}

EXP_LABELS = {
    "E1": "E1: Baseline",
    "E2": "E2: +CLAHE",
    "E3": "E3: +Gabor",
    "E4": "E4: +Morph",
    "E5": "E5: Full",
}

SAVE_DIR = Path("experiments/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

csv_path = Path("experiments/results.csv")
if not csv_path.exists():
    print("No results.csv found. Run: python main.py --experiment")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

exp_order = ["E1", "E2", "E3", "E4", "E5"]
summary   = df.groupby("experiment_id").mean(numeric_only=True).reindex(exp_order)


# ─────────────────────────────────────────────
# Fig 1 — Mean Dice comparison
# ─────────────────────────────────────────────

def plot_dice_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor("#161b22")

    x      = np.arange(len(exp_order))
    colors = [EXP_COLORS[e] for e in exp_order]
    bars   = ax.bar(x, summary["vessel_dice"], color=colors,
                    width=0.6, edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, summary["vessel_dice"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS[e] for e in exp_order], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Dice Score (5 images)", fontsize=12)
    ax.set_title("Vessel Segmentation — Ablation Study (E1 → E5)\nMean Dice Score per Pipeline Stage",
                 fontsize=13, fontweight="bold", pad=16)
    ax.grid(axis="y", zorder=0)
    ax.axhline(0.7, color="#38bdf8", linewidth=1, linestyle="--", alpha=0.6,
               label="Dice = 0.70 (reference)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "01_dice_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_dice_comparison.png")


# ─────────────────────────────────────────────
# Fig 2 — Sensitivity / Specificity trade-off
# ─────────────────────────────────────────────

def plot_sens_spec():
    fig, ax = plt.subplots(figsize=(8, 6))

    for eid in exp_order:
        sub   = summary.loc[eid]
        color = EXP_COLORS[eid]
        label = EXP_LABELS[eid]
        ax.scatter(sub["vessel_sensitivity"], sub["vessel_specificity"],
                   s=180, color=color, label=label, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(label,
                    (sub["vessel_sensitivity"], sub["vessel_specificity"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color)

    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.94, 1.005)
    ax.set_xlabel("Sensitivity (Recall)", fontsize=12)
    ax.set_ylabel("Specificity", fontsize=12)
    ax.set_title("Sensitivity vs Specificity Trade-off\nVessel Segmentation (mean over 5 images)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True)
    ax.legend(fontsize=10, loc="lower left")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "02_sensitivity_specificity.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_sensitivity_specificity.png")


# ─────────────────────────────────────────────
# Fig 3 — Per-image Dice scores grouped by experiment
# ─────────────────────────────────────────────

def plot_per_image_dice():
    images = sorted(df["image"].unique())
    n_img  = len(images)
    n_exp  = len(exp_order)
    x      = np.arange(n_img)
    width  = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, eid in enumerate(exp_order):
        sub    = df[df["experiment_id"] == eid].set_index("image")
        vals   = [sub.loc[img, "vessel_dice"] if img in sub.index else 0.0
                  for img in images]
        offset = (i - n_exp / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, color=EXP_COLORS[eid],
               label=EXP_LABELS[eid], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([img.replace("_training", "") for img in images], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_xlabel("Image", fontsize=12)
    ax.set_title("Per-Image Vessel Dice Scores by Experiment Stage",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=10, ncol=5)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "03_per_image_dice.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_per_image_dice.png")


# ─────────────────────────────────────────────
# Fig 4 — Precision vs Recall
# ─────────────────────────────────────────────

def plot_precision_recall():
    fig, ax = plt.subplots(figsize=(8, 6))

    for eid in exp_order:
        sub   = summary.loc[eid]
        color = EXP_COLORS[eid]
        ax.scatter(sub["vessel_sensitivity"], sub["vessel_precision"],
                   s=180, color=color, label=EXP_LABELS[eid], zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(EXP_LABELS[eid],
                    (sub["vessel_sensitivity"], sub["vessel_precision"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=color)

    # F1 iso-curves
    for f1_target in [0.5, 0.6, 0.7, 0.8]:
        r_vals = np.linspace(0.01, 1.0, 200)
        p_vals = (f1_target * r_vals) / (2 * r_vals - f1_target + 1e-8)
        mask   = (p_vals >= 0) & (p_vals <= 1)
        ax.plot(r_vals[mask], p_vals[mask], "--", alpha=0.3, color="#94a3b8",
                linewidth=1)
        ax.text(0.97, f1_target / (2 - f1_target) + 0.015,
                f"F₁={f1_target}", fontsize=8, color="#64748b")

    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve (mean per experiment)\nWith F₁ iso-lines",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "04_precision_recall.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_precision_recall.png")


# ─────────────────────────────────────────────
# Fig 5 — Metric heatmap
# ─────────────────────────────────────────────

def plot_metric_heatmap():
    metrics = ["vessel_dice", "vessel_iou", "vessel_sensitivity",
               "vessel_specificity", "vessel_precision", "vessel_f1"]
    labels  = ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "F1"]

    data = summary[metrics].values  # shape (5, 6)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(len(exp_order)))
    ax.set_yticklabels([EXP_LABELS[e] for e in exp_order], fontsize=12)

    for i in range(len(exp_order)):
        for j in range(len(labels)):
            val = data[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, color="black" if val > 0.6 else "white",
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.ax.tick_params(colors="#c9d1d9")
    cbar.set_label("Score", color="#c9d1d9", fontsize=11)

    ax.set_title("Metric Heatmap — Vessel Segmentation (Mean over 5 images)",
                 fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "05_metric_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved 05_metric_heatmap.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating experiment figures …\n")
    plot_dice_comparison()
    plot_sens_spec()
    plot_per_image_dice()
    plot_precision_recall()
    plot_metric_heatmap()
    print(f"\n✅  All figures saved to {SAVE_DIR}/")
