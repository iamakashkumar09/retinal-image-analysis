# Retinal Image Analysis Platform

> **Retinal Blood Vessel Segmentation & Candidate Hard-Exudate Detection Using Classical Digital Image Processing**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)

---

## ⚠️ Disclaimer

**This is a research / decision-support prototype. It is NOT a medical diagnosis tool.**
All outputs are labelled "candidate regions." Clinical review by a qualified ophthalmologist is required.

---

## 📖 Abstract

This project implements a complete, explainable classical Digital Image Processing (DIP) pipeline for analysing retinal fundus photographs. It performs two parallel tasks:

1. **Blood-vessel segmentation** using a Gabor filter bank
2. **Hard-exudate candidate detection** using white top-hat morphology

Results are evaluated against annotated ground-truth masks using standard segmentation metrics (Dice, IoU, Sensitivity, Specificity, F1). An interactive Streamlit web interface and a downloadable PDF report are included.

---

## 🎯 Objectives

- Demonstrate explainable classical DIP for retinal image analysis
- Apply Gabor filtering and morphological operators to real-world images
- Evaluate results quantitatively using pixel-level segmentation metrics
- Build an interactive web application around the pipeline
- Clearly communicate the limitations of a non-clinical prototype

---

## 🗂️ Datasets

| Dataset | Task | Images | Source |
|---------|------|--------|--------|
| **DRIVE** | Vessel segmentation | 40 (768×584 px) | https://drive.grand-challenge.org/ |
| **IDRiD** | Hard-exudate detection | 81 annotated (4288×2848 px) | https://idrid.grand-challenge.org/ |

**Why two datasets?** DRIVE is optimised for vessel extraction. IDRiD provides pixel-level exudate masks. Forcing one dataset to cover both tasks would misrepresent the respective benchmark conditions.

### Dataset Download
Register at the official Grand Challenge sites and place files in:
```
data/
├── DRIVE/
│   ├── training/images/       ← .tif files (01-20)
│   ├── training/fov_masks/    ← _fov.gif or .png
│   ├── training/vessel_masks/ ← _vessel.png (binary)
│   └── test/images/
└── IDRiD/
    ├── images/                ← IDRiD_001.jpg … IDRiD_081.jpg
    ├── exudate_masks/         ← IDRiD_001_EX.png …
    ├── optic_disc_masks/      ← IDRiD_001_OD.png …
    └── metadata/fovea.csv
```

**Until datasets are downloaded**, generate synthetic images:
```bash
python generate_synthetic_data.py
```

---

## 🏗️ Pipeline Architecture

```
Input (RGB fundus image)
    │
    ├── Preprocessing
    │     ├── Green channel extraction
    │     ├── CLAHE (local contrast)
    │     ├── Background subtraction (illumination correction)
    │     └── Lab L-channel (for exudate branch)
    │
    ├── Vessel Branch
    │     ├── Gabor filter bank (12 orientations)
    │     ├── Max-response pooling
    │     ├── Otsu threshold
    │     ├── Morphological cleanup (open + close)
    │     └── FOV masking → vessel_mask
    │
    ├── Exudate Branch
    │     ├── White top-hat morphology
    │     ├── Otsu threshold
    │     ├── Morphological cleanup
    │     ├── Optic-disc suppression (evaluation mode)
    │     └── Connected components → candidate_mask + regions
    │
    ├── Measurements
    │     ├── Vessel density (% of FOV)
    │     └── Region stats (count, area, centroid, bounding box)
    │
    ├── Evaluation (optional — requires GT mask)
    │     └── Dice, IoU, Sensitivity, Specificity, Precision, F1
    │
    ├── Streamlit UI (app.py)
    └── PDF Report (ReportLab)
```

---

## 🔬 Algorithms

| Stage | Algorithm | Why |
|-------|-----------|-----|
| Preprocessing | CLAHE | Local contrast improvement without noise amplification |
| Preprocessing | Gaussian BG subtraction | Normalises slow illumination variation |
| Vessels | Gabor filter bank (12 angles) | Responds to oriented elongated structures |
| Vessels | Max-response pooling | Catches vessels at any orientation |
| Vessels | Otsu threshold | Automatic histogram-based binarisation |
| Vessels | Open + Close | Noise removal and gap bridging |
| Exudates | White top-hat | Highlights bright local spots vs. background |
| Exudates | Connected components | Converts mask to measurable individual regions |
| Evaluation | Dice / IoU | Overlap metrics robust to class imbalance |

---

## ⚙️ Installation

```bash
# Clone or navigate to the project
cd retinal-dip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Generate synthetic test data (if datasets not downloaded)
```bash
python generate_synthetic_data.py
```

### 2. Run the Streamlit web interface
```bash
streamlit run app.py
```

### 3. Analyse a single image (CLI)
```bash
python main.py --image data/DRIVE/training/images/001_training.tif \
               --fov   data/DRIVE/training/fov_masks/001_training_fov.png \
               --vessel-gt data/DRIVE/training/vessel_masks/001_training_vessel.png \
               --report
```

### 4. Batch process all DRIVE images
```bash
python main.py --batch-drive
```

### 5. Batch process all IDRiD images
```bash
python main.py --batch-idrid
```

### 6. Run experiment sweep E1–E5
```bash
python main.py --experiment
```

---

## 📊 Evaluation Methodology

Metrics are computed at the **pixel level** using a confusion matrix (TP, FP, FN, TN):

| Metric | Formula |
|--------|---------|
| Dice | 2·TP / (2·TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| F1 | 2·P·R / (P + R) |

**Why not accuracy?** Vessel and lesion pixels are a minority class. Accuracy would be dominated by background pixels and could be deceptively high even when all target pixels are missed.

For DRIVE, metrics are restricted to pixels **within the FOV mask**.

---

## 🧪 Experiments

Five ablation experiments (E1–E5) demonstrate the contribution of each pipeline stage:

| Exp | Green | CLAHE | Gabor | Morphology | FOV | Purpose |
|-----|-------|-------|-------|------------|-----|---------|
| E1 | ✓ | ✗ | ✗ | ✗ | ✗ | Intensity baseline |
| E2 | ✓ | ✓ | ✗ | ✗ | ✗ | + Local contrast |
| E3 | ✓ | ✓ | ✓ | ✗ | ✗ | + Oriented filtering |
| E4 | ✓ | ✓ | ✓ | ✓ | ✗ | + Morphological cleanup |
| E5 | ✓ | ✓ | ✓ | ✓ | ✓ | Full pipeline |

Results are saved to `experiments/results.csv`.

---

## 🏛️ Project Structure

```
retinal-dip/
├── data/                     ← Dataset folders (DRIVE + IDRiD)
├── src/
│   ├── preprocessing.py      ← CLAHE, illumination correction
│   ├── vessels.py            ← Gabor bank, vessel segmentation
│   ├── exudates.py           ← Top-hat, connected components
│   ├── morphology.py         ← Reusable morphology helpers
│   ├── evaluation.py         ← Dice, IoU, Sensitivity, F1
│   ├── visualization.py      ← Overlays, pipeline figures
│   ├── report.py             ← PDF report generation
│   └── pipeline.py           ← Integrated analyze_retina()
├── outputs/                  ← Saved masks, overlays, reports
├── experiments/              ← CSV result tables
├── app.py                    ← Streamlit UI
├── main.py                   ← CLI / batch pipeline
├── generate_synthetic_data.py ← Synthetic dataset generator
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- Uses classical DIP algorithms; not trained or validated clinically
- Top-hat detects bright regions, not semantic lesion categories
- Gabor-based vessel segmentation may miss fine capillaries
- Results are sensitive to image quality and parameter choices
- Optic-disc suppression requires a GT disc mask in evaluation mode
- This system does not diagnose diabetic retinopathy

---

## 📚 References

1. [DRIVE - Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)
2. [IDRiD - Indian Diabetic Retinopathy Image Dataset](https://idrid.grand-challenge.org/Data/)
3. Gabor, D. (1946). Theory of communication. *Journal of the IEE*, 93(26), 429-457.
4. Soille, P. (2004). *Morphological Image Analysis*. Springer.

---

*Retinal DIP Project · IPCV Academic Research Prototype · Not for clinical use.*
