# 🩺 Retinal Image Analysis Platform

> **Classical Digital Image Processing · Blood Vessel Segmentation · Hard-Exudate Candidate Detection**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

> ⚠️ **RESEARCH / DECISION-SUPPORT PROTOTYPE — NOT A MEDICAL DIAGNOSIS TOOL.**
> All outputs are candidate regions only. Clinical review by a qualified ophthalmologist is required.

---

## 🎯 What Does This Project Do? (Simple Answer)

This project takes a **photograph of the back of a human eye (retina)** and automatically:

1. **Finds and highlights all the blood vessels** in the image (shown in 🟢 green)
2. **Detects bright spots called "hard exudates"** that can indicate eye disease (shown in 🟡 yellow)
3. **Measures how much of the eye is covered by vessels** (vessel density %)
4. **Scores its own accuracy** by comparing against known ground-truth masks
5. **Generates a PDF medical-style report** of all findings

---

## 🧠 Why Does This Matter?

Diseases like **Diabetic Retinopathy** damage the blood vessels in the retina. As the disease progresses:
- Blood vessels start leaking → bright fatty deposits called **hard exudates** appear
- Vessels may grow abnormally or block

By detecting vessels and exudates automatically using **Digital Image Processing (DIP)**, doctors can be assisted in screening large numbers of patients faster.

---

## 📂 What Goes In, What Comes Out

```
INPUT:  A retinal fundus photograph (JPG / PNG / TIFF)
        └─ Color image of the back of the eye

OUTPUT:
   ├── 🟢 Vessel Mask         — which pixels are blood vessels
   ├── 🟡 Exudate Mask        — which pixels might be exudates
   ├── 🎨 Overlay Image       — original photo with colored highlights
   ├── 📐 Measurements        — vessel density %, candidate count, areas
   ├── 📊 Accuracy Metrics    — Dice, IoU, Sensitivity, Specificity
   └── 📄 PDF Report          — printable summary of all findings
```

---

## 🔬 Complete Step-by-Step Pipeline

### STEP 1 — Upload the Image
**File:** `app.py` (Upload & Analyze tab)

- You upload a color retinal photograph
- The system checks it is a valid 3-channel (RGB) color image, at least 100×100 pixels
- The image is stored in memory for processing

---

### STEP 2 — Preprocessing
**File:** `retinal-dip/src/preprocessing.py`

The raw photo is messy — uneven lighting, low contrast. We clean it up using two parallel paths:

#### Path A — For Vessel Detection:
```
Original RGB image
    ↓
Extract GREEN channel only
    (vessels show best contrast in the green channel of a color eye photo)
    ↓
Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    (boosts local contrast so faint vessels become visible)
    ↓
Subtract blurred background (Gaussian filter)
    (removes the slow uneven illumination across the retina)
    ↓
Result: "corrected" grayscale image — vessels stand out clearly
```

#### Path B — For Exudate Detection:
```
Original RGB image
    ↓
Convert to Lab color space (separates brightness from color)
    ↓
Take the L (Luminance) channel only
    (hard exudates are BRIGHT, so we focus on brightness)
    ↓
Apply CLAHE to the L channel
    ↓
Result: "L_clahe" image — bright spots are enhanced
```

---

### STEP 3 — FOV (Field-of-View) Mask
**File:** `retinal-dip/src/vessels.py` → `estimate_fov()`

Retinal photos have a **dark circular border** — the camera lens frame. We must ignore those black pixels or they'll confuse the analysis.

```
Detect the circular boundary of the retinal image
    ↓
Create a binary mask (white = valid retina, black = outside lens)
    ↓
All measurements only count pixels INSIDE this circle
```

---

### STEP 4 — Blood Vessel Segmentation (Gabor Filter Bank)
**File:** `retinal-dip/src/vessels.py`

This is the heart of the vessel detection. We use a **Gabor filter** — a mathematical filter that responds to **elongated line-like structures** at specific angles.

```
Corrected grayscale image
    ↓
Apply Gabor filter at 12 different angles (0°, 15°, 30°, 45° ... 165°)
    (each filter detects vessels running in one direction)
    ↓
Take the MAXIMUM response across all 12 angles
    (wherever there's a vessel at ANY angle, it lights up)
    ↓
Apply Otsu Thresholding
    (automatically find the best cutoff to separate vessel vs background)
    ↓
Morphological Cleanup:
    - Opening: remove tiny noise specks
    - Closing: fill small gaps in vessels
    ↓
Apply FOV mask (only keep detections inside the eye circle)
    ↓
Result: vessel_mask — a black/white image where white = vessel
```

**Why Gabor?** Blood vessels are long, thin, curved structures. Gabor filters are mathematically designed to detect exactly such oriented patterns.

---

### STEP 5 — Exudate Candidate Detection (White Top-Hat)
**File:** `retinal-dip/src/exudates.py`

Hard exudates are **small bright spots** that are brighter than their local background. We use **White Top-Hat morphology** to find them:

```
L_clahe (brightness-enhanced channel)
    ↓
White Top-Hat Transform:
    image − morphological_opening(image)
    = small bright spots that are LOCALLY brighter than their surroundings
    ↓
Apply Otsu Threshold to binarize
    ↓
Morphological cleanup (remove tiny noise)
    ↓
Suppress optic disc area
    (the disc is also bright — we remove it to avoid false detections)
    ↓
Connected Components Analysis:
    label each separate blob of white pixels
    measure area, centroid (center), bounding box of each blob
    filter out blobs smaller than minimum area threshold
    ↓
Result: candidate_mask + list of individual regions with measurements
```

---

### STEP 6 — Measurements
**File:** `retinal-dip/src/pipeline.py`

| Measurement | What It Means |
|------------|---------------|
| **Vessel Density** | What % of the retinal area is covered by vessels |
| **Candidate Count** | How many separate bright spots were found |
| **Largest Region** | Area (in pixels) of the biggest candidate spot |
| **Total Area** | Combined pixel area of all candidate spots |
| **Centroid** | Center coordinates (X, Y) of each candidate |
| **Bounding Box** | Rectangle around each candidate (X, Y, Width, Height) |

---

### STEP 7 — Overlay Visualization
**File:** `retinal-dip/src/visualization.py`

```
Original color photo
    + vessel_mask pixels → tinted GREEN  (alpha blended)
    + exudate_mask pixels → tinted YELLOW (alpha blended)
    ↓
Combined overlay image
```

---

### STEP 8 — Evaluation (when Ground Truth is available)
**File:** `retinal-dip/src/evaluation.py`

Every pixel falls into one of four categories:

```
                  Predicted: Vessel    Predicted: Not Vessel
GT: Vessel        TP (True Positive)   FN (False Negative)
GT: Not Vessel    FP (False Positive)  TN (True Negative)
```

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Dice** | 2·TP / (2·TP+FP+FN) | Overlap quality — 0=no overlap, 1=perfect |
| **IoU** | TP / (TP+FP+FN) | Intersection over Union |
| **Sensitivity** | TP / (TP+FN) | How many real vessels did we find? |
| **Specificity** | TN / (TN+FP) | How many non-vessels did we correctly ignore? |
| **Precision** | TP / (TP+FP) | Of what we called vessel, how many really were? |
| **F1** | 2·P·R / (P+R) | Harmonic mean of Precision and Recall |

---

### STEP 9 — PDF Report
**File:** `retinal-dip/src/report.py`

A ReportLab-generated PDF containing:
- Project title + disclaimer
- Original image + all result images
- Measurements table
- Candidate region table (ID, area, centroid, bounding box)
- Evaluation metrics table (if GT provided)
- Medical disclaimer footer

---

## 🧪 The 5 Experiments (Ablation Study)

To prove each step adds value, we ran 5 experiments switching stages ON/OFF one at a time:

| Exp | What's Active | Mean Dice | Key Finding |
|-----|--------------|-----------|-------------|
| **E1** | Just Otsu threshold | 0.663 | Baseline — no processing |
| **E2** | + CLAHE | **0.825** | **CLAHE is the biggest improvement (+24%)** |
| **E3** | + Gabor | 0.601 | Gabor alone needs cleanup |
| **E4** | + Morphology | 0.681 | Cleanup improves Gabor significantly |
| **E5** | **ALL stages + FOV** | **0.742** | Full pipeline is balanced & robust |

> **Key Finding:** CLAHE (local contrast enhancement) is the single most important step.
> It boosts Dice by 24% over the raw baseline on its own.

---

## 📊 What the Results Mean

### Vessel Results (on synthetic DRIVE-style data)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Dice** | ~0.74 | Good overlap with expert ground truth |
| **Sensitivity** | ~0.85 | We find 85% of all real vessel pixels |
| **Specificity** | ~0.99 | Almost never wrongly label background as vessel |
| **Vessel Density** | ~4–7% | Typical healthy retinal vessel coverage |

### Exudate Results (on synthetic IDRiD-style data)
- Candidate detection is validated on real IDRiD data after download
- Typical classical DIP exudate Dice: 0.40–0.60 on real annotated data
- The candidate list gives a doctor a shortlist of suspicious areas to review

---

## 🖥️ How to Use the App (Step by Step)

### 1. Install dependencies
```bash
cd retinal-dip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate synthetic test data (if real datasets not downloaded)
```bash
python generate_synthetic_data.py
```

### 3. Run the Streamlit web app
```bash
streamlit run app.py
# Opens at → http://localhost:8501
```

### 4. Using the 5 tabs in the app

| Tab | What To Do |
|-----|-----------|
| **📤 Upload & Analyze** | Upload a retinal image → click Analyze Retina |
| **📊 Results** | View vessel mask, exudate mask, overlay, measurements table |
| **🎯 Evaluation** | Upload ground-truth mask → see Dice/IoU/Sensitivity scores |
| **🔬 Debug View** | See every intermediate pipeline image (CLAHE, Gabor, top-hat…) |
| **📄 PDF Report** | Generate & download a PDF report of all findings |

### 5. CLI batch processing
```bash
# Single image with report
python main.py --image data/DRIVE/training/images/001_training.tif \
               --vessel-gt data/DRIVE/training/vessel_masks/001_training_vessel.png \
               --report

# Batch all DRIVE images
python main.py --batch-drive

# Batch all IDRiD images
python main.py --batch-idrid

# Run ablation experiments E1–E5
python main.py --experiment

# Generate result charts
python plot_experiments.py
```

### 6. Run unit tests
```bash
python test_pipeline.py
# Expected: 26/26 PASS ✅
```

---

## 📁 Complete File Structure

```
IPCV-Project/
├── README.md                          ← You are here
└── retinal-dip/
    │
    ├── src/                           ← All algorithm modules
    │   ├── preprocessing.py           ← CLAHE, green channel, illumination fix
    │   ├── morphology.py              ← Morphological operation helpers
    │   ├── vessels.py                 ← Gabor filter bank, vessel segmentation
    │   ├── exudates.py                ← White top-hat, candidate detection
    │   ├── evaluation.py              ← Dice, IoU, Sensitivity, F1 metrics
    │   ├── visualization.py           ← Overlay, pipeline grid figure
    │   ├── report.py                  ← PDF report via ReportLab
    │   └── pipeline.py                ← analyze_retina() — the main API
    │
    ├── app.py                         ← Streamlit web interface
    ├── main.py                        ← CLI: single / batch / experiment modes
    ├── generate_synthetic_data.py     ← Creates test images without real datasets
    ├── plot_experiments.py            ← 5 publication-quality result charts
    ├── test_pipeline.py               ← 26 automated unit tests
    ├── requirements.txt               ← Python dependencies
    │
    ├── data/
    │   ├── DRIVE/training/images/     ← Retinal images (.tif)
    │   ├── DRIVE/training/fov_masks/  ← FOV masks
    │   ├── DRIVE/training/vessel_masks/ ← Ground truth vessel masks
    │   └── IDRiD/images/              ← IDRiD color images (.jpg)
    │
    ├── outputs/
    │   ├── vessels/                   ← Predicted vessel masks
    │   ├── exudates/                  ← Predicted exudate masks
    │   ├── overlays/                  ← Combined overlay images
    │   └── reports/                   ← Generated PDF reports
    │
    └── experiments/
        ├── results.csv                ← E1–E5 ablation numbers
        ├── vessel_experiments.csv     ← Batch DRIVE results
        ├── exudate_experiments.csv    ← Batch IDRiD results
        └── figures/                   ← 5 publication-quality charts
            ├── 01_dice_comparison.png
            ├── 02_sensitivity_specificity.png
            ├── 03_per_image_dice.png
            ├── 04_precision_recall.png
            └── 05_metric_heatmap.png
```

---

## 🔑 Technologies Used

| Technology | Role in This Project |
|-----------|---------------------|
| **OpenCV** | Image I/O, CLAHE, Gabor filter, morphology, connected components |
| **NumPy** | All numerical array operations |
| **scikit-image** | Additional image processing utilities |
| **Matplotlib** | Experiment figures and pipeline charts |
| **Streamlit** | Interactive web application UI |
| **ReportLab** | PDF report generation |
| **Pandas** | Reading/writing CSV experiment results |

---

## 🗂️ Datasets

| Dataset | Task | Images | Link |
|---------|------|--------|------|
| **DRIVE** | Vessel segmentation | 40 (768×584 px) | https://drive.grand-challenge.org/ |
| **IDRiD** | Hard-exudate detection | 81 annotated | https://idrid.grand-challenge.org/ |

> If you haven't downloaded the datasets yet, run `python generate_synthetic_data.py` to generate realistic synthetic versions for immediate testing.

---

## ⚠️ Limitations

- Uses classical DIP algorithms — not trained or clinically validated
- Top-hat detects bright regions, not confirmed lesion categories
- Results are sensitive to image quality and lighting conditions
- Optic disc suppression requires a GT disc mask in evaluation mode
- This system does **not** diagnose diabetic retinopathy

---

## 📚 References

1. [DRIVE - Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)
2. [IDRiD - Indian Diabetic Retinopathy Image Dataset](https://idrid.grand-challenge.org/)
3. Gabor, D. (1946). Theory of communication. *Journal of the IEE*, 93(26), 429–457.
4. Soille, P. (2004). *Morphological Image Analysis*. Springer.

---

*Retinal DIP Project · IPCV Academic Research Prototype · Not for clinical use.*