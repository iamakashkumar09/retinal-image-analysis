# Retinal Image Analysis — Next.js Frontend

> Modern Next.js 14 UI replacing the original Streamlit app.
> Talks to a FastAPI Python backend for all image processing.

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Icons | Lucide React |
| HTTP | Axios |
| Backen
d | FastAPI (Python) |

---

## Folder Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx          ← Root layout + metadata
│   │   ├── page.tsx            ← Main page (tabs + sidebar)
│   │   └── globals.css         ← Tailwind base + custom styles
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Sidebar.tsx     ← Collapsible algorithm parameter panel
│   │   │   ├── MetricCard.tsx  ← Stat display card
│   │   │   ├── ImagePanel.tsx  ← Base64 image display
│   │   │   └── DropZone.tsx    ← Drag-and-drop file uploader
│   │   └── tabs/
│   │       ├── UploadTab.tsx   ← File upload + analyze button
│   │       ├── ResultsTab.tsx  ← Masks, overlay, region table
│   │       ├── EvaluationTab.tsx ← GT mask comparison + metrics
│   │       ├── DebugTab.tsx    ← Intermediate pipeline images
│   │       └── ReportTab.tsx   ← PDF report generation
│   └── lib/
│       ├── api.ts              ← All API calls (analyzeRetina, evaluateMask, generateReport)
│       └── utils.ts            ← cn(), b64ToSrc(), metricColor(), downloadBlob()
├── package.json
├── next.config.mjs             ← Proxy /api/py/* → http://localhost:8000/*
├── tailwind.config.ts
└── tsconfig.json
```

---

## Setup & Run

### Prerequisites

1. **Node.js 18+** — [Download here](https://nodejs.org/)
2. **Python 3.10+** with the project dependencies

### Step 1 — Install Python API dependencies

```bash
cd retinal-dip
pip install -r requirements_api.txt
```

### Step 2 — Start the FastAPI backend

```bash
cd retinal-dip
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
You can explore the auto-generated docs at `http://localhost:8000/docs`.

### Step 3 — Install Next.js dependencies

```bash
cd retinal-dip/frontend
npm install
```

### Step 4 — Start the Next.js dev server

```bash
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## How it Works

```
Browser (Next.js :3000)
    │
    │  POST /api/py/analyze   (multipart/form-data)
    │  ← Next.js rewrites to → http://localhost:8000/analyze
    ▼
FastAPI (Python :8000)
    │
    │  Runs analyze_retina() from src/pipeline.py
    │  Returns JSON with base64-encoded PNG images + measurements
    ▼
Browser renders results
```

The `next.config.mjs` file contains a rewrite rule so the browser never needs to know the Python server URL directly:
```js
{ source: '/api/py/:path*', destination: 'http://localhost:8000/:path*' }
```

---

## API Endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Full pipeline analysis (returns base64 images + stats) |
| `POST` | `/evaluate` | Metric evaluation against GT mask |
| `POST` | `/report` | Generate PDF report (returns binary PDF) |

---

## ⚠️ Disclaimer

This is a **research / decision-support prototype** — NOT a medical diagnosis tool.
All candidate regions require clinical review by a qualified ophthalmologist.

