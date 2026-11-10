"use client";

import { useState } from "react";
import { Loader2, Zap, AlertTriangle } from "lucide-react";
import DropZone from "@/components/ui/DropZone";
import { analyzeRetina, AnalysisResult } from "@/lib/api";
import { AlgoParams } from "@/components/ui/Sidebar";
import { b64ToSrc } from "@/lib/utils";

interface UploadTabProps {
  params: AlgoParams;
  onResult: (r: AnalysisResult) => void;
}

const STEPS = [
  "⚙️ Preprocessing (CLAHE + illumination correction)…",
  "🔬 Running Gabor filter bank…",
  "🩸 Segmenting blood vessels…",
  "🟡 Running white top-hat exudate detector…",
  "📐 Analysing candidate regions…",
  "🖼️ Generating overlay…",
];

export default function UploadTab({ params, onResult }: UploadTabProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [fovFile, setFovFile] = useState<File | null>(null);
  const [discFile, setDiscFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stepIdx, setStepIdx] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);

  const handleImageChange = (file: File | null) => {
    setImageFile(file);
    setError(null);
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      const img = new window.Image();
      img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
      img.src = url;
    } else {
      setPreview(null);
      setDims(null);
    }
  };

  const handleAnalyze = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setStepIdx(0);

    // Animate steps
    for (let i = 0; i < STEPS.length; i++) {
      setStepIdx(i);
      await new Promise((r) => setTimeout(r, 350));
    }

    try {
      const result = await analyzeRetina(imageFile, params, fovFile, discFile);
      onResult(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed. Is the Python API running?";
      setError(msg);
    } finally {
      setLoading(false);
      setStepIdx(-1);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left — main image upload */}
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
          Upload Retinal Image
        </p>

        <DropZone
          label="Upload a retinal fundus photograph"
          accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
          value={imageFile}
          onChange={handleImageChange}
          hint="Supports JPEG, PNG, TIFF — minimum 100×100 px"
        />

        {preview && (
          <div className="mt-4 animate-fadeIn">
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-green-500/15 border border-green-500/40 text-green-400">
                ✓ Valid Image
              </span>
              {dims && (
                <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-brand-sky/15 border border-brand-sky/40 text-brand-sky">
                  {dims.w} × {dims.h} px
                </span>
              )}
            </div>
            <div className="rounded-xl overflow-hidden border border-white/[0.08]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt="Uploaded fundus" className="w-full object-contain max-h-80" />
            </div>
          </div>
        )}
      </div>

      {/* Right — optional masks + analyze */}
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
          Optional Masks
        </p>

        <div className="space-y-3 mb-6">
          <div>
            <p className="text-xs font-medium text-slate-400 mb-1">
              FOV Mask <span className="text-slate-600">(optional — auto-estimated if omitted)</span>
            </p>
            <DropZone label="FOV mask" accept=".png,.bmp,.tif" value={fovFile} onChange={setFovFile} hint="PNG/BMP/TIFF circular field-of-view mask" compact />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 mb-1">
              Optic-Disc Mask <span className="text-slate-600">(optional — suppresses disc false positives)</span>
            </p>
            <DropZone label="Optic-disc mask" accept=".png,.bmp,.tif" value={discFile} onChange={setDiscFile} hint="PNG/BMP/TIFF optic disc region mask" compact />
          </div>
        </div>

        <div className="border-t border-white/[0.06] pt-4">
          <button
            onClick={handleAnalyze}
            disabled={!imageFile || loading}
            className="w-full py-3 px-6 rounded-xl font-semibold text-sm text-white transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-2"
            style={{
              background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
              boxShadow: "0 4px 15px rgba(56,189,248,0.25)",
            }}
            onMouseOver={(e) => { if (!loading && imageFile) { (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 6px 25px rgba(56,189,248,0.4)"; (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-2px)"; } }}
            onMouseOut={(e) => { (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 4px 15px rgba(56,189,248,0.25)"; (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)"; }}
          >
            {loading ? (
              <><Loader2 size={16} className="animate-spin" /> Analysing…</>
            ) : (
              <><Zap size={16} /> Analyze Retina</>
            )}
          </button>
        </div>

        {/* Progress steps */}
        {loading && (
          <div className="mt-4 space-y-1.5 animate-fadeIn">
            {STEPS.map((step, i) => (
              <div
                key={i}
                className={`flex items-center gap-2 text-xs py-1.5 px-3 rounded-lg transition-all duration-300 ${
                  i < stepIdx
                    ? "text-green-400 bg-green-500/5"
                    : i === stepIdx
                    ? "text-brand-sky bg-brand-sky/10 font-medium"
                    : "text-slate-600"
                }`}
              >
                {i < stepIdx ? "✓" : i === stepIdx ? <Loader2 size={10} className="animate-spin" /> : "○"}
                {step}
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 flex items-start gap-3 p-3 rounded-xl bg-red-500/10 border border-red-500/30 animate-fadeIn">
            <AlertTriangle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-xs font-semibold text-red-400">Analysis Failed</p>
              <p className="text-xs text-red-300/70 mt-0.5">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
