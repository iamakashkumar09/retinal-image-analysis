"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import DropZone from "@/components/ui/DropZone";
import { evaluateMask, Metrics, AnalysisResult } from "@/lib/api";
import { metricColor } from "@/lib/utils";

interface EvaluationTabProps {
  result: AnalysisResult | null;
}

function MetricDisplay({ label, value }: { label: string; value: number }) {
  const color = metricColor(value);
  return (
    <div className="bg-gradient-to-br from-[#1e2d3d] to-[#162130] border border-white/10 rounded-xl p-4 text-center transition-all hover:border-brand-sky/40">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-2">{label}</p>
      <p className="text-2xl font-bold font-mono" style={{ color }}>{value.toFixed(4)}</p>
    </div>
  );
}

export default function EvaluationTab({ result }: EvaluationTabProps) {
  const [vesselGt, setVesselGt] = useState<File | null>(null);
  const [exudateGt, setExudateGt] = useState<File | null>(null);
  const [vesselMetrics, setVesselMetrics] = useState<Metrics | null>(null);
  const [exudateMetrics, setExudateMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState<"vessel" | "exudate" | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-slate-400 font-medium">Run analysis first</p>
        <p className="text-sm text-slate-600 mt-1">Go to the Upload & Analyze tab and run the pipeline.</p>
      </div>
    );
  }

  const evaluate = async (type: "vessel" | "exudate") => {
    const file = type === "vessel" ? vesselGt : exudateGt;
    const predB64 = type === "vessel" ? result.vessel_mask_b64 : result.exudate_mask_b64;
    if (!file) return;

    setLoading(type);
    setError(null);
    try {
      const metrics = await evaluateMask(predB64, file, type);
      if (type === "vessel") setVesselMetrics(metrics);
      else setExudateMetrics(metrics);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading(null);
    }
  };

  const renderMetrics = (metrics: Metrics | null, type: "vessel" | "exudate") => {
    const title = type === "vessel" ? "Vessel Segmentation" : "Exudate Candidate Detection";
    if (!metrics) return null;
    return (
      <div className="mt-6 animate-fadeIn">
        <p className="text-sm font-semibold text-slate-300 mb-3">📊 {title} Metrics</p>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-4">
          <MetricDisplay label="Dice" value={metrics.dice} />
          <MetricDisplay label="IoU" value={metrics.iou} />
          <MetricDisplay label="Sensitivity" value={metrics.sensitivity} />
          <MetricDisplay label="Specificity" value={metrics.specificity} />
          <MetricDisplay label="Precision" value={metrics.precision} />
          <MetricDisplay label="F1" value={metrics.f1} />
        </div>

        {/* Confusion matrix */}
        <details className="bg-white/[0.03] border border-white/[0.08] rounded-xl overflow-hidden">
          <summary className="px-4 py-3 text-xs font-medium text-slate-400 cursor-pointer hover:text-slate-200 transition-colors">
            Confusion Matrix
          </summary>
          <div className="p-4">
            <table className="text-xs font-mono border-collapse">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-slate-500"></th>
                  <th className="px-3 py-2 text-slate-400">Pred +</th>
                  <th className="px-3 py-2 text-slate-400">Pred −</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="px-3 py-2 text-slate-400 font-semibold">GT +</td>
                  <td className="px-3 py-2 text-green-400">{metrics.tp.toLocaleString()}</td>
                  <td className="px-3 py-2 text-red-400">{metrics.fn.toLocaleString()}</td>
                </tr>
                <tr>
                  <td className="px-3 py-2 text-slate-400 font-semibold">GT −</td>
                  <td className="px-3 py-2 text-red-400">{metrics.fp.toLocaleString()}</td>
                  <td className="px-3 py-2 text-green-400">{metrics.tn.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>

        {Boolean((metrics as Metrics & { warnings?: string[] }).warnings?.length) && (
          <div className="mt-3 space-y-1">
            {((metrics as Metrics & { warnings?: string[] }).warnings ?? []).map((w: string, i: number) => (
              <p key={i} className="text-xs text-yellow-400 bg-yellow-500/10 px-3 py-1.5 rounded-lg border border-yellow-500/20">
                ⚠️ {w}
              </p>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-2">
          Ground-Truth Evaluation
        </p>
        <p className="text-xs text-slate-500 mb-4">
          Upload a ground-truth binary mask to evaluate prediction quality.
          GT masks are available in the DRIVE (vessel GT) and IDRiD (exudate GT) datasets.
        </p>
      </div>

      {error && (
        <p className="text-xs text-red-400 bg-red-500/10 px-3 py-2 rounded-lg border border-red-500/20">{error}</p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Vessel GT */}
        <div>
          <p className="text-xs font-medium text-slate-300 mb-2">Vessel Ground Truth <span className="text-slate-600">(DRIVE training set)</span></p>
          <DropZone label="Vessel GT mask" accept=".png,.bmp,.tif" value={vesselGt} onChange={setVesselGt} compact />
          <button
            onClick={() => evaluate("vessel")}
            disabled={!vesselGt || loading !== null}
            className="mt-2 w-full py-2 px-4 rounded-lg text-xs font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 bg-brand-sky/10 hover:bg-brand-sky/20 border border-brand-sky/30 text-brand-sky"
          >
            {loading === "vessel" ? <><Loader2 size={12} className="animate-spin" /> Evaluating…</> : "Evaluate Vessel Mask"}
          </button>
          {renderMetrics(vesselMetrics, "vessel")}
        </div>

        {/* Exudate GT */}
        <div>
          <p className="text-xs font-medium text-slate-300 mb-2">Exudate Ground Truth <span className="text-slate-600">(IDRiD pixel-level subset)</span></p>
          <DropZone label="Exudate GT mask" accept=".png,.bmp,.tif" value={exudateGt} onChange={setExudateGt} compact />
          <button
            onClick={() => evaluate("exudate")}
            disabled={!exudateGt || loading !== null}
            className="mt-2 w-full py-2 px-4 rounded-lg text-xs font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/30 text-yellow-400"
          >
            {loading === "exudate" ? <><Loader2 size={12} className="animate-spin" /> Evaluating…</> : "Evaluate Exudate Mask"}
          </button>
          {renderMetrics(exudateMetrics, "exudate")}
        </div>
      </div>
    </div>
  );
}
