"use client";

import { AnalysisResult } from "@/lib/api";
import MetricCard from "@/components/ui/MetricCard";
import ImagePanel from "@/components/ui/ImagePanel";

interface ResultsTabProps {
  result: AnalysisResult | null;
}

export default function ResultsTab({ result }: ResultsTabProps) {
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-full bg-brand-sky/10 flex items-center justify-center mb-4">
          <span className="text-2xl">📊</span>
        </div>
        <p className="text-slate-400 font-medium">No results yet</p>
        <p className="text-sm text-slate-600 mt-1">Upload an image and click <strong>Analyze Retina</strong> to see results.</p>
      </div>
    );
  }

  const { vessel_density, region_stats, regions, original_b64, vessel_mask_b64, exudate_mask_b64, overlay_b64 } = result;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Metric cards */}
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
          Quantitative Measurements
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="Vessel Density" value={`${vessel_density.toFixed(2)}%`} icon="🩸" sub="% of FOV pixels" />
          <MetricCard label="Candidates" value={region_stats.count} icon="🟡" sub="exudate regions" />
          <MetricCard label="Largest Region" value={`${region_stats.largest_area} px`} icon="📐" />
          <MetricCard label="Total Area" value={`${region_stats.total_area} px`} icon="📊" />
        </div>
      </div>

      {/* Segmentation images */}
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
          Segmentation Results
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <ImagePanel b64={original_b64} label="Original Fundus" badge="Input" badgeColor="#94a3b8" />
          <ImagePanel b64={vessel_mask_b64} label="Blood Vessel Mask" badge="Vessels" badgeColor="#4ade80" />
          <ImagePanel b64={exudate_mask_b64} label="Exudate Candidate Mask" badge="Candidates" badgeColor="#facc15" />
        </div>
      </div>

      {/* Overlay */}
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-3">
          Combined Overlay
        </p>
        <p className="text-xs text-slate-500 mb-3">
          🟢 <strong className="text-slate-400">Green</strong> = Detected vessels &nbsp;|&nbsp; 🟡 <strong className="text-slate-400">Yellow</strong> = Candidate exudate regions
        </p>
        <ImagePanel b64={overlay_b64} label="Vessel + Exudate Overlay" />
      </div>

      {/* Candidate region table */}
      {regions.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
            Candidate Region Table
          </p>
          <div className="overflow-x-auto rounded-xl border border-white/[0.08]">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-white/[0.04] text-slate-400">
                  {["ID", "Area (px)", "Centroid X", "Centroid Y", "BBox X", "BBox Y", "Width", "Height"].map((h) => (
                    <th key={h} className="px-3 py-2.5 text-left font-semibold tracking-wide">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {regions.map((r, i) => (
                  <tr key={r.id} className={`border-t border-white/[0.05] hover:bg-white/[0.03] transition-colors ${i % 2 === 0 ? "" : "bg-white/[0.015]"}`}>
                    <td className="px-3 py-2 font-mono text-brand-sky">{r.id}</td>
                    <td className="px-3 py-2 font-mono">{r.area}</td>
                    <td className="px-3 py-2 font-mono">{r.centroid[0].toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono">{r.centroid[1].toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono">{r.bbox[0]}</td>
                    <td className="px-3 py-2 font-mono">{r.bbox[1]}</td>
                    <td className="px-3 py-2 font-mono">{r.bbox[2]}</td>
                    <td className="px-3 py-2 font-mono">{r.bbox[3]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-600 mt-2">
            ⚠️ Candidate regions only — not confirmed clinical lesions. Professional clinical review is required.
          </p>
        </div>
      )}
    </div>
  );
}
