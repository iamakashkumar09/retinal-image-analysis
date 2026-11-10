"use client";

import { useState } from "react";
import { FileText, Loader2, Download } from "lucide-react";
import { generateReport, AnalysisResult, getApiErrorMessage } from "@/lib/api";
import { downloadBlob } from "@/lib/utils";

interface ReportTabProps {
  result: AnalysisResult | null;
}

const REPORT_CONTENTS = [
  "Project title and research disclaimer",
  "Image identifier and dimensions",
  "Original fundus photograph",
  "Preprocessed image",
  "Blood-vessel segmentation mask",
  "Candidate exudate mask",
  "Combined overlay image",
  "Candidate-region table (area, centroid, bounding box)",
  "Vessel density and region statistics",
  "Evaluation metrics (when ground truth provided)",
  "System limitations and non-diagnostic disclaimer",
];

export default function ReportTab({ result }: ReportTabProps) {
  const [imageId, setImageId] = useState("retinal_image_001");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-slate-400 font-medium">Run analysis first to generate a report.</p>
      </div>
    );
  }

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const blob = await generateReport(result, imageId);
      downloadBlob(blob, `retinal_analysis_${imageId}.pdf`);
      setSuccess(true);
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, "Report generation failed."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl space-y-6 animate-fadeIn">
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-4">
          PDF Report Generation
        </p>
      </div>

      {/* Image ID input */}
      <div>
        <label className="block text-xs font-medium text-slate-300 mb-1.5">Image Identifier (for report)</label>
        <input
          type="text"
          value={imageId}
          onChange={(e) => setImageId(e.target.value)}
          className="w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-brand-sky/50 focus:ring-1 focus:ring-brand-sky/20 transition-all"
          placeholder="e.g. retinal_image_001"
        />
      </div>

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={loading}
        className="w-full py-3 px-6 rounded-xl font-semibold text-sm text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        style={{
          background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
          boxShadow: "0 4px 15px rgba(56,189,248,0.25)",
        }}
      >
        {loading ? (
          <><Loader2 size={16} className="animate-spin" /> Building PDF…</>
        ) : (
          <><FileText size={16} /> Generate PDF Report</>
        )}
      </button>

      {/* Success */}
      {success && (
        <div className="flex items-center gap-2 text-sm text-green-400 bg-green-500/10 border border-green-500/20 rounded-xl px-4 py-3 animate-fadeIn">
          <Download size={16} />
          PDF report downloaded successfully!
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="text-xs text-red-400 bg-red-500/10 px-3 py-2 rounded-lg border border-red-500/20">{error}</p>
      )}

      {/* Contents list */}
      <div>
        <p className="text-xs font-semibold text-slate-400 mb-3">📑 Report Contents</p>
        <ul className="space-y-1.5">
          {REPORT_CONTENTS.map((item) => (
            <li key={item} className="flex items-center gap-2 text-xs text-slate-400">
              <span className="text-green-400 flex-shrink-0">✅</span>
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
