"use client";

import { AnalysisResult } from "@/lib/api";
import ImagePanel from "@/components/ui/ImagePanel";
import { b64ToSrc } from "@/lib/utils";

interface DebugTabProps {
  result: AnalysisResult | null;
}

export default function DebugTab({ result }: DebugTabProps) {
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <p className="text-slate-400 font-medium">Run analysis first</p>
        <p className="text-sm text-slate-600 mt-1">Intermediate pipeline images will appear here.</p>
      </div>
    );
  }

  const panels: { b64: string; label: string }[] = [
    { b64: result.green_b64, label: "Green Channel" },
    { b64: result.clahe_b64, label: "CLAHE-enhanced" },
    { b64: result.corrected_b64, label: "Illumination-corrected" },
    { b64: result.L_b64, label: "Lab L-channel" },
    { b64: result.L_clahe_b64, label: "Lab L-channel + CLAHE" },
    { b64: result.response_norm_b64, label: "Gabor max-response" },
    { b64: result.tophat_b64, label: "Top-hat response" },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-2">
          Intermediate Pipeline Images
        </p>
        <p className="text-xs text-slate-500 mb-4 italic">
          These intermediate outputs are for research inspection only. They are not shown in the main results view.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {panels.map(({ b64, label }) => (
          <ImagePanel key={label} b64={b64} label={label} />
        ))}
      </div>

      {/* Gabor colorized */}
      {result.response_norm_b64 && (
        <div>
          <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 border-b border-white/[0.06] pb-1 mb-3">
            Gabor Response (Info)
          </p>
          <p className="text-xs text-slate-500 mb-3">
            Jet colormap applied — blue = low response, red = high response (vessel-like regions).
          </p>
          <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={b64ToSrc(result.response_norm_b64)}
              alt="Gabor response"
              className="w-full object-contain max-h-96 rounded-lg"
            />
          </div>
        </div>
      )}
    </div>
  );
}
