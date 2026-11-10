"use client";

import { useCallback, useState } from "react";
import { Upload, X, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropZoneProps {
  label: string;
  accept?: string;
  value: File | null;
  onChange: (f: File | null) => void;
  hint?: string;
  compact?: boolean;
}

export default function DropZone({ label, accept, value, onChange, hint, compact = false }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onChange(file);
    },
    [onChange]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0] ?? null;
      onChange(file);
    },
    [onChange]
  );

  if (compact) {
    return (
      <label className="flex items-center gap-3 cursor-pointer p-3 rounded-xl bg-white/[0.03] border border-white/[0.08] hover:border-brand-sky/30 transition-all group">
        <input type="file" accept={accept} className="hidden" onChange={handleChange} />
        <div className="p-1.5 rounded-lg bg-brand-sky/10 group-hover:bg-brand-sky/20 transition-colors">
          {value ? <CheckCircle size={14} className="text-green-400" /> : <Upload size={14} className="text-brand-sky" />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-slate-300 truncate">{value ? value.name : label}</p>
          {hint && !value && <p className="text-[10px] text-slate-600">{hint}</p>}
        </div>
        {value && (
          <button
            onClick={(e) => { e.preventDefault(); onChange(null); }}
            className="p-1 rounded hover:bg-white/10 text-slate-500 hover:text-red-400 transition-colors"
          >
            <X size={12} />
          </button>
        )}
      </label>
    );
  }

  return (
    <label
      className={cn(
        "flex flex-col items-center justify-center gap-3 cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200 p-8",
        dragging
          ? "border-brand-sky bg-brand-sky/10 scale-[1.01]"
          : value
          ? "border-green-500/40 bg-green-500/5"
          : "border-white/10 bg-white/[0.02] hover:border-brand-sky/40 hover:bg-brand-sky/5"
      )}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input type="file" accept={accept} className="hidden" onChange={handleChange} />
      {value ? (
        <>
          <CheckCircle size={32} className="text-green-400" />
          <div className="text-center">
            <p className="text-sm font-medium text-green-400">{value.name}</p>
            <p className="text-xs text-slate-500 mt-1">{(value.size / 1024).toFixed(1)} KB</p>
          </div>
          <button
            onClick={(e) => { e.preventDefault(); onChange(null); }}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-red-400 transition-colors"
          >
            <X size={12} /> Remove
          </button>
        </>
      ) : (
        <>
          <div className="p-4 rounded-2xl bg-brand-sky/10">
            <Upload size={28} className="text-brand-sky" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-300">{label}</p>
            {hint && <p className="text-xs text-slate-500 mt-1">{hint}</p>}
            <p className="text-xs text-slate-600 mt-2">Drag & drop or click to browse</p>
          </div>
        </>
      )}
    </label>
  );
}
