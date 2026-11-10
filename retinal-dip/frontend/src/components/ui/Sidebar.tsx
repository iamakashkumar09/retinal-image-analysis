"use client";

import { useState } from "react";
import { Settings, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface SliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  help?: string;
}

function ParamSlider({ label, min, max, step, value, onChange, help }: SliderProps) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-slate-300">{label}</label>
        <span className="text-xs font-mono text-brand-sky bg-white/5 px-1.5 py-0.5 rounded">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-brand-sky"
        style={{ background: `linear-gradient(to right, #38bdf8 0%, #38bdf8 ${((value - min) / (max - min)) * 100}%, #30363d ${((value - min) / (max - min)) * 100}%, #30363d 100%)` }}
      />
      {help && <p className="text-[10px] text-slate-500 mt-1">{help}</p>}
    </div>
  );
}

interface SelectSliderProps {
  label: string;
  options: number[];
  value: number;
  onChange: (v: number) => void;
  help?: string;
}

function SelectSlider({ label, options, value, onChange, help }: SelectSliderProps) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-slate-300">{label}</label>
        <span className="text-xs font-mono text-brand-sky bg-white/5 px-1.5 py-0.5 rounded">{value}</span>
      </div>
      <div className="flex gap-1 flex-wrap">
        {options.map((opt) => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={cn(
              "px-2 py-0.5 rounded text-xs font-medium transition-all",
              value === opt
                ? "bg-brand-sky text-black"
                : "bg-white/5 text-slate-400 hover:bg-white/10"
            )}
          >
            {opt}
          </button>
        ))}
      </div>
      {help && <p className="text-[10px] text-slate-500 mt-1">{help}</p>}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 border-b border-white/[0.06] pb-1 mb-3">
      {children}
    </p>
  );
}

export interface AlgoParams {
  gaborSigma: number;
  gaborLambda: number;
  gaborGamma: number;
  nOrientations: number;
  vesselMorph: number;
  tophatSize: number;
  exudateMorph: number;
  minArea: number;
  claheClip: number;
  bgSigma: number;
}

interface SidebarProps {
  params: AlgoParams;
  onChange: (p: AlgoParams) => void;
}

export default function Sidebar({ params, onChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  const set = <K extends keyof AlgoParams>(key: K, val: AlgoParams[K]) =>
    onChange({ ...params, [key]: val });

  return (
    <aside
      className={cn(
        "relative flex-shrink-0 h-screen sticky top-0 flex flex-col transition-all duration-300 overflow-hidden",
        "bg-gradient-to-b from-[#161b22] to-[#1c2128]",
        "border-r border-white/[0.06]",
        collapsed ? "w-12" : "w-72"
      )}
    >
      {/* Toggle button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute top-4 right-2 z-10 p-1 rounded bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>

      {collapsed ? (
        <div className="flex justify-center pt-4">
          <Settings size={18} className="text-slate-500" />
        </div>
      ) : (
        <div className="p-4 overflow-y-auto flex-1">
          <div className="flex items-center gap-2 mb-5">
            <Settings size={16} className="text-brand-sky" />
            <h2 className="text-sm font-semibold text-white">Algorithm Parameters</h2>
          </div>

          {/* Vessel Branch */}
          <SectionLabel>🩸 Vessel Branch (Gabor)</SectionLabel>
          <ParamSlider label="Gabor σ (sigma)" min={1} max={8} step={0.5} value={params.gaborSigma} onChange={(v) => set("gaborSigma", v)} help="Gaussian envelope spread" />
          <ParamSlider label="Gabor λ (lambda)" min={4} max={16} step={1} value={params.gaborLambda} onChange={(v) => set("gaborLambda", v)} help="Sinusoidal wavelength" />
          <ParamSlider label="Gabor γ (gamma)" min={0.3} max={1.2} step={0.1} value={params.gaborGamma} onChange={(v) => set("gaborGamma", v)} help="Aspect ratio (elongation)" />
          <SelectSlider label="Orientations" options={[6, 8, 12, 16, 18]} value={params.nOrientations} onChange={(v) => set("nOrientations", v)} help="Number of Gabor filter angles" />
          <ParamSlider label="Vessel morph kernel" min={1} max={7} step={2} value={params.vesselMorph} onChange={(v) => set("vesselMorph", v)} help="Morphological cleanup kernel size" />

          {/* Exudate Branch */}
          <div className="mt-2">
            <SectionLabel>🟡 Exudate Branch</SectionLabel>
            <ParamSlider label="Top-hat kernel" min={7} max={31} step={2} value={params.tophatSize} onChange={(v) => set("tophatSize", v)} help="Structuring element size" />
            <ParamSlider label="Exudate morph kernel" min={1} max={7} step={2} value={params.exudateMorph} onChange={(v) => set("exudateMorph", v)} />
            <ParamSlider label="Min candidate area (px)" min={5} max={100} step={5} value={params.minArea} onChange={(v) => set("minArea", v)} help="Discard candidates smaller than this" />
          </div>

          {/* Preprocessing */}
          <div className="mt-2">
            <SectionLabel>🔧 Preprocessing</SectionLabel>
            <ParamSlider label="CLAHE clip limit" min={0.5} max={5} step={0.5} value={params.claheClip} onChange={(v) => set("claheClip", v)} />
            <ParamSlider label="BG correction σ" min={5} max={30} step={1} value={params.bgSigma} onChange={(v) => set("bgSigma", v)} help="Gaussian sigma for background subtraction" />
          </div>

          <p className="text-[10px] text-slate-600 text-center mt-6 border-t border-white/[0.06] pt-4">
            🔬 Research prototype<br />Not a medical diagnostic tool
          </p>
        </div>
      )}
    </aside>
  );
}
