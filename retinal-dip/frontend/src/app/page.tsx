"use client";

import { useState } from "react";
import { Upload, BarChart2, Target, FlaskConical, FileText } from "lucide-react";
import Sidebar, { AlgoParams } from "@/components/ui/Sidebar";
import UploadTab from "@/components/tabs/UploadTab";
import ResultsTab from "@/components/tabs/ResultsTab";
import EvaluationTab from "@/components/tabs/EvaluationTab";
import DebugTab from "@/components/tabs/DebugTab";
import ReportTab from "@/components/tabs/ReportTab";
import { AnalysisResult } from "@/lib/api";
import { cn } from "@/lib/utils";

const DEFAULT_PARAMS: AlgoParams = {
  gaborSigma: 4,
  gaborLambda: 10,
  gaborGamma: 0.5,
  nOrientations: 12,
  vesselMorph: 3,
  tophatSize: 15,
  exudateMorph: 3,
  minArea: 20,
  claheClip: 2,
  bgSigma: 15,
};

const TABS = [
  { id: "upload", label: "Upload & Analyze", icon: Upload },
  { id: "results", label: "Results", icon: BarChart2 },
  { id: "evaluation", label: "Evaluation", icon: Target },
  { id: "debug", label: "Debug View", icon: FlaskConical },
  { id: "report", label: "PDF Report", icon: FileText },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function HomePage() {
  const [params, setParams] = useState<AlgoParams>(DEFAULT_PARAMS);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("upload");

  const handleResult = (r: AnalysisResult) => {
    setResult(r);
    setActiveTab("results");
  };

  return (
    <div className="flex min-h-screen bg-[#0d1117]">
      {/* Sidebar */}
      <Sidebar params={params} onChange={setParams} />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="sticky top-0 z-10 bg-[#0d1117]/90 backdrop-blur-md border-b border-white/[0.06]">
          <div className="px-6 py-4">
            {/* Title */}
            <div
              className="rounded-2xl px-6 py-5 mb-4"
              style={{
                background: "linear-gradient(135deg, #1a2a3a 0%, #0d2137 100%)",
                border: "1px solid rgba(56,189,248,0.3)",
                boxShadow: "0 8px 32px rgba(56,189,248,0.1)",
              }}
            >
              <h1 className="text-2xl font-bold gradient-text">🩺 Retinal Image Analysis Platform</h1>
              <p className="text-sm text-slate-500 mt-1">
                Classical Digital Image Processing · Blood Vessel Segmentation · Hard-Exudate Candidate Detection
              </p>
            </div>

            {/* Disclaimer */}
            <div
              className="rounded-xl px-4 py-2.5 mb-4 text-center text-xs font-medium text-red-300"
              style={{
                background: "linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06))",
                border: "1px solid rgba(239,68,68,0.4)",
              }}
            >
              ⚠️ RESEARCH / DECISION-SUPPORT PROTOTYPE — NOT A MEDICAL DIAGNOSIS.
              Candidate regions detected by classical DIP algorithms.
              Clinical review by a qualified ophthalmologist is required.
            </div>

            {/* Tabs */}
            <div className="flex gap-1 overflow-x-auto">
              {TABS.map(({ id, label, icon: Icon }) => {
                const hasResult = result !== null;
                const disabled = id !== "upload" && !hasResult;
                return (
                  <button
                    key={id}
                    onClick={() => !disabled && setActiveTab(id)}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-200",
                      activeTab === id
                        ? "bg-brand-sky/15 text-brand-sky border border-brand-sky/30"
                        : disabled
                        ? "text-slate-700 cursor-not-allowed"
                        : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] border border-transparent"
                    )}
                  >
                    <Icon size={13} />
                    {label}
                    {id !== "upload" && !hasResult && (
                      <span className="text-[9px] text-slate-700 ml-1">•</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </header>

        {/* Tab content */}
        <main className="flex-1 p-6">
          {activeTab === "upload" && (
            <UploadTab params={params} onResult={handleResult} />
          )}
          {activeTab === "results" && <ResultsTab result={result} />}
          {activeTab === "evaluation" && <EvaluationTab result={result} />}
          {activeTab === "debug" && <DebugTab result={result} />}
          {activeTab === "report" && <ReportTab result={result} />}
        </main>

        {/* Footer */}
        <footer className="border-t border-white/[0.05] px-6 py-3 text-center">
          <p className="text-[10px] text-slate-700">
            Retinal DIP Project · Classical Digital Image Processing · IPCV Academic Research Prototype · Not for clinical use.
          </p>
        </footer>
      </div>
    </div>
  );
}
