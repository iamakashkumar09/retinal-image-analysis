import { b64ToSrc } from "@/lib/utils";
import Image from "next/image";

interface ImagePanelProps {
  b64: string;
  label: string;
  badge?: string;
  badgeColor?: string;
}

export default function ImagePanel({ b64, label, badge, badgeColor = "#38bdf8" }: ImagePanelProps) {
  if (!b64) {
    return (
      <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-3 flex flex-col items-center justify-center min-h-[180px]">
        <div className="skeleton w-full h-full rounded-lg min-h-[140px]" />
        <p className="text-xs text-slate-600 mt-2">{label}</p>
      </div>
    );
  }

  return (
    <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-3 flex flex-col animate-fadeIn">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-medium text-slate-400">{label}</p>
        {badge && (
          <span
            className="text-[10px] font-semibold px-2 py-0.5 rounded-full border"
            style={{ color: badgeColor, borderColor: `${badgeColor}66`, background: `${badgeColor}22` }}
          >
            {badge}
          </span>
        )}
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={b64ToSrc(b64)}
        alt={label}
        className="w-full rounded-lg object-contain"
        style={{ maxHeight: 300 }}
      />
    </div>
  );
}
