interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  icon?: string;
}

export default function MetricCard({ label, value, sub, color = "#38bdf8", icon }: MetricCardProps) {
  return (
    <div className="bg-gradient-to-br from-[#1e2d3d] to-[#162130] border border-brand-sky/20 rounded-xl p-4 text-center transition-all duration-250 hover:border-brand-sky/50 hover:shadow-lg hover:shadow-brand-sky/10 hover:-translate-y-0.5">
      {icon && <div className="text-xl mb-1">{icon}</div>}
      <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-2">{label}</p>
      <p className="text-2xl font-bold" style={{ color }}>{value}</p>
      {sub && <p className="text-[10px] text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}
