import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function b64ToSrc(b64: string, mime = "image/png") {
  return `data:${mime};base64,${b64}`;
}

export function metricColor(val: number): string {
  if (val > 0.7) return "#4ade80";
  if (val > 0.4) return "#facc15";
  return "#f87171";
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
