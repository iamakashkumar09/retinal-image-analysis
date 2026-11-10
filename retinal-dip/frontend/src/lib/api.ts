import axios from "axios";

const BASE = "/api/py";

export function getApiErrorMessage(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data;
    if (data && typeof data === "object" && "detail" in data && typeof data.detail === "string" && data.detail.trim()) {
      return data.detail;
    }
    if (err.response?.status === 500 && typeof data === "string" && data.includes("Internal Server Error")) {
      return "Backend proxy error — make sure the Python FastAPI server (port 8000) is running.";
    }
    if (!err.response) {
      return "Cannot reach the API server. Make sure both Next.js and FastAPI (port 8000) are running.";
    }
    return err.message;
  }
  return err instanceof Error ? err.message : fallback;
}

export interface AnalysisParams {
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

export interface Region {
  id: number;
  area: number;
  centroid: [number, number];
  bbox: [number, number, number, number];
}

export interface RegionStats {
  count: number;
  total_area: number;
  largest_area: number;
  avg_area: number;
}

export interface AnalysisResult {
  vessel_density: number;
  region_stats: RegionStats;
  regions: Region[];
  original_b64: string;
  vessel_mask_b64: string;
  exudate_mask_b64: string;
  overlay_b64: string;
  green_b64: string;
  clahe_b64: string;
  corrected_b64: string;
  L_b64: string;
  L_clahe_b64: string;
  response_norm_b64: string;
  tophat_b64: string;
}

export interface Metrics {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  dice: number;
  iou: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  f1: number;
}

export async function analyzeRetina(
  imageFile: File,
  params: AnalysisParams,
  fovFile?: File | null,
  discFile?: File | null
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("image", imageFile);
  if (fovFile) form.append("fov_mask", fovFile);
  if (discFile) form.append("disc_mask", discFile);
  form.append("gabor_sigma", String(params.gaborSigma));
  form.append("gabor_lambda", String(params.gaborLambda));
  form.append("gabor_gamma", String(params.gaborGamma));
  form.append("n_orientations", String(params.nOrientations));
  form.append("vessel_morph", String(params.vesselMorph));
  form.append("tophat_size", String(params.tophatSize));
  form.append("exudate_morph", String(params.exudateMorph));
  form.append("min_area", String(params.minArea));
  form.append("clahe_clip", String(params.claheClip));
  form.append("bg_sigma", String(params.bgSigma));

  const { data } = await axios.post<AnalysisResult>(`${BASE}/analyze`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function evaluateMask(
  predB64: string,
  gtFile: File,
  maskType: "vessel" | "exudate"
): Promise<Metrics> {
  const form = new FormData();
  form.append("pred_b64", predB64);
  form.append("gt_mask", gtFile);
  form.append("mask_type", maskType);

  const { data } = await axios.post<Metrics>(`${BASE}/evaluate`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function generateReport(
  resultPayload: object,
  imageId: string
): Promise<Blob> {
  const { data } = await axios.post(
    `${BASE}/report`,
    { result: resultPayload, image_id: imageId },
    { responseType: "blob" }
  );
  return data;
}
