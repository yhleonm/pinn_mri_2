export type B0FieldStrength = 1.5 | 3.0 | 7.0;
export type TrajectoryType = "cartesian" | "radial" | "poisson";
export type AccelerationFactor = 4 | 8;

export interface CrossSectionProfile {
  row_idx: number;
  x_mm?: number[];
  sigma_gt: number[];
  sigma_helmholtz: number[];
  sigma_cr_pinn: number[];
}

export interface ThermalCurves {
  time_min: number[];
  t_tumor_c: number[];
  t_ventricle_c: number[];
  t_healthy_c: number[];
  delta_t_tumor_c?: number[];
  cem43_tumor_min?: number[];
}

export interface ElectroThermalMapsPayload {
  grid_size: number;
  b0_tesla: number;
  larmor_mhz: number;
  dataset_source?: string;
  dataset_title?: string;
  has_ground_truth?: boolean;
  mask: number[][];
  mag_gt: number[][];
  mag_zf: number[][];
  mag_pred: number[][];
  sigma_gt: number[][];
  sigma_pred: number[][];
  sigma_helmholtz_ref: number[][];
  perfusion_wb_gt: number[][];
  sar_map: number[][];
  temp_map_c: number[][];
  cem43_map: number[][];
  cross_section_profile?: CrossSectionProfile;
  thermal_curves?: ThermalCurves;
}

export interface TrainingTelemetryEvent {
  type: "epoch" | "completed" | "stopped" | "error";
  epoch: number;
  total_epochs: number;
  total_loss: number;
  loss_kspace: number;
  loss_ept: number;
  loss_coil: number;
  loss_harmonic: number;
  lambda_ept: number;
  lambda_harm: number;
  psnr_db: number;
  ssim: number;
  nrmse_sigma: number;
  nrmse_eps: number;
  tumor_sigma_mean: number;
  ventricle_sigma_mean: number;
  max_temp_c: number;
  max_cem43_min: number;
  iec_safety_status: string;
  ept_active: boolean;
  elapsed_ms: number;
  thermal_timeline?: ThermalCurves;
  profile_1d?: CrossSectionProfile;
  maps_f32_b64?: Record<string, string>;
  message?: string;
}

export interface TrainingConfigRequest {
  b0_tesla: B0FieldStrength;
  trajectory: TrajectoryType;
  acceleration: AccelerationFactor;
  snr_db: number;
  epochs: number;
  learning_rate: number;
  lambda_ept: number;
  use_convection_reaction: boolean;
  use_gradnorm: boolean;
  exposure_minutes: number;
  slice_idx: number;
}

export interface ThermalSimulationResponse {
  duration_min: number;
  tumor_perfusion_ratio: number;
  max_temp_c: number;
  max_delta_t_c: number;
  max_cem43_min: number;
  iec_safety_status: string;
  timeline: ThermalCurves;
  profile_1d: CrossSectionProfile;
  maps_f32_b64: Record<string, string>;
}

export function decodeF32B64ToMatrix(
  b64: string | undefined,
  height = 128,
  width = 128,
): number[][] {
  if (!b64) {
    return Array.from({ length: height }, () => new Array(width).fill(0));
  }
  const binary = window.atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const floats = new Float32Array(bytes.buffer);
  const matrix: number[][] = [];
  for (let r = 0; r < height; r++) {
    const row: number[] = new Array(width);
    const offset = r * width;
    for (let c = 0; c < width; c++) {
      row[c] = floats[offset + c] ?? 0;
    }
    matrix.push(row);
  }
  return matrix;
}
