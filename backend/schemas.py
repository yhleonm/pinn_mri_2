"""Pydantic Schemas for Electro-Thermal MRI-PINN 2.0, Pennes Bioheat Simulation, and 1D Profiles."""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PhantomConfigRequest(BaseModel):
    """Parameters to generate or update multi-coil k-space and subsampling mask."""

    height: int = Field(default=128, ge=64, le=256)
    width: int = Field(default=128, ge=64, le=256)
    num_coils: int = Field(default=8, ge=2, le=32)
    b0_tesla: float = Field(default=3.0, ge=0.5, le=11.7)
    fov_x_m: float = Field(default=0.24, gt=0.05, le=0.50)
    fov_y_m: float = Field(default=0.24, gt=0.05, le=0.50)
    noise_snr_db: float = Field(default=36.0, ge=10.0, le=60.0)
    mask_type: Literal["cartesian", "radial", "poisson"] = "poisson"
    acceleration: int = Field(default=4, ge=1, le=16)
    acs_lines: int = Field(default=18, ge=4, le=64)
    exposure_duration_min: float = Field(default=30.0, ge=1.0, le=60.0)
    tumor_perfusion_ratio: float = Field(default=0.25, ge=0.05, le=1.0)


class ThermalSimulationRequest(BaseModel):
    """Interactive request for Pennes Bioheat transient exposure simulation (/api/thermal-simulation)."""

    duration_min: float = Field(default=30.0, ge=1.0, le=60.0)
    tumor_perfusion_ratio: float = Field(default=0.25, ge=0.05, le=1.0)
    rf_duty_scale: float = Field(default=4.2, ge=0.5, le=12.0)
    cross_section_row: Optional[int] = Field(default=None, ge=0, le=255)


class ThermalSimulationResponse(BaseModel):
    """Response from interactive Pennes Bioheat simulation."""

    duration_min: float
    tumor_perfusion_ratio: float
    max_temp_c: float
    max_delta_t_c: float
    max_cem43_min: float
    iec_safety_status: str
    timeline: Dict[str, List[float]]
    profile_1d: Dict[str, Any]
    maps_f32_b64: Dict[str, str]


class TrainingStartConfig(BaseModel):
    """Physical and neural hyperparameters to start GPU PINN training via WebSocket (/ws/train)."""

    b0_tesla: float = Field(default=3.0, ge=0.5, le=11.7)
    mask_type: Literal["cartesian", "radial", "poisson"] = "poisson"
    acceleration: int = Field(default=4, ge=1, le=16)
    acs_lines: int = Field(default=18, ge=4, le=64)
    epochs: int = Field(default=55, ge=5, le=500)
    warmup_epochs: int = Field(default=12, ge=0, le=100)
    use_convection_reaction: bool = True
    use_gradnorm: bool = True
    exposure_duration_min: float = Field(default=30.0, ge=1.0, le=60.0)
    tumor_perfusion_ratio: float = Field(default=0.25, ge=0.05, le=1.0)
    omega_anatomy: float = Field(default=30.0, ge=5.0, le=60.0)
    omega_coil: float = Field(default=5.0, ge=1.0, le=20.0)
    omega_ept: float = Field(default=18.0, ge=2.0, le=40.0)
    lr_anatomy: float = Field(default=7.5e-4, gt=1e-6, le=1e-2)
    lr_coil: float = Field(default=1.2e-3, gt=1e-6, le=1e-2)
    lr_ept: float = Field(default=2.8e-3, gt=1e-6, le=1e-2)
    lambda_kspace: float = Field(default=1.0, ge=0.01, le=20.0)
    lambda_ept_sigma: float = Field(default=0.06, ge=0.0, le=5.0)
    lambda_ept_eps: float = Field(default=0.015, ge=0.0, le=5.0)
    lambda_coil_smooth: float = Field(default=0.02, ge=0.0, le=2.0)
    lambda_harmonic_b0: float = Field(default=0.012, ge=0.0, le=2.0)
    telemetry_interval: int = Field(default=2, ge=1, le=25)


class DatasetStateResponse(BaseModel):
    """Current GPU volume state, baseline metrics, thermal dosimetry, 1D profile, and Float32 Base64 maps."""

    source: str
    height: int
    width: int
    num_coils: int
    b0_tesla: float
    larmor_freq_mhz: float
    fov_x_m: float
    fov_y_m: float
    mask_type: str
    acceleration: int
    acs_lines: int
    sampling_fraction: float
    zero_filled_psnr: float
    zero_filled_ssim: float
    gpu_name: str
    vram_allocated_mb: float
    max_temp_c: float
    max_cem43_min: float
    iec_safety_status: str
    thermal_timeline: Dict[str, List[float]]
    profile_1d: Dict[str, Any]
    maps_f32_b64: Dict[str, str]


class EpochTelemetryPayload(BaseModel):
    """Real-time telemetry message emitted by WebSocket (/ws/train) on each epoch."""

    type: Literal["epoch", "completed", "stopped", "error"]
    epoch: int
    total_epochs: int
    total_loss: float
    loss_kspace: float
    loss_ept: float
    loss_coil: float
    loss_harmonic: float
    psnr_db: float
    ssim: float
    nrmse_sigma: float
    nrmse_eps: float
    tumor_sigma_mean: float
    ventricle_sigma_mean: float
    max_temp_c: float
    max_cem43_min: float
    iec_safety_status: str
    lambda_ept: float
    lambda_harm: float
    ept_active: bool
    elapsed_ms: float
    thermal_timeline: Optional[Dict[str, List[float]]] = None
    profile_1d: Optional[Dict[str, Any]] = None
    maps_f32_b64: Optional[Dict[str, str]] = None
    message: Optional[str] = None
