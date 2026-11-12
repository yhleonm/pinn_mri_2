"""REST Endpoints for K-Space management, Pennes Bioheat Simulation, 1D Profiles, and .h5 Datasets."""

from __future__ import annotations
import base64
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.data.kspace_loader import (
    SyntheticPhantomDataset,
    _build_physiological_perfusion_and_rois,
    _gabriel_dispersion_scale,
    generate_numerical_brain_phantom,
    generate_realistic_brain_ept_phantom,
    generate_shepp_logan_ept_phantom,
    load_h5_kspace_slice,
)
from src.data.masks import create_sampling_mask
from src.physics.cr_ept_operator import DifferentiableCrEPT
from src.physics.pennes_bioheat import PennesBioheatSolver
from src.physics.sense_operator import DifferentiableSENSE
from src.training.trainer import (
    ElectroThermalPINNTrainer,
    compute_psnr,
    compute_ssim,
    separate_harmonic_b0_and_sigma_prior,
    unwrap_phase_2d,
)
from ..schemas import (
    DatasetStateResponse,
    PhantomConfigRequest,
    ThermalSimulationRequest,
    ThermalSimulationResponse,
)

router = APIRouter(prefix="/api", tags=["electro-thermal-kspace"])


def tensor_to_f32_b64(tensor: torch.Tensor) -> str:
    """Converts a 2D PyTorch tensor to little-endian IEEE-754 float32 Base64 for zero-copy WebGL/Canvas decoding."""
    arr = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    return base64.b64encode(arr.tobytes()).decode("ascii")


class GPUSessionManager:
    """Persistent GPU session manager for k-space, ElectroThermalMRIPINN, and Pennes Bioheat solver."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sense_op = DifferentiableSENSE().to(self.device)
        self.bioheat_solver = PennesBioheatSolver().to(self.device)
        self.dataset: Optional[SyntheticPhantomDataset] = None
        self.source: str = "cortical_brain_tumor_8coils_3T.h5"
        self.mask_type: str = "poisson"
        self.acceleration: int = 4
        self.acs_lines: int = 18
        self.noise_snr_db: float = 36.0
        self.exposure_duration_min: float = 30.0
        self.tumor_perfusion_ratio: float = 0.25
        self.rf_duty_scale: float = 5.6
        self.cross_section_row: int = 57
        self.trainer: Optional[ElectroThermalPINNTrainer] = None
        self.last_maps: Dict[str, Any] = {}

    def initialize_default(self) -> None:
        """Initializes the anthropomorphic cortical glioma phantom with bioheat maps on GPU."""
        self.dataset = generate_realistic_brain_ept_phantom(
            height=128,
            width=128,
            num_coils=8,
            fov_x_m=0.24,
            fov_y_m=0.24,
            b0_tesla=3.0,
            noise_snr_db=36.0,
            mask_type="poisson",
            acceleration=4,
            acs_lines=18,
            device=self.device,
        )
        self.source = "cortical_brain_tumor_8coils_3T.h5"
        self._precompute_initial_priors()

    def _precompute_initial_priors(self) -> None:
        """Computes initial Helmholtz vs Convection-Reaction maps and Pennes Bioheat state before training."""
        if self.dataset is None:
            return
        ds = self.dataset
        with torch.no_grad():
            c_ref = self.sense_op.estimate_initial_coil_maps(ds.kspace_masked, acs_width=self.acs_lines)
            m_it = self.sense_op.adjoint(ds.kspace_masked, coil_maps=c_ref)
            for _ in range(40):
                k_it = self.sense_op.forward(m_it, c_ref, mask=ds.mask)
                m_it = m_it - 0.85 * self.sense_op.adjoint(k_it - ds.kspace_masked, coil_maps=c_ref)
            u_s = unwrap_phase_2d(torch.angle(m_it), mag=torch.abs(m_it))
            _, _, sig_cr = separate_harmonic_b0_and_sigma_prior(
                m_it, ds.fov_x_m, ds.fov_y_m, ds.b0_tesla, unwrapped_phase=u_s, use_convection_reaction=True
            )
            _, _, sig_helm = separate_harmonic_b0_and_sigma_prior(
                m_it, ds.fov_x_m, ds.fov_y_m, ds.b0_tesla, unwrapped_phase=u_s, use_convection_reaction=False
            )
            self.last_maps["sigma_helmholtz"] = sig_helm * ds.tissue_mask_gt
            self.last_maps["sigma_cr_init"] = sig_cr * ds.tissue_mask_gt

    def update_mask_and_b0(
        self,
        mask_type: str,
        acceleration: int,
        acs_lines: int,
        b0_tesla: float,
        exposure_duration_min: Optional[float] = None,
        tumor_perfusion_ratio: Optional[float] = None,
    ) -> None:
        """Updates k-space sampling mask P(k), B0 field, and physiological perfusion parameters."""
        if self.dataset is None:
            self.initialize_default()
        assert self.dataset is not None

        _, h, w = self.dataset.kspace_full.shape
        old_b0 = float(self.dataset.b0_tesla)
        self.mask_type = mask_type
        self.acceleration = acceleration
        self.acs_lines = acs_lines
        self.dataset.b0_tesla = b0_tesla
        if exposure_duration_min is not None:
            self.exposure_duration_min = float(exposure_duration_min)
        if tumor_perfusion_ratio is not None:
            self.tumor_perfusion_ratio = float(tumor_perfusion_ratio)

        if abs(b0_tesla - old_b0) > 1e-3 and old_b0 > 0.1:
            sigma_scale, eps_scale = _gabriel_dispersion_scale(b0_tesla=float(b0_tesla), ref_b0=old_b0)
            scale_b0 = (float(b0_tesla) / old_b0) * sigma_scale
            self.dataset.sigma_gt = self.dataset.sigma_gt * sigma_scale
            self.dataset.eps_r_gt = torch.where(
                self.dataset.tissue_mask_gt > 0.5,
                torch.clamp(self.dataset.eps_r_gt * eps_scale, min=1.0, max=95.0),
                self.dataset.eps_r_gt,
            )
            self.dataset.phi_tr_gt = self.dataset.phi_tr_gt * scale_b0
            self.dataset.phi_total_gt = self.dataset.phi_tr_gt + self.dataset.phi_b0_gt
            m_cplx_new = torch.polar(self.dataset.m_mag_gt, self.dataset.phi_total_gt)
            self.dataset.kspace_full = self.sense_op.forward(
                m_cplx_new, self.dataset.coil_maps_gt, mask=None
            )

        wb_gt, q_met_gt, rho_gt, tumor_mask_gt, vent_mask_gt = _build_physiological_perfusion_and_rois(
            self.dataset.sigma_gt,
            self.dataset.tissue_mask_gt,
            b0_tesla=b0_tesla,
            tumor_perfusion_ratio=self.tumor_perfusion_ratio,
        )
        self.dataset.perfusion_wb_gt = wb_gt
        self.dataset.q_met_gt = q_met_gt
        self.dataset.rho_gt = rho_gt
        self.dataset.tumor_mask_gt = tumor_mask_gt
        self.dataset.ventricle_mask_gt = vent_mask_gt

        mask_t = create_sampling_mask(
            height=h,
            width=w,
            mask_type=mask_type,
            acceleration=acceleration,
            acs_lines=acs_lines,
            device=self.device,
        )
        kspace_masked = self.dataset.kspace_full * mask_t.unsqueeze(0)
        coil_est = self.sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
        zero_filled = self.sense_op.adjoint(kspace_masked, coil_maps=coil_est)

        self.dataset.mask = mask_t
        self.dataset.kspace_masked = kspace_masked
        self.dataset.zero_filled = zero_filled
        self._precompute_initial_priors()

    def compute_current_profile_1d(
        self,
        sigma_cr: torch.Tensor,
        sigma_helm: torch.Tensor,
        row_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        assert self.dataset is not None
        ds = self.dataset
        _, h, w = ds.kspace_full.shape
        r_idx = row_idx if row_idx is not None else self.cross_section_row
        r_idx = max(0, min(h - 1, int(r_idx)))
        self.cross_section_row = r_idx

        x_mm = torch.linspace(-ds.fov_x_m * 500.0, ds.fov_x_m * 500.0, steps=w).tolist()
        t_row = ds.tissue_mask_gt[r_idx, :]
        gt_line = (ds.sigma_gt[r_idx, :] * t_row).detach().cpu().tolist()
        helm_line = (sigma_helm[r_idx, :] * t_row).detach().cpu().tolist()
        cr_line = (sigma_cr[r_idx, :] * t_row).detach().cpu().tolist()

        return {
            "row_idx": r_idx,
            "x_mm": [round(v, 2) for v in x_mm],
            "sigma_gt": [round(v, 4) for v in gt_line],
            "sigma_helmholtz": [round(v, 4) for v in helm_line],
            "sigma_cr_pinn": [round(v, 4) for v in cr_line],
        }

    def build_state_response(self) -> DatasetStateResponse:
        if self.dataset is None:
            self.initialize_default()
        assert self.dataset is not None

        ds = self.dataset
        nc, h, w = ds.kspace_full.shape
        zf_mag = torch.abs(ds.zero_filled)
        zf_psnr = compute_psnr(zf_mag, ds.m_mag_gt)
        zf_ssim = compute_ssim(zf_mag, ds.m_mag_gt)

        k_full_mag = torch.sqrt(torch.sum(torch.abs(ds.kspace_full) ** 2, dim=0))
        k_log_full = torch.log1p(k_full_mag * 25.0)
        k_masked_mag = torch.sqrt(torch.sum(torch.abs(ds.kspace_masked) ** 2, dim=0))
        k_log_masked = torch.log1p(k_masked_mag * 25.0)

        gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        vram_mb = (
            torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)
            if self.device.type == "cuda"
            else 0.0
        )

        m_pred = self.last_maps.get("m_mag", zf_mag)
        sigma_helm = self.last_maps.get("sigma_helmholtz", ds.sigma_gt * 0.82)
        sigma_pred = self.last_maps.get("sigma", self.last_maps.get("sigma_cr_init", ds.sigma_gt * 0.0))
        eps_pred = self.last_maps.get("eps_r", ds.eps_r_gt * 0.0)
        phi_tr_pred = self.last_maps.get("phi_tr", torch.angle(ds.zero_filled))

        sar_gt = (
            DifferentiableCrEPT.compute_local_sar(
                sigma=ds.sigma_gt,
                b1_mag=ds.m_mag_gt,
                coords_norm=None,
                b0_tesla=ds.b0_tesla,
            )
            * ds.tissue_mask_gt
        )
        sar_pred = self.last_maps.get(
            "sar",
            DifferentiableCrEPT.compute_local_sar(
                sigma=sigma_pred if float(sigma_pred.max().item()) > 0.05 else ds.sigma_gt,
                b1_mag=m_pred,
                coords_norm=None,
                b0_tesla=ds.b0_tesla,
            )
            * ds.tissue_mask_gt,
        )

        # Ground truth & predicted Pennes bioheat maps
        self.bioheat_solver.rf_duty_scale = self.rf_duty_scale
        wb_map = ds.perfusion_wb_gt if ds.perfusion_wb_gt is not None else torch.full_like(ds.sigma_gt, 9.0)
        thermal_gt = self.bioheat_solver.simulate_transient_exposure(
            sar_w_kg=sar_gt,
            perfusion_wb=wb_map,
            tissue_mask=ds.tissue_mask_gt,
            tumor_mask=ds.tumor_mask_gt,
            ventricle_mask=ds.ventricle_mask_gt,
            rho_kg_m3=ds.rho_gt,
            q_met=ds.q_met_gt,
            duration_min=self.exposure_duration_min,
        )
        thermal_pred = self.bioheat_solver.simulate_transient_exposure(
            sar_w_kg=sar_pred,
            perfusion_wb=wb_map,
            tissue_mask=ds.tissue_mask_gt,
            tumor_mask=ds.tumor_mask_gt,
            ventricle_mask=ds.ventricle_mask_gt,
            rho_kg_m3=ds.rho_gt,
            q_met=ds.q_met_gt,
            duration_min=self.exposure_duration_min,
        )

        temp_c_pred = self.last_maps.get("temperature_c", thermal_pred["temperature_c_2d"])
        cem43_pred = self.last_maps.get("cem43_min", thermal_pred["cem43_min_2d"])
        timeline = self.last_maps.get("thermal_timeline", thermal_pred["timeline"])
        iec_status = str(self.last_maps.get("iec_safety_status", thermal_pred["iec_safety_status"]))
        profile_1d = self.compute_current_profile_1d(sigma_cr=sigma_pred, sigma_helm=sigma_helm)

        maps_f32_b64 = {
            "zero_filled_mag": tensor_to_f32_b64(zf_mag),
            "m_mag": tensor_to_f32_b64(m_pred),
            "sigma": tensor_to_f32_b64(sigma_pred),
            "sigma_helmholtz": tensor_to_f32_b64(sigma_helm),
            "eps_r": tensor_to_f32_b64(eps_pred),
            "sar": tensor_to_f32_b64(sar_pred),
            "temperature_c": tensor_to_f32_b64(temp_c_pred),
            "cem43_min": tensor_to_f32_b64(cem43_pred),
            "perfusion_wb": tensor_to_f32_b64(wb_map),
            "phi_tr": tensor_to_f32_b64(phi_tr_pred),
            "m_mag_gt": tensor_to_f32_b64(ds.m_mag_gt),
            "sigma_gt": tensor_to_f32_b64(ds.sigma_gt),
            "eps_r_gt": tensor_to_f32_b64(ds.eps_r_gt),
            "sar_gt": tensor_to_f32_b64(sar_gt),
            "temperature_c_gt": tensor_to_f32_b64(thermal_gt["temperature_c_2d"]),
            "cem43_min_gt": tensor_to_f32_b64(thermal_gt["cem43_min_2d"]),
            "phi_tr_gt": tensor_to_f32_b64(ds.phi_tr_gt),
            "phi_b0_gt": tensor_to_f32_b64(ds.phi_b0_gt),
            "kspace_log_full": tensor_to_f32_b64(k_log_full),
            "kspace_log_masked": tensor_to_f32_b64(k_log_masked),
            "mask": tensor_to_f32_b64(ds.mask),
            "tissue_mask": tensor_to_f32_b64(ds.tissue_mask_gt),
        }

        sampling_frac = float(torch.mean(ds.mask).item())
        return DatasetStateResponse(
            source=self.source,
            height=h,
            width=w,
            num_coils=nc,
            b0_tesla=ds.b0_tesla,
            larmor_freq_mhz=round(42.58 * ds.b0_tesla, 2),
            fov_x_m=ds.fov_x_m,
            fov_y_m=ds.fov_y_m,
            mask_type=self.mask_type,
            acceleration=self.acceleration,
            acs_lines=self.acs_lines,
            sampling_fraction=sampling_frac,
            zero_filled_psnr=zf_psnr,
            zero_filled_ssim=zf_ssim,
            gpu_name=gpu_name,
            vram_allocated_mb=round(vram_mb, 2),
            max_temp_c=float(temp_c_pred.max().item()),
            max_cem43_min=float(cem43_pred.max().item()),
            iec_safety_status=iec_status,
            thermal_timeline=timeline,
            profile_1d=profile_1d,
            maps_f32_b64=maps_f32_b64,
        )


gpu_session = GPUSessionManager()


@router.get("/state", response_model=DatasetStateResponse)
async def get_current_state() -> DatasetStateResponse:
    """Returns current GPU k-space state, electro-thermal maps, 1D cross-section, and bioheat curves."""
    return gpu_session.build_state_response()


@router.post("/phantom", response_model=DatasetStateResponse)
async def generate_or_update_phantom(req: PhantomConfigRequest) -> DatasetStateResponse:
    """Generates or updates the multi-coil phantom with requested B0, trajectory, and bioheat parameters."""
    gpu_session.mask_type = req.mask_type
    gpu_session.acceleration = req.acceleration
    gpu_session.acs_lines = req.acs_lines
    gpu_session.noise_snr_db = req.noise_snr_db
    gpu_session.exposure_duration_min = req.exposure_duration_min
    gpu_session.tumor_perfusion_ratio = req.tumor_perfusion_ratio
    gpu_session.source = f"synthetic_shepp_logan_{req.num_coils}ch"
    gpu_session.last_maps.clear()

    gpu_session.dataset = generate_shepp_logan_ept_phantom(
        height=req.height,
        width=req.width,
        num_coils=req.num_coils,
        fov_x_m=req.fov_x_m,
        fov_y_m=req.fov_y_m,
        b0_tesla=req.b0_tesla,
        noise_snr_db=req.noise_snr_db,
        mask_type=req.mask_type,
        acceleration=req.acceleration,
        acs_lines=req.acs_lines,
        device=gpu_session.device,
    )
    gpu_session._precompute_initial_priors()
    return gpu_session.build_state_response()


@router.post("/load-sample-h5", response_model=DatasetStateResponse)
async def load_sample_h5(
    req: PhantomConfigRequest,
    sample_name: Optional[str] = None,
) -> DatasetStateResponse:
    """Loads anthropomorphic or real patient in-vivo multi-coil .h5 datasets (Brain, Knee, Torso, Cardiac)."""
    name_str = str(sample_name or "").lower().strip()
    if name_str in ("knee", "rodilla"):
        sample_path = Path("data/real_in_vivo_knee_8coils_3T.h5")
    elif name_str in ("torso", "abdomen"):
        sample_path = Path("data/real_in_vivo_torso_8coils_3T.h5")
    elif name_str in ("cardiac", "cardiaco", "corazon"):
        sample_path = Path("data/real_in_vivo_cardiac_6coils_3T.h5")
    elif name_str in ("brain_clinical", "clinical"):
        sample_path = Path("data/clinical_in_vivo_brain_8coils.h5")
    else:
        sample_path = Path("data/cortical_brain_tumor_8coils_3T.h5")

    if not sample_path.exists():
        sample_path = Path("data/raw") / sample_path.name

    if not sample_path.exists():
        gpu_session.initialize_default()
        return gpu_session.build_state_response()

    gpu_session.dataset = load_h5_kspace_slice(
        h5_path=sample_path,
        mask_type=req.mask_type,
        acceleration=req.acceleration,
        acs_lines=req.acs_lines,
        fov_x_m=req.fov_x_m,
        fov_y_m=req.fov_y_m,
        b0_tesla=req.b0_tesla,
        device=gpu_session.device,
    )
    gpu_session.source = sample_path.name
    gpu_session.mask_type = req.mask_type
    gpu_session.acceleration = req.acceleration
    gpu_session.acs_lines = req.acs_lines
    gpu_session.exposure_duration_min = req.exposure_duration_min
    gpu_session.tumor_perfusion_ratio = req.tumor_perfusion_ratio
    gpu_session.last_maps.clear()
    gpu_session._precompute_initial_priors()
    return gpu_session.build_state_response()


@router.post("/load-uc-berkeley", response_model=DatasetStateResponse)
async def load_uc_berkeley_real(req: PhantomConfigRequest) -> DatasetStateResponse:
    """Loads the public 8-coil in-vivo human brain dataset from UC Berkeley / BART ISMRM."""
    ucb_path = Path("data/external_public_brain_raw.h5")
    if not ucb_path.exists():
        ucb_path = Path("data/clinical_in_vivo_brain_8coils.h5")
    if not ucb_path.exists():
        raise HTTPException(status_code=404, detail="UC Berkeley dataset not found in data/")

    gpu_session.dataset = load_h5_kspace_slice(
        h5_path=ucb_path,
        mask_type=req.mask_type,
        acceleration=req.acceleration,
        acs_lines=req.acs_lines,
        fov_x_m=req.fov_x_m,
        fov_y_m=req.fov_y_m,
        b0_tesla=req.b0_tesla,
        device=gpu_session.device,
    )
    gpu_session.source = ucb_path.name
    gpu_session.mask_type = req.mask_type
    gpu_session.acceleration = req.acceleration
    gpu_session.acs_lines = req.acs_lines
    gpu_session.exposure_duration_min = req.exposure_duration_min
    gpu_session.tumor_perfusion_ratio = req.tumor_perfusion_ratio
    gpu_session.last_maps.clear()
    gpu_session._precompute_initial_priors()
    return gpu_session.build_state_response()


@router.post("/upload-h5", response_model=DatasetStateResponse)
async def upload_h5_file(
    file: UploadFile = File(...),
    mask_type: str = "poisson",
    acceleration: int = 4,
    acs_lines: int = 18,
    b0_tesla: float = 3.0,
) -> DatasetStateResponse:
    """Uploads a custom .h5 file and initializes multi-physics priors on GPU."""
    if not file.filename or not file.filename.endswith((".h5", ".hdf5")):
        raise HTTPException(status_code=400, detail="File must have .h5 or .hdf5 extension")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / Path(file.filename).name
    content = await file.read()
    dest_path.write_bytes(content)

    gpu_session.dataset = load_h5_kspace_slice(
        h5_path=dest_path,
        mask_type=mask_type,
        acceleration=acceleration,
        acs_lines=acs_lines,
        b0_tesla=b0_tesla,
        device=gpu_session.device,
    )
    gpu_session.source = dest_path.name
    gpu_session.mask_type = mask_type
    gpu_session.acceleration = acceleration
    gpu_session.acs_lines = acs_lines
    gpu_session.last_maps.clear()
    gpu_session._precompute_initial_priors()
    return gpu_session.build_state_response()


@router.post("/thermal-simulation", response_model=ThermalSimulationResponse)
async def simulate_thermal_exposure(req: ThermalSimulationRequest) -> ThermalSimulationResponse:
    """
    Real-time interactive Pennes Bioheat simulation endpoint (/api/thermal-simulation).
    Recomputes T(x, t) [°C], Delta T(t), cumulative CEM43 [min], IEC 60601-2-33 alerts,
    and 1D Cross-Section Profile for any user-selected exposure duration (0-30+ min),
    intratumoral hypoperfusion ratio, or cross-section row.
    """
    if gpu_session.dataset is None:
        gpu_session.initialize_default()
    assert gpu_session.dataset is not None
    ds = gpu_session.dataset

    gpu_session.exposure_duration_min = req.duration_min
    gpu_session.tumor_perfusion_ratio = req.tumor_perfusion_ratio
    gpu_session.rf_duty_scale = req.rf_duty_scale
    gpu_session.bioheat_solver.rf_duty_scale = req.rf_duty_scale

    wb_gt, q_met_gt, rho_gt, tumor_mask_gt, vent_mask_gt = _build_physiological_perfusion_and_rois(
        ds.sigma_gt,
        ds.tissue_mask_gt,
        b0_tesla=ds.b0_tesla,
        tumor_perfusion_ratio=req.tumor_perfusion_ratio,
    )
    ds.perfusion_wb_gt = wb_gt
    ds.q_met_gt = q_met_gt
    ds.rho_gt = rho_gt
    ds.tumor_mask_gt = tumor_mask_gt
    ds.ventricle_mask_gt = vent_mask_gt

    sigma_cr = gpu_session.last_maps.get("sigma", gpu_session.last_maps.get("sigma_cr_init", ds.sigma_gt))
    sigma_helm = gpu_session.last_maps.get("sigma_helmholtz", ds.sigma_gt * 0.82)
    m_mag = gpu_session.last_maps.get("m_mag", torch.abs(ds.zero_filled))

    sar_map = gpu_session.last_maps.get(
        "sar",
        DifferentiableCrEPT.compute_local_sar(
            sigma=sigma_cr,
            b1_mag=m_mag,
            coords_norm=None,
            b0_tesla=ds.b0_tesla,
        )
        * ds.tissue_mask_gt,
    )

    thermal_out = gpu_session.bioheat_solver.simulate_transient_exposure(
        sar_w_kg=sar_map,
        perfusion_wb=wb_gt,
        tissue_mask=ds.tissue_mask_gt,
        tumor_mask=tumor_mask_gt,
        ventricle_mask=vent_mask_gt,
        rho_kg_m3=rho_gt,
        q_met=q_met_gt,
        duration_min=req.duration_min,
    )

    gpu_session.last_maps["temperature_c"] = thermal_out["temperature_c_2d"]
    gpu_session.last_maps["cem43_min"] = thermal_out["cem43_min_2d"]
    gpu_session.last_maps["thermal_timeline"] = thermal_out["timeline"]
    gpu_session.last_maps["iec_safety_status"] = thermal_out["iec_safety_status"]

    profile_1d = gpu_session.compute_current_profile_1d(
        sigma_cr=sigma_cr,
        sigma_helm=sigma_helm,
        row_idx=req.cross_section_row,
    )

    return ThermalSimulationResponse(
        duration_min=req.duration_min,
        tumor_perfusion_ratio=req.tumor_perfusion_ratio,
        max_temp_c=float(thermal_out["max_temp_c"]),
        max_delta_t_c=float(thermal_out["max_delta_t_c"]),
        max_cem43_min=float(thermal_out["max_cem43_min"]),
        iec_safety_status=str(thermal_out["iec_safety_status"]),
        timeline=thermal_out["timeline"],
        profile_1d=profile_1d,
        maps_f32_b64={
            "temperature_c": tensor_to_f32_b64(thermal_out["temperature_c_2d"]),
            "cem43_min": tensor_to_f32_b64(thermal_out["cem43_min_2d"]),
            "perfusion_wb": tensor_to_f32_b64(wb_gt),
        },
    )
