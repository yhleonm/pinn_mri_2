"""Multi-Coil K-Space Loader (.h5 fastMRI / UC Berkeley) and Dispersive Electro-Thermal Brain Phantom."""

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..physics.sense_operator import DifferentiableSENSE
from .masks import create_sampling_mask


@dataclass
class SyntheticPhantomDataset:
    """Physical data container for Multi-Coil MRI-EPT and Pennes Electro-Thermal Dosimetry."""

    kspace_full: torch.Tensor          # (Nc, H, W) complex64
    kspace_masked: torch.Tensor        # (Nc, H, W) complex64
    mask: torch.Tensor                 # (H, W) float32
    zero_filled: torch.Tensor          # (H, W) complex64
    coil_maps_gt: torch.Tensor         # (Nc, H, W) complex64
    m_mag_gt: torch.Tensor             # (H, W) float32
    phi_tr_gt: torch.Tensor            # (H, W) float32
    phi_b0_gt: torch.Tensor            # (H, W) float32
    phi_total_gt: torch.Tensor         # (H, W) float32
    sigma_gt: torch.Tensor             # (H, W) float32 [S/m]
    eps_r_gt: torch.Tensor             # (H, W) float32 [dimensionless]
    tissue_mask_gt: torch.Tensor       # (H, W) float32
    fov_x_m: float
    fov_y_m: float
    b0_tesla: float
    # Electro-Thermal & ROI fields (pinn_mri_2)
    perfusion_wb_gt: Optional[torch.Tensor] = None   # (H, W) float32 [kg/(m^3*s)]
    q_met_gt: Optional[torch.Tensor] = None          # (H, W) float32 [W/m^3]
    rho_gt: Optional[torch.Tensor] = None            # (H, W) float32 [kg/m^3]
    tumor_mask_gt: Optional[torch.Tensor] = None     # (H, W) float32 binary ROI
    ventricle_mask_gt: Optional[torch.Tensor] = None # (H, W) float32 binary ROI


def _gaussian_filter_2d(arr: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """Separable 2D Gaussian filter in PyTorch/NumPy without SciPy dependency."""
    if sigma <= 0.05:
        return arr.astype(np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    t_in = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    t_pad = F.pad(t_in, (radius, radius, radius, radius), mode="reflect")
    t_k = torch.from_numpy(kernel_2d).float().unsqueeze(0).unsqueeze(0)
    t_out = F.conv2d(t_pad, t_k)
    return t_out.squeeze().numpy().astype(np.float32)


def _dst1_1d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Exact Type-I Discrete Sine Transform (DST-I) via antisymmetric FFT."""
    n = x.shape[axis]
    pad_shape = list(x.shape)
    pad_shape[axis] = 1
    zeros = np.zeros(pad_shape, dtype=x.dtype)
    odd_ext = np.concatenate([zeros, x, zeros, -np.flip(x, axis=axis)], axis=axis)
    fft_out = np.fft.fft(odd_ext, axis=axis)
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(1, n + 1)
    return (-0.5 * fft_out[tuple(slc)].imag).astype(np.float64)


def _solve_poisson_dirichlet_2d(
    rhs: np.ndarray,
    dx_m: float,
    dy_m: float,
) -> np.ndarray:
    """Analytically solves 2D homogeneous Dirichlet Poisson equation nabla^2 u = rhs via 2D DST-I."""
    h, w = rhs.shape
    rhs_f64 = rhs.astype(np.float64)
    rhs_hat = _dst1_1d(_dst1_1d(rhs_f64, axis=0), axis=1)

    ky = np.arange(1, h + 1, dtype=np.float64)[:, None]
    kx = np.arange(1, w + 1, dtype=np.float64)[None, :]

    eig_y = (2.0 * np.cos(np.pi * ky / (h + 1)) - 2.0) / (dx_m ** 2)
    eig_x = (2.0 * np.cos(np.pi * kx / (w + 1)) - 2.0) / (dy_m ** 2)
    denom = eig_y + eig_x
    denom = np.where(np.abs(denom) < 1e-12, -1.0, denom)

    u_hat = rhs_hat / denom
    norm_factor = (2.0 / (h + 1)) * (2.0 / (w + 1))
    u = norm_factor * _dst1_1d(_dst1_1d(u_hat, axis=0), axis=1)
    return u.astype(np.float32)


def _gabriel_dispersion_scale(b0_tesla: float, ref_b0: float = 3.0) -> Tuple[float, float]:
    """Cole-Cole dielectric dispersion scaling (Gabriel et al., 1996) relative to 3.0 T."""
    if abs(float(b0_tesla) - float(ref_b0)) < 1e-3:
        return 1.0, 1.0
    ratio = max(float(b0_tesla), 0.5) / max(float(ref_b0), 0.5)
    sigma_factor = float(ratio ** 0.115)
    eps_factor = float(ratio ** (-0.185))
    return sigma_factor, eps_factor


def _build_physiological_perfusion_and_rois(
    sigma_2d: torch.Tensor,
    tissue_mask_2d: torch.Tensor,
    b0_tesla: float = 3.0,
    wb_healthy: float = 9.0,
    tumor_perfusion_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs realistic oncological blood perfusion map omega_b(x) [kg/(m^3*s)],
    metabolic heat Q_m(x) [W/m^3], tissue density rho(x) [kg/m^3], tumor_mask, and ventricle_mask.

    Models necrotic/hypoxic intratumoral hypoperfusion:
        omega_{b,tumor} = tumor_perfusion_ratio * omega_{b,healthy} (~0.25 * 9.0 = 2.25 kg/(m^3*s))
    """
    device = sigma_2d.device
    h, w = sigma_2d.shape
    sig_disp, _ = _gabriel_dispersion_scale(b0_tesla, ref_b0=3.0)
    sigma_3t_eq = sigma_2d / max(sig_disp, 1e-3)

    y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Ventricles (CSF interior ROI: sigma_3T ~ 1.65 S/m)
    th_v1 = math.radians(16.0)
    xr_v1 = (xx + 0.11) * math.cos(th_v1) + (yy + 0.04) * math.sin(th_v1)
    yr_v1 = -(xx + 0.11) * math.sin(th_v1) + (yy + 0.04) * math.cos(th_v1)
    v1_geom = ((xr_v1 / 0.065) ** 2 + (yr_v1 / 0.185) ** 2) <= 1.0

    th_v2 = math.radians(-16.0)
    xr_v2 = (xx - 0.11) * math.cos(th_v2) + (yy + 0.04) * math.sin(th_v2)
    yr_v2 = -(xx - 0.11) * math.sin(th_v2) + (yy + 0.04) * math.cos(th_v2)
    v2_geom = ((xr_v2 / 0.065) ** 2 + (yr_v2 / 0.185) ** 2) <= 1.0

    vent_region = (torch.abs(xx) < 0.28) & (torch.abs(yy) < 0.28)
    ventricle_mask = (((sigma_3t_eq > 1.45) & vent_region) | ((v1_geom | v2_geom) & (sigma_3t_eq > 1.25))).float() * tissue_mask_2d
    if float(ventricle_mask.sum().item()) < 6.0:
        ventricle_mask = ((v1_geom | v2_geom) & (tissue_mask_2d > 0.5)).float()

    # Tumor ROI (Glioma core matching Paper 1 exact evaluation ROI: sigma_3T = 1.20-1.22 S/m)
    t_geom = (((xx - 0.36) / 0.115) ** 2 + ((yy + 0.16) / 0.110) ** 2) <= 1.0
    t_shepp = (((xx - 0.24) / 0.135) ** 2 + ((yy + 0.18) / 0.115) ** 2) <= 1.0
    tumor_mask = ((t_geom & (sigma_3t_eq > 0.95)) | (t_shepp & (sigma_3t_eq > 0.95) & (~t_geom))).float() * tissue_mask_2d
    if float(tumor_mask.sum().item()) < 6.0:
        tumor_mask = (t_geom & (tissue_mask_2d > 0.5)).float()

    # Peritumoral edema halo
    edema_region = (xx > 0.08) & (yy < 0.08) & (tumor_mask < 0.5) & (ventricle_mask < 0.5)
    edema_mask = ((sigma_3t_eq > 0.62) & edema_region & (tissue_mask_2d > 0.5)).float()

    # Healthy cerebral blood perfusion: gray matter ~10.5, white matter ~7.5, average ~9.0 kg/(m^3*s)
    wb_map = torch.full((h, w), float(wb_healthy), device=device, dtype=torch.float32) * tissue_mask_2d
    # Higher perfusion in cortical gray matter
    cortex_mask = ((sigma_3t_eq > 0.48) & (sigma_3t_eq <= 0.68) & (tissue_mask_2d > 0.5)).float()
    wb_map = wb_map + cortex_mask * 1.5
    # Reduced perfusion in peritumoral vasogenic edema (~50% of normal)
    wb_map = torch.where(edema_mask > 0.5, torch.full_like(wb_map, 0.50 * wb_healthy), wb_map)
    # Hypoperfused necrotic/hypoxic glioma core (omega_{b,tumor} = 0.25 * omega_{b,healthy})
    wb_tumor = float(tumor_perfusion_ratio) * float(wb_healthy)
    wb_map = torch.where(tumor_mask > 0.5, torch.full_like(wb_map, wb_tumor), wb_map)
    # CSF in ventricles has negligible direct capillary blood perfusion (~1.2 kg/(m^3*s) choroid plexus exchange)
    wb_map = torch.where(ventricle_mask > 0.5, torch.full_like(wb_map, 1.8), wb_map)
    wb_map = wb_map * tissue_mask_2d

    # Metabolic heat generation Q_m [W/m^3]
    q_met = torch.full((h, w), 9700.0, device=device, dtype=torch.float32) * tissue_mask_2d
    q_met = torch.where(tumor_mask > 0.5, torch.full_like(q_met, 12800.0), q_met)
    q_met = torch.where(ventricle_mask > 0.5, torch.full_like(q_met, 500.0), q_met)

    # Tissue mass density rho [kg/m^3]
    rho = torch.full((h, w), 1040.0, device=device, dtype=torch.float32) * tissue_mask_2d
    rho = torch.where(ventricle_mask > 0.5, torch.full_like(rho, 1007.0), rho)
    rho = torch.where(tissue_mask_2d <= 0.5, torch.full_like(rho, 1.2), rho)

    return wb_map, q_met, rho, tumor_mask, ventricle_mask


def generate_shepp_logan_ept_phantom(
    height: int = 128,
    width: int = 128,
    num_coils: int = 8,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    noise_snr_db: float = 35.0,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    seed: int = 42,
    device: torch.device | str = "cuda",
) -> SyntheticPhantomDataset:
    """Generates Analytical 8-Coil Shepp-Logan EPT + Bioheat Phantom."""
    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    ellipses = [
        (0.0, 0.0, 0.72, 0.86, 0.0, 0.85, 0.18, 22.0),
        (0.0, -0.01, 0.66, 0.79, 0.0, 0.75, 0.42, 48.0),
        (0.0, -0.01, 0.60, 0.72, 0.0, 0.68, 0.55, 56.0),
        (0.0, -0.02, 0.48, 0.58, 0.0, 0.62, 0.34, 41.0),
        (-0.18, 0.02, 0.11, 0.24, 18.0, 0.95, 1.65, 78.0),
        (0.18, 0.02, 0.11, 0.24, -18.0, 0.95, 1.65, 78.0),
        (0.0, 0.28, 0.15, 0.12, 0.0, 0.78, 0.62, 58.0),
        (-0.08, -0.45, 0.08, 0.06, 0.0, 0.82, 0.60, 55.0),
        (0.08, -0.45, 0.08, 0.06, 0.0, 0.82, 0.60, 55.0),
        (0.24, -0.18, 0.15, 0.13, -12.0, 1.00, 1.20, 68.0),
    ]

    m_mag = np.zeros((height, width), dtype=np.float32)
    sigma = np.full((height, width), 0.02, dtype=np.float32)
    eps_r = np.full((height, width), 1.0, dtype=np.float32)
    tissue_mask = np.zeros((height, width), dtype=np.float32)

    for idx, (cx, cy, a, b, theta_deg, mag_v, sig_v, eps_v) in enumerate(ellipses):
        theta = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = xx - cx
        dy = yy - cy
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        inside = ((xr / a) ** 2 + (yr / b) ** 2) <= 1.0
        m_mag[inside] = mag_v
        sigma[inside] = sig_v
        eps_r[inside] = eps_v
        if idx == 0:
            tissue_mask[inside] = 1.0

    sig_disp, eps_disp = _gabriel_dispersion_scale(b0_tesla, ref_b0=3.0)
    sigma[tissue_mask > 0.5] *= sig_disp
    eps_r[tissue_mask > 0.5] = np.clip(eps_r[tissue_mask > 0.5] * eps_disp, 1.0, 100.0)

    m_mag = _gaussian_filter_2d(m_mag, sigma=0.85)
    sigma_smooth = _gaussian_filter_2d(sigma, sigma=0.85)
    eps_r_smooth = _gaussian_filter_2d(eps_r, sigma=0.85)

    mu_0 = 4.0 * math.pi * 1e-7
    f_0 = 42.58e6 * b0_tesla
    omega = 2.0 * math.pi * f_0
    dx_m = fov_x_m / width
    dy_m = fov_y_m / height

    rhs_poisson = (mu_0 * omega * sigma_smooth * tissue_mask).astype(np.float32)
    phi_plus_raw = _solve_poisson_dirichlet_2d(rhs_poisson, dx_m=dx_m, dy_m=dy_m)
    phi_tr_raw = (2.0 * phi_plus_raw).astype(np.float32)

    x_m = xx * (fov_x_m / 2.0)
    y_m = yy * (fov_y_m / 2.0)
    harm_basis = np.stack(
        [np.ones_like(x_m), x_m, y_m, x_m ** 2 - y_m ** 2, x_m * y_m], axis=-1
    )
    ring_mask = (tissue_mask > 0.5) & (((xx / 0.58) ** 2 + (yy / 0.70) ** 2) > 1.0)
    harm_coeffs, _, _, _ = np.linalg.lstsq(harm_basis[ring_mask], phi_tr_raw[ring_mask], rcond=None)
    phi_tr = (phi_tr_raw - np.tensordot(harm_basis, harm_coeffs, axes=([-1], [0]))).astype(np.float32)

    phi_b0 = (14.0 * (x_m ** 2 - y_m ** 2) + 10.0 * (x_m * y_m) + 0.35 * x_m - 0.25 * y_m).astype(np.float32)
    phi_total = phi_tr + phi_b0

    coil_maps = np.zeros((num_coils, height, width), dtype=np.complex64)
    radius_coil = 1.35
    for c in range(num_coils):
        angle = 2.0 * math.pi * c / num_coils
        cx_c = radius_coil * math.cos(angle)
        cy_c = radius_coil * math.sin(angle)
        dist_sq = (xx - cx_c) ** 2 + (yy - cy_c) ** 2
        amp = np.exp(-0.85 * dist_sq)
        phase_c = 0.65 * ((xx - cx_c) * math.sin(angle) - (yy - cy_c) * math.cos(angle))
        coil_maps[c] = (amp * np.exp(1j * phase_c)).astype(np.complex64)

    common_coil_phase = np.angle(np.sum(coil_maps, axis=0, keepdims=True))
    coil_maps = coil_maps * np.exp(-1j * common_coil_phase)
    sos = np.sqrt(np.sum(np.abs(coil_maps) ** 2, axis=0, keepdims=True) + 1e-8)
    coil_maps = (coil_maps / sos).astype(np.complex64)

    sense_op = DifferentiableSENSE()
    m_mag_t = torch.from_numpy(m_mag).to(device=device, dtype=torch.float32)
    phi_tr_t = torch.from_numpy(phi_tr).to(device=device, dtype=torch.float32)
    phi_b0_t = torch.from_numpy(phi_b0).to(device=device, dtype=torch.float32)
    phi_total_t = torch.from_numpy(phi_total).to(device=device, dtype=torch.float32)
    sigma_t = torch.from_numpy(sigma_smooth).to(device=device, dtype=torch.float32)
    eps_r_t = torch.from_numpy(eps_r_smooth).to(device=device, dtype=torch.float32)
    tissue_mask_t = torch.from_numpy(tissue_mask).to(device=device, dtype=torch.float32)
    coil_maps_t = torch.from_numpy(coil_maps).to(device=device, dtype=torch.complex64)

    m_complex_gt = torch.polar(m_mag_t, phi_total_t)
    kspace_clean = sense_op.forward(m_complex_gt, coil_maps_t, mask=None)

    signal_power = torch.mean(torch.abs(kspace_clean) ** 2).item()
    snr_linear = 10.0 ** (noise_snr_db / 10.0)
    noise_std = math.sqrt(signal_power / max(snr_linear, 1e-6) / 2.0)

    noise_real = rng.normal(0.0, noise_std, size=kspace_clean.shape).astype(np.float32)
    noise_imag = rng.normal(0.0, noise_std, size=kspace_clean.shape).astype(np.float32)
    noise_t = torch.complex(
        torch.from_numpy(noise_real).to(device=device),
        torch.from_numpy(noise_imag).to(device=device),
    )
    kspace_full = kspace_clean + noise_t

    mask_t = create_sampling_mask(
        height=height,
        width=width,
        mask_type=mask_type,
        acceleration=acceleration,
        acs_lines=acs_lines,
        seed=seed,
        device=device,
    )
    kspace_masked = kspace_full * mask_t.unsqueeze(0)
    coil_est = sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
    zero_filled = sense_op.adjoint(kspace_masked, coil_maps=coil_est)

    wb_gt, q_met_gt, rho_gt, tumor_mask_gt, vent_mask_gt = _build_physiological_perfusion_and_rois(
        sigma_t, tissue_mask_t, b0_tesla=b0_tesla
    )

    return SyntheticPhantomDataset(
        kspace_full=kspace_full,
        kspace_masked=kspace_masked,
        mask=mask_t,
        zero_filled=zero_filled,
        coil_maps_gt=coil_maps_t,
        m_mag_gt=m_mag_t,
        phi_tr_gt=phi_tr_t,
        phi_b0_gt=phi_b0_t,
        phi_total_gt=phi_total_t,
        sigma_gt=sigma_t,
        eps_r_gt=eps_r_t,
        tissue_mask_gt=tissue_mask_t,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        b0_tesla=b0_tesla,
        perfusion_wb_gt=wb_gt,
        q_met_gt=q_met_gt,
        rho_gt=rho_gt,
        tumor_mask_gt=tumor_mask_gt,
        ventricle_mask_gt=vent_mask_gt,
    )


def generate_realistic_brain_ept_phantom(
    height: int = 128,
    width: int = 128,
    num_coils: int = 8,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    noise_snr_db: float = 36.0,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    seed: int = 42,
    contrast_mode: str = "coupled",
    device: torch.device | str = "cuda",
) -> SyntheticPhantomDataset:
    """
    Generates an anthropomorphic 2D human brain phantom with cortical gyri/sulci,
    deep basal nuclei, lateral ventricles (CSF: sigma = 1.65 S/m), and a focal
    hyper-conductive & hypoperfused glioma in the right temporoparietal lobe
    (sigma_3T = 1.22 S/m, omega_{b,tumor} = 0.25 * omega_{b,healthy}).
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    angle_map = np.arctan2(yy, xx)
    r_ell = np.sqrt((xx / 0.72) ** 2 + (yy / 0.86) ** 2)

    m_mag = np.zeros((height, width), dtype=np.float32)
    sigma = np.full((height, width), 0.02, dtype=np.float32)
    eps_r = np.full((height, width), 1.0, dtype=np.float32)
    tissue_mask = np.zeros((height, width), dtype=np.float32)

    skull_mask = r_ell <= 1.0
    m_mag[skull_mask] = 0.82
    sigma[skull_mask] = 0.18
    eps_r[skull_mask] = 22.0
    tissue_mask[skull_mask] = 1.0

    gyri_mod = 0.028 * np.cos(12.0 * angle_map) + 0.018 * np.sin(18.0 * angle_map)
    cortex_outer = r_ell <= (0.91 + 0.35 * gyri_mod)
    m_mag[cortex_outer] = 0.72
    sigma[cortex_outer] = 0.56
    eps_r[cortex_outer] = 56.0

    wm_mod = 0.045 * np.cos(10.0 * angle_map + 0.4) + 0.030 * np.sin(16.0 * angle_map)
    wm_mask = r_ell <= (0.74 + wm_mod)
    m_mag[wm_mask] = 0.61
    sigma[wm_mask] = 0.34
    eps_r[wm_mask] = 40.0

    fissure = (np.abs(xx) < 0.018) & (np.abs(yy) < 0.72) & cortex_outer
    m_mag[fissure] = 0.80
    sigma[fissure] = 0.62
    eps_r[fissure] = 60.0

    deep_nuclei = [
        (-0.14, 0.06, 0.09, 0.14, 14.0, 0.70, 0.54, 54.0),
        (0.14, 0.06, 0.09, 0.14, -14.0, 0.70, 0.54, 54.0),
        (-0.26, -0.06, 0.07, 0.15, 10.0, 0.69, 0.52, 53.0),
        (0.26, 0.12, 0.06, 0.11, -12.0, 0.69, 0.52, 53.0),
    ]
    for cx_n, cy_n, a_n, b_n, th_n, mv, sv, ev in deep_nuclei:
        th = math.radians(th_n)
        xr = (xx - cx_n) * math.cos(th) + (yy - cy_n) * math.sin(th)
        yr = -(xx - cx_n) * math.sin(th) + (yy - cy_n) * math.cos(th)
        ind = ((xr / a_n) ** 2 + (yr / b_n) ** 2) <= 1.0
        m_mag[ind] = mv
        sigma[ind] = sv
        eps_r[ind] = ev

    ventricles = [
        (-0.11, -0.04, 0.075, 0.21, 16.0, 0.94, 1.65, 78.0),
        (0.11, -0.04, 0.075, 0.21, -16.0, 0.94, 1.65, 78.0),
    ]
    for cx_v, cy_v, a_v, b_v, th_v, mv, sv, ev in ventricles:
        th = math.radians(th_v)
        xr = (xx - cx_v) * math.cos(th) + (yy - cy_v) * math.sin(th)
        yr = -(xx - cx_v) * math.sin(th) + (yy - cy_v) * math.cos(th)
        ind = ((xr / a_v) ** 2 + (yr / b_v) ** 2) <= 1.0
        m_mag[ind] = mv
        sigma[ind] = sv
        eps_r[ind] = ev

    edema_dist = ((xx - 0.36) / 0.165) ** 2 + ((yy + 0.16) / 0.155) ** 2
    edema_mask = (edema_dist <= 1.0) & wm_mask
    tumor_dist = ((xx - 0.36) / 0.125) ** 2 + ((yy + 0.16) / 0.120) ** 2
    tumor_mask = tumor_dist <= 1.0

    cmode = contrast_mode.lower().strip()
    if cmode == "sigma_only":
        m_mag[edema_mask] = 0.61
        sigma[edema_mask] = 0.72
        eps_r[edema_mask] = 60.0
        m_mag[tumor_mask] = 0.61
        sigma[tumor_mask] = 1.22
        eps_r[tumor_mask] = 68.0
    elif cmode == "mag_only":
        m_mag[edema_mask] = 0.78
        sigma[edema_mask] = 0.34
        eps_r[edema_mask] = 40.0
        m_mag[tumor_mask] = 1.00
        sigma[tumor_mask] = 0.34
        eps_r[tumor_mask] = 40.0
    else:
        m_mag[edema_mask] = 0.78
        sigma[edema_mask] = 0.72
        eps_r[edema_mask] = 60.0
        m_mag[tumor_mask] = 1.00
        sigma[tumor_mask] = 1.22
        eps_r[tumor_mask] = 68.0

    sig_disp, eps_disp = _gabriel_dispersion_scale(b0_tesla, ref_b0=3.0)
    sigma[tissue_mask > 0.5] *= sig_disp
    eps_r[tissue_mask > 0.5] = np.clip(eps_r[tissue_mask > 0.5] * eps_disp, 1.0, 100.0)

    m_mag = _gaussian_filter_2d(m_mag, sigma=1.05)
    sigma_smooth = _gaussian_filter_2d(sigma, sigma=1.10)
    eps_r_smooth = _gaussian_filter_2d(eps_r, sigma=1.10)

    mu_0 = 4.0 * math.pi * 1e-7
    f_0 = 42.58e6 * b0_tesla
    omega = 2.0 * math.pi * f_0
    dx_m = fov_x_m / width
    dy_m = fov_y_m / height

    rhs_poisson = (mu_0 * omega * sigma_smooth * tissue_mask).astype(np.float32)
    phi_plus_raw = _solve_poisson_dirichlet_2d(rhs_poisson, dx_m=dx_m, dy_m=dy_m)
    phi_tr_raw = (2.0 * phi_plus_raw).astype(np.float32)

    x_m = xx * (fov_x_m / 2.0)
    y_m = yy * (fov_y_m / 2.0)
    harm_basis = np.stack(
        [np.ones_like(x_m), x_m, y_m, x_m ** 2 - y_m ** 2, x_m * y_m], axis=-1
    )
    ring_mask = (tissue_mask > 0.5) & (((xx / 0.58) ** 2 + (yy / 0.70) ** 2) > 1.0)
    harm_coeffs, _, _, _ = np.linalg.lstsq(harm_basis[ring_mask], phi_tr_raw[ring_mask], rcond=None)
    phi_tr = (phi_tr_raw - np.tensordot(harm_basis, harm_coeffs, axes=([-1], [0]))).astype(np.float32)

    phi_b0 = (14.0 * (x_m ** 2 - y_m ** 2) + 10.0 * (x_m * y_m) + 0.35 * x_m - 0.25 * y_m).astype(np.float32)
    phi_total = phi_tr + phi_b0

    coil_maps = np.zeros((num_coils, height, width), dtype=np.complex64)
    radius_coil = 1.35
    for c in range(num_coils):
        angle = 2.0 * math.pi * c / num_coils
        cx_c = radius_coil * math.cos(angle)
        cy_c = radius_coil * math.sin(angle)
        dist_sq = (xx - cx_c) ** 2 + (yy - cy_c) ** 2
        amp = np.exp(-0.85 * dist_sq)
        phase_c = 0.65 * ((xx - cx_c) * math.sin(angle) - (yy - cy_c) * math.cos(angle))
        coil_maps[c] = (amp * np.exp(1j * phase_c)).astype(np.complex64)

    common_coil_phase = np.angle(np.sum(coil_maps, axis=0, keepdims=True))
    coil_maps = coil_maps * np.exp(-1j * common_coil_phase)
    sos = np.sqrt(np.sum(np.abs(coil_maps) ** 2, axis=0, keepdims=True) + 1e-8)
    coil_maps = (coil_maps / sos).astype(np.complex64)

    sense_op = DifferentiableSENSE()
    m_mag_t = torch.from_numpy(m_mag).to(device=device, dtype=torch.float32)
    phi_tr_t = torch.from_numpy(phi_tr).to(device=device, dtype=torch.float32)
    phi_b0_t = torch.from_numpy(phi_b0).to(device=device, dtype=torch.float32)
    phi_total_t = torch.from_numpy(phi_total).to(device=device, dtype=torch.float32)
    sigma_t = torch.from_numpy(sigma_smooth).to(device=device, dtype=torch.float32)
    eps_r_t = torch.from_numpy(eps_r_smooth).to(device=device, dtype=torch.float32)
    tissue_mask_t = torch.from_numpy(tissue_mask).to(device=device, dtype=torch.float32)
    coil_maps_t = torch.from_numpy(coil_maps).to(device=device, dtype=torch.complex64)

    m_complex_gt = torch.polar(m_mag_t, phi_total_t)
    kspace_clean = sense_op.forward(m_complex_gt, coil_maps_t, mask=None)

    signal_power = torch.mean(torch.abs(kspace_clean) ** 2).item()
    snr_linear = 10.0 ** (noise_snr_db / 10.0)
    noise_std = math.sqrt(signal_power / max(snr_linear, 1e-6) / 2.0)

    noise_real = rng.normal(0.0, noise_std, size=kspace_clean.shape).astype(np.float32)
    noise_imag = rng.normal(0.0, noise_std, size=kspace_clean.shape).astype(np.float32)
    noise_t = torch.complex(
        torch.from_numpy(noise_real).to(device=device),
        torch.from_numpy(noise_imag).to(device=device),
    )
    kspace_full = kspace_clean + noise_t

    mask_t = create_sampling_mask(
        height=height,
        width=width,
        mask_type=mask_type,
        acceleration=acceleration,
        acs_lines=acs_lines,
        seed=seed,
        device=device,
    )
    kspace_masked = kspace_full * mask_t.unsqueeze(0)
    coil_est = sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
    zero_filled = sense_op.adjoint(kspace_masked, coil_maps=coil_est)

    wb_gt, q_met_gt, rho_gt, tumor_mask_gt, vent_mask_gt = _build_physiological_perfusion_and_rois(
        sigma_t, tissue_mask_t, b0_tesla=b0_tesla
    )

    return SyntheticPhantomDataset(
        kspace_full=kspace_full,
        kspace_masked=kspace_masked,
        mask=mask_t,
        zero_filled=zero_filled,
        coil_maps_gt=coil_maps_t,
        m_mag_gt=m_mag_t,
        phi_tr_gt=phi_tr_t,
        phi_b0_gt=phi_b0_t,
        phi_total_gt=phi_total_t,
        sigma_gt=sigma_t,
        eps_r_gt=eps_r_t,
        tissue_mask_gt=tissue_mask_t,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        b0_tesla=b0_tesla,
        perfusion_wb_gt=wb_gt,
        q_met_gt=q_met_gt,
        rho_gt=rho_gt,
        tumor_mask_gt=tumor_mask_gt,
        ventricle_mask_gt=vent_mask_gt,
    )


def load_h5_kspace_slice(
    h5_path: str | Path,
    slice_idx: Optional[int] = None,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    seed: int = 42,
    device: torch.device | str = "cuda",
) -> SyntheticPhantomDataset:
    """Loads a 2D multi-coil k-space slice from a fastMRI/UC Berkeley .h5 file and attaches physiological bioheat maps."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f".h5 file not found at: {path}")

    with h5py.File(path, "r") as hf:
        if "kspace" not in hf:
            available = list(hf.keys())
            raise KeyError(f"File {path.name} missing 'kspace' key. Keys: {available}")

        raw_kspace = hf["kspace"][:]
        if raw_kspace.ndim == 4 and np.iscomplexobj(raw_kspace):
            s_idx = slice_idx if slice_idx is not None else raw_kspace.shape[0] // 2
            slice_k = raw_kspace[s_idx]
        elif raw_kspace.ndim == 3 and np.iscomplexobj(raw_kspace):
            slice_k = raw_kspace
        elif raw_kspace.ndim >= 4 and raw_kspace.shape[-1] == 2:
            s_idx = slice_idx if slice_idx is not None else raw_kspace.shape[0] // 2
            sub = raw_kspace[s_idx]
            cplx = sub[..., 0] + 1j * sub[..., 1]
            slice_k = np.moveaxis(cplx, -1, 0) if cplx.shape[0] > cplx.shape[-1] else cplx
        else:
            raise ValueError(f"Unrecognized 'kspace' shape: {raw_kspace.shape}")

        sigma_ref = hf["sigma_gt"][:] if "sigma_gt" in hf else None
        eps_ref = hf["eps_r_gt"][:] if "eps_r_gt" in hf else None
        m_mag_ref = hf["m_mag_gt"][:] if "m_mag_gt" in hf else None
        phi_tr_ref = hf["phi_tr_gt"][:] if "phi_tr_gt" in hf else None
        phi_b0_ref = hf["phi_b0_gt"][:] if "phi_b0_gt" in hf else None
        tissue_ref = hf["tissue_mask_gt"][:] if "tissue_mask_gt" in hf else None
        stored_b0 = float(hf.attrs.get("b0_tesla", 3.0))

    slice_k = slice_k.astype(np.complex64)
    nc, h, w = slice_k.shape
    sense_op = DifferentiableSENSE()

    kspace_full = torch.from_numpy(slice_k).to(device=device, dtype=torch.complex64)
    coil_est_full = sense_op.estimate_initial_coil_maps(kspace_full, acs_width=acs_lines)
    ref_complex = sense_op.adjoint(kspace_full, coil_maps=coil_est_full)

    if m_mag_ref is None:
        p995_mag = torch.quantile(torch.abs(ref_complex).flatten(), 0.995).clamp(min=1e-8)
        kspace_full = kspace_full / p995_mag
        ref_complex = ref_complex / p995_mag
        m_mag_gt = torch.clamp(torch.abs(ref_complex), 0.0, 1.0)
    else:
        m_mag_gt = torch.from_numpy(m_mag_ref.astype(np.float32)).to(device=device)

    phi_total_gt = torch.angle(ref_complex)
    if tissue_ref is not None:
        tissue_mask_gt = torch.from_numpy(tissue_ref.astype(np.float32)).to(device=device)
    else:
        tissue_mask_gt = (m_mag_gt > 0.1 * m_mag_gt.max()).float()

    sig_disp, eps_disp = _gabriel_dispersion_scale(b0_tesla, ref_b0=stored_b0)
    if sigma_ref is not None:
        sigma_gt = torch.from_numpy(sigma_ref.astype(np.float32)).to(device=device)
        if abs(sig_disp - 1.0) > 1e-3:
            sigma_gt = sigma_gt * (tissue_mask_gt * sig_disp + (1.0 - tissue_mask_gt))
    else:
        sigma_gt = torch.full((h, w), 0.50 * sig_disp, device=device, dtype=torch.float32) * tissue_mask_gt

    if eps_ref is not None:
        eps_r_gt = torch.from_numpy(eps_ref.astype(np.float32)).to(device=device)
        if abs(eps_disp - 1.0) > 1e-3:
            eps_r_gt = torch.clamp(eps_r_gt * (tissue_mask_gt * eps_disp + (1.0 - tissue_mask_gt)), 1.0, 100.0)
    else:
        eps_r_gt = torch.full((h, w), 50.0 * eps_disp, device=device, dtype=torch.float32) * tissue_mask_gt

    b0_scale = (float(b0_tesla) / max(stored_b0, 1e-3)) * sig_disp
    phi_tr_gt = (
        torch.from_numpy(phi_tr_ref.astype(np.float32)).to(device=device) * b0_scale
        if phi_tr_ref is not None
        else phi_total_gt
    )
    phi_b0_gt = (
        torch.from_numpy(phi_b0_ref.astype(np.float32)).to(device=device)
        if phi_b0_ref is not None
        else torch.zeros_like(phi_total_gt)
    )

    if phi_tr_ref is not None and abs(b0_scale - 1.0) > 1e-3:
        phi_total_gt = phi_tr_gt + phi_b0_gt
        m_cplx_scaled = torch.polar(m_mag_gt, phi_total_gt)
        kspace_full = sense_op.forward(m_cplx_scaled, coil_est_full, mask=None)

    mask_t = create_sampling_mask(
        height=h,
        width=w,
        mask_type=mask_type,
        acceleration=acceleration,
        acs_lines=acs_lines,
        seed=seed,
        device=device,
    )
    kspace_masked = kspace_full * mask_t.unsqueeze(0)
    coil_est = sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
    zero_filled = sense_op.adjoint(kspace_masked, coil_maps=coil_est)

    wb_gt, q_met_gt, rho_gt, tumor_mask_gt, vent_mask_gt = _build_physiological_perfusion_and_rois(
        sigma_gt, tissue_mask_gt, b0_tesla=b0_tesla
    )

    return SyntheticPhantomDataset(
        kspace_full=kspace_full,
        kspace_masked=kspace_masked,
        mask=mask_t,
        zero_filled=zero_filled,
        coil_maps_gt=coil_est_full,
        m_mag_gt=m_mag_gt,
        phi_tr_gt=phi_tr_gt,
        phi_b0_gt=phi_b0_gt,
        phi_total_gt=phi_total_gt,
        sigma_gt=sigma_gt,
        eps_r_gt=eps_r_gt,
        tissue_mask_gt=tissue_mask_gt,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        b0_tesla=b0_tesla,
        perfusion_wb_gt=wb_gt,
        q_met_gt=q_met_gt,
        rho_gt=rho_gt,
        tumor_mask_gt=tumor_mask_gt,
        ventricle_mask_gt=vent_mask_gt,
    )


def generate_numerical_brain_phantom(
    height: int = 128,
    width: int = 128,
    num_coils: int = 8,
    fov_x_m: float = 0.24,
    fov_y_m: float = 0.24,
    b0_tesla: float = 3.0,
    noise_snr_db: float = 33.0,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    seed: int = 108,
    contrast_mode: str = "coupled",
    device: torch.device | str = "cuda",
) -> SyntheticPhantomDataset:
    """Generates anthropomorphic 2D brain phantom with sub-voxel parenchymal texture and correlated coil noise."""
    base_ds = generate_realistic_brain_ept_phantom(
        height=height,
        width=width,
        num_coils=num_coils,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        b0_tesla=b0_tesla,
        noise_snr_db=noise_snr_db,
        mask_type=mask_type,
        acceleration=acceleration,
        acs_lines=acs_lines,
        seed=seed,
        contrast_mode=contrast_mode,
        device=device,
    )
    rng = np.random.default_rng(seed)
    tex_raw = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    tex_smooth = _gaussian_filter_2d(tex_raw, sigma=1.8)
    tex_smooth = tex_smooth / (np.std(tex_smooth) + 1e-6)

    t_mask = base_ds.tissue_mask_gt.detach().cpu().numpy()
    m_mag_np = np.clip(
        base_ds.m_mag_gt.detach().cpu().numpy() * (1.0 + 0.030 * tex_smooth * t_mask),
        0.0,
        1.0,
    )
    sigma_np = np.clip(
        base_ds.sigma_gt.detach().cpu().numpy() * (1.0 + 0.020 * tex_smooth * t_mask),
        0.02,
        2.2,
    )

    m_mag_t = torch.from_numpy(m_mag_np).to(device=device, dtype=torch.float32)
    sigma_t = torch.from_numpy(sigma_np).to(device=device, dtype=torch.float32)
    m_complex_gt = torch.polar(m_mag_t, base_ds.phi_total_gt)

    sense_op = DifferentiableSENSE()
    kspace_clean = sense_op.forward(m_complex_gt, base_ds.coil_maps_gt, mask=None)

    signal_power = torch.mean(torch.abs(kspace_clean) ** 2).item()
    snr_linear = 10.0 ** (noise_snr_db / 10.0)
    noise_std = math.sqrt(signal_power / max(snr_linear, 1e-6) / 2.0)

    idx = np.arange(num_coils)
    cov_mat = (0.22 ** np.abs(idx[:, None] - idx[None, :])).astype(np.float32)
    chol = np.linalg.cholesky(cov_mat)

    z_real = rng.normal(0.0, noise_std, size=(num_coils, height * width)).astype(np.float32)
    z_imag = rng.normal(0.0, noise_std, size=(num_coils, height * width)).astype(np.float32)
    corr_real = (chol @ z_real).reshape(num_coils, height, width)
    corr_imag = (chol @ z_imag).reshape(num_coils, height, width)
    noise_t = torch.complex(
        torch.from_numpy(corr_real).to(device=device),
        torch.from_numpy(corr_imag).to(device=device),
    )

    kspace_full = kspace_clean + noise_t
    kspace_masked = kspace_full * base_ds.mask.unsqueeze(0)
    coil_est = sense_op.estimate_initial_coil_maps(kspace_masked, acs_width=acs_lines)
    zero_filled = sense_op.adjoint(kspace_masked, coil_maps=coil_est)

    return SyntheticPhantomDataset(
        kspace_full=kspace_full,
        kspace_masked=kspace_masked,
        mask=base_ds.mask,
        zero_filled=zero_filled,
        coil_maps_gt=base_ds.coil_maps_gt,
        m_mag_gt=m_mag_t,
        phi_tr_gt=base_ds.phi_tr_gt,
        phi_b0_gt=base_ds.phi_b0_gt,
        phi_total_gt=base_ds.phi_total_gt,
        sigma_gt=sigma_t,
        eps_r_gt=base_ds.eps_r_gt,
        tissue_mask_gt=base_ds.tissue_mask_gt,
        fov_x_m=fov_x_m,
        fov_y_m=fov_y_m,
        b0_tesla=b0_tesla,
        perfusion_wb_gt=base_ds.perfusion_wb_gt,
        q_met_gt=base_ds.q_met_gt,
        rho_gt=base_ds.rho_gt,
        tumor_mask_gt=base_ds.tumor_mask_gt,
        ventricle_mask_gt=base_ds.ventricle_mask_gt,
    )


generate_clinical_in_vivo_brain_phantom = generate_numerical_brain_phantom
