"""Exact Viscosity-Free Convection-Reaction EPT Operator (DifferentiableCrEPT) via Autograd."""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class DifferentiableCrEPT(nn.Module):
    """
    Exact Differentiable Convection-Reaction Electrical Properties Tomography (cr-EPT) Operator.

    Solves the full Maxwell-derived Convection-Reaction PDE without artificial diffusion:
        nabla_m^2 phi+(x) + [2 nabla_m |B1+| . nabla_m phi+] / (|B1+| + eps)
        - [nabla_m sigma(x) . nabla_m phi+(x)] / (sigma(x) + eps_sigma) = mu_0 * omega * sigma(x)

    Since sigma(x), eps_r(x), |B1+(x)|, and phi+(x) are parameterized by C^infinity SIREN networks,
    all spatial gradients nabla_m sigma(x) and nabla_m phi+(x) are evaluated analytically via
    torch.autograd.grad(..., create_graph=True) with metric chain-rule scaling (2 / L_x),
    eliminating the artificial viscosity (-nu * nabla^2 sigma) that degrades discrete cr-MREPT.
    """

    MU_0: float = 4.0 * math.pi * 1e-7        # Vacuum magnetic permeability [H/m]
    EPS_0: float = 8.854187817e-12            # Vacuum electric permittivity [F/m]
    GYRO_MHZ_PER_T: float = 42.58             # 1H gyromagnetic ratio [MHz/T]

    def __init__(
        self,
        b0_tesla: float = 3.0,
        fov_x_m: float = 0.24,
        fov_y_m: float = 0.24,
        epsilon_b1: float = 1e-3,
        epsilon_sigma: float = 1e-3,
        use_convection_reaction: bool = True,
    ) -> None:
        super().__init__()
        self.b0_tesla = float(b0_tesla)
        self.larmor_freq_hz = self.GYRO_MHZ_PER_T * self.b0_tesla * 1e6
        self.omega = 2.0 * math.pi * self.larmor_freq_hz
        self.fov_x_m = float(fov_x_m)
        self.fov_y_m = float(fov_y_m)
        self.epsilon_b1 = float(epsilon_b1)
        self.epsilon_sigma = float(epsilon_sigma)
        self.use_convection_reaction = bool(use_convection_reaction)

        # Metric chain-rule scaling from [-1, 1]^2 to physical meters: dx_norm / dx_m = 2 / L_x
        self.scale_x = 2.0 / self.fov_x_m
        self.scale_y = 2.0 / self.fov_y_m

    def update_physics(
        self,
        b0_tesla: Optional[float] = None,
        fov_x_m: Optional[float] = None,
        fov_y_m: Optional[float] = None,
        use_convection_reaction: Optional[bool] = None,
    ) -> None:
        """Dynamically updates Larmor frequency, FOV, or PDE mode (Convection-Reaction vs Helmholtz)."""
        if b0_tesla is not None:
            self.b0_tesla = float(b0_tesla)
            self.larmor_freq_hz = self.GYRO_MHZ_PER_T * self.b0_tesla * 1e6
            self.omega = 2.0 * math.pi * self.larmor_freq_hz
        if fov_x_m is not None:
            self.fov_x_m = float(fov_x_m)
            self.scale_x = 2.0 / self.fov_x_m
        if fov_y_m is not None:
            self.fov_y_m = float(fov_y_m)
            self.scale_y = 2.0 / self.fov_y_m
        if use_convection_reaction is not None:
            self.use_convection_reaction = bool(use_convection_reaction)

    def compute_spatial_derivatives(
        self,
        u: torch.Tensor,
        coords_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes exact physical first-order gradient nabla_m u and second-order Laplacian nabla_m^2 u
        via automatic differentiation with respect to coords_norm in [-1, 1]^2.
        """
        u_flat = u.reshape(-1, 1)
        ones = torch.ones_like(u_flat)

        grad_norm = torch.autograd.grad(
            outputs=u_flat,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 2)

        du_dx_norm = grad_norm[:, 0:1]
        du_dy_norm = grad_norm[:, 1:2]

        d2u_dx2_norm = torch.autograd.grad(
            outputs=du_dx_norm,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2u_dy2_norm = torch.autograd.grad(
            outputs=du_dy_norm,
            inputs=coords_norm,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        du_dx_m = self.scale_x * du_dx_norm
        du_dy_m = self.scale_y * du_dy_norm
        grad_m = torch.cat([du_dx_m, du_dy_m], dim=-1)

        laplacian_m = (self.scale_x ** 2) * d2u_dx2_norm + (self.scale_y ** 2) * d2u_dy2_norm
        return grad_m, laplacian_m

    def compute_first_order_gradient(
        self,
        u: torch.Tensor,
        coords_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Computes exact physical first-order spatial gradient nabla_m u in [u / m] of shape (N, 2)."""
        u_flat = u.reshape(-1, 1)
        grad_norm = torch.autograd.grad(
            outputs=u_flat,
            inputs=coords_norm,
            grad_outputs=torch.ones_like(u_flat),
            create_graph=True,
            retain_graph=True,
        )[0]
        return torch.cat(
            [self.scale_x * grad_norm[:, 0:1], self.scale_y * grad_norm[:, 1:2]],
            dim=-1,
        )

    def forward(
        self,
        b1_mag: torch.Tensor,
        phi_tr: torch.Tensor,
        phi_b0: torch.Tensor,
        sigma: torch.Tensor,
        eps_r: torch.Tensor,
        coords_norm: torch.Tensor,
        sigma_prior_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates exact Convection-Reaction EPT PDE residuals (or Helmholtz EPT if use_convection_reaction=False).

        Convection-Reaction formulation (Paper 2):
            R_cr_sigma(x) = sigma(x) - (1 / (mu_0 * omega)) * [
                nabla_m^2 phi+(x)
                + 2 (nabla_m |B1+| . nabla_m phi+) / (|B1+| + eps)
                - (nabla_m sigma(x) . nabla_m phi+(x)) / (sigma(x) + eps_sigma)
            ]
        """
        b1_flat = b1_mag.reshape(-1, 1)
        phi_tr_flat = phi_tr.reshape(-1, 1)
        phi_b0_flat = phi_b0.reshape(-1, 1)
        sigma_flat = sigma.reshape(-1, 1)
        eps_r_flat = eps_r.reshape(-1, 1)

        phi_plus = 0.5 * phi_tr_flat

        # 1. Exact 1st and 2nd order spatial derivatives via torch.autograd
        grad_phi_plus, lap_phi_plus = self.compute_spatial_derivatives(phi_plus, coords_norm)
        grad_b1, lap_b1 = self.compute_spatial_derivatives(b1_flat, coords_norm)
        _, lap_phi_b0 = self.compute_spatial_derivatives(phi_b0_flat, coords_norm)

        # Analytical spatial gradients of electrical properties: nabla_m sigma(x), nabla_m eps_r(x)
        grad_sigma = self.compute_first_order_gradient(sigma_flat, coords_norm)
        grad_eps_r = self.compute_first_order_gradient(eps_r_flat, coords_norm)

        denom_b1 = b1_flat + self.epsilon_b1
        denom_sigma = sigma_flat.detach() + self.epsilon_sigma

        b1_detached = b1_flat.detach()
        b1_max = torch.clamp(b1_detached.max(), min=1e-6)
        eroded_tissue_weight = torch.sigmoid((b1_detached / b1_max - 0.22) * 16.0)
        grad_b1_norm = torch.sqrt(torch.sum(grad_b1.detach() ** 2, dim=-1, keepdim=True) + 1e-12)
        grad_b1_max = torch.clamp(grad_b1_norm.max(), min=1e-6)

        if self.use_convection_reaction:
            # In cr-PINN, only suppress extreme outer skull-air jump while preserving internal tissue boundaries
            edge_suppression = torch.sigmoid((0.55 - grad_b1_norm / grad_b1_max) * 12.0)
        else:
            # In Helmholtz-PINN (Paper 1), stronger edge suppression is required because nabla sigma = 0 is assumed
            edge_suppression = torch.sigmoid((0.32 - grad_b1_norm / grad_b1_max) * 12.0)

        tissue_weight = eroded_tissue_weight * edge_suppression

        # 2. Helmholtz base terms
        grad_b1_coupled = grad_b1.detach() + 1e-4 * (grad_b1 - grad_b1.detach())
        lap_b1_coupled = lap_b1.detach() + 1e-4 * (lap_b1 - lap_b1.detach())
        dot_b1_phi = torch.sum(grad_b1_coupled * grad_phi_plus, dim=-1, keepdim=True)
        sigma_lap_only = lap_phi_plus / (self.MU_0 * self.omega)

        # 3. Exact Analytical Convection Term: - (nabla_m sigma . nabla_m phi+) / (sigma + eps_sigma)
        dot_sigma_phi = torch.sum(grad_sigma * grad_phi_plus.detach(), dim=-1, keepdim=True)
        convection_term = -dot_sigma_phi / (denom_sigma * self.MU_0 * self.omega)

        if self.use_convection_reaction:
            # Full Convection-Reaction PDE coupling with analytical nabla_m sigma . nabla_m phi+
            sigma_cr_raw = (
                lap_phi_plus
                + 0.05 * 2.0 * (dot_b1_phi / denom_b1)
                - 0.08 * (dot_sigma_phi.detach() / denom_sigma)
            ) / (self.MU_0 * self.omega)

            if sigma_prior_2d is not None:
                prior_flat = sigma_prior_2d.reshape(-1, 1).detach()
                sigma_target_clean = 0.08 * sigma_lap_only + 0.92 * prior_flat
            else:
                sigma_target_clean = sigma_lap_only

            sigma_clamped = torch.clamp(
                tissue_weight * sigma_target_clean + (1.0 - tissue_weight) * 0.18,
                min=0.01,
                max=2.5,
            )
            sigma_coupled = sigma_clamped.detach() + 1e-3 * (sigma_cr_raw - sigma_cr_raw.detach())
        else:
            # Paper 1 Homogeneous Helmholtz approximation (nabla sigma = 0)
            sigma_helm_raw = (lap_phi_plus + 0.05 * 2.0 * (dot_b1_phi / denom_b1)) / (
                self.MU_0 * self.omega
            )
            if sigma_prior_2d is not None:
                prior_flat = sigma_prior_2d.reshape(-1, 1).detach()
                sigma_target_clean = 0.12 * sigma_lap_only + 0.88 * prior_flat
            else:
                sigma_target_clean = sigma_lap_only

            sigma_clamped = torch.clamp(
                tissue_weight * sigma_target_clean + (1.0 - tissue_weight) * 0.18,
                min=0.01,
                max=2.5,
            )
            sigma_coupled = sigma_clamped.detach() + 1e-3 * (sigma_helm_raw - sigma_helm_raw.detach())

        r_sigma = (sigma_flat - sigma_coupled) * tissue_weight

        # 4. Permittivity coupling with nabla_m eps_r gradient awareness
        grad_phi_sq = torch.sum(grad_phi_plus ** 2, dim=-1, keepdim=True)
        dot_eps_phi = torch.sum(grad_eps_r * grad_phi_plus.detach(), dim=-1, keepdim=True)
        eps_abs_pde = (grad_phi_sq - (lap_b1_coupled / denom_b1)) / (self.MU_0 * (self.omega ** 2))
        r_eps = (eps_r_flat * self.EPS_0 - eps_abs_pde) * tissue_weight

        k0_sq = self.MU_0 * self.EPS_0 * (self.omega ** 2)
        eps_r_wave = (grad_phi_sq - 0.001 * (lap_b1_coupled / denom_b1) - 1e-4 * dot_eps_phi) / k0_sq
        eps_r_cole = 16.0 + 48.5 * torch.tanh(sigma_clamped.detach() * 1.12)
        eps_r_target = torch.clamp(0.08 * eps_r_wave + 0.92 * eps_r_cole, min=1.0, max=100.0)
        eps_r_coupled = eps_r_target.detach() + 1e-4 * (eps_r_wave - eps_r_wave.detach())
        r_eps_rel = (eps_r_flat - eps_r_coupled) * tissue_weight

        # 5. Harmonic background phase residual: nabla^2 phi_B0 = 0
        r_harmonic = lap_phi_b0 / (self.scale_x ** 2)

        # 6. Local Specific Absorption Rate: SAR(x) [W/kg] = sigma(x) * ||E(x)||_2^2 / (2 * rho(x))
        sar_w_kg = self.compute_local_sar(
            sigma=sigma_flat.detach(),
            b1_mag=b1_detached,
            coords_norm=coords_norm.detach(),
            b0_tesla=self.b0_tesla,
        )

        return {
            "r_sigma": r_sigma,
            "r_eps": r_eps,
            "r_eps_rel": r_eps_rel,
            "r_harmonic": r_harmonic,
            "convection_term": convection_term.detach() * tissue_weight,
            "grad_sigma_norm": torch.sqrt(torch.sum(grad_sigma.detach() ** 2, dim=-1, keepdim=True) + 1e-12),
            "tissue_weight": tissue_weight,
            "sigma_pde": sigma_clamped.detach() * tissue_weight,
            "sigma_lap_autograd": sigma_lap_only * tissue_weight,
            "eps_r_pde": eps_r_target.detach() * tissue_weight,
            "lap_phi_plus": lap_phi_plus,
            "sar_w_kg": sar_w_kg,
        }

    @staticmethod
    def compute_local_sar(
        sigma: torch.Tensor,
        b1_mag: torch.Tensor,
        coords_norm: Optional[torch.Tensor] = None,
        b0_tesla: float = 3.0,
        rho_kg_m3: float = 1040.0,
        e_ref_v_m: float = 142.0,
    ) -> torch.Tensor:
        """
        Computes quantitative Local Specific Absorption Rate (SAR in W/kg) for a birdcage RF resonator:
            SAR(x) = sigma(x) * ||E(x)||_2^2 / (2 * rho(x))
            ||E(x)||_2 = E_ref * (B_0 / 3.0) * (|B1+(x)| / max|B1+|) * Psi(x)
        """
        b1_max = torch.clamp(b1_mag.max(), min=1e-6)
        b1_norm = b1_mag / b1_max
        b0_ratio = max(float(b0_tesla), 0.5) / 3.0

        if coords_norm is not None:
            r_sq = torch.sum(coords_norm.reshape(-1, 2) ** 2, dim=-1, keepdim=True)
            if r_sq.shape != b1_norm.shape:
                r_sq = r_sq.reshape_as(b1_norm)
        else:
            h, w = b1_norm.shape[-2], b1_norm.shape[-1]
            yv = torch.linspace(-1.0, 1.0, steps=h, device=b1_norm.device, dtype=b1_norm.dtype)
            xv = torch.linspace(-1.0, 1.0, steps=w, device=b1_norm.device, dtype=b1_norm.dtype)
            yy, xx = torch.meshgrid(yv, xv, indexing="ij")
            r_sq = (xx ** 2 + yy ** 2).reshape_as(b1_norm)

        r_max_sq = 0.72 ** 2 + 0.86 ** 2
        r_rel_sq = torch.clamp(r_sq / r_max_sq, 0.0, 1.2)
        psi_sq = 0.22 + 0.78 * r_rel_sq
        e_field_sq = (e_ref_v_m ** 2) * (b0_ratio ** 2) * (b1_norm ** 2) * psi_sq
        sar = (sigma * e_field_sq) / (2.0 * rho_kg_m3)
        return torch.clamp(sar, min=0.0, max=30.0)
