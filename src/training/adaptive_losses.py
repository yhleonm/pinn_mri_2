"""Sobolev H^2 + Convection-Reaction Loss with Wang, Teng & Perdikaris (2021) GradNorm Auto-Balancing."""

from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradNormWeighting:
    """
    Self-Adaptive Loss Balancing via Gradient Norm Ratios (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021).

    Updates the physics weights lambda_ept and lambda_harm every `update_every` epochs according to:
        hat_lambda_i = mean(|nabla_theta L_data|) / (mean(|nabla_theta (lambda_i * L_i)|) + eps) * lambda_i
        lambda_i^{(k)} = alpha * lambda_i^{(k-1)} + (1 - alpha) * clip(hat_lambda_i, min_val, max_val)
    """

    def __init__(
        self,
        init_lambda_ept: float = 0.06,
        init_lambda_harm: float = 0.012,
        update_every: int = 5,
        ema_alpha: float = 0.85,
        enabled: bool = True,
    ) -> None:
        self.lambda_ept = float(init_lambda_ept)
        self.lambda_harm = float(init_lambda_harm)
        self.init_lambda_ept = float(init_lambda_ept)
        self.init_lambda_harm = float(init_lambda_harm)
        self.update_every = max(1, int(update_every))
        self.ema_alpha = float(ema_alpha)
        self.enabled = bool(enabled)
        self.last_grad_ratio_ept: float = 1.0
        self.last_grad_ratio_harm: float = 1.0

    @staticmethod
    def _grad_norm(loss_term: torch.Tensor, params: List[nn.Parameter]) -> float:
        grads = torch.autograd.grad(
            outputs=loss_term,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        sq_sum = 0.0
        count = 0
        for g in grads:
            if g is not None:
                sq_sum += float(torch.sum(g.detach() ** 2).item())
                count += g.numel()
        if count == 0:
            return 1e-6
        return float((sq_sum ** 0.5) / (count ** 0.5) + 1e-8)

    def maybe_update(
        self,
        epoch: int,
        warmup_epochs: int,
        loss_data: torch.Tensor,
        loss_ept_raw: torch.Tensor,
        loss_harm_raw: torch.Tensor,
        shared_params: Dict[str, List[nn.Parameter]],
    ) -> Dict[str, float]:
        """Updates lambda_ept and lambda_harm every `update_every` epochs after warmup."""
        if not self.enabled or epoch < warmup_epochs:
            return {
                "lambda_ept": self.lambda_ept,
                "lambda_harm": self.lambda_harm,
                "grad_ratio_ept": self.last_grad_ratio_ept,
                "grad_ratio_harm": self.last_grad_ratio_harm,
            }

        if (epoch - warmup_epochs) % self.update_every == 0:
            g_data = self._grad_norm(loss_data, shared_params["data"])
            g_ept = self._grad_norm(loss_ept_raw, shared_params["ept"])
            g_harm = self._grad_norm(loss_harm_raw, shared_params["harm"])

            # Target balanced gradient norms relative to data fidelity
            target_ept = 0.065 * (g_data / max(g_ept, 1e-5))
            target_harm = 0.015 * (g_data / max(g_harm, 1e-5))

            clipped_ept = max(0.035, min(0.095, target_ept))
            clipped_harm = max(0.008, min(0.028, target_harm))

            self.lambda_ept = self.ema_alpha * self.lambda_ept + (1.0 - self.ema_alpha) * clipped_ept
            self.lambda_harm = self.ema_alpha * self.lambda_harm + (1.0 - self.ema_alpha) * clipped_harm
            self.last_grad_ratio_ept = float(g_data / max(g_ept * self.lambda_ept, 1e-7))
            self.last_grad_ratio_harm = float(g_data / max(g_harm * self.lambda_harm, 1e-7))

        return {
            "lambda_ept": self.lambda_ept,
            "lambda_harm": self.lambda_harm,
            "grad_ratio_ept": self.last_grad_ratio_ept,
            "grad_ratio_harm": self.last_grad_ratio_harm,
        }


class AdaptiveElectroThermalLoss(nn.Module):
    """
    Multi-Physics Composite Loss with Sobolev H^2 phase regularization, Exact Convection-Reaction EPT residual,
    Pennes Bioheat consistency, and Wang et al. (2021) GradNorm auto-adaptive weighting.
    """

    def __init__(
        self,
        lambda_kspace: float = 1.0,
        lambda_ept_sigma: float = 0.06,
        lambda_ept_eps: float = 0.015,
        lambda_coil_smooth: float = 0.02,
        lambda_harmonic_b0: float = 0.012,
        warmup_epochs: int = 12,
        b0_tesla: float = 3.0,
        fov_x_m: float = 0.24,
        grid_width: int = 128,
        use_gradnorm: bool = True,
        gradnorm_update_every: int = 5,
        gradnorm_ema_alpha: float = 0.85,
        use_convection_reaction: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_kspace = float(lambda_kspace)
        self.lambda_ept_sigma = float(lambda_ept_sigma)
        self.lambda_ept_eps = float(lambda_ept_eps)
        self.lambda_coil_smooth = float(lambda_coil_smooth)
        self.lambda_harmonic_b0 = float(lambda_harmonic_b0)
        self.warmup_epochs = int(warmup_epochs)
        self.b0_tesla = float(b0_tesla)
        self.use_convection_reaction = bool(use_convection_reaction)

        dx_m = float(fov_x_m) / float(grid_width)
        mu_0 = 4.0 * 3.141592653589793e-7
        omega = 2.0 * 3.141592653589793 * 42.58e6 * self.b0_tesla
        self.lap_to_sigma_scale = (dx_m ** 2) * 2.0 * mu_0 * omega

        self.gradnorm = GradNormWeighting(
            init_lambda_ept=self.lambda_ept_sigma,
            init_lambda_harm=self.lambda_harmonic_b0,
            update_every=gradnorm_update_every,
            ema_alpha=gradnorm_ema_alpha,
            enabled=use_gradnorm,
        )

    @staticmethod
    def _huber_tv_smoothness(field_2d: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
        dx = field_2d[..., :, 1:] - field_2d[..., :, :-1]
        dy = field_2d[..., 1:, :] - field_2d[..., :-1, :]
        tv_x = F.smooth_l1_loss(dx, torch.zeros_like(dx), beta=delta, reduction="mean")
        tv_y = F.smooth_l1_loss(dy, torch.zeros_like(dy), beta=delta, reduction="mean")
        return tv_x + tv_y

    @staticmethod
    def _spatial_tv_smoothness(field_2d: torch.Tensor) -> torch.Tensor:
        dy = field_2d[..., 1:, :] - field_2d[..., :-1, :]
        dx = field_2d[..., :, 1:] - field_2d[..., :, :-1]
        return torch.mean(dx ** 2) + torch.mean(dy ** 2)

    @staticmethod
    def _edge_aware_smoothness(
        prop_2d: torch.Tensor,
        anat_mag_2d: torch.Tensor,
        edge_alpha: float = 22.0,
    ) -> torch.Tensor:
        """Anatomical and dielectric edge-guided regularizer preserving sharp boundaries."""
        anat_det = anat_mag_2d.detach()
        prop_det = prop_2d.detach()
        gx_anat = torch.maximum(
            torch.abs(anat_det[:, 1:] - anat_det[:, :-1]),
            0.75 * torch.abs(prop_det[:, 1:] - prop_det[:, :-1]),
        )
        gy_anat = torch.maximum(
            torch.abs(anat_det[1:, :] - anat_det[:-1, :]),
            0.75 * torch.abs(prop_det[1:, :] - prop_det[:-1, :]),
        )
        wx = torch.exp(-edge_alpha * gx_anat)
        wy = torch.exp(-edge_alpha * gy_anat)

        dx_p = (prop_2d[:, 1:] - prop_2d[:, :-1]) ** 2
        dy_p = (prop_2d[1:, :] - prop_2d[:-1, :]) ** 2
        return torch.mean(wx * dx_p) + torch.mean(wy * dy_p)

    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        ept_outputs: Dict[str, torch.Tensor],
        epoch: int,
        shared_params: Optional[List[nn.Parameter]] = None,
        coil_acs_ref: Optional[torch.Tensor] = None,
        phi_b0_ref: Optional[torch.Tensor] = None,
        phi_tr_ref: Optional[torch.Tensor] = None,
        sense_ref_complex: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluates composite electro-thermal loss and updates GradNorm weights when active."""
        # 1. Complex k-space data consistency
        diff_k = (kspace_pred - kspace_target) * mask.unsqueeze(0)
        active_elements = torch.clamp(torch.sum(mask) * kspace_pred.shape[0], min=1.0)
        loss_kspace = torch.sum(torch.abs(diff_k) ** 2) / active_elements

        if sense_ref_complex is not None:
            m_mag_2d = model_outputs["m_mag_2d"]
            phi_tr_2d = model_outputs["phi_tr_2d"]
            phi_b0_2d = model_outputs["phi_b0_2d"]
            ref_mag = torch.abs(sense_ref_complex)
            fg_weight = (ref_mag > 0.08 * ref_mag.max()).float()
            bg_weight = 1.0 - fg_weight
            loss_spatial_mag = torch.mean((m_mag_2d - ref_mag) ** 2) + 0.5 * torch.mean(
                bg_weight * (m_mag_2d ** 2)
            )
            if phi_tr_ref is not None:
                ref_phase = torch.angle(sense_ref_complex)
                phase_total_pred = phi_tr_2d + phi_b0_2d
                loss_complex_phase = torch.sum(
                    (1.0 - torch.cos(phase_total_pred - ref_phase)) * fg_weight
                ) / torch.clamp(fg_weight.sum(), min=1.0)

                b0_norm = max(self.b0_tesla / 3.0, 1.0)
                diff_phi = (phi_tr_2d - phi_tr_ref) / b0_norm
                d2x_ref = phi_tr_ref[1:-1, 2:] - 2.0 * phi_tr_ref[1:-1, 1:-1] + phi_tr_ref[1:-1, :-2]
                d2y_ref = phi_tr_ref[2:, 1:-1] - 2.0 * phi_tr_ref[1:-1, 1:-1] + phi_tr_ref[:-2, 1:-1]
                sigma_ref_interior = torch.clamp(
                    (d2x_ref + d2y_ref) / max(self.lap_to_sigma_scale, 1e-6), 0.01, 2.5
                ).detach()

                sigma_aut_2d = ept_outputs["sigma_lap_autograd"].reshape_as(phi_tr_2d)[1:-1, 1:-1]
                loss_sobolev_h2 = torch.mean((sigma_aut_2d - sigma_ref_interior) ** 2)

                gx_diff = (phi_tr_2d[:, 1:] - phi_tr_2d[:, :-1]) - (
                    phi_tr_ref[:, 1:] - phi_tr_ref[:, :-1]
                )
                gy_diff = (phi_tr_2d[1:, :] - phi_tr_2d[:-1, :]) - (
                    phi_tr_ref[1:, :] - phi_tr_ref[:-1, :]
                )
                loss_grad_h1 = torch.mean(gx_diff ** 2) + torch.mean(gy_diff ** 2)

                loss_spatial_phase = (
                    0.45 * loss_complex_phase
                    + 0.85 * torch.mean(diff_phi ** 2)
                    + 0.65 * loss_sobolev_h2
                    + 0.15 * loss_grad_h1
                )
            else:
                ref_phase = torch.angle(sense_ref_complex)
                loss_spatial_phase = torch.sum(
                    (1.0 - torch.cos((phi_tr_2d + phi_b0_2d) - ref_phase)) * fg_weight
                ) / torch.clamp(fg_weight.sum(), min=1.0)
            loss_kspace = 0.25 * loss_kspace + 0.85 * loss_spatial_mag + 1.50 * loss_spatial_phase

        # 2. Receiver coil smoothness + ACS consistency
        coil_maps_2d = model_outputs["coil_maps_2d"]
        loss_coil_grad = self._spatial_tv_smoothness(coil_maps_2d.real) + self._spatial_tv_smoothness(
            coil_maps_2d.imag
        )
        if coil_acs_ref is not None:
            loss_coil_acs = torch.mean(torch.abs(coil_maps_2d - coil_acs_ref) ** 2)
            loss_coil = loss_coil_grad + 0.8 * loss_coil_acs
        else:
            loss_coil = loss_coil_grad

        # 3. Harmonic background phase constraint: ||nabla^2 phi_B0||_2^2
        r_harmonic = ept_outputs["r_harmonic"]
        loss_harmonic = torch.mean(r_harmonic ** 2)
        if phi_b0_ref is not None:
            phi_b0_2d = model_outputs["phi_b0_2d"]
            loss_harmonic = loss_harmonic + 0.5 * torch.mean((phi_b0_2d - phi_b0_ref) ** 2)

        # 4. Convection-Reaction (or Helmholtz) EPT PDE residual
        r_sigma = ept_outputs["r_sigma"]
        r_eps_rel = ept_outputs["r_eps_rel"]
        tissue_weight = ept_outputs["tissue_weight"]
        tissue_count = torch.clamp(torch.sum(tissue_weight), min=1.0)

        loss_ept_sigma = torch.sum(
            F.smooth_l1_loss(r_sigma, torch.zeros_like(r_sigma), reduction="none", beta=0.20)
        ) / tissue_count

        r_eps_scaled = r_eps_rel / 50.0
        loss_ept_eps = torch.sum(
            F.smooth_l1_loss(r_eps_scaled, torch.zeros_like(r_eps_scaled), reduction="none", beta=0.5)
        ) / tissue_count

        sigma_2d = model_outputs["sigma_2d"]
        eps_r_2d = model_outputs["eps_r_2d"]
        m_mag_2d = model_outputs["m_mag_2d"]
        tw_2d = tissue_weight.reshape_as(sigma_2d)
        bg_mask = 1.0 - tw_2d

        # In cr-PINN, use sharper edge-preserving TV so step edges at tumor and ventricles are not rounded
        tv_weight = 0.015 if self.use_convection_reaction else 0.06
        edge_alpha = 26.0 if self.use_convection_reaction else 14.0

        loss_huber_tv = 5e-4 * (
            self._huber_tv_smoothness(sigma_2d, delta=0.03)
            + 0.02 * self._huber_tv_smoothness(eps_r_2d, delta=1.5)
        )
        loss_ept_tv = (
            self._edge_aware_smoothness(sigma_2d, m_mag_2d, edge_alpha=edge_alpha)
            + 1e-4 * self._edge_aware_smoothness(eps_r_2d, m_mag_2d, edge_alpha=edge_alpha)
            + loss_huber_tv
        )
        loss_ept_bg = torch.mean(bg_mask * ((sigma_2d - 0.02) ** 2 + 1e-4 * (eps_r_2d - 1.0) ** 2))
        loss_ept_combined = loss_ept_sigma + tv_weight * loss_ept_tv + 0.25 * loss_ept_bg

        # 5. Self-adaptive GradNorm update (Wang et al., 2021) every 5 epochs
        if shared_params is not None and epoch >= self.warmup_epochs:
            gn_state = self.gradnorm.maybe_update(
                epoch=epoch,
                warmup_epochs=self.warmup_epochs,
                loss_data=loss_spatial_mag if "loss_spatial_mag" in locals() else loss_kspace,
                loss_ept_raw=loss_ept_sigma,
                loss_harm_raw=loss_harmonic,
                shared_params=shared_params,
            )
            eff_lambda_sigma = gn_state["lambda_ept"]
            eff_lambda_harm = gn_state["lambda_harm"]
            eff_lambda_eps = self.lambda_ept_eps * (eff_lambda_sigma / max(self.lambda_ept_sigma, 1e-6))
        elif epoch < self.warmup_epochs:
            eff_lambda_sigma = 0.0
            eff_lambda_eps = 0.0
            eff_lambda_harm = self.lambda_harmonic_b0
        else:
            eff_lambda_sigma = self.lambda_ept_sigma
            eff_lambda_eps = self.lambda_ept_eps
            eff_lambda_harm = self.lambda_harmonic_b0

        loss_pde_ept = eff_lambda_sigma * loss_ept_combined + eff_lambda_eps * loss_ept_eps

        total_loss = (
            self.lambda_kspace * loss_kspace
            + self.lambda_coil_smooth * loss_coil
            + eff_lambda_harm * loss_harmonic
            + loss_pde_ept
        )

        return {
            "total_loss": total_loss,
            "loss_kspace": loss_kspace.detach(),
            "loss_ept": (loss_ept_sigma + loss_ept_eps).detach(),
            "loss_ept_sigma": loss_ept_sigma.detach(),
            "loss_ept_eps": loss_ept_eps.detach(),
            "loss_coil": loss_coil.detach(),
            "loss_harmonic": loss_harmonic.detach(),
            "lambda_ept_active": torch.tensor(float(eff_lambda_sigma), device=total_loss.device),
            "lambda_harm_active": torch.tensor(float(eff_lambda_harm), device=total_loss.device),
            "ept_active": torch.tensor(1.0 if epoch >= self.warmup_epochs else 0.0, device=total_loss.device),
        }
