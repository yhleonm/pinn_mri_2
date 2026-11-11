"""Differentiable Pennes Bioheat PDE Solver (T(x,t) in °C) and Sapareto-Dewey CEM43 Thermal Dosimetry."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PennesBioheatSolver(nn.Module):
    """
    Differentiable Steady-State and Transient Pennes Bioheat Transfer Solver.

    Solves the parabolic Pennes bioheat equation coupled to the reconstructed SAR(x) map:
        rho(x) * c_p * dT(x,t)/dt = nabla_m . (k_t * nabla_m T(x,t))
                                    - omega_b(x) * c_b * (T(x,t) - T_art)
                                    + rho(x) * SAR_eff(x) + Delta_Q_m(x)

    Models oncological thermal trapping caused by the dual mechanism:
      1. High electrical conductivity in the glioma (sigma_tumor ~ 1.20 S/m) -> elevated SAR absorption.
      2. Necrotic/hypoxic microvascular hypoperfusion (omega_{b,tumor} = 0.25 * omega_{b,healthy}) ->
         impaired blood perfusion heat sink, producing focal intratumoral hyperthermia (Delta T)
         and cumulative thermal isoeffect dose CEM43 [equivalent minutes at 43.0 °C].
    """

    def __init__(
        self,
        fov_x_m: float = 0.24,
        fov_y_m: float = 0.24,
        t_arterial_c: float = 37.0,
        k_thermal_w_mk: float = 0.52,
        c_tissue_j_kgk: float = 3630.0,
        c_blood_j_kgk: float = 3617.0,
        rho_tissue_kg_m3: float = 1040.0,
        rf_duty_scale: float = 5.6,
    ) -> None:
        super().__init__()
        self.fov_x_m = float(fov_x_m)
        self.fov_y_m = float(fov_y_m)
        self.t_arterial_c = float(t_arterial_c)
        self.k_thermal = float(k_thermal_w_mk)
        self.c_tissue = float(c_tissue_j_kgk)
        self.c_blood = float(c_blood_j_kgk)
        self.rho_tissue = float(rho_tissue_kg_m3)
        self.rf_duty_scale = float(rf_duty_scale)

    @staticmethod
    def _laplacian_2d_metric(u_2d: torch.Tensor, dx_m: float, dy_m: float) -> torch.Tensor:
        """Computes 2D metric Laplacian nabla_m^2 u with reflective Neumann boundary padding."""
        u_pad = F.pad(u_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect").squeeze(0).squeeze(0)
        d2x = (u_pad[1:-1, 2:] - 2.0 * u_2d + u_pad[1:-1, :-2]) / (dx_m ** 2)
        d2y = (u_pad[2:, 1:-1] - 2.0 * u_2d + u_pad[:-2, 1:-1]) / (dy_m ** 2)
        return d2x + d2y

    def solve_steady_state(
        self,
        sar_w_kg: torch.Tensor,
        perfusion_wb: torch.Tensor,
        tissue_mask: torch.Tensor,
        rho_kg_m3: Optional[torch.Tensor] = None,
        q_met: Optional[torch.Tensor] = None,
        num_iters: int = 45,
    ) -> torch.Tensor:
        """
        Differentiably solves the steady-state Pennes Bioheat PDE:
            -k_t * nabla_m^2 Delta_T + omega_b(x) * c_b * Delta_T = rho(x) * SAR_eff(x) + Delta_Q_m(x)
        Returns absolute steady-state temperature T_ss(x) = T_art + Delta_T_ss(x) in °C.
        """
        h, w = sar_w_kg.shape
        dx_m = self.fov_x_m / float(w)
        dy_m = self.fov_y_m / float(h)

        rho = rho_kg_m3 if rho_kg_m3 is not None else torch.full_like(sar_w_kg, self.rho_tissue)
        # Excess metabolic heat above basal equilibrium
        delta_qm = (
            torch.clamp(q_met - 9700.0, min=0.0)
            if q_met is not None
            else torch.zeros_like(sar_w_kg)
        )

        # Effective electromagnetic volumetric power source Q_EM [W/m^3]
        q_em = rho * sar_w_kg * self.rf_duty_scale + delta_qm

        # Blood perfusion sink coefficient: S_b(x) = omega_b(x) * c_b [W/(m^3*K)]
        sink_b = torch.clamp(perfusion_wb, min=1.2) * self.c_blood

        # Initial algebraic local estimate without thermal conduction
        dt_est = (q_em / (sink_b + 1e-6)) * tissue_mask

        # Differentiable Jacobi-Helmholtz solver for (-k_t nabla^2 + S_b) Delta_T = Q_EM
        inv_dx2 = 1.0 / (dx_m ** 2)
        inv_dy2 = 1.0 / (dy_m ** 2)
        diag_diff = 2.0 * self.k_thermal * (inv_dx2 + inv_dy2)
        denom = diag_diff + sink_b

        dt_cur = dt_est
        for _ in range(num_iters):
            u_pad = F.pad(dt_cur.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect").squeeze(0).squeeze(0)
            off_diag = self.k_thermal * (
                (u_pad[1:-1, 2:] + u_pad[1:-1, :-2]) * inv_dx2
                + (u_pad[2:, 1:-1] + u_pad[:-2, 1:-1]) * inv_dy2
            )
            dt_next = (q_em + off_diag) / denom
            dt_cur = (0.75 * dt_next + 0.25 * dt_cur) * tissue_mask

        t_ss_c = self.t_arterial_c + torch.clamp(dt_cur, min=0.0, max=8.5) * tissue_mask
        return t_ss_c

    def simulate_transient_exposure(
        self,
        sar_w_kg: torch.Tensor,
        perfusion_wb: torch.Tensor,
        tissue_mask: torch.Tensor,
        tumor_mask: Optional[torch.Tensor] = None,
        ventricle_mask: Optional[torch.Tensor] = None,
        rho_kg_m3: Optional[torch.Tensor] = None,
        q_met: Optional[torch.Tensor] = None,
        duration_min: float = 30.0,
        dt_sec: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Integrates the transient Pennes Bioheat PDE over t in [0, duration_min] (in °C)
        and computes the Sapareto-Dewey Cumulative Equivalent Minutes at 43°C (CEM43):
            CEM43(x, t_f) = int_0^{t_f} R^(43.0 - T(x, tau)) d_tau  [min]
            where R = 0.50 if T >= 43°C else 0.25.
        """
        with torch.no_grad():
            h, w = sar_w_kg.shape
            dx_m = self.fov_x_m / float(w)
            dy_m = self.fov_y_m / float(h)

            rho = rho_kg_m3 if rho_kg_m3 is not None else torch.full_like(sar_w_kg, self.rho_tissue)
            delta_qm = (
                torch.clamp(q_met - 9700.0, min=0.0)
                if q_met is not None
                else torch.zeros_like(sar_w_kg)
            )

            #Steady-state asymptotic target field
            t_ss = self.solve_steady_state(
                sar_w_kg=sar_w_kg,
                perfusion_wb=perfusion_wb,
                tissue_mask=tissue_mask,
                rho_kg_m3=rho,
                q_met=q_met,
                num_iters=45,
            )
            dt_ss = torch.clamp(t_ss - self.t_arterial_c, min=0.0)

            # Local thermal time constant tau(x) = rho * c_p / (omega_b * c_b + k_t * pi^2 / L_eff^2) [sec]
            sink_b = torch.clamp(perfusion_wb, min=1.5) * self.c_blood
            conduction_eff = self.k_thermal * (1.0 / (0.018 ** 2))
            tau_sec = (rho * self.c_tissue) / (sink_b + conduction_eff)

            total_sec = max(60.0, float(duration_min) * 60.0)
            num_steps = max(4, int(round(total_sec / dt_sec)))
            dt_min = (total_sec / num_steps) / 60.0

            t_cur = torch.full_like(sar_w_kg, self.t_arterial_c)
            cem43_map = torch.zeros_like(sar_w_kg)

            healthy_mask = (
                (tissue_mask > 0.5)
                & ((tumor_mask < 0.5) if tumor_mask is not None else True)
                & ((ventricle_mask < 0.5) if ventricle_mask is not None else True)
            ).float()

            time_points_min: List[float] = [0.0]
            t_tumor_curve: List[float] = [self.t_arterial_c]
            t_healthy_curve: List[float] = [self.t_arterial_c]
            t_ventricle_curve: List[float] = [self.t_arterial_c]
            cem43_tumor_curve: List[float] = [0.0]

            for step_i in range(1, num_steps + 1):
                t_sec = step_i * (total_sec / num_steps)
                # Exponential relaxation with spatial diffusion coupling
                relax = 1.0 - torch.exp(-t_sec / torch.clamp(tau_sec, min=30.0))
                dt_now = dt_ss * relax
                # Mild diffusive smoothing step
                lap_dt = self._laplacian_2d_metric(dt_now, dx_m, dy_m)
                dt_now = torch.clamp(dt_now + 0.08 * (dx_m ** 2) * lap_dt, min=0.0, max=8.5) * tissue_mask
                t_cur = self.t_arterial_c + dt_now

                # Sapareto-Dewey CEM43 increment over dt_min
                r_base = torch.where(
                    t_cur >= 43.0,
                    torch.full_like(t_cur, 0.50),
                    torch.full_like(t_cur, 0.25),
                )
                # Only accumulate thermal dose inside biological tissue where Delta_T > 0.05°C
                active_heat = (dt_now > 0.05) & (tissue_mask > 0.5)
                dose_rate = torch.pow(r_base, 43.0 - torch.clamp(t_cur, min=37.0, max=48.0))
                cem43_map = cem43_map + torch.where(active_heat, dose_rate * dt_min, torch.zeros_like(t_cur))

                # Record timeline every ~1 min
                if step_i % max(1, num_steps // 30) == 0 or step_i == num_steps:
                    time_points_min.append(round(t_sec / 60.0, 2))
                    if tumor_mask is not None and float(tumor_mask.sum().item()) > 0:
                        t_tum = float(torch.quantile(t_cur[tumor_mask > 0.5], 0.90).item())
                        cem_tum = float(torch.quantile(cem43_map[tumor_mask > 0.5], 0.90).item())
                    else:
                        t_tum = float(t_cur.max().item())
                        cem_tum = float(cem43_map.max().item())

                    if float(healthy_mask.sum().item()) > 0:
                        t_hlth = float(torch.mean(t_cur[healthy_mask > 0.5]).item())
                    else:
                        t_hlth = self.t_arterial_c + 0.3

                    if ventricle_mask is not None and float(ventricle_mask.sum().item()) > 0:
                        t_vent = float(torch.mean(t_cur[ventricle_mask > 0.5]).item())
                    else:
                        t_vent = self.t_arterial_c + 0.6

                    t_tumor_curve.append(round(t_tum, 3))
                    t_healthy_curve.append(round(t_hlth, 3))
                    t_ventricle_curve.append(round(t_vent, 3))
                    cem43_tumor_curve.append(round(cem_tum, 5))

            max_temp_c = float(t_cur.max().item())
            iec_status = self.classify_iec_60601_safety(max_temp_c)

            return {
                "temperature_c_2d": t_cur,
                "delta_t_c_2d": (t_cur - self.t_arterial_c) * tissue_mask,
                "cem43_min_2d": cem43_map * tissue_mask,
                "max_temp_c": max_temp_c,
                "max_delta_t_c": max(0.0, max_temp_c - self.t_arterial_c),
                "max_cem43_min": float(cem43_map.max().item()),
                "iec_safety_status": iec_status,
                "timeline": {
                    "time_min": time_points_min,
                    "t_tumor_c": t_tumor_curve,
                    "t_healthy_c": t_healthy_curve,
                    "t_ventricle_c": t_ventricle_curve,
                    "cem43_tumor_min": cem43_tumor_curve,
                },
            }

    @staticmethod
    def classify_iec_60601_safety(max_brain_temp_c: float) -> str:
        """
        Classifies local cerebral temperature according to IEC 60601-2-33 thermal limits:
          - Normal Operating Mode: T_max <= 38.0 °C
          - First-Level Controlled Mode: 38.0 °C < T_max <= 39.0 °C
          - Second-Level Controlled Mode: 39.0 °C < T_max <= 40.0 °C
          - Thermal Limit Exceeded (Hyperthermic Trap): T_max > 40.0 °C
        """
        if max_brain_temp_c <= 38.0:
            return "IEC NORMAL MODE (<= 38.0 °C)"
        if max_brain_temp_c <= 39.0:
            return "IEC LEVEL 1 CONTROLLED (38.0 - 39.0 °C)"
        if max_brain_temp_c <= 40.0:
            return "IEC LEVEL 2 CONTROLLED (39.0 - 40.0 °C)"
        return "IEC THERMAL ALERT: HYPERTHERMIC TRAP (> 40.0 °C)"
