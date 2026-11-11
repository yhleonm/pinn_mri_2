"""4-Head Multi-Physics Coupled Electro-Thermal MRI-PINN Architecture (ElectroThermalMRIPINN)."""

from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .siren import SIRENNetwork


def bounded_softplus(
    x: torch.Tensor,
    low: float,
    high: float,
    center_offset: float = 0.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Doubly bounded C^infinity Softplus activation on [low, high]."""
    z = x + center_offset
    span = high - low
    return low + F.softplus(z, beta=beta) - F.softplus(z - span, beta=beta)


class AnatomyAndPhaseSIREN(nn.Module):
    """
    Heads 1 & 2: Continuous SIREN networks predicting:
      - |m(x)|: Spinor anatomical magnitude (omega_0 = 30 for sharp gyri/tumor/ventricle boundaries).
      - phi_tr(x): Transceiver RF phase 2*phi+(x) in Sobolev H^2 space.
      - phi_B0(x): Harmonic background susceptibility phase (nabla^2 phi_B0 = 0).
    """

    def __init__(
        self,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.mag_net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=0.5,
        )
        self.phase_tr_net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=3,
            out_features=1,
            first_omega_0=5.0,
            hidden_omega_0=5.0,
            final_scale=0.15,
        )
        self.b0_net = SIRENNetwork(
            in_features=2,
            hidden_features=32,
            hidden_layers=2,
            out_features=1,
            first_omega_0=2.5,
            hidden_omega_0=2.5,
            final_scale=0.15,
        )

    def forward(
        self, coords_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_mag = self.mag_net(coords_norm)
        phi_tr = self.phase_tr_net(coords_norm)
        m_mag = F.softplus(raw_mag - 0.35, beta=6.0)
        phi_b0 = self.b0_net(coords_norm)
        phi_total = phi_tr + phi_b0
        return m_mag, phi_tr, phi_b0, phi_total


class CoilSensitivityNet(nn.Module):
    """
    Head 3: Low-frequency SIREN (omega_0 = 5) predicting Nc complex receiver coil maps C_j(x)
    with exact Sum-of-Squares normalization (sum_j |C_j(x)|^2 = 1) and VBC phase gauge fixing.
    """

    def __init__(
        self,
        num_coils: int = 8,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_coils = int(num_coils)
        self.net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=2 * self.num_coils,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=1.0,
        )

    def forward(
        self, coords_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(coords_norm)
        raw_real = raw[..., : self.num_coils]
        raw_imag = raw[..., self.num_coils :]

        sos = torch.sqrt(
            torch.sum(raw_real ** 2 + raw_imag ** 2, dim=-1, keepdim=True) + 1e-8
        )
        coil_real = raw_real / sos
        coil_imag = raw_imag / sos
        coil_complex_raw = torch.complex(coil_real, coil_imag)

        common_sum = torch.sum(coil_complex_raw, dim=-1, keepdim=True)
        common_phase = torch.angle(common_sum)
        demod = torch.polar(torch.ones_like(common_phase), -common_phase)
        coil_complex = coil_complex_raw * demod
        return coil_complex, coil_complex.real, coil_complex.imag


class ElectroThermalPropertiesNet(nn.Module):
    """
    Head 4: Multi-physics SIREN head parameterizing:
      - Electrical conductivity: sigma(x) in [0.01, 3.0] S/m (C^infinity for exact nabla_m sigma).
      - Relative permittivity: eps_r(x) in [1.0, 100.0] dimensionless.
      - Steady-state bioheat increment: delta_t_net(x) in [0.0, 8.0] °C.
    """

    def __init__(
        self,
        hidden_features: int = 96,
        hidden_layers: int = 3,
        omega_0: float = 18.0,
        sigma_bounds: Tuple[float, float] = (0.01, 3.0),
        eps_r_bounds: Tuple[float, float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        self.sigma_min, self.sigma_max = float(sigma_bounds[0]), float(sigma_bounds[1])
        self.eps_min, self.eps_max = float(eps_r_bounds[0]), float(eps_r_bounds[1])

        self.net = SIRENNetwork(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=3,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
            final_scale=0.5,
        )

    def forward(
        self, coords_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(coords_norm)
        raw_sigma = out[..., 0:1]
        raw_eps = out[..., 1:2]
        raw_dt = out[..., 2:3]

        sigma = bounded_softplus(
            raw_sigma,
            low=self.sigma_min,
            high=self.sigma_max,
            center_offset=0.25,
            beta=2.0,
        )
        eps_r = bounded_softplus(
            raw_eps * 25.0,
            low=self.eps_min,
            high=self.eps_max,
            center_offset=50.0,
            beta=0.2,
        )
        delta_t_c = bounded_softplus(
            raw_dt,
            low=0.0,
            high=8.0,
            center_offset=0.2,
            beta=2.0,
        )
        return sigma, eps_r, delta_t_c


class ElectroThermalMRIPINN(nn.Module):
    """
    4-Head Multi-Physics Coupled Electro-Thermal PINN for Calibrationless SENSE Reconstruction,
    Exact Convection-Reaction EPT (nabla sigma != 0), and Pennes Bioheat Dosimetry.
    """

    def __init__(
        self,
        num_coils: int = 8,
        omega_anatomy: float = 30.0,
        omega_coil: float = 5.0,
        omega_ept: float = 18.0,
        hidden_features_anatomy: int = 128,
        hidden_layers_anatomy: int = 4,
        hidden_features_coil: int = 64,
        hidden_layers_coil: int = 3,
        hidden_features_ept: int = 96,
        hidden_layers_ept: int = 3,
        sigma_bounds: Tuple[float, float] = (0.01, 3.0),
        eps_r_bounds: Tuple[float, float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        self.num_coils = int(num_coils)

        self.anatomy_branch = AnatomyAndPhaseSIREN(
            hidden_features=hidden_features_anatomy,
            hidden_layers=hidden_layers_anatomy,
            omega_0=omega_anatomy,
        )
        self.coil_branch = CoilSensitivityNet(
            num_coils=num_coils,
            hidden_features=hidden_features_coil,
            hidden_layers=hidden_layers_coil,
            omega_0=omega_coil,
        )
        self.ept_branch = ElectroThermalPropertiesNet(
            hidden_features=hidden_features_ept,
            hidden_layers=hidden_layers_ept,
            omega_0=omega_ept,
            sigma_bounds=sigma_bounds,
            eps_r_bounds=eps_r_bounds,
        )

    def shared_gradnorm_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """Returns head projection parameters for Wang et al. (2021) gradient-norm balancing without double 2nd-order graph traversal."""
        return {
            "data": [self.anatomy_branch.mag_net.final_linear.weight],
            "ept": [self.ept_branch.net.final_linear.weight],
            "harm": [self.anatomy_branch.b0_net.final_linear.weight],
        }

    @staticmethod
    def create_normalized_grid(
        height: int,
        width: int,
        device: torch.device | str = "cuda",
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """Creates normalized spatial coordinate grid (x_norm, y_norm) in [-1, 1]^2 of shape (H*W, 2)."""
        y_vals = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x_vals = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing="ij")
        coords_norm = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        if requires_grad:
            coords_norm.requires_grad_(True)
        return coords_norm

    def forward(
        self,
        coords_norm: torch.Tensor,
        grid_shape: Tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluates all 4 physical heads on coords_norm (N, 2) in [-1, 1]^2."""
        m_mag, phi_tr, phi_b0, phi_total = self.anatomy_branch(coords_norm)
        coil_complex, coil_real, coil_imag = self.coil_branch(coords_norm)
        sigma, eps_r, delta_t_c = self.ept_branch(coords_norm)

        m_complex = torch.polar(m_mag.squeeze(-1), phi_total.squeeze(-1))

        outputs: Dict[str, torch.Tensor] = {
            "m_mag": m_mag,
            "phi_tr": phi_tr,
            "phi_b0": phi_b0,
            "phi_total": phi_total,
            "m_complex": m_complex,
            "coil_complex": coil_complex,
            "coil_real": coil_real,
            "coil_imag": coil_imag,
            "sigma": sigma,
            "eps_r": eps_r,
            "delta_t_c": delta_t_c,
        }

        if grid_shape is not None:
            h, w = grid_shape
            outputs["m_complex_2d"] = m_complex.reshape(h, w)
            outputs["coil_maps_2d"] = coil_complex.transpose(0, 1).reshape(self.num_coils, h, w)
            outputs["m_mag_2d"] = m_mag.reshape(h, w)
            outputs["phi_tr_2d"] = phi_tr.reshape(h, w)
            outputs["phi_b0_2d"] = phi_b0.reshape(h, w)
            outputs["sigma_2d"] = sigma.reshape(h, w)
            outputs["eps_r_2d"] = eps_r.reshape(h, w)
            outputs["delta_t_2d"] = delta_t_c.reshape(h, w)

        return outputs
