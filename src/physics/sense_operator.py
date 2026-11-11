"""Differentiable Multi-Coil SENSE Operator (Forward & Adjoint) with SVD + Virtual Body Coil (VBC)."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class DifferentiableSENSE(nn.Module):
    """
    Differentiable Forward (E) and Adjoint (E^H) multi-coil MRI SENSE operator
    with SVD electronic cable phase alignment and Virtual Body Coil (VBC) gauge fixing.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def fft2c(image: torch.Tensor) -> torch.Tensor:
        """Centered orthonormal 2D Fast Fourier Transform."""
        return torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(image, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

    @staticmethod
    def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
        """Centered orthonormal 2D Inverse Fast Fourier Transform."""
        return torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(kspace, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ),
            dim=(-2, -1),
        )

    def forward(
        self,
        m_complex: torch.Tensor,
        coil_maps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies Forward SENSE operator: s_hat_j(k) = P(k) * F_ortho{ C_j(x) * m(x) }."""
        if m_complex.ndim == 2:
            m_expanded = m_complex.unsqueeze(0)
        else:
            m_expanded = m_complex

        coil_images = coil_maps * m_expanded
        kspace_pred = self.fft2c(coil_images)

        if mask is not None:
            mask_expanded = mask.unsqueeze(0) if mask.ndim == 2 else mask
            kspace_pred = kspace_pred * mask_expanded

        return kspace_pred

    def adjoint(
        self,
        masked_kspace: torch.Tensor,
        coil_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies Adjoint SENSE operator: m_zf(x) = sum_j conj(C_j(x)) * F_ortho^{-1}{ P(k) * s_j(k) }."""
        coil_images = self.ifft2c(masked_kspace)

        if coil_maps is not None:
            return torch.sum(torch.conj(coil_maps) * coil_images, dim=-3)

        rss_mag = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=-3) + 1e-12)
        weighted_complex = torch.sum(coil_images * torch.abs(coil_images), dim=-3)
        phase = torch.angle(weighted_complex)
        return torch.polar(rss_mag, phase)

    def estimate_initial_coil_maps(
        self,
        kspace: torch.Tensor,
        acs_width: int = 20,
    ) -> torch.Tensor:
        """
        Estimates smooth reference coil sensitivity maps from central ACS band
        using 2D Hann windowing, SVD electronic cable dephasing alignment,
        Sum-of-Squares (SoS = 1) normalization, and Virtual Body Coil (VBC) common-mode gauge fixing.
        """
        nc, h, w = kspace.shape
        device = kspace.device

        win_y = torch.zeros(h, device=device, dtype=torch.float32)
        win_x = torch.zeros(w, device=device, dtype=torch.float32)
        cy, cx = h // 2, w // 2
        ry = min(acs_width, h // 2)
        rx = min(acs_width, w // 2)

        win_y[cy - ry : cy + ry] = torch.hann_window(2 * ry, device=device)
        win_x[cx - rx : cx + rx] = torch.hann_window(2 * rx, device=device)
        window_2d = torch.outer(win_y, win_x).unsqueeze(0)

        lowres_kspace = kspace * window_2d
        lowres_imgs = self.ifft2c(lowres_kspace)

        center_phases = torch.angle(lowres_imgs[:, cy, cx])
        phase_spread = torch.max(
            torch.abs(
                torch.atan2(
                    torch.sin(center_phases - center_phases[0]),
                    torch.cos(center_phases - center_phases[0]),
                )
            )
        )
        if float(phase_spread.item()) > 0.8:
            flat_low = lowres_imgs.reshape(nc, -1)
            cov = torch.matmul(flat_low, torch.conj(flat_low.T))
            _, eigvecs = torch.linalg.eigh(cov)
            dom_vec = eigvecs[:, -1]
            cable_phasor = torch.polar(torch.ones(nc, device=device), -torch.angle(dom_vec)).view(
                nc, 1, 1
            )
            lowres_aligned = lowres_imgs * cable_phasor
            common_sum = torch.sum(lowres_aligned, dim=0, keepdim=True)
        else:
            common_sum = torch.sum(lowres_imgs, dim=0, keepdim=True)

        common_phase = torch.angle(common_sum)
        phase_demod = torch.polar(torch.ones_like(common_phase), -common_phase)

        sos = torch.sqrt(torch.sum(torch.abs(lowres_imgs) ** 2, dim=0, keepdim=True) + 1e-8)
        return (lowres_imgs * phase_demod) / sos
