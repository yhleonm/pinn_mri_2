"""Sinusoidal Representation Networks (SIREN) with exact Sitzmann et al. (2020) initialization."""

from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """
    Periodic sinusoidal activation layer sin(omega_0 * (W x + b)) with exact Sitzmann et al. (2020)
    weight initialization to preserve C^infinity spatial derivatives of arbitrary order.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENNetwork(nn.Module):
    """Coordinate-based continuous SIREN MLP with linear output projection."""

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        out_features: int = 1,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        final_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.first_omega_0 = float(first_omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)

        layers: List[nn.Module] = [
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        self.trunk = nn.Sequential(*layers)

        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = (math.sqrt(6.0 / hidden_features) / hidden_omega_0) * final_scale
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

    def forward(self, coords_norm: torch.Tensor) -> torch.Tensor:
        features = self.trunk(coords_norm)
        return self.final_linear(features)
