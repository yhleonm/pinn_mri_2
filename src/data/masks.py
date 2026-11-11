"""2D Sub-sampling Masks (Cartesian, Radial Golden-Angle, Poisson-Disc) with Autocalibration (ACS)."""

from __future__ import annotations
import math
import numpy as np
import torch


def _cartesian_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """1D variable-density Cartesian mask along phase-encoding columns."""
    if rng is None:
        rng = np.random.default_rng(42)

    mask = np.zeros((height, width), dtype=np.float32)
    center = width // 2
    target_lines = max(4, int(round(width / max(1, acceleration))))
    eff_acs = min(acs_lines, max(4, target_lines - 4))
    half_acs = max(2, eff_acs // 2)
    acs_start = max(0, center - half_acs)
    acs_end = min(width, center + half_acs)
    mask[:, acs_start:acs_end] = 1.0

    remaining_lines = max(0, target_lines - (acs_end - acs_start))

    if remaining_lines > 0:
        left_cols = np.arange(0, acs_start)
        right_cols = np.arange(acs_end, width)
        n_left = remaining_lines // 2
        n_right = remaining_lines - n_left

        for cols, n_pick, flip in ((left_cols, n_left, True), (right_cols, n_right, False)):
            if n_pick > 0 and len(cols) > 0:
                u = np.linspace(0.0, 1.0, n_pick) ** 0.85
                if flip:
                    raw_idx = len(cols) - 1 - np.round(u * (len(cols) - 1)).astype(int)
                else:
                    raw_idx = np.round(u * (len(cols) - 1)).astype(int)
                idx = np.unique(np.clip(raw_idx, 0, len(cols) - 1))
                if len(idx) < n_pick:
                    avail = np.setdiff1d(np.arange(len(cols)), idx)
                    extra = avail[: (n_pick - len(idx))]
                    idx = np.concatenate([idx, extra])
                mask[:, cols[idx[:n_pick]]] = 1.0

    return mask


def _radial_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
) -> np.ndarray:
    """Golden-Angle radial mask calibrated to exact 1/R sampling fraction with central ACS disc."""
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[:height, :width]
    target_fraction = 1.0 / max(1.0, float(acceleration))
    eff_acs = min(acs_lines, 18 if acceleration <= 4 else 12)
    r_acs = max(4.0, eff_acs / 2.0)
    acs_disc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_acs ** 2)

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    max_r = math.hypot(cy, cx)
    num_samples = int(max_r * 2.5)
    radii = np.linspace(-max_r, max_r, num_samples)

    best_mask = np.zeros((height, width), dtype=np.float32)
    best_diff = 1.0
    min_spokes = 6
    max_spokes = int(height * 1.6)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[acs_disc] = 1.0

    for s in range(max_spokes):
        theta = s * golden_angle
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        xs = np.round(cx + radii * cos_t).astype(int)
        ys = np.round(cy + radii * sin_t).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[valid], xs[valid]] = 1.0
        if s + 1 >= min_spokes:
            frac = float(mask.mean())
            diff = abs(frac - target_fraction)
            if diff < best_diff:
                best_diff = diff
                best_mask = mask.copy()
            if frac > target_fraction + 0.01:
                break

    return best_mask


def _poisson_disc_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    acs_lines: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """2D variable-density Poisson-Disc mask with central ACS window."""
    if rng is None:
        rng = np.random.default_rng(42)

    mask = np.zeros((height, width), dtype=np.float32)
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    r_norm = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    r_acs = max(4.0, acs_lines / 2.0)
    acs_region = (np.abs(yy - cy) <= r_acs) & (np.abs(xx - cx) <= r_acs)
    mask[acs_region] = 1.0

    target_total = int(round((height * width) / max(1, acceleration)))
    current_count = int(mask.sum())
    needed = max(0, target_total - current_count)

    if needed > 0:
        weights = np.clip((1.0 - 0.65 * np.clip(r_norm, 0.0, 1.4)) ** 2.2, 1e-4, 1.0)
        weights[acs_region] = 0.0
        flat_w = weights.ravel()
        flat_w /= flat_w.sum()
        chosen_idx = rng.choice(
            height * width,
            size=min(needed, int((~acs_region).sum())),
            replace=False,
            p=flat_w,
        )
        mask.ravel()[chosen_idx] = 1.0

    return mask


def create_sampling_mask(
    height: int,
    width: int,
    mask_type: str = "cartesian",
    acceleration: int = 4,
    acs_lines: int = 18,
    seed: int = 42,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Creates binary 2D k-space sampling mask P(k) on target device."""
    if acceleration <= 1:
        return torch.ones((height, width), dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    mtype = mask_type.lower().strip()

    if mtype == "cartesian":
        mask_np = _cartesian_mask(height, width, acceleration, acs_lines, rng)
    elif mtype == "radial":
        mask_np = _radial_mask(height, width, acceleration, acs_lines)
    elif mtype in ("poisson", "poisson_disc", "poisson-disc"):
        mask_np = _poisson_disc_mask(height, width, acceleration, acs_lines, rng)
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    return torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
